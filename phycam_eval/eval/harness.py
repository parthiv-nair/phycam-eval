"""Backend-agnostic, schema-v3 detector evaluation orchestration.

The harness deliberately knows nothing about Torch, COCO, or a particular
detector.  User-supplied callables load an image, render one camera condition
and stochastic realization, preprocess the rendered frame, and infer on an
ordered batch.  The harness owns ordering, boundary validation, provenance
retention, and cardinality checks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Set
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .._canonical import canonical_sha256, freeze_json_value, json_value, positive_int
from ..domains import ColorSpace, DataMode, Domain
from ..frame import Frame
from .model_provenance import (
    validate_detector_execution_identity,
    validate_model_identity,
)
from .preprocess import DetectorInput
from .protocol import (
    camera_stage_graph_signature,
    embedded_integer_selection_identity,
    evaluation_source_record,
    ordered_integer_selection_identity,
    ordered_source_selection_identity,
    source_content_identity,
    target_annotation_identity,
    validate_camera_provenance,
    validate_embedded_integer_selection_identity,
    validate_ordered_source_selection_identity,
)

ITERATION_ORDER = ("image_id", "condition_id", "realization_id")


def _immutable_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    frozen = freeze_json_value(value)
    if not isinstance(frozen, MappingProxyType):
        raise TypeError(f"{label} must be a mapping")
    return frozen


def _ordered_ids(values: Iterable[int], *, kind: str) -> tuple[tuple[int, ...], Mapping[str, Any]]:
    if isinstance(values, (Mapping, Set)) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{kind} IDs require an explicitly ordered iterable")
    materialized = tuple(values)
    identity = embedded_integer_selection_identity(materialized, selection_kind=kind)
    normalized = tuple(int(value) for value in materialized)
    return normalized, _immutable_mapping(identity, label=f"{kind} selection identity")


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class _ConditionEntry:
    condition_sha256: str
    callback_value: Mapping[str, Any]
    record: Mapping[str, Any]


def _condition_selection_identity(values: Iterable[str]) -> Mapping[str, Any]:
    if isinstance(values, (Mapping, Set)) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError("condition hashes require an explicitly ordered iterable")
    hashes = tuple(
        _sha256(value, label=f"condition_hashes[{index}]") for index, value in enumerate(values)
    )
    if not hashes:
        raise ValueError("condition_sha256 selection must contain at least one value")
    if len(set(hashes)) != len(hashes):
        raise ValueError("condition_sha256 selection must not contain duplicates")
    payload = {
        "kind": "condition_sha256",
        "ordered_values": list(hashes),
        "encoding": "repository canonical bytes v1",
    }
    identity = {
        "kind": payload["kind"],
        "count": len(hashes),
        "first": hashes[0],
        "last": hashes[-1],
        "ordered_values": list(hashes),
        "selection_sha256": canonical_sha256(payload),
        "encoding": payload["encoding"],
    }
    return _immutable_mapping(identity, label="condition selection identity")


def _ordered_conditions(
    values: Iterable[Mapping[str, Any]],
) -> tuple[tuple[_ConditionEntry, ...], Mapping[str, Any]]:
    if isinstance(values, (Mapping, Set)) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError("conditions require an explicitly ordered iterable of embedded records")
    raw = tuple(values)
    if not raw:
        raise ValueError("condition_sha256 selection must contain at least one value")
    entries: list[_ConditionEntry] = []
    hashes: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise TypeError("current conditions require embedded condition records, not hashes")
        normalized = json_value(value)
        if normalized.get("schema_version") != 2:
            raise ValueError("embedded condition records require schema_version 2")
        digest = _sha256(
            normalized.get("condition_sha256"),
            label=f"condition_ids[{index}].condition_sha256",
        )
        payload = {key: item for key, item in normalized.items() if key != "condition_sha256"}
        if canonical_sha256(payload) != digest:
            raise ValueError("condition_sha256 does not match condition record")
        _sha256(
            normalized.get("fixed_profile_sha256"),
            label="fixed_profile_sha256",
        )
        try:
            DataMode(normalized.get("data_mode"))
        except (TypeError, ValueError) as exc:
            raise ValueError("condition record requires a supported data_mode") from exc
        bound_realizations = normalized.get("ordered_realization_ids")
        if not isinstance(bound_realizations, list):
            raise TypeError("condition record requires ordered_realization_ids")
        ordered_integer_selection_identity(
            bound_realizations,
            selection_kind="realization_id",
        )
        expected_selection_hash = canonical_sha256({"ordered_realization_ids": bound_realizations})
        if normalized.get("realization_selection_sha256") != expected_selection_hash:
            raise ValueError("realization_selection_sha256 does not match condition record")
        record = _immutable_mapping(normalized, label="condition record")
        entries.append(_ConditionEntry(digest, record, record))
        hashes.append(digest)
    return tuple(entries), _condition_selection_identity(hashes)


def _preprocessing_signature(value: Mapping[str, Any]) -> tuple[str, str]:
    normalized = json_value(value)
    supplied = normalized.get("preprocessing_sha256")
    payload = {key: item for key, item in normalized.items() if key != "preprocessing_sha256"}
    expected = canonical_sha256(payload)
    if supplied != expected:
        raise ValueError("preprocessing_sha256 does not match preprocessing contract")
    implementation = normalized.get("implementation_id")
    if not isinstance(implementation, str) or not implementation:
        raise ValueError("preprocessing contract requires an implementation_id")
    return supplied, implementation


def _frame_boundary_matches(frame: Frame, provenance: Mapping[str, Any]) -> None:
    descriptor = provenance.get("output_frame")
    if not isinstance(descriptor, Mapping):
        raise ValueError("camera provenance requires an output_frame descriptor")
    if canonical_sha256(descriptor) != canonical_sha256(frame.descriptor()):
        raise ValueError("camera output frame drifted from its provenance descriptor")


def _stage_graph_signature(provenance: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return camera_stage_graph_signature(provenance)


def _immutable_evaluation_payload(value: Any) -> tuple[Any, dict[str, Any]]:
    """Snapshot one supported source payload and describe its exact bytes."""

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("evaluation source arrays must not use object dtype")
        contiguous = np.ascontiguousarray(value)
        payload = contiguous.tobytes(order="C")
        immutable = np.frombuffer(payload, dtype=contiguous.dtype).reshape(contiguous.shape)
        return immutable, {
            "method": "numpy_c_order_array_bytes_sha256.v1",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, bytes):
        return value, {
            "method": "immutable_bytes_sha256.v1",
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    raise TypeError("evaluation source payload must be a NumPy array or immutable bytes")


def _provenance_realization_id(provenance: Mapping[str, Any], *, expected: int) -> int:
    """Bind a selected stochastic realization to renderer-owned provenance."""

    candidates: list[int] = []

    def append(value: object, *, label: str) -> None:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{label} must be an integer")
        candidates.append(int(value))

    if "realization_id" in provenance:
        append(provenance["realization_id"], label="camera provenance realization_id")
    for field in ("capture_condition", "renderer_capture_condition"):
        condition = provenance.get(field)
        if not isinstance(condition, Mapping):
            continue
        if "realization_id" in condition:
            append(condition["realization_id"], label=f"{field}.realization_id")
        if "realization" in condition:
            append(condition["realization"], label=f"{field}.realization")
    if not candidates:
        raise ValueError("camera provenance does not attest the selected realization_id")
    if any(value != candidates[0] for value in candidates):
        raise ValueError("camera provenance contains contradictory realization IDs")
    if candidates[0] != expected:
        raise ValueError("camera provenance realization_id does not match selected realization")
    return candidates[0]


@dataclass(frozen=True, slots=True)
class EvaluationSource:
    """One loaded image plus exact source-content and annotation identities.

    The arbitrary ``payload`` is passed to the camera renderer and is excluded
    from JSON serialization.  Its portable source record is included in both
    the ordered selection identity and every evaluation leaf.
    """

    image_id: int
    payload: Any
    source_identity: Mapping[str, Any]
    target_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        payload, payload_identity = _immutable_evaluation_payload(self.payload)
        record = evaluation_source_record(
            image_id=self.image_id,
            source_identity=self.source_identity,
            target_identity=self.target_identity,
        )
        if record["source_identity"]["content"].get("evaluation_payload") != payload_identity:
            raise ValueError("evaluation source payload bytes do not match source identity")
        object.__setattr__(self, "image_id", record["image_id"])
        object.__setattr__(self, "payload", payload)
        object.__setattr__(
            self,
            "source_identity",
            _immutable_mapping(record["source_identity"], label="source identity"),
        )
        object.__setattr__(
            self,
            "target_identity",
            _immutable_mapping(record["target_identity"], label="target identity"),
        )

    @classmethod
    def from_payloads(
        cls,
        *,
        image_id: int,
        payload: Any,
        source_content: Mapping[str, Any],
        target_annotations: Mapping[str, Any],
    ) -> "EvaluationSource":
        """Build self-addressed source and target records from portable payloads."""

        _, payload_identity = _immutable_evaluation_payload(payload)
        content = json_value(source_content)
        supplied_payload_identity = content.get("evaluation_payload")
        if supplied_payload_identity is not None and supplied_payload_identity != payload_identity:
            raise ValueError("source_content evaluation_payload does not match payload bytes")
        content["evaluation_payload"] = payload_identity
        return cls(
            image_id=image_id,
            payload=payload,
            source_identity=source_content_identity(content),
            target_identity=target_annotation_identity(target_annotations),
        )

    @property
    def record(self) -> Mapping[str, Any]:
        return _immutable_mapping(
            evaluation_source_record(
                image_id=self.image_id,
                source_identity=self.source_identity,
                target_identity=self.target_identity,
            ),
            label="evaluation source record",
        )


@dataclass(frozen=True, slots=True)
class CameraSample:
    """Rendered camera frame paired with its complete immutable provenance."""

    frame: Frame
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, Frame):
            raise TypeError("frame must be a Frame")
        if self.frame.domain is not Domain.DISPLAY_RGB:
            raise ValueError("camera output must use DISPLAY_RGB domain")
        if self.frame.color_space is not ColorSpace.SRGB:
            raise ValueError("camera output must use SRGB color space")
        if not np.issubdtype(self.frame.dtype, np.floating):
            raise TypeError("camera output must remain floating point")
        validated = validate_camera_provenance(self.provenance)
        _frame_boundary_matches(self.frame, validated)
        signature = _stage_graph_signature(validated)
        if signature[-1][4] != self.frame.metadata.units:
            raise ValueError("camera stage graph terminal units drifted from output frame")
        object.__setattr__(
            self,
            "provenance",
            _immutable_mapping(validated, label="camera provenance"),
        )


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One leaf of the image -> condition -> realization hierarchy."""

    ordinal: int
    image_ordinal: int
    condition_ordinal: int
    realization_ordinal: int
    image_id: int
    condition_id: str
    realization_id: int
    condition_record: Mapping[str, Any]
    source_identity: Mapping[str, Any]
    target_identity: Mapping[str, Any]
    camera_provenance: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    preprocessing_geometry: Mapping[str, Any]
    detector_output: Any

    def __post_init__(self) -> None:
        for name in (
            "ordinal",
            "image_ordinal",
            "condition_ordinal",
            "realization_ordinal",
        ):
            object.__setattr__(
                self,
                name,
                positive_int(getattr(self, name), field_name=name, allow_zero=True),
            )
        for name in ("image_id", "realization_id"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        condition_id = _sha256(self.condition_id, label="condition_id")
        object.__setattr__(self, "condition_id", condition_id)
        if not isinstance(self.condition_record, Mapping):
            raise TypeError("current evaluation records require an embedded condition record")
        condition_entries, _ = _ordered_conditions((self.condition_record,))
        normalized_condition = json_value(condition_entries[0].record)
        if normalized_condition.get("condition_sha256") != condition_id:
            raise ValueError("condition record does not match condition_id")
        object.__setattr__(
            self,
            "condition_record",
            _immutable_mapping(normalized_condition, label="condition record"),
        )
        source_record = evaluation_source_record(
            image_id=self.image_id,
            source_identity=self.source_identity,
            target_identity=self.target_identity,
        )
        object.__setattr__(
            self,
            "source_identity",
            _immutable_mapping(source_record["source_identity"], label="source identity"),
        )
        object.__setattr__(
            self,
            "target_identity",
            _immutable_mapping(source_record["target_identity"], label="target identity"),
        )
        camera_provenance = validate_camera_provenance(self.camera_provenance)
        if camera_provenance.get("capture_condition_sha256") != condition_id:
            raise ValueError("camera provenance does not match evaluation condition_id")
        condition = json_value(self.condition_record)
        if condition["fixed_profile_sha256"] != camera_provenance.get("camera_profile_sha256"):
            raise ValueError("condition is bound to a different camera profile")
        if condition["data_mode"] != camera_provenance.get("data_mode"):
            raise ValueError("condition data mode drifted from camera provenance")
        if self.realization_id not in condition["ordered_realization_ids"]:
            raise ValueError("record realization_id is absent from its condition binding")
        _provenance_realization_id(camera_provenance, expected=self.realization_id)
        object.__setattr__(
            self,
            "camera_provenance",
            _immutable_mapping(camera_provenance, label="camera provenance"),
        )
        _preprocessing_signature(self.preprocessing)
        object.__setattr__(
            self,
            "preprocessing",
            _immutable_mapping(self.preprocessing, label="preprocessing"),
        )
        object.__setattr__(
            self,
            "preprocessing_geometry",
            _immutable_mapping(
                self.preprocessing_geometry,
                label="preprocessing geometry",
            ),
        )
        object.__setattr__(self, "detector_output", freeze_json_value(self.detector_output))

    @property
    def hierarchy_key(self) -> tuple[int, str, int]:
        return self.image_id, self.condition_id, self.realization_id

    @property
    def condition_sha256(self) -> str:
        return self.condition_id

    @property
    def source_record_sha256(self) -> str:
        return evaluation_source_record(
            image_id=self.image_id,
            source_identity=self.source_identity,
            target_identity=self.target_identity,
        )["source_record_sha256"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "ordinal": self.ordinal,
            "image_ordinal": self.image_ordinal,
            "condition_ordinal": self.condition_ordinal,
            "realization_ordinal": self.realization_ordinal,
            "image_id": self.image_id,
            "condition_id": self.condition_id,
            "condition_sha256": self.condition_id,
            "realization_id": self.realization_id,
            "condition_record": json_value(self.condition_record),
            "source_identity": json_value(self.source_identity),
            "target_identity": json_value(self.target_identity),
            "source_record_sha256": self.source_record_sha256,
            "camera_provenance": json_value(self.camera_provenance),
            "preprocessing": json_value(self.preprocessing),
            "preprocessing_geometry": json_value(self.preprocessing_geometry),
            "detector_output": json_value(self.detector_output),
        }


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """Immutable result table bound to sources, camera, model, and execution."""

    records: tuple[EvaluationRecord, ...]
    image_selection: Mapping[str, Any]
    condition_selection: Mapping[str, Any]
    realization_selection: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    camera_profile_sha256: str
    detector_model: Mapping[str, Any]
    detector_execution: Mapping[str, Any]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if any(not isinstance(record, EvaluationRecord) for record in records):
            raise TypeError("records must contain EvaluationRecord values")
        if tuple(record.ordinal for record in records) != tuple(range(len(records))):
            raise ValueError("record ordinals must be contiguous and ordered")
        object.__setattr__(self, "records", records)
        validated_image_selection = validate_ordered_source_selection_identity(self.image_selection)
        validated_realization_selection = validate_embedded_integer_selection_identity(
            self.realization_selection,
            selection_kind="realization_id",
        )
        raw_condition_selection = json_value(self.condition_selection)
        validated_condition_selection = _condition_selection_identity(
            raw_condition_selection.get("ordered_values", ())
        )
        if raw_condition_selection != json_value(validated_condition_selection):
            raise ValueError("condition selection identity does not match ordered conditions")
        object.__setattr__(
            self,
            "image_selection",
            _immutable_mapping(validated_image_selection, label="image_selection"),
        )
        object.__setattr__(
            self,
            "realization_selection",
            _immutable_mapping(
                validated_realization_selection,
                label="realization_selection",
            ),
        )
        object.__setattr__(
            self,
            "condition_selection",
            _immutable_mapping(
                validated_condition_selection,
                label="condition_selection",
            ),
        )
        run_preprocessing_hash, _ = _preprocessing_signature(self.preprocessing)
        object.__setattr__(
            self,
            "preprocessing",
            _immutable_mapping(self.preprocessing, label="preprocessing"),
        )
        profile_hash = self.camera_profile_sha256
        if (
            not isinstance(profile_hash, str)
            or len(profile_hash) != 64
            or any(char not in "0123456789abcdef" for char in profile_hash)
        ):
            raise ValueError("camera_profile_sha256 must be a lowercase SHA-256 digest")
        model = validate_model_identity(self.detector_model)
        execution = validate_detector_execution_identity(
            self.detector_execution,
            model=model,
        )
        object.__setattr__(
            self,
            "detector_model",
            _immutable_mapping(model, label="detector model"),
        )
        object.__setattr__(
            self,
            "detector_execution",
            _immutable_mapping(execution, label="detector execution"),
        )
        source_by_id = {
            item["image_id"]: item for item in validated_image_selection["ordered_values"]
        }
        image_ids = tuple(source_by_id)
        condition_ids = tuple(raw_condition_selection["ordered_values"])
        realization_ids = tuple(validated_realization_selection["ordered_values"])
        expected_axes = tuple(
            (
                image_ordinal,
                condition_ordinal,
                realization_ordinal,
                image_id,
                condition_id,
                realization_id,
            )
            for image_ordinal, image_id in enumerate(image_ids)
            for condition_ordinal, condition_id in enumerate(condition_ids)
            for realization_ordinal, realization_id in enumerate(realization_ids)
        )
        observed_axes = tuple(
            (
                record.image_ordinal,
                record.condition_ordinal,
                record.realization_ordinal,
                record.image_id,
                record.condition_id,
                record.realization_id,
            )
            for record in records
        )
        if observed_axes != expected_axes:
            raise ValueError("evaluation records do not match the ordered Cartesian selection")
        graph_signatures: dict[str, tuple[tuple[Any, ...], ...]] = {}
        graph_hashes: dict[str, str] = {}
        for record in records:
            selected = source_by_id.get(record.image_id)
            if selected is None or record.source_record_sha256 != selected["source_record_sha256"]:
                raise ValueError("evaluation record source identity drifted from selection")
            if record.camera_provenance.get("camera_profile_sha256") != profile_hash:
                raise ValueError("evaluation record camera profile drifted from run identity")
            record_preprocessing_hash, _ = _preprocessing_signature(record.preprocessing)
            if record_preprocessing_hash != run_preprocessing_hash:
                raise ValueError("evaluation record preprocessing drifted from run identity")
            current_signature = _stage_graph_signature(record.camera_provenance)
            prior_signature = graph_signatures.setdefault(record.condition_id, current_signature)
            if current_signature != prior_signature:
                raise ValueError(
                    "camera stage graph implementation drifted within an evaluation condition"
                )
            current_graph_hash = record.camera_provenance["stage_graph_sha256"]
            prior_graph_hash = graph_hashes.setdefault(record.condition_id, current_graph_hash)
            if current_graph_hash != prior_graph_hash:
                raise ValueError(
                    "camera stage graph identity drifted within an evaluation condition"
                )
            bound_realizations = record.condition_record["ordered_realization_ids"]
            if tuple(bound_realizations) != realization_ids:
                raise ValueError("condition realization selection drifted from run selection")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 3,
            "iteration_order": list(ITERATION_ORDER),
            "image_selection": json_value(self.image_selection),
            "condition_selection": json_value(self.condition_selection),
            "realization_selection": json_value(self.realization_selection),
            "preprocessing": json_value(self.preprocessing),
            "camera_profile_sha256": self.camera_profile_sha256,
            "detector_model": json_value(self.detector_model),
            "detector_execution": json_value(self.detector_execution),
            "records": [record.to_dict() for record in self.records],
        }
        return {**payload, "evaluation_sha256": canonical_sha256(payload)}

    @property
    def evaluation_sha256(self) -> str:
        return self.to_dict()["evaluation_sha256"]


def _detector_outputs(value: Any, *, expected: int) -> tuple[Any, ...]:
    if isinstance(value, (Mapping, str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise TypeError("detector callable must return one ordered iterable of outputs")
    outputs = tuple(value)
    if len(outputs) != expected:
        raise ValueError(
            "detector output cardinality mismatch: "
            f"received {len(outputs)} outputs for {expected} inputs"
        )
    return outputs


def run_detector_evaluation(
    *,
    image_ids: Iterable[int],
    condition_ids: Iterable[Mapping[str, Any]],
    realization_ids: Iterable[int],
    load_image: Callable[[int], EvaluationSource],
    render_camera: Callable[
        [Any, int, Mapping[str, Any], int],
        CameraSample,
    ],
    preprocess_frame: Callable[[Frame], DetectorInput],
    detect_batch: Callable[[tuple[DetectorInput, ...]], Iterable[Any]],
    detector_model: Mapping[str, Any],
    detector_execution: Mapping[str, Any] | Callable[[], Mapping[str, Any]],
    batch_size: int = 1,
) -> EvaluationRun:
    """Evaluate an explicitly ordered Cartesian product through callable adapters.

    Iteration is image-major, then condition, then stochastic realization.  A
    detector batch is only a transport grouping; records always retain this
    normative leaf order.  Images are loaded once in declared order.
    """

    for name, operation in (
        ("load_image", load_image),
        ("render_camera", render_camera),
        ("preprocess_frame", preprocess_frame),
        ("detect_batch", detect_batch),
    ):
        if not callable(operation):
            raise TypeError(f"{name} must be callable")
    resolved_batch_size = positive_int(batch_size, field_name="batch_size")
    images, _ = _ordered_ids(image_ids, kind="image_id")
    conditions, condition_selection = _ordered_conditions(condition_ids)
    realizations, realization_selection = _ordered_ids(
        realization_ids,
        kind="realization_id",
    )
    validated_model = validate_model_identity(detector_model)
    if not isinstance(detector_execution, Mapping) and not callable(detector_execution):
        raise TypeError("detector_execution must be a mapping or zero-argument callable")
    if isinstance(detector_execution, Mapping):
        validate_detector_execution_identity(detector_execution, model=validated_model)

    pending: list[
        tuple[
            int,
            int,
            int,
            int,
            str,
            int,
            Mapping[str, Any],
            EvaluationSource,
            CameraSample,
            DetectorInput,
        ]
    ] = []
    records: list[EvaluationRecord] = []
    profile_hash: str | None = None
    graph_signatures: dict[str, tuple[tuple[Any, ...], ...]] = {}
    graph_hashes: dict[str, str] = {}
    source_records: list[Mapping[str, Any]] = []
    preprocessing_hash: str | None = None
    preprocessing_contract: Mapping[str, Any] | None = None

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        detector_inputs = tuple(item[-1] for item in pending)
        outputs = _detector_outputs(detect_batch(detector_inputs), expected=len(pending))
        for item, detector_output in zip(pending, outputs):
            (
                image_ordinal,
                condition_ordinal,
                realization_ordinal,
                image_id,
                condition_id,
                realization_id,
                condition_record,
                evaluation_source,
                camera_sample,
                detector_input,
            ) = item
            records.append(
                EvaluationRecord(
                    ordinal=len(records),
                    image_ordinal=image_ordinal,
                    condition_ordinal=condition_ordinal,
                    realization_ordinal=realization_ordinal,
                    image_id=image_id,
                    condition_id=condition_id,
                    realization_id=realization_id,
                    condition_record=condition_record,
                    source_identity=evaluation_source.source_identity,
                    target_identity=evaluation_source.target_identity,
                    camera_provenance=camera_sample.provenance,
                    preprocessing=detector_input.preprocessing,
                    preprocessing_geometry=detector_input.geometry.to_dict(),
                    detector_output=detector_output,
                )
            )
        pending = []

    for image_ordinal, image_id in enumerate(images):
        # Loading at the image-major boundary keeps only one source image live.
        # This matters for native-resolution datasets and does not change the
        # normative image -> condition -> realization iteration order.
        source = load_image(image_id)
        if not isinstance(source, EvaluationSource):
            raise TypeError("load_image must return an EvaluationSource")
        if source.image_id != image_id:
            raise ValueError("loaded EvaluationSource image_id drifted from selection")
        source_records.append(source.record)
        for condition_ordinal, condition in enumerate(conditions):
            for realization_ordinal, realization_id in enumerate(realizations):
                sample = render_camera(
                    source.payload,
                    image_id,
                    condition.callback_value,
                    realization_id,
                )
                if not isinstance(sample, CameraSample):
                    raise TypeError("render_camera must return a CameraSample")

                _provenance_realization_id(sample.provenance, expected=realization_id)
                rendered_condition = sample.provenance.get("capture_condition_sha256")
                if rendered_condition != condition.condition_sha256:
                    raise ValueError(
                        "rendered capture_condition_sha256 does not match selected condition"
                    )

                current_profile_hash = sample.provenance["camera_profile_sha256"]
                if profile_hash is None:
                    profile_hash = current_profile_hash
                elif current_profile_hash != profile_hash:
                    raise ValueError("camera profile identity drifted across evaluation samples")
                if condition.record["fixed_profile_sha256"] != current_profile_hash:
                    raise ValueError("condition is bound to a different camera profile")
                if condition.record["data_mode"] != sample.provenance["data_mode"]:
                    raise ValueError("condition data mode drifted from camera provenance")
                if tuple(condition.record["ordered_realization_ids"]) != realizations:
                    raise ValueError(
                        "condition realization selection drifted from harness selection"
                    )
                current_signature = _stage_graph_signature(sample.provenance)
                prior_signature = graph_signatures.setdefault(
                    condition.condition_sha256,
                    current_signature,
                )
                if current_signature != prior_signature:
                    raise ValueError(
                        "camera stage graph implementation drifted within an evaluation condition"
                    )
                current_graph_hash = sample.provenance["stage_graph_sha256"]
                prior_graph_hash = graph_hashes.setdefault(
                    condition.condition_sha256,
                    current_graph_hash,
                )
                if current_graph_hash != prior_graph_hash:
                    raise ValueError(
                        "camera stage graph identity drifted within an evaluation condition"
                    )

                detector_input = preprocess_frame(sample.frame)
                if not isinstance(detector_input, DetectorInput):
                    raise TypeError("preprocess_frame must return a DetectorInput")
                if detector_input.frame.metadata.data_mode is not sample.frame.metadata.data_mode:
                    raise ValueError("detector preprocessing changed the camera data mode")
                if detector_input.geometry.input_shape != sample.frame.shape[:2]:
                    raise ValueError("detector preprocessing geometry drifted from camera output")
                current_preprocessing_hash, _ = _preprocessing_signature(
                    detector_input.preprocessing
                )
                if preprocessing_hash is None:
                    preprocessing_hash = current_preprocessing_hash
                    preprocessing_contract = detector_input.preprocessing
                elif current_preprocessing_hash != preprocessing_hash:
                    raise ValueError("preprocessing identity drifted across evaluation samples")

                pending.append(
                    (
                        image_ordinal,
                        condition_ordinal,
                        realization_ordinal,
                        image_id,
                        condition.condition_sha256,
                        realization_id,
                        condition.record,
                        source,
                        sample,
                        detector_input,
                    )
                )
                if len(pending) == resolved_batch_size:
                    flush()
    flush()

    assert profile_hash is not None
    assert preprocessing_contract is not None
    image_selection = ordered_source_selection_identity(source_records)
    resolved_execution = (
        detector_execution() if callable(detector_execution) else detector_execution
    )
    validated_execution = validate_detector_execution_identity(
        resolved_execution,
        model=validated_model,
    )
    expected_count = len(images) * len(conditions) * len(realizations)
    if len(records) != expected_count:
        raise RuntimeError("evaluation record count violated the Cartesian product contract")
    return EvaluationRun(
        records=tuple(records),
        image_selection=image_selection,
        condition_selection=condition_selection,
        realization_selection=realization_selection,
        preprocessing=preprocessing_contract,
        camera_profile_sha256=profile_hash,
        detector_model=validated_model,
        detector_execution=validated_execution,
    )


__all__ = [
    "CameraSample",
    "EvaluationSource",
    "EvaluationRecord",
    "EvaluationRun",
    "ITERATION_ORDER",
    "run_detector_evaluation",
]
