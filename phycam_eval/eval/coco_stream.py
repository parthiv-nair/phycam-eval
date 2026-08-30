"""Streaming native-COCO LDR inference into deterministic prediction shards.

This runner deliberately bypasses the provenance-heavy Cartesian evaluation
harness. It decodes, renders, and preprocesses no more than one detector batch
at a time, immediately maps detector boxes back to stored native coordinates,
and retains only compact JSON prediction records until the shard is published.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .._canonical import canonical_sha256, json_value, positive_int
from ..capture import LDRCaptureSeverity, build_ldr_pipeline, make_ldr_input_frame
from ..domains import DataMode
from ..experiments.conditions import (
    BaselineCondition,
    BaselineKind,
    ExperimentCondition,
    MechanismComparatorCondition,
    to_ldr_capture_severity,
)
from ..frame import Frame
from ..profiles import CameraProfile
from .coco import (
    LazyNativeCOCOSubset,
    NativeCOCODataset,
    NativeCOCOSubset,
    detector_output_to_native_prediction,
)
from .metrics import yolo_to_coco_category_ids
from .model_provenance import validate_model_identity
from .preprocess import DetectorInput, LetterboxConfig, letterbox
from .shards import (
    PredictionShard,
    make_prediction_record,
    make_prediction_shard_header,
    validate_existing_prediction_shard,
    validate_prediction_shard,
    write_prediction_shard,
)

_LABEL_SPACES = {"coco_sparse", "coco80_contiguous"}
_NATIVE_ROI_SOURCE_ADAPTER = "native_active_sensor_roi_v1"
_COCO_MAXIMUM_DETECTIONS_PER_IMAGE = 100


def _publication_raw_category_ids(
    detector_output: Mapping[str, Any],
    *,
    image_id: int,
    label_space: str,
) -> np.ndarray:
    """Validate the detector-budget output before geometry can discard candidates."""

    if not isinstance(detector_output, Mapping):
        raise TypeError("detector_output must be a mapping")
    missing = {"boxes", "labels", "scores"}.difference(detector_output)
    if missing:
        raise ValueError(f"detector_output is missing keys: {sorted(missing)}")
    boxes = np.asarray(detector_output["boxes"])
    labels = np.asarray(detector_output["labels"])
    scores = np.asarray(detector_output["scores"])
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("detector boxes must have shape (N, 4)")
    if (
        labels.ndim != 1
        or scores.ndim != 1
        or len(labels) != len(boxes)
        or len(scores) != len(boxes)
    ):
        raise ValueError("detector labels and scores must have one entry per box")
    if len(boxes) > _COCO_MAXIMUM_DETECTIONS_PER_IMAGE:
        raise ValueError(
            f"raw detector prediction for image {image_id} has {len(boxes)} outputs; "
            f"COCO publication permits at most {_COCO_MAXIMUM_DETECTIONS_PER_IMAGE} per image"
        )
    if label_space == "coco80_contiguous":
        return yolo_to_coco_category_ids(labels)
    if label_space != "coco_sparse":
        raise ValueError("label_space must be 'coco_sparse' or 'coco80_contiguous'")
    try:
        numeric = labels.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("detector labels must contain positive integers") from exc
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric <= 0.0)
    ):
        raise ValueError("detector labels must contain positive integers")
    return numeric.astype(np.int64)


def _ordered_shard_image_ids(
    values: Sequence[int],
    *,
    available: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("image_ids must be an explicit ordered sequence")
    image_ids = tuple(
        positive_int(value, field_name=f"image_ids[{index}]", allow_zero=True)
        for index, value in enumerate(values)
    )
    if not image_ids:
        raise ValueError("image_ids must not be empty")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("image_ids must not contain duplicates")
    available_ids = set(available)
    unknown = [image_id for image_id in image_ids if image_id not in available_ids]
    if unknown:
        raise ValueError(f"shard image IDs are outside the native COCO dataset: {unknown}")
    return image_ids


def _condition_contract(
    *,
    condition: ExperimentCondition,
    profile: CameraProfile,
    preprocessing: LetterboxConfig,
    label_space: str,
) -> tuple[Mapping[str, Any], LDRCaptureSeverity | None, Any | None]:
    if not isinstance(condition, ExperimentCondition):
        raise TypeError("condition must be an ExperimentCondition")
    condition.binding.assert_profile(profile)
    if condition.data_mode is not DataMode.LDR_REDEGRADATION:
        raise ValueError("condition must be bound to LDR_REDEGRADATION mode")
    if condition.realization_ids != (0,):
        raise ValueError("deterministic LDR shards require exactly realization_ids=(0,)")

    untouched = (
        isinstance(condition, BaselineCondition) and condition.kind is BaselineKind.UNTOUCHED_INPUT
    )
    comparator_config = None
    comparator_contract = None
    if isinstance(condition, MechanismComparatorCondition):
        # Import lazily: the comparator package also exposes evaluation helpers,
        # so importing it while ``eval`` is being initialized would form a cycle.
        from ..comparators.capture import comparator_config_from_dict

        comparator_config = comparator_config_from_dict(condition.match["config"])
        severity = None
        render_mode = "mechanism_matched_comparator"
        geometry_policy = "profile_sensor_or_native_active_roi"
        comparator_contract = {
            "match_sha256": condition.match["match_sha256"],
            "comparator_family": condition.comparator_family,
            "target_edge_waves_ref": condition.target_edge_waves_ref,
            "config_sha256": condition.match["config_sha256"],
            "common_neutral_branch": "physical_W0_linear_rgb_optical",
            "insertion_boundary": "after_common_neutral_before_gamut_tone_encode",
            "comparator_is_physical_model": False,
        }
    elif untouched:
        severity = None
        render_mode = "untouched_input"
        geometry_policy = "stored_native_image"
    else:
        severity = to_ldr_capture_severity(condition, profile=profile)
        if any(value != 0.0 for value in severity.angular_velocity_rad_s):
            raise ValueError("motion is unsupported for native-COCO condition shards")
        render_mode = (
            "modeled_neutral"
            if isinstance(condition, BaselineCondition)
            and condition.kind is BaselineKind.MODELED_NEUTRAL
            else "physical_static_ldr"
        )
        geometry_policy = "profile_sensor_or_native_active_roi"

    source_adapter = profile.fixed_parameters.get("source_adapter")
    if untouched:
        formation_operator = None
        pixel_integration_owner = None
        formation_selection_policy = None
    elif source_adapter == _NATIVE_ROI_SOURCE_ADAPTER:
        formation_operator = "exact_equal_grid_cell_average_transfer_v1"
        pixel_integration_owner = "collapsed_source_reconstruction_and_target_photosite_transfer"
        formation_selection_policy = {
            "policy_id": "native_active_roi_requires_equal_source_sensor_grid_v1",
            "equal_grid_operator": formation_operator,
            "non_equal_grid_action": "reject",
        }
    else:
        # Stored-image shapes are known only when a frame is decoded.  The
        # camera pipeline therefore selects the truthful exactly-once operator
        # from source/sensor geometry at render time.  Binding the policy (not
        # one falsely universal operator) keeps the compact shard header valid
        # for both equal-grid and strictly oversampled inputs.
        formation_operator = "geometry_selected_exactly_once_transfer_v1"
        pixel_integration_owner = "geometry_selected_exactly_once_v1"
        formation_selection_policy = {
            "policy_id": "equal_grid_cell_average_else_strictly_oversampled_quadrature_v1",
            "equal_grid": {
                "operator": "exact_equal_grid_cell_average_transfer_v1",
                "pixel_integration_owner": (
                    "collapsed_source_reconstruction_and_target_photosite_transfer"
                ),
            },
            "strictly_finer_in_both_axes": {
                "operator": "continuous_psf_quadrature_then_target_area_average_v1",
                "pixel_integration_owner": "target_photosite_area_resampler",
            },
            "all_other_geometry": "reject",
        }

    contract_payload = {
        "schema_version": 4,
        "record_type": "native_coco_ldr_render_preprocess_contract",
        "render": {
            "implementation_id": "eval.coco_stream.static_ldr.v4",
            "mode": render_mode,
            "camera_graph_bypassed": untouched,
            "deterministic": True,
            "realization_id": 0,
            "severity": None if severity is None else severity.to_dict(),
            "mechanism_comparator": comparator_contract,
            "source_adapter": source_adapter,
            "source_reconstruction": (None if untouched else "piecewise_constant_cell_average"),
            "optical_formation_operator": formation_operator,
            "optical_formation_selection_policy": formation_selection_policy,
            "pixel_integration_owner": pixel_integration_owner,
            "pixel_integration_count": 0 if untouched else 1,
            "camera_output_geometry_policy": geometry_policy,
        },
        "preprocessing": json_value(preprocessing.identity),
        "prediction": {
            "coordinate_space": "native_stored_image",
            "box_convention": "continuous_xyxy_image_edges",
            "recovery": "letterbox_inverse_then_full_window_axis_scale.v2",
            "label_space": label_space,
        },
    }
    contract = {
        **contract_payload,
        "contract_sha256": canonical_sha256(contract_payload),
    }
    condition_record = condition.to_dict()
    binding_payload = {
        "schema_version": 4,
        "record_type": "native_coco_ldr_condition_shard_binding",
        "fixed_profile_sha256": profile.profile_hash,
        "condition_sha256": condition.condition_hash,
        "condition": condition_record,
        "render_preprocess_contract": contract,
    }
    return (
        {
            **binding_payload,
            "condition_binding_sha256": canonical_sha256(binding_payload),
        },
        severity,
        comparator_config,
    )


def _detector_outputs(value: Any, *, expected: int) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (Mapping, str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise TypeError("detect_batch must return one ordered iterable of output mappings")
    outputs = tuple(value)
    if len(outputs) != expected:
        raise ValueError(
            "detector output cardinality mismatch: "
            f"received {len(outputs)} outputs for {expected} inputs"
        )
    if not all(isinstance(output, Mapping) for output in outputs):
        raise TypeError("detect_batch outputs must be mappings")
    return outputs


def _render_frame(
    source: np.ndarray,
    *,
    image_id: int,
    profile: CameraProfile,
    severity: LDRCaptureSeverity | None,
    comparator_config: Any | None,
    pipeline_cache: dict[tuple[int, int] | None, Any],
) -> Frame:
    input_frame = make_ldr_input_frame(source, image_id=str(image_id))
    if severity is None and comparator_config is None:
        return input_frame
    active_shape = (
        input_frame.shape[:2]
        if profile.fixed_parameters.get("source_adapter") == _NATIVE_ROI_SOURCE_ADAPTER
        else None
    )
    pipeline = pipeline_cache.get(active_shape)
    if pipeline is None:
        if comparator_config is None:
            if severity is None:  # pragma: no cover - narrowed above
                raise RuntimeError("missing LDR render specification")
            pipeline, _ = build_ldr_pipeline(
                profile,
                severity,
                active_sensor_shape=active_shape,
            )
        else:
            from ..comparators.capture import build_ldr_comparator_pipeline

            pipeline, _ = build_ldr_comparator_pipeline(
                profile,
                comparator_config,
                active_sensor_shape=active_shape,
            )
        pipeline_cache[active_shape] = pipeline
    if not pipeline.deterministic:
        raise RuntimeError("LDR render pipeline unexpectedly contains a stochastic stage")
    return pipeline.run(input_frame)


def _native_prediction_record(
    *,
    image_id: int,
    detector_output: Mapping[str, Any],
    detector_input: DetectorInput,
    native_shape: Sequence[int],
    label_space: str,
    allowed_category_ids: frozenset[int],
) -> dict[str, Any]:
    raw_category_ids = _publication_raw_category_ids(
        detector_output,
        image_id=image_id,
        label_space=label_space,
    )
    raw_undeclared = sorted(
        set(int(value) for value in raw_category_ids).difference(allowed_category_ids)
    )
    if raw_undeclared:
        raise ValueError(
            f"raw detector prediction for image {image_id} contains category IDs outside "
            f"subset.category_ids: {raw_undeclared}"
        )
    prediction = detector_output_to_native_prediction(
        image_id=image_id,
        detector_output=detector_output,
        geometry=detector_input.geometry,
        native_shape=native_shape,
        label_space=label_space,
    )
    labels = np.asarray(prediction["labels"], dtype=np.int64)
    if len(labels) > _COCO_MAXIMUM_DETECTIONS_PER_IMAGE:
        raise ValueError(
            f"native prediction for image {image_id} has {len(labels)} outputs; "
            f"COCO publication permits at most {_COCO_MAXIMUM_DETECTIONS_PER_IMAGE} per image"
        )
    undeclared = sorted(set(int(value) for value in labels).difference(allowed_category_ids))
    if undeclared:
        raise ValueError(
            f"native prediction for image {image_id} contains category IDs outside "
            f"subset.category_ids: {undeclared}"
        )
    return make_prediction_record(
        image_id,
        {
            "boxes": np.asarray(prediction["boxes"]).tolist(),
            "labels": labels.tolist(),
            "scores": np.asarray(prediction["scores"]).tolist(),
        },
    )


def run_coco_ldr_condition_shard(
    *,
    subset: NativeCOCODataset,
    condition: ExperimentCondition,
    image_ids: Sequence[int],
    profile: CameraProfile,
    preprocessing: LetterboxConfig,
    model: Mapping[str, Any],
    run: Mapping[str, Any],
    detect_batch: Callable[[tuple[DetectorInput, ...]], Iterable[Mapping[str, Any]]],
    shard_path: str | Path,
    label_space: str = "coco_sparse",
    batch_size: int = 1,
    receipt_path: str | Path | None = None,
) -> PredictionShard:
    """Run or safely resume one deterministic static-LDR condition shard.

    A complete existing shard is fully validated against the expected header
    and returned before any image decode, camera render, or detector call.
    Incomplete, corrupt, or identity-drifted publications fail closed.
    """

    if not isinstance(subset, (NativeCOCOSubset, LazyNativeCOCOSubset)):
        raise TypeError("subset must be a native COCO dataset")
    if not isinstance(profile, CameraProfile):
        raise TypeError("profile must be a CameraProfile")
    if profile.data_mode is not DataMode.LDR_REDEGRADATION:
        raise ValueError("native-COCO JPEG shards require an LDR_REDEGRADATION profile")
    if not isinstance(preprocessing, LetterboxConfig):
        raise TypeError("preprocessing must be a LetterboxConfig")
    if not isinstance(run, Mapping):
        raise TypeError("run identity must be a mapping")
    if not callable(detect_batch):
        raise TypeError("detect_batch must be callable")
    if label_space not in _LABEL_SPACES:
        raise ValueError("label_space must be 'coco_sparse' or 'coco80_contiguous'")
    resolved_batch_size = positive_int(batch_size, field_name="batch_size")
    selected = _ordered_shard_image_ids(image_ids, available=subset.image_ids)
    validated_model = validate_model_identity(model)
    condition_binding, severity, comparator_config = _condition_contract(
        condition=condition,
        profile=profile,
        preprocessing=preprocessing,
        label_space=label_space,
    )
    header = make_prediction_shard_header(
        run=run,
        dataset=subset.identity,
        model=validated_model,
        camera_profile_sha256=profile.profile_hash,
        condition=condition_binding,
        image_ids=selected,
    )
    existing = validate_existing_prediction_shard(
        shard_path,
        expected_header=header,
        receipt_path=receipt_path,
    )
    if existing is not None:
        return existing

    records: list[Mapping[str, Any]] = []
    allowed_category_ids = frozenset(int(value) for value in subset.category_ids)
    pipeline_cache: dict[tuple[int, int] | None, Any] = {}
    for start in range(0, len(selected), resolved_batch_size):
        batch_ids = selected[start : start + resolved_batch_size]
        detector_inputs: list[DetectorInput] = []
        native_shapes: list[tuple[int, int]] = []
        for image_id in batch_ids:
            native_shape = subset.image_shape(image_id)
            source = subset.image(image_id)
            rendered = _render_frame(
                source,
                image_id=image_id,
                profile=profile,
                severity=severity,
                comparator_config=comparator_config,
                pipeline_cache=pipeline_cache,
            )
            detector_input = letterbox(rendered, preprocessing)
            if detector_input.geometry.input_shape != rendered.shape[:2]:
                raise RuntimeError("letterbox geometry drifted from the rendered camera output")
            detector_inputs.append(detector_input)
            native_shapes.append(native_shape)
            del detector_input, source, rendered

        inputs = tuple(detector_inputs)
        outputs = _detector_outputs(detect_batch(inputs), expected=len(inputs))
        for image_id, native_shape, detector_input, detector_output in zip(
            batch_ids,
            native_shapes,
            inputs,
            outputs,
        ):
            records.append(
                _native_prediction_record(
                    image_id=image_id,
                    detector_output=detector_output,
                    detector_input=detector_input,
                    native_shape=native_shape,
                    label_space=label_space,
                    allowed_category_ids=allowed_category_ids,
                )
            )
        del detector_input, detector_output, detector_inputs, inputs, outputs

    write_prediction_shard(
        shard_path,
        header=header,
        predictions=records,
        receipt_path=receipt_path,
    )
    return validate_prediction_shard(
        shard_path,
        receipt_path=receipt_path,
        expected_header=header,
    )


__all__ = ["run_coco_ldr_condition_shard"]
