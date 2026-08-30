"""Local orchestration for the native-COCO publication study.

The pilot runner is intentionally small and eager.  This module owns the
publication-scale layer above the streaming condition runner: it freezes the
dataset selection, camera profiles, mechanism-matched arms, detector
allocations, analysis contract, and deterministic shard layout before any
model is run. Detector runs use resumable prediction shards and bind their
scientific configuration, dataset, checkpoint, preprocessing, and runtime.

Records contain no dataset, checkpoint, repository, or output paths.  Paths
are local execution inputs; portable identities are hashes plus disclosed
contracts. Existing outputs are accepted only when their scientific identities
match the requested run.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from .._canonical import (
    canonical_sha256,
    finite_float,
    freeze_json_value,
    json_value,
    nfc_string,
    positive_int,
)
from ..capture import make_ldr_input_frame
from ..comparators.matching import MechanismMatch, match_common_neutral_comparators
from ..domains import DataMode
from ..experiments.conditions import (
    BaselineCondition,
    BaselineKind,
    ConditionBinding,
    ExperimentCondition,
    MechanismComparatorCondition,
    PhysicalDefocusCondition,
    condition_from_dict,
)
from ..profiles import CameraProfile
from .coco import LazyNativeCOCOSubset, NativeCOCODataset
from .coco_stream import run_coco_ldr_condition_shard
from .locking import advisory_target_lock
from .model_provenance import model_identity, validate_model_identity
from .preprocess import DetectorInput, LetterboxConfig, letterbox
from .protocol import (
    DEFAULT_INFERENCE_SEED,
    configure_deterministic_inference,
    deterministic_inference_execution_contract,
    image_selection_identity,
    runtime_reproducibility_identity,
    torch_device_attestation_matches,
)
from .shards import PredictionShardMerge, merge_prediction_shards
from .torchvision_coco_postprocess import COCO_SPARSE_CATEGORY_IDS

PUBLICATION_DEFOCUS_GRID: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
PUBLICATION_ANCHOR_GRID: tuple[float, ...] = (0.5, 1.5, 3.0)
COCO_IOU_THRESHOLDS: tuple[float, ...] = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)
COCO_CONFIDENCE_FLOOR = 0.001
COCO_MAXIMUM_DETECTIONS = 100
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20_260_715
MAX_BOOTSTRAP_SEED = 2**64 - 1
DEFAULT_SHARD_SIZE = 250
PUBLICATION_IMAGE_COUNT = 5_000
PUBLICATION_EXECUTION_CELL_COUNT = 67
STUDY_PLAN_VERSION = 11
STUDY_IMPLEMENTATION_ID = "phycam.native_coco_static_ldr_publication_study.v11"
PUBLICATION_PROTOCOL_REQUIREMENTS_ID = "phycam.full_coco_publication_protocol.v11"
FULL_COCO_VAL2017_DATASET_SHA256 = (
    "5fdea8630d7bde63ddee06b530acb3d9e4624f5b2e85f9c5f6362d288d638ff6"
)
FULL_COCO_VAL2017_SELECTION_SHA256 = (
    "9d99596a43d19b52d6598313c1f0135d46989e70cb6e9077b98490bbc2635f2c"
)
COCO_VAL2017_ANNOTATION_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
PUBLICATION_PROFILE_HASHES = (
    ("primary", "350cd7c431e3df38e8e5ab5fdcd24036a714a7799289aee455df49df6fccac9c"),
    ("replication", "e74bc1e7a255441fbad1f678c2a733785f256d26b68bb40ccabe61ee92061383"),
)
_NATIVE_ROI_SOURCE_ADAPTER = "native_active_sensor_roi_v1"
_EXECUTION_ENGINE_IMPLEMENTATION_ID = "phycam.study_execution_engine.v3"
_OFFICIAL_DATASET_CLASS_ID = "phycam_eval.eval.coco.LazyNativeCOCOSubset"
_OFFICIAL_RUNTIME_FACTORY_ID = "phycam_eval.eval.protocol.runtime_reproducibility_identity"
_OFFICIAL_SHARD_RUNNER_ID = "phycam_eval.eval.coco_stream.run_coco_ldr_condition_shard"
_OFFICIAL_SHARD_MERGER_ID = "phycam_eval.eval.shards.merge_prediction_shards"
_OFFICIAL_DETECTOR_ADAPTER_CLASS_IDS = {
    "yolov8n": "phycam_eval.eval.detectors.UltralyticsYOLOAdapter",
    "fasterrcnn_r50_fpn": "phycam_eval.eval.detectors.TorchvisionFasterRCNNAdapter",
    "retinanet_r50_fpn_v2": "phycam_eval.eval.detectors.TorchvisionRetinaNetAdapter",
    "detr_r50": "phycam_eval.eval.detectors.HuggingFaceDETRAdapter",
}
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_PLAN_KEYS = {
    "schema_version",
    "record_type",
    "implementation_id",
    "study_id",
    "dataset",
    "image_selection",
    "profiles",
    "design",
    "arms",
    "detector_allocations",
    "analysis_protocol",
    "evidence_tier",
    "execution_cells",
    "study_plan_sha256",
    "plan_version",
}
_ALLOCATION_POLICIES = {"all_arms", "baselines_and_physical_anchors"}
_KNOWN_COMPARATOR_FAMILIES = {
    "gaussian",
    "adapted_quadratic_cosine",
    "adapted_sampled_incoherent",
}
_PUBLICATION_MODEL_CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    "yolov8n": {
        "model_id": "yolov8n.pt",
        "revision": None,
        "artifacts": (
            {
                "name": "yolov8n.pt",
                "bytes": 6_549_796,
                "sha256": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
            },
        ),
    },
    "fasterrcnn_r50_fpn": {
        "model_id": "torchvision/fasterrcnn_resnet50_fpn@COCO_V1",
        "revision": None,
        "artifacts": (
            {
                "name": "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
                "bytes": 167_502_836,
                "sha256": "258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384",
            },
        ),
    },
    "retinanet_r50_fpn_v2": {
        "model_id": "torchvision/retinanet_resnet50_fpn_v2@COCO_V1",
        "revision": None,
        "artifacts": (
            {
                "name": "retinanet_resnet50_fpn_v2_coco-5905b1c5.pth",
                "bytes": 153_130_989,
                "sha256": "5905b1c544219215e544dbe319720397bc4e68de61a733a59350d7976645b769",
            },
        ),
    },
    "detr_r50": {
        "model_id": "facebook/detr-resnet-50",
        "revision": "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b",
        "artifacts": (
            {
                "name": "config.json",
                "bytes": 4_592,
                "sha256": "e7bcf3992363f27717a863f14b193140ad2e41d4338ee012730e58a92cae17e6",
            },
            {
                "name": "model.safetensors",
                "bytes": 166_587_896,
                "sha256": "830f5e2eeaada8c8c8281779dcc8ab12833972eb8514ed0a35be6c1d4420ad81",
            },
            {
                "name": "preprocessor_config.json",
                "bytes": 290,
                "sha256": "0673fea2a6d3cf92cdbab3c7426c0ecdf8a4729a2a4d5199033dcd66a2b8759b",
            },
        ),
    },
}


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    result = nfc_string(value, field_name=label)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ValueError(f"{label} must match {_IDENTIFIER.pattern!r}")
    return result


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _plan_contract_id() -> str:
    return PUBLICATION_PROTOCOL_REQUIREMENTS_ID


def _probability(value: object, *, label: str, positive: bool = False) -> float:
    result = finite_float(value, field_name=label)
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not lower_ok or result > 1.0:
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{label} must lie in {interval}")
    return result


def _float_grid(values: Iterable[float], *, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be an ordered iterable of real values")
    result = tuple(
        finite_float(value, field_name=f"{label}[{index}]") for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{label} must not be empty")
    if any(value <= 0.0 for value in result):
        raise ValueError(f"{label} values must be positive")
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError(f"{label} must be strictly increasing and unique")
    return result


def _image_ids(values: Iterable[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("image_ids must be an ordered iterable of integers")
    result = tuple(
        positive_int(value, field_name=f"image_ids[{index}]", allow_zero=True)
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError("image_ids must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("image_ids must be unique")
    return result


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = freeze_json_value(value)
    if not isinstance(frozen, MappingProxyType):
        raise TypeError("record must be a mapping")
    return frozen


def _number_token(value: float) -> str:
    text = format(value, ".17g").lower()
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def _selection_record(image_ids: Sequence[int]) -> dict[str, Any]:
    ordered = _image_ids(image_ids)
    payload = {"ordered_image_ids": list(ordered)}
    return {
        "ordered_image_ids": list(ordered),
        "count": len(ordered),
        "first": ordered[0],
        "last": ordered[-1],
        "selection_sha256": canonical_sha256(payload),
    }


def _validate_dataset_identity(
    value: Mapping[str, Any], *, image_ids: Sequence[int]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("dataset identity must be a mapping")
    record = json_value(value)
    if record.get("schema_version") != 2 or record.get("record_type") != "native_coco_subset":
        raise ValueError("study dataset must be a schema-v2 native_coco_subset")
    supplied = record.get("dataset_sha256")
    payload = {key: item for key, item in record.items() if key != "dataset_sha256"}
    if supplied != canonical_sha256(payload):
        raise ValueError("dataset_sha256 does not match the native COCO identity")
    if tuple(record.get("ordered_image_ids", ())) != tuple(image_ids):
        raise ValueError("dataset identity image order differs from the study selection")
    if record.get("dataset") != "COCO" or record.get("split") != "val2017":
        raise ValueError("publication study requires the native COCO val2017 split")
    expected_keys = {
        "schema_version",
        "record_type",
        "dataset",
        "split",
        "annotation_artifact",
        "ordered_image_ids",
        "image_selection",
        "image_artifacts",
        "category_ids",
        "filtered_nonpositive_annotations",
        "decode_contract",
        "target_contract",
        "dataset_sha256",
    }
    if set(record) != expected_keys:
        raise ValueError("native COCO dataset identity has missing or unknown fields")
    if record.get("image_selection") != image_selection_identity(image_ids):
        raise ValueError("native COCO image-selection identity does not match")

    def validate_artifact(artifact: object, *, label: str, expected_prefix: str) -> None:
        if not isinstance(artifact, Mapping) or set(artifact) != {"name", "bytes", "sha256"}:
            raise ValueError(f"{label} must be a portable artifact identity")
        name = artifact.get("name")
        if not isinstance(name, str) or not name.startswith(expected_prefix):
            raise ValueError(f"{label} has an unexpected published name")
        size = artifact.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{label} bytes must be a nonnegative integer")
        _sha256(artifact.get("sha256"), label=f"{label} sha256")

    annotation = record.get("annotation_artifact")
    validate_artifact(
        annotation,
        label="COCO annotation artifact",
        expected_prefix="annotations/instances_val2017.json",
    )
    if annotation["name"] != "annotations/instances_val2017.json":
        raise ValueError("COCO annotation artifact name is noncanonical")
    artifacts = record.get("image_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(image_ids):
        raise ValueError("native COCO image artifacts must align with selected image IDs")
    for index, artifact in enumerate(artifacts):
        validate_artifact(
            artifact,
            label=f"COCO image artifact {index}",
            expected_prefix="val2017/",
        )
    names = [artifact["name"] for artifact in artifacts]
    if len(names) != len(set(names)):
        raise ValueError("native COCO image artifact names must be unique")
    categories = record.get("category_ids")
    if (
        not isinstance(categories, list)
        or not categories
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in categories
        )
        or categories != sorted(set(categories))
    ):
        raise ValueError("native COCO category IDs must be sorted unique positive integers")
    filtered = record.get("filtered_nonpositive_annotations")
    if isinstance(filtered, bool) or not isinstance(filtered, int) or filtered < 0:
        raise ValueError("filtered_nonpositive_annotations must be a nonnegative integer")
    if record.get("decode_contract") != {
        "implementation_id": "pillow.rgb.native.float32.v1",
        "layout": "HWC_RGB",
        "range": [0.0, 1.0],
        "resize": None,
        "uint8_to_float": "exact_divide_by_255",
    }:
        raise ValueError("native COCO decode contract drifted")
    if record.get("target_contract") != {
        "coordinate_space": "native_stored_image",
        "box_convention": "continuous_xyxy_image_edges",
        "area": "official_coco_annotation_area",
    }:
        raise ValueError("native COCO target contract drifted")
    return record


def _validate_publication_profile(profile: CameraProfile) -> None:
    if not isinstance(profile, CameraProfile):
        raise TypeError("profiles must contain CameraProfile values")
    if profile.data_mode is not DataMode.LDR_REDEGRADATION:
        raise ValueError("publication profiles must use LDR_REDEGRADATION mode")
    if profile.fixed_parameters.get("source_adapter") != _NATIVE_ROI_SOURCE_ADAPTER:
        raise ValueError("publication profiles must preserve native aspect via the active ROI")
    if profile.readout.line_time_s != 0.0:
        raise ValueError("static publication profiles must have zero line_time_s")
    maximum = profile.fixed_parameters.get("active_sensor_roi")
    if not isinstance(maximum, Mapping) or json_value(maximum) != {
        "maximum_shape_pixels": list(profile.sensor.sensor_shape_pixels),
        "origin_policy": "upper_left",
        "stored_sample_to_photosite": "one_to_one",
    }:
        raise ValueError("publication profiles must disclose the active_sensor_roi contract")
    calibration = profile.calibration_reference
    if (
        calibration is None
        or "synthetic" not in calibration.lower()
        or ("not hardware calibrated" not in calibration.lower())
    ):
        raise ValueError("publication profiles must be disclosed as synthetic, not calibrated")


@dataclass(frozen=True, slots=True)
class DetectorAllocation:
    """Predeclared detector-to-profile/arm allocation and inference contract."""

    detector_id: str
    model_backend: str
    model_id: str
    model_revision: str | None
    model_artifacts: tuple[Mapping[str, Any], ...]
    profile_ids: tuple[str, ...]
    arm_policy: str
    anchor_edge_waves_ref: tuple[float, ...] = ()
    label_space: str = "coco_sparse"
    confidence_threshold: float = COCO_CONFIDENCE_FLOOR
    nms_iou_threshold: float | None = None
    input_shape: tuple[int, int] = (640, 640)
    pad_value: tuple[float, float, float] | float = (0.5, 0.5, 0.5)
    requested_device: str = "cpu"
    batch_size: int = 1
    shard_size: int = DEFAULT_SHARD_SIZE
    backend_settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "detector_id", _identifier(self.detector_id, label="detector_id"))
        if not isinstance(self.model_backend, str) or not self.model_backend:
            raise ValueError("model_backend must be a nonempty string")
        object.__setattr__(
            self,
            "model_backend",
            nfc_string(self.model_backend, field_name="model_backend"),
        )
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a nonempty string")
        normalized_model_id = nfc_string(self.model_id, field_name="model_id")
        normalized_revision = self.model_revision
        if normalized_revision is not None:
            if not isinstance(normalized_revision, str) or not normalized_revision:
                raise ValueError("model_revision must be a nonempty string when supplied")
            normalized_revision = nfc_string(
                normalized_revision,
                field_name="model_revision",
            )
        artifact_contract = model_identity(
            backend=self.model_backend,
            model_id=normalized_model_id,
            revision=normalized_revision,
            artifacts=self.model_artifacts,
            implementation={},
        )
        object.__setattr__(self, "model_id", normalized_model_id)
        object.__setattr__(self, "model_revision", normalized_revision)
        object.__setattr__(
            self,
            "model_artifacts",
            tuple(_immutable_mapping(value) for value in artifact_contract["artifacts"]),
        )
        profile_ids = tuple(
            _identifier(value, label=f"profile_ids[{index}]")
            for index, value in enumerate(self.profile_ids)
        )
        if not profile_ids or len(profile_ids) != len(set(profile_ids)):
            raise ValueError("profile_ids must be nonempty and unique")
        object.__setattr__(self, "profile_ids", profile_ids)
        if self.arm_policy not in _ALLOCATION_POLICIES:
            raise ValueError(f"arm_policy must be one of {sorted(_ALLOCATION_POLICIES)}")
        anchors = tuple(self.anchor_edge_waves_ref)
        if self.arm_policy == "all_arms":
            if anchors:
                raise ValueError("all_arms allocations must not declare anchor coordinates")
        else:
            anchors = _float_grid(anchors, label="anchor_edge_waves_ref")
        object.__setattr__(self, "anchor_edge_waves_ref", anchors)
        if self.label_space not in {"coco_sparse", "coco80_contiguous"}:
            raise ValueError("label_space must be 'coco_sparse' or 'coco80_contiguous'")
        object.__setattr__(
            self,
            "confidence_threshold",
            _probability(self.confidence_threshold, label="confidence_threshold"),
        )
        if self.nms_iou_threshold is not None:
            object.__setattr__(
                self,
                "nms_iou_threshold",
                _probability(self.nms_iou_threshold, label="nms_iou_threshold", positive=True),
            )
        try:
            height, width = self.input_shape
        except (TypeError, ValueError) as exc:
            raise TypeError("input_shape must contain height and width") from exc
        object.__setattr__(
            self,
            "input_shape",
            (
                positive_int(height, field_name="input_shape[0]"),
                positive_int(width, field_name="input_shape[1]"),
            ),
        )
        normalized_preprocessing = LetterboxConfig(self.input_shape, self.pad_value)
        object.__setattr__(self, "pad_value", normalized_preprocessing.pad_value)
        if not isinstance(self.requested_device, str):
            raise TypeError("requested_device must be a string")
        requested_device = self.requested_device.strip().lower()
        if re.fullmatch(r"(?:cpu|mps|cuda(?::[0-9]+)?)", requested_device) is None:
            raise ValueError("requested_device must be cpu, mps, cuda, or cuda:<index>")
        object.__setattr__(self, "requested_device", requested_device)
        object.__setattr__(
            self, "batch_size", positive_int(self.batch_size, field_name="batch_size")
        )
        object.__setattr__(
            self, "shard_size", positive_int(self.shard_size, field_name="shard_size")
        )
        settings = freeze_json_value(self.backend_settings)
        if not isinstance(settings, MappingProxyType):
            raise TypeError("backend_settings must be a mapping")
        object.__setattr__(self, "backend_settings", settings)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "record_type": "coco_detector_allocation",
            "detector_id": self.detector_id,
            "model_backend": self.model_backend,
            "model_contract": {
                "model_id": self.model_id,
                "revision": self.model_revision,
                "artifacts": [json_value(value) for value in self.model_artifacts],
            },
            "ordered_profile_ids": list(self.profile_ids),
            "arm_selection": {
                "policy": self.arm_policy,
                "physical_anchor_edge_waves_ref": list(self.anchor_edge_waves_ref),
            },
            "detector_contract": {
                "label_space": self.label_space,
                "confidence_threshold": self.confidence_threshold,
                "nms_iou_threshold": self.nms_iou_threshold,
                "evaluation_maximum_detections": COCO_MAXIMUM_DETECTIONS,
                "backend_settings": json_value(self.backend_settings),
            },
            "preprocessing": {
                "output_shape_hw": list(self.input_shape),
                "pad_value_srgb": list(self.pad_value),
                "letterbox_identity": json_value(
                    LetterboxConfig(self.input_shape, self.pad_value).identity
                ),
            },
            "execution": {
                "requested_detector_device": self.requested_device,
                "batch_size": self.batch_size,
                "images_per_shard": self.shard_size,
            },
        }

    @property
    def allocation_sha256(self) -> str:
        return canonical_sha256(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        return {**payload, "allocation_sha256": canonical_sha256(payload)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DetectorAllocation":
        if not isinstance(value, Mapping):
            raise TypeError("detector allocation must be a mapping")
        record = json_value(value)
        expected_keys = {
            "schema_version",
            "record_type",
            "detector_id",
            "model_backend",
            "model_contract",
            "ordered_profile_ids",
            "arm_selection",
            "detector_contract",
            "preprocessing",
            "execution",
            "allocation_sha256",
        }
        if set(record) != expected_keys:
            raise ValueError("detector allocation has missing or unknown fields")
        if record.get("schema_version") != 3 or record.get("record_type") != (
            "coco_detector_allocation"
        ):
            raise ValueError("detector allocation requires the schema-v3 allocation type")
        selection = record.get("arm_selection")
        model_contract = record.get("model_contract")
        contract = record.get("detector_contract")
        preprocessing = record.get("preprocessing")
        execution = record.get("execution")
        if not all(
            isinstance(item, Mapping)
            for item in (model_contract, selection, contract, preprocessing, execution)
        ):
            raise TypeError("detector allocation nested contracts must be mappings")
        if set(model_contract) != {"model_id", "revision", "artifacts"}:
            raise ValueError("detector allocation model contract is noncanonical")
        if set(selection) != {"policy", "physical_anchor_edge_waves_ref"}:
            raise ValueError("detector allocation arm selection is noncanonical")
        if set(contract) != {
            "label_space",
            "confidence_threshold",
            "nms_iou_threshold",
            "evaluation_maximum_detections",
            "backend_settings",
        }:
            raise ValueError("detector allocation detector contract is noncanonical")
        if contract.get("evaluation_maximum_detections") != COCO_MAXIMUM_DETECTIONS:
            raise ValueError("detector allocation must use COCO maxDets=100")
        if set(preprocessing) != {
            "output_shape_hw",
            "pad_value_srgb",
            "letterbox_identity",
        }:
            raise ValueError("detector allocation preprocessing is noncanonical")
        if set(execution) != {
            "requested_detector_device",
            "batch_size",
            "images_per_shard",
        }:
            raise ValueError("detector allocation execution contract is noncanonical")
        result = cls(
            detector_id=record.get("detector_id"),
            model_backend=record.get("model_backend"),
            model_id=model_contract.get("model_id"),
            model_revision=model_contract.get("revision"),
            model_artifacts=tuple(model_contract.get("artifacts", ())),
            profile_ids=tuple(record.get("ordered_profile_ids", ())),
            arm_policy=selection.get("policy"),
            anchor_edge_waves_ref=tuple(selection.get("physical_anchor_edge_waves_ref", ())),
            label_space=contract.get("label_space"),
            confidence_threshold=contract.get("confidence_threshold"),
            nms_iou_threshold=contract.get("nms_iou_threshold"),
            input_shape=tuple(preprocessing.get("output_shape_hw", ())),
            pad_value=tuple(preprocessing.get("pad_value_srgb", ())),
            requested_device=execution.get("requested_detector_device"),
            batch_size=execution.get("batch_size"),
            shard_size=execution.get("images_per_shard"),
            backend_settings=contract.get("backend_settings"),
        )
        if preprocessing.get("letterbox_identity") != json_value(
            LetterboxConfig(result.input_shape, result.pad_value).identity
        ):
            raise ValueError("detector allocation letterbox identity does not match its settings")
        if result.to_dict() != record:
            raise ValueError("detector allocation identity does not match its payload")
        return result

    def accepts(self, profile_id: str, condition: ExperimentCondition) -> bool:
        if profile_id not in self.profile_ids:
            return False
        if self.arm_policy == "all_arms":
            return True
        if isinstance(condition, BaselineCondition):
            return True
        return isinstance(condition, PhysicalDefocusCondition) and any(
            condition.edge_waves_ref == anchor for anchor in self.anchor_edge_waves_ref
        )


def publication_detector_allocations(
    profile_ids: Sequence[str],
    *,
    primary_profile_id: str,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> tuple[DetectorAllocation, ...]:
    """Return the predeclared full/anchor detector allocation for the paper."""

    profiles = tuple(_identifier(value, label="profile_id") for value in profile_ids)
    primary = _identifier(primary_profile_id, label="primary_profile_id")
    if not profiles or len(profiles) != len(set(profiles)):
        raise ValueError("profile_ids must be nonempty and unique")
    if primary not in profiles:
        raise ValueError("primary_profile_id must be one of profile_ids")
    size = positive_int(shard_size, field_name="shard_size")
    return (
        DetectorAllocation(
            detector_id="yolov8n",
            model_backend="ultralytics-yolo",
            model_id=_PUBLICATION_MODEL_CONTRACTS["yolov8n"]["model_id"],
            model_revision=_PUBLICATION_MODEL_CONTRACTS["yolov8n"]["revision"],
            model_artifacts=tuple(_PUBLICATION_MODEL_CONTRACTS["yolov8n"]["artifacts"]),
            profile_ids=profiles,
            arm_policy="all_arms",
            label_space="coco80_contiguous",
            nms_iou_threshold=0.7,
            pad_value=(114.0 / 255.0,) * 3,
            requested_device="mps",
            batch_size=16,
            shard_size=size,
            backend_settings={
                "adapter": "ultralytics.bchw_float_rgb.v4",
                "ultralytics_version": "8.4.37",
                "nms_time_limit_policy": (
                    "isolated_upstream_postprocess_nms_proxy_max_time_img_inf_attested.v1"
                ),
                "upstream_postprocess_source_sha256": (
                    "f134eaac019aa291fec1361f6d01cef6d0000a4393d8ef8c345543474fe8a1c6"
                ),
                "upstream_nms_source_sha256": (
                    "6533165fa8bd6bd3775cf14e02ea34906c3a2c2199b0a588ff4bddd514a23cc9"
                ),
                "maximum_detections_before_coco_limit": 300,
                "maximum_detections": COCO_MAXIMUM_DETECTIONS,
                "global_detection_cap": (
                    "stable_descending_score_mergesort_before_native_mapping.v1"
                ),
                "class_agnostic_nms": False,
                "test_time_augmentation": False,
                "half_precision": False,
            },
        ),
        DetectorAllocation(
            detector_id="fasterrcnn_r50_fpn",
            model_backend="torchvision-fasterrcnn",
            model_id=_PUBLICATION_MODEL_CONTRACTS["fasterrcnn_r50_fpn"]["model_id"],
            model_revision=_PUBLICATION_MODEL_CONTRACTS["fasterrcnn_r50_fpn"]["revision"],
            model_artifacts=tuple(_PUBLICATION_MODEL_CONTRACTS["fasterrcnn_r50_fpn"]["artifacts"]),
            profile_ids=(primary,),
            arm_policy="baselines_and_physical_anchors",
            anchor_edge_waves_ref=PUBLICATION_ANCHOR_GRID,
            nms_iou_threshold=0.5,
            pad_value=(0.485, 0.456, 0.406),
            requested_device="cpu",
            shard_size=size,
            backend_settings={
                "adapter": "torchvision.fasterrcnn.fixed_input_float_rgb.v2",
                "torchvision_version": "0.26.0",
                "maximum_detections": 100,
                "postprocessor_implementation": (
                    "torchvision.coco_sparse_prebudget_postprocess.v1"
                ),
                "postprocessor_binding": "roi_heads.postprocess_detections",
                "allowed_category_ids": COCO_SPARSE_CATEGORY_IDS,
                "coco_sparse_filter_policy": ("valid_sparse_coco_before_detections_per_img"),
                "coco_sparse_logit_masking": False,
                "coco_sparse_internal_cap_inflation": False,
                "upstream_postprocessor_source_sha256": (
                    "ad194bab11211e5ee3ec88297abe9708f69ad49b10bbd1b7dec81eb039b8f8a9"
                ),
                "repository_postprocessor_source_sha256": (
                    "bb2e5baf68bc5afb0bcd9e9aa337270c862826df4c06165d52ab0966eaa9767c"
                ),
                "repository_postprocessor_callable": (
                    "phycam_eval.eval.torchvision_coco_postprocess."
                    "fasterrcnn_coco_sparse_postprocess_detections"
                ),
            },
        ),
        DetectorAllocation(
            detector_id="retinanet_r50_fpn_v2",
            model_backend="torchvision-retinanet",
            model_id=_PUBLICATION_MODEL_CONTRACTS["retinanet_r50_fpn_v2"]["model_id"],
            model_revision=_PUBLICATION_MODEL_CONTRACTS["retinanet_r50_fpn_v2"]["revision"],
            model_artifacts=tuple(
                _PUBLICATION_MODEL_CONTRACTS["retinanet_r50_fpn_v2"]["artifacts"]
            ),
            profile_ids=(primary,),
            arm_policy="baselines_and_physical_anchors",
            anchor_edge_waves_ref=PUBLICATION_ANCHOR_GRID,
            nms_iou_threshold=0.5,
            pad_value=(0.485, 0.456, 0.406),
            requested_device="cpu",
            shard_size=size,
            backend_settings={
                "adapter": ("torchvision.retinanet_resnet50_fpn_v2.fixed_input_float_rgb.v2"),
                "torchvision_version": "0.26.0",
                "maximum_detections": 100,
                "postprocessor_implementation": (
                    "torchvision.coco_sparse_prebudget_postprocess.v1"
                ),
                "postprocessor_binding": "postprocess_detections",
                "allowed_category_ids": COCO_SPARSE_CATEGORY_IDS,
                "coco_sparse_filter_policy": (
                    "valid_sparse_coco_before_each_level_topk_and_detections_per_img"
                ),
                "coco_sparse_logit_masking": False,
                "coco_sparse_internal_cap_inflation": False,
                "upstream_postprocessor_source_sha256": (
                    "3a9ab809a3822c8c86175e7268059fd0e4048b3d034fab1c18fe4e1bc0ab299d"
                ),
                "repository_postprocessor_source_sha256": (
                    "5e210ba9a6cc091fe74bcb52217ef69d51eaccc6cd9292fd3ba31c3ef4bb8a53"
                ),
                "repository_postprocessor_callable": (
                    "phycam_eval.eval.torchvision_coco_postprocess."
                    "retinanet_coco_sparse_postprocess_detections"
                ),
            },
        ),
        DetectorAllocation(
            detector_id="detr_r50",
            model_backend="huggingface-transformers-detr",
            model_id=_PUBLICATION_MODEL_CONTRACTS["detr_r50"]["model_id"],
            model_revision=_PUBLICATION_MODEL_CONTRACTS["detr_r50"]["revision"],
            model_artifacts=tuple(_PUBLICATION_MODEL_CONTRACTS["detr_r50"]["artifacts"]),
            profile_ids=(primary,),
            arm_policy="baselines_and_physical_anchors",
            anchor_edge_waves_ref=PUBLICATION_ANCHOR_GRID,
            nms_iou_threshold=None,
            pad_value=(0.485, 0.456, 0.406),
            requested_device="cpu",
            shard_size=size,
            backend_settings={
                "adapter": "transformers.detr_resnet50.detector_input_manual_normalize.v2",
                "maximum_detections": 100,
            },
        ),
    )


def _profile_record(profile_id: str, profile: CameraProfile) -> dict[str, Any]:
    payload = {
        "profile_id": _identifier(profile_id, label="profile_id"),
        "camera_profile_sha256": profile.profile_hash,
        "camera_profile": profile.to_dict(),
    }
    return {**payload, "profile_record_sha256": canonical_sha256(payload)}


def _arm_record(*, profile_id: str, arm_id: str, condition: ExperimentCondition) -> dict[str, Any]:
    payload = {
        "profile_id": _identifier(profile_id, label="profile_id"),
        "arm_id": _identifier(arm_id, label="arm_id"),
        "condition_sha256": condition.condition_hash,
        "condition": condition.to_dict(),
    }
    return {**payload, "arm_sha256": canonical_sha256(payload)}


def _match_record(value: MechanismMatch | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, MechanismMatch):
        return value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("comparator matcher must return MechanismMatch values or mappings")
    return json_value(value)


def _condition_arms(
    profile_id: str,
    profile: CameraProfile,
    defocus_grid: Sequence[float],
    matcher: Callable[[CameraProfile, float], Sequence[MechanismMatch | Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    binding = ConditionBinding.from_profile(profile, (0,))
    result = [
        _arm_record(
            profile_id=profile_id,
            arm_id="untouched_input",
            condition=BaselineCondition(binding, BaselineKind.UNTOUCHED_INPUT),
        ),
        _arm_record(
            profile_id=profile_id,
            arm_id="modeled_neutral",
            condition=BaselineCondition(binding, BaselineKind.MODELED_NEUTRAL),
        ),
    ]
    matches_by_waves: dict[float, tuple[dict[str, Any], ...]] = {}
    for waves in defocus_grid:
        raw_matches = matcher(profile, waves)
        if isinstance(raw_matches, (str, bytes, bytearray)) or not isinstance(
            raw_matches, Sequence
        ):
            raise TypeError("comparator matcher must return an ordered sequence")
        matches = tuple(_match_record(value) for value in raw_matches)
        families = tuple(match.get("comparator_family") for match in matches)
        if len(matches) != len(_KNOWN_COMPARATOR_FAMILIES) or set(families) != (
            _KNOWN_COMPARATOR_FAMILIES
        ):
            raise ValueError("each defocus coordinate requires exactly the three comparators")
        matches_by_waves[waves] = tuple(
            sorted(matches, key=lambda item: str(item["comparator_family"]).encode("utf-8"))
        )

    for waves in defocus_grid:
        token = _number_token(waves)
        result.append(
            _arm_record(
                profile_id=profile_id,
                arm_id=f"physical_w_{token}",
                condition=PhysicalDefocusCondition(binding, waves),
            )
        )
        for match in matches_by_waves[waves]:
            if float(match.get("target_edge_waves_ref")) != waves:
                raise ValueError("comparator match target differs from its defocus coordinate")
            family = str(match["comparator_family"])
            result.append(
                _arm_record(
                    profile_id=profile_id,
                    arm_id=f"comparator_{family}_w_{token}",
                    condition=MechanismComparatorCondition(binding, match),
                )
            )
    return tuple(result)


def _cell_record(
    allocation: DetectorAllocation,
    arm: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "record_type": "coco_study_execution_cell",
        "detector_id": allocation.detector_id,
        "detector_allocation_sha256": allocation.allocation_sha256,
        "profile_id": arm["profile_id"],
        "arm_id": arm["arm_id"],
        "condition_sha256": arm["condition_sha256"],
    }
    return {**payload, "cell_sha256": canonical_sha256(payload)}


def _analysis_protocol(
    *,
    primary_profile_id: str,
    primary_detector_id: str,
    defocus_grid: Sequence[float],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return {
        "primary_metric": {
            "name": "COCO_AP",
            "iou_thresholds": list(COCO_IOU_THRESHOLDS),
            "area_range": "all",
            "maximum_detections": COCO_MAXIMUM_DETECTIONS,
        },
        "primary_estimand": {
            "name": "physical_minus_gaussian_AP_curve_auc",
            "primary_profile_id": primary_profile_id,
            "primary_detector_id": primary_detector_id,
            "ordered_edge_waves_ref": list(defocus_grid),
            "integration": "trapezoidal_over_declared_edge_waves_grid",
            "contrast_orientation": "physical_AP_minus_matched_gaussian_AP",
        },
        "uncertainty": {
            "method": "paired_image_cluster_cached_coco_percentile_bootstrap_v2",
            "resampling_unit": "native_COCO_image_id",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "two_sided": True,
        },
        "multiplicity": {
            "primary_contrasts": 1,
            "secondary_results": "descriptive_with_interval_no_hypothesis_test",
        },
    }


def publication_reproduction_contract() -> dict[str, Any]:
    """Return the dataset-free, machine-readable publication rerun contract.

    The summary is derived from the same constants and detector allocations
    used by :func:`build_publication_study_plan`. It lets an auditor inspect
    the intended defaults without first obtaining COCO or model artifacts. It
    describes a reproduction of the same frozen scientific protocol, not an
    attestation for the historical run that produced the manuscript values.
    """

    allocations = publication_detector_allocations(
        ("primary", "replication"),
        primary_profile_id="primary",
    )
    payload = {
        "schema_version": 1,
        "record_type": "phycam_publication_reproduction_contract",
        "implementation_id": STUDY_IMPLEMENTATION_ID,
        "protocol_requirements_id": PUBLICATION_PROTOCOL_REQUIREMENTS_ID,
        "dataset": {
            "name": "COCO val2017",
            "ordered_selection": "all_images_sorted_by_image_id",
            "expected_image_count": PUBLICATION_IMAGE_COUNT,
            "dataset_sha256": FULL_COCO_VAL2017_DATASET_SHA256,
            "selection_sha256": FULL_COCO_VAL2017_SELECTION_SHA256,
            "annotation_sha256": COCO_VAL2017_ANNOTATION_SHA256,
        },
        "rendering_scope": {
            "mode": "decoded_ldr_srgb_redegradation",
            "physical_arm": (
                "representative_rgb_pupil_psf_otf_with_exact_equal_grid_cell_average"
            ),
            "disabled_in_reported_study": [
                "sensor_noise",
                "cfa_and_demosaicing",
                "gain_and_adc",
                "motion_and_rolling_readout",
                "separate_tone_curve",
            ],
        },
        "design": {
            "ordered_profile_hashes": [
                {"profile_id": profile_id, "camera_profile_sha256": digest}
                for profile_id, digest in PUBLICATION_PROFILE_HASHES
            ],
            "ordered_edge_waves_ref": list(PUBLICATION_DEFOCUS_GRID),
            "physical_anchor_edge_waves_ref": list(PUBLICATION_ANCHOR_GRID),
            "expected_execution_cell_count": PUBLICATION_EXECUTION_CELL_COUNT,
            "detector_allocations": [allocation.to_dict() for allocation in allocations],
        },
        "analysis_protocol": _analysis_protocol(
            primary_profile_id="primary",
            primary_detector_id="yolov8n",
            defocus_grid=PUBLICATION_DEFOCUS_GRID,
            bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
            bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        ),
        "evidence_scope": {
            "matching_rerun_tier": "publication_protocol_reproduction",
            "confirmatory_eligible": False,
            "historical_run_attested_here": False,
        },
    }
    return {**payload, "protocol_contract_sha256": canonical_sha256(payload)}


def _validated_bootstrap_replicates(value: object) -> int:
    replicates = positive_int(value, field_name="bootstrap_replicates")
    if replicates < 2:
        raise ValueError("bootstrap_replicates must be at least two")
    return replicates


def _validated_bootstrap_seed(value: object) -> int:
    seed = positive_int(value, field_name="bootstrap_seed", allow_zero=True)
    if seed > MAX_BOOTSTRAP_SEED:
        raise ValueError(f"bootstrap_seed must not exceed {MAX_BOOTSTRAP_SEED}")
    return seed


def derive_study_evidence_tier_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a complete plan payload against the published scientific protocol."""

    implementation_id = record.get("implementation_id")
    current_v11 = (
        implementation_id == STUDY_IMPLEMENTATION_ID
        and record.get("plan_version") == STUDY_PLAN_VERSION
    )
    analysis = record["analysis_protocol"]
    primary = analysis["primary_estimand"]
    uncertainty = analysis["uncertainty"]
    profiles = tuple(
        (str(value["profile_id"]), str(value["camera_profile_sha256"]))
        for value in record["profiles"]
    )
    allocations = tuple(json_value(value) for value in record["detector_allocations"])
    expected_allocations = tuple(
        allocation.to_dict()
        for allocation in publication_detector_allocations(
            ("primary", "replication"),
            primary_profile_id="primary",
        )
    )
    waves = tuple(float(value) for value in record["design"]["ordered_edge_waves_ref"])
    canonical_arms = tuple(
        arm
        for profile_record in record["profiles"]
        for arm in _condition_arms(
            str(profile_record["profile_id"]),
            CameraProfile.from_dict(profile_record["camera_profile"]),
            waves,
            match_common_neutral_comparators,
        )
    )
    checks = {
        "exact_publication_implementation": (
            current_v11
            and record["design"]["physical_arm"]
            == "exact_equal_grid_cell_average_representative_rgb_pupil_defocus_v1"
            and record["design"]["comparator_matching"]
            == (
                "rec709_luminance_first_downward_mtf50_on_exact_equal_grid_"
                "cell_average_common_W0_branch_v1"
            )
        ),
        "study_id_is_frozen_publication_v1": (
            record["study_id"] == "phycam_coco_static_ldr_publication_v1"
        ),
        "complete_coco_val2017_dataset_identity": (
            record["dataset"]["dataset_sha256"] == FULL_COCO_VAL2017_DATASET_SHA256
        ),
        "complete_coco_val2017_plan_selection": (
            record["image_selection"]["count"] == PUBLICATION_IMAGE_COUNT
            and record["image_selection"]["selection_sha256"] == FULL_COCO_VAL2017_SELECTION_SHA256
        ),
        "official_coco_val2017_annotation_artifact": (
            record["dataset"]["annotation_artifact"]["sha256"] == COCO_VAL2017_ANNOTATION_SHA256
        ),
        "exact_primary_and_replication_profiles": profiles == PUBLICATION_PROFILE_HASHES,
        "exact_publication_defocus_grid": (
            tuple(record["design"]["ordered_edge_waves_ref"]) == PUBLICATION_DEFOCUS_GRID
        ),
        "exact_default_detector_allocations": allocations == expected_allocations,
        "exact_canonical_condition_arms": (
            json_value(record["arms"]) == json_value(canonical_arms)
        ),
        "complete_default_execution_cell_allocation": (
            len(record["execution_cells"]) == PUBLICATION_EXECUTION_CELL_COUNT
        ),
        "prespecified_primary_detector_and_profile": (
            primary["primary_detector_id"] == "yolov8n"
            and primary["primary_profile_id"] == "primary"
        ),
        "prespecified_primary_estimand": (
            primary["name"] == "physical_minus_gaussian_AP_curve_auc"
            and primary["contrast_orientation"] == "physical_AP_minus_matched_gaussian_AP"
            and primary["integration"] == "trapezoidal_over_declared_edge_waves_grid"
        ),
        "prespecified_bootstrap_iterations_and_seed": (
            uncertainty["replicates"] == DEFAULT_BOOTSTRAP_REPLICATES
            and uncertainty["seed"] == DEFAULT_BOOTSTRAP_SEED
        ),
        "prespecified_paired_uncertainty_method": (
            uncertainty["method"] == "paired_image_cluster_cached_coco_percentile_bootstrap_v2"
        ),
    }
    protocol_match = all(checks.values())
    return {
        "contract_id": _plan_contract_id(),
        "publication_protocol_match": protocol_match,
        "confirmatory_eligible": False,
        "tier": (
            "publication_protocol_reproduction"
            if protocol_match
            else "exploratory_protocol_variant"
        ),
        "checks": checks,
    }


def build_publication_study_plan(
    *,
    dataset: Mapping[str, Any],
    image_ids: Sequence[int],
    profiles: Mapping[str, CameraProfile],
    primary_profile_id: str,
    detector_allocations: Sequence[DetectorAllocation] | None = None,
    defocus_grid: Sequence[float] = PUBLICATION_DEFOCUS_GRID,
    primary_detector_id: str = "yolov8n",
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    study_id: str = "phycam_coco_static_ldr_publication_v1",
    matcher: Callable[
        [CameraProfile, float], Sequence[MechanismMatch | Mapping[str, Any]]
    ] = match_common_neutral_comparators,
) -> "CocoStudyPlan":
    """Build the complete deterministic publication plan before inference."""

    replicates = _validated_bootstrap_replicates(bootstrap_replicates)
    seed = _validated_bootstrap_seed(bootstrap_seed)
    selected = _image_ids(image_ids)
    dataset_record = _validate_dataset_identity(dataset, image_ids=selected)
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("profiles must be a nonempty mapping")
    profile_items = tuple(
        sorted(
            (
                (_identifier(profile_id, label="profile_id"), profile)
                for profile_id, profile in profiles.items()
            ),
            key=lambda item: item[0].encode("utf-8"),
        )
    )
    if len({profile_id for profile_id, _ in profile_items}) != len(profile_items):
        raise ValueError("profile IDs must be unique after Unicode normalization")
    primary_profile = _identifier(primary_profile_id, label="primary_profile_id")
    if primary_profile not in {profile_id for profile_id, _ in profile_items}:
        raise ValueError("primary_profile_id is not present in profiles")
    for _, profile in profile_items:
        _validate_publication_profile(profile)
    waves = _float_grid(defocus_grid, label="defocus_grid")
    if not callable(matcher):
        raise TypeError("matcher must be callable")
    profile_records = tuple(
        _profile_record(profile_id, profile) for profile_id, profile in profile_items
    )
    arms = tuple(
        arm
        for profile_id, profile in profile_items
        for arm in _condition_arms(profile_id, profile, waves, matcher)
    )
    if len({(arm["profile_id"], arm["arm_id"]) for arm in arms}) != len(arms):
        raise RuntimeError("generated study arm IDs are not unique")

    if detector_allocations is None:
        allocations = publication_detector_allocations(
            tuple(profile_id for profile_id, _ in profile_items),
            primary_profile_id=primary_profile,
        )
    else:
        if isinstance(detector_allocations, (str, bytes, bytearray)) or not isinstance(
            detector_allocations, Sequence
        ):
            raise TypeError("detector_allocations must be an ordered sequence")
        allocations = tuple(detector_allocations)
        if not allocations or not all(
            isinstance(value, DetectorAllocation) for value in allocations
        ):
            raise TypeError("detector_allocations must contain DetectorAllocation values")
    if len({allocation.detector_id for allocation in allocations}) != len(allocations):
        raise ValueError("detector allocation IDs must be unique")
    known_profiles = {profile_id for profile_id, _ in profile_items}
    for allocation in allocations:
        unknown = set(allocation.profile_ids).difference(known_profiles)
        if unknown:
            raise ValueError(f"detector allocation references unknown profiles: {sorted(unknown)}")
        if any(anchor not in waves for anchor in allocation.anchor_edge_waves_ref):
            raise ValueError("detector allocation anchor lies outside the defocus grid")
    primary_detector = _identifier(primary_detector_id, label="primary_detector_id")
    if primary_detector not in {allocation.detector_id for allocation in allocations}:
        raise ValueError("primary_detector_id is not declared in detector allocations")
    primary_allocation = next(
        allocation for allocation in allocations if allocation.detector_id == primary_detector
    )
    if primary_profile not in primary_allocation.profile_ids or (
        primary_allocation.arm_policy != "all_arms"
    ):
        raise ValueError(
            "primary estimand requires its detector to run every arm on the primary profile"
        )

    cells = tuple(
        _cell_record(allocation, arm)
        for allocation in allocations
        for arm in arms
        if allocation.accepts(
            str(arm["profile_id"]),
            condition_from_dict(arm["condition"]),
        )
    )
    if not cells or len({cell["cell_sha256"] for cell in cells}) != len(cells):
        raise RuntimeError("generated execution cells are empty or nonunique")
    normalized_study_id = _identifier(study_id, label="study_id")
    payload = {
        "schema_version": 4,
        "record_type": "phycam_coco_publication_study_plan",
        "plan_version": STUDY_PLAN_VERSION,
        "implementation_id": STUDY_IMPLEMENTATION_ID,
        "study_id": normalized_study_id,
        "dataset": dataset_record,
        "image_selection": _selection_record(selected),
        "profiles": list(profile_records),
        "design": {
            "data_tier": DataMode.LDR_REDEGRADATION.value,
            "geometry": "native_stored_aspect_active_sensor_roi",
            "motion": "disabled_static_optics_only",
            "ordered_edge_waves_ref": list(waves),
            "baseline_arms": ["untouched_input", "modeled_neutral"],
            "physical_arm": ("exact_equal_grid_cell_average_representative_rgb_pupil_defocus_v1"),
            "comparator_families": sorted(_KNOWN_COMPARATOR_FAMILIES),
            "comparator_matching": (
                "rec709_luminance_first_downward_mtf50_on_exact_equal_grid_"
                "cell_average_common_W0_branch_v1"
            ),
            "realization_ids": [0],
            "inference_execution": deterministic_inference_execution_contract(
                DEFAULT_INFERENCE_SEED
            ),
        },
        "arms": list(arms),
        "detector_allocations": [allocation.to_dict() for allocation in allocations],
        "analysis_protocol": _analysis_protocol(
            primary_profile_id=primary_profile,
            primary_detector_id=primary_detector,
            defocus_grid=waves,
            bootstrap_replicates=replicates,
            bootstrap_seed=seed,
        ),
        "execution_cells": list(cells),
    }
    payload["evidence_tier"] = derive_study_evidence_tier_record(payload)
    return CocoStudyPlan({**payload, "study_plan_sha256": canonical_sha256(payload)})


def _validate_profile_record(value: Mapping[str, Any]) -> tuple[str, CameraProfile, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("study profiles must be mappings")
    record = json_value(value)
    if set(record) != {
        "profile_id",
        "camera_profile_sha256",
        "camera_profile",
        "profile_record_sha256",
    }:
        raise ValueError("study profile record has missing or unknown fields")
    profile_id = _identifier(record.get("profile_id"), label="profile_id")
    profile = CameraProfile.from_dict(record.get("camera_profile"))
    _validate_publication_profile(profile)
    if record.get("camera_profile_sha256") != profile.profile_hash:
        raise ValueError("study profile hash does not match its embedded profile")
    payload = {key: item for key, item in record.items() if key != "profile_record_sha256"}
    if record.get("profile_record_sha256") != canonical_sha256(payload):
        raise ValueError("study profile record identity does not match")
    return profile_id, profile, record


def _validate_arm_record(
    value: Mapping[str, Any], *, profiles: Mapping[str, CameraProfile]
) -> tuple[str, str, ExperimentCondition, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("study arms must be mappings")
    record = json_value(value)
    if set(record) != {
        "profile_id",
        "arm_id",
        "condition_sha256",
        "condition",
        "arm_sha256",
    }:
        raise ValueError("study arm has missing or unknown fields")
    profile_id = _identifier(record.get("profile_id"), label="profile_id")
    arm_id = _identifier(record.get("arm_id"), label="arm_id")
    if profile_id not in profiles:
        raise ValueError("study arm references an unknown profile")
    condition = condition_from_dict(record.get("condition"))
    condition.binding.assert_profile(profiles[profile_id])
    if condition.realization_ids != (0,):
        raise ValueError("publication arms require realization_ids=(0,)")
    if isinstance(condition, BaselineCondition):
        expected_arm_id = condition.kind.value
    elif isinstance(condition, PhysicalDefocusCondition):
        expected_arm_id = f"physical_w_{_number_token(condition.edge_waves_ref)}"
    elif isinstance(condition, MechanismComparatorCondition):
        expected_arm_id = (
            f"comparator_{condition.comparator_family}_w_"
            f"{_number_token(condition.target_edge_waves_ref)}"
        )
        match = condition.match
        rebuilt_match = MechanismMatch(
            comparator_family=match.get("comparator_family"),
            target_edge_waves_ref=match.get("target_edge_waves_ref"),
            target_mtf50_cycles_per_pixel=match.get("target_mtf50_cycles_per_pixel"),
            neutral_mtf_at_target=match.get("neutral_mtf_at_target"),
            config=match.get("config"),
            config_sha256=match.get("config_sha256"),
            achieved_mtf50_cycles_per_pixel=match.get("achieved_mtf50_cycles_per_pixel"),
            relative_match_error=match.get("relative_match_error"),
            camera_profile_sha256=match.get("camera_profile_sha256"),
            neutral_model_sha256=match.get("neutral_model_sha256"),
            target_model_sha256=match.get("target_model_sha256"),
        )
        if rebuilt_match.to_dict() != json_value(match):
            raise ValueError("study comparator match has a noncanonical matching contract")
        if rebuilt_match.relative_match_error > 0.005:
            raise ValueError("study comparator exceeds the predeclared 0.5% MTF50 tolerance")
    else:
        raise TypeError("publication study arms support only baselines, defocus, and comparators")
    if arm_id != expected_arm_id:
        raise ValueError("study arm ID does not describe its condition")
    if record.get("condition_sha256") != condition.condition_hash:
        raise ValueError("study arm condition hash does not match")
    payload = {key: item for key, item in record.items() if key != "arm_sha256"}
    if record.get("arm_sha256") != canonical_sha256(payload):
        raise ValueError("study arm identity does not match")
    return profile_id, arm_id, condition, record


def _validate_plan_record(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("study plan must be a mapping")
    record = json_value(value)
    if record.get("schema_version") != 4 or record.get("record_type") != (
        "phycam_coco_publication_study_plan"
    ):
        raise ValueError("study plan requires the schema-v4 publication plan type")
    if record.get("implementation_id") != STUDY_IMPLEMENTATION_ID:
        raise ValueError("study plan implementation is unsupported")
    if set(record) != _PLAN_KEYS:
        raise ValueError("study plan has missing or unknown fields")
    if record.get("plan_version") != STUDY_PLAN_VERSION:
        raise ValueError(f"study plan requires plan_version {STUDY_PLAN_VERSION}")
    _identifier(record.get("study_id"), label="study_id")
    selection = record.get("image_selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "ordered_image_ids",
        "count",
        "first",
        "last",
        "selection_sha256",
    }:
        raise ValueError("study plan image selection is noncanonical")
    selected = _image_ids(selection.get("ordered_image_ids", ()))
    if _selection_record(selected) != selection:
        raise ValueError("study plan image selection identity does not match")
    _validate_dataset_identity(record.get("dataset"), image_ids=selected)

    raw_profiles = record.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("study plan profiles must be a nonempty array")
    parsed_profiles = tuple(_validate_profile_record(value) for value in raw_profiles)
    profile_ids = tuple(item[0] for item in parsed_profiles)
    if tuple(sorted(profile_ids, key=lambda item: item.encode("utf-8"))) != profile_ids:
        raise ValueError("study plan profiles are not in canonical profile-ID order")
    if len(set(profile_ids)) != len(profile_ids):
        raise ValueError("study plan profile IDs are not unique")
    profiles = {item[0]: item[1] for item in parsed_profiles}

    design = record.get("design")
    if not isinstance(design, Mapping) or set(design) != {
        "data_tier",
        "geometry",
        "motion",
        "ordered_edge_waves_ref",
        "baseline_arms",
        "physical_arm",
        "comparator_families",
        "comparator_matching",
        "realization_ids",
        "inference_execution",
    }:
        raise ValueError("study plan design is noncanonical")
    waves = _float_grid(design.get("ordered_edge_waves_ref", ()), label="defocus_grid")
    expected_design = {
        "data_tier": DataMode.LDR_REDEGRADATION.value,
        "geometry": "native_stored_aspect_active_sensor_roi",
        "motion": "disabled_static_optics_only",
        "ordered_edge_waves_ref": list(waves),
        "baseline_arms": ["untouched_input", "modeled_neutral"],
        "physical_arm": "exact_equal_grid_cell_average_representative_rgb_pupil_defocus_v1",
        "comparator_families": sorted(_KNOWN_COMPARATOR_FAMILIES),
        "comparator_matching": (
            "rec709_luminance_first_downward_mtf50_on_exact_equal_grid_"
            "cell_average_common_W0_branch_v1"
        ),
        "realization_ids": [0],
        "inference_execution": deterministic_inference_execution_contract(DEFAULT_INFERENCE_SEED),
    }
    if design != expected_design:
        raise ValueError("study plan design contract drifted")

    raw_arms = record.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ValueError("study plan arms must be a nonempty array")
    parsed_arms = tuple(_validate_arm_record(value, profiles=profiles) for value in raw_arms)
    arm_keys = tuple((item[0], item[1]) for item in parsed_arms)
    if len(set(arm_keys)) != len(arm_keys):
        raise ValueError("study plan arm IDs are not unique within profiles")
    expected_arm_keys = tuple(
        (profile_id, arm_id)
        for profile_id in profile_ids
        for arm_id in (
            "untouched_input",
            "modeled_neutral",
            *(
                arm_id
                for waves_value in waves
                for arm_id in (
                    f"physical_w_{_number_token(waves_value)}",
                    *(
                        f"comparator_{family}_w_{_number_token(waves_value)}"
                        for family in sorted(_KNOWN_COMPARATOR_FAMILIES)
                    ),
                )
            ),
        )
    )
    if arm_keys != expected_arm_keys:
        raise ValueError("study plan arms are not in canonical profile/condition order")
    expected_arm_count = len(profiles) * (2 + 4 * len(waves))
    if len(parsed_arms) != expected_arm_count:
        raise ValueError(
            "study plan does not contain the complete baseline/physical/comparator grid"
        )
    for profile_id in profile_ids:
        profile_conditions = [item[2] for item in parsed_arms if item[0] == profile_id]
        baseline_kinds = {
            condition.kind
            for condition in profile_conditions
            if isinstance(condition, BaselineCondition)
        }
        physical_waves = {
            condition.edge_waves_ref
            for condition in profile_conditions
            if isinstance(condition, PhysicalDefocusCondition)
        }
        comparator_pairs = {
            (condition.target_edge_waves_ref, condition.comparator_family)
            for condition in profile_conditions
            if isinstance(condition, MechanismComparatorCondition)
        }
        expected_pairs = {
            (waves_value, family) for waves_value in waves for family in _KNOWN_COMPARATOR_FAMILIES
        }
        if (
            baseline_kinds != set(BaselineKind)
            or physical_waves != set(waves)
            or (comparator_pairs != expected_pairs)
        ):
            raise ValueError("study plan arm grid is incomplete or duplicated")

    raw_allocations = record.get("detector_allocations")
    if not isinstance(raw_allocations, list) or not raw_allocations:
        raise ValueError("study plan detector allocations must be a nonempty array")
    allocations = tuple(DetectorAllocation.from_dict(value) for value in raw_allocations)
    if len({allocation.detector_id for allocation in allocations}) != len(allocations):
        raise ValueError("study plan detector allocations are not unique")
    for allocation in allocations:
        if set(allocation.profile_ids).difference(profiles):
            raise ValueError("study plan detector allocation references an unknown profile")
        if any(anchor not in waves for anchor in allocation.anchor_edge_waves_ref):
            raise ValueError("study plan detector anchor lies outside its design grid")

    expected_cells = [
        _cell_record(allocation, arm_record)
        for allocation in allocations
        for profile_id, _, condition, arm_record in parsed_arms
        if allocation.accepts(profile_id, condition)
    ]
    if record.get("execution_cells") != expected_cells:
        raise ValueError("study plan execution cells do not match its arms and allocations")

    analysis = record.get("analysis_protocol")
    if not isinstance(analysis, Mapping):
        raise TypeError("study plan analysis protocol must be a mapping")
    try:
        primary = analysis["primary_estimand"]
        uncertainty = analysis["uncertainty"]
    except KeyError as exc:
        raise ValueError("study plan analysis protocol is incomplete") from exc
    if not isinstance(primary, Mapping) or not isinstance(uncertainty, Mapping):
        raise TypeError("study plan analysis protocol sections must be mappings")
    expected_analysis = _analysis_protocol(
        primary_profile_id=_identifier(
            primary.get("primary_profile_id"), label="primary_profile_id"
        ),
        primary_detector_id=_identifier(
            primary.get("primary_detector_id"), label="primary_detector_id"
        ),
        defocus_grid=waves,
        bootstrap_replicates=_validated_bootstrap_replicates(uncertainty.get("replicates")),
        bootstrap_seed=_validated_bootstrap_seed(uncertainty.get("seed")),
    )
    if analysis != expected_analysis:
        raise ValueError("study plan analysis protocol drifted")
    if primary["primary_profile_id"] not in profiles:
        raise ValueError("primary profile is not in the study plan")
    if primary["primary_detector_id"] not in {allocation.detector_id for allocation in allocations}:
        raise ValueError("primary detector is not in the study plan")
    primary_allocation = next(
        allocation
        for allocation in allocations
        if allocation.detector_id == primary["primary_detector_id"]
    )
    if primary["primary_profile_id"] not in primary_allocation.profile_ids or (
        primary_allocation.arm_policy != "all_arms"
    ):
        raise ValueError("primary detector allocation cannot identify the primary estimand")

    evidence_tier = record.get("evidence_tier")
    if not isinstance(evidence_tier, Mapping):
        raise TypeError("study plan evidence_tier must be a mapping")
    expected_evidence_tier = derive_study_evidence_tier_record(record)
    if evidence_tier != expected_evidence_tier:
        raise ValueError("study plan evidence tier differs from the frozen eligibility contract")

    supplied = record.get("study_plan_sha256")
    payload = {key: item for key, item in record.items() if key != "study_plan_sha256"}
    if supplied != canonical_sha256(payload):
        raise ValueError("study_plan_sha256 does not match the plan payload")
    return record


@dataclass(frozen=True, slots=True)
class CocoStudyPlan:
    """Validated immutable publication study plan."""

    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", _immutable_mapping(_validate_plan_record(self.record)))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CocoStudyPlan":
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        return json_value(self.record)

    @property
    def study_plan_sha256(self) -> str:
        return str(self.record["study_plan_sha256"])

    @property
    def image_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.record["image_selection"]["ordered_image_ids"])

    @property
    def confirmatory_eligible(self) -> bool:
        return bool(self.record["evidence_tier"]["confirmatory_eligible"])

    @property
    def profiles(self) -> Mapping[str, CameraProfile]:
        return MappingProxyType(
            {
                str(value["profile_id"]): CameraProfile.from_dict(value["camera_profile"])
                for value in self.record["profiles"]
            }
        )

    @property
    def allocations(self) -> tuple[DetectorAllocation, ...]:
        return tuple(
            DetectorAllocation.from_dict(value) for value in self.record["detector_allocations"]
        )

    def allocation(self, detector_id: str) -> DetectorAllocation:
        selected = _identifier(detector_id, label="detector_id")
        for allocation in self.allocations:
            if allocation.detector_id == selected:
                return allocation
        raise KeyError(f"detector {selected!r} is not allocated in this study")

    def cells(self, detector_id: str | None = None) -> tuple[Mapping[str, Any], ...]:
        if detector_id is None:
            return tuple(_immutable_mapping(value) for value in self.record["execution_cells"])
        selected = _identifier(detector_id, label="detector_id")
        return tuple(
            _immutable_mapping(value)
            for value in self.record["execution_cells"]
            if value["detector_id"] == selected
        )

    def condition(self, profile_id: str, arm_id: str) -> ExperimentCondition:
        selected_profile = _identifier(profile_id, label="profile_id")
        selected_arm = _identifier(arm_id, label="arm_id")
        for value in self.record["arms"]:
            if value["profile_id"] == selected_profile and value["arm_id"] == selected_arm:
                return condition_from_dict(json_value(value["condition"]))
        raise KeyError(f"study arm {selected_profile}/{selected_arm} is not declared")


def _record_bytes(value: Mapping[str, Any]) -> bytes:
    normalized = json_value(value)
    return (
        json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_or_validate(path: Path, value: Mapping[str, Any]) -> bool:
    payload = _record_bytes(value)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"publication path is not a regular file: {path}")
        if path.read_bytes() != payload:
            raise ValueError(f"existing publication identity drifted: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # The temporary and destination files share one directory, so a
            # hard-link publication is atomic and, unlike os.replace, cannot
            # overwrite a concurrent writer's result.
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"concurrent publication path is unsafe: {path}") from None
            if path.read_bytes() != payload:
                raise ValueError(f"concurrent publication identity drifted: {path}")
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def materialize_study_plan(path: str | Path, plan: CocoStudyPlan) -> bool:
    """Publish a plan atomically, or validate the exact existing plan for resume."""

    if not isinstance(plan, CocoStudyPlan):
        raise TypeError("plan must be a CocoStudyPlan")
    return _publish_or_validate(Path(path), plan.to_dict())


def load_study_plan(path: str | Path) -> CocoStudyPlan:
    """Read and fully validate one canonical materialized study plan."""

    plan_path = Path(path)
    if not plan_path.is_file():
        raise FileNotFoundError(f"study plan is missing: {plan_path}")
    try:
        raw = plan_path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to parse study plan: {plan_path}") from exc
    plan = CocoStudyPlan.from_dict(value)
    if raw != _record_bytes(plan.to_dict()):
        raise ValueError("study plan file is not in canonical deterministic JSON form")
    return plan


def validate_detector_model_contract(
    allocation: DetectorAllocation,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate model provenance and its declared detector-boundary settings."""

    if not isinstance(allocation, DetectorAllocation):
        raise TypeError("allocation must be a DetectorAllocation")
    validated = validate_model_identity(model)
    if validated["backend"] != allocation.model_backend:
        raise ValueError("model backend differs from the detector allocation")
    expected_model_contract = {
        "model_id": allocation.model_id,
        "revision": allocation.model_revision,
        "artifacts": [json_value(value) for value in allocation.model_artifacts],
    }
    observed_model_contract = {
        "model_id": validated["model_id"],
        "revision": validated["revision"],
        "artifacts": validated["artifacts"],
    }
    if observed_model_contract != expected_model_contract:
        raise ValueError(
            "model ID, revision, or artifact bytes differ from the frozen detector allocation"
        )
    implementation = validated.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("model identity must disclose its adapter implementation contract")
    if implementation.get("confidence_threshold") != allocation.confidence_threshold:
        raise ValueError("model confidence threshold differs from the detector allocation")
    if implementation.get("output_label_space") != allocation.label_space:
        raise ValueError("model output label space differs from the detector allocation")
    if allocation.nms_iou_threshold is None:
        postprocessing = implementation.get("postprocessing")
        if not isinstance(postprocessing, Mapping) or postprocessing.get("nms") is not None:
            raise ValueError("non-NMS detector allocation requires an explicit nms=null contract")
    elif implementation.get("nms_iou_threshold") != allocation.nms_iou_threshold:
        raise ValueError("model NMS threshold differs from the detector allocation")
    maximum = implementation.get("maximum_detections")
    if maximum is not None and maximum != COCO_MAXIMUM_DETECTIONS:
        raise ValueError("model maximum detections conflicts with COCO maxDets=100")
    input_contract = implementation.get("input_contract")
    if isinstance(input_contract, Mapping) and input_contract.get("shape_hw") is not None:
        if tuple(input_contract["shape_hw"]) != allocation.input_shape:
            raise ValueError("model detector-input shape differs from the allocation")
    for key, expected in allocation.backend_settings.items():
        if json_value(implementation.get(key)) != json_value(expected):
            raise ValueError(f"model backend setting {key!r} differs from the allocation")
    return validated


def _runtime_identity_valid(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime identity must be a mapping")
    record = json_value(value)
    supplied = record.get("runtime_identity_sha256")
    payload = {key: item for key, item in record.items() if key != "runtime_identity_sha256"}
    if supplied != canonical_sha256(payload):
        raise ValueError("runtime identity hash does not match its payload")
    if record.get("schema_version") != 2 or record.get("record_type") != (
        "runtime_reproducibility_identity"
    ):
        raise ValueError("run requires a schema-v2 runtime reproducibility identity")
    device = record.get("detector_device")
    if not isinstance(device, Mapping) or not torch_device_attestation_matches(
        device.get("resolved_requested"), device.get("resolved_actual")
    ):
        raise ValueError("runtime detector device attestation is invalid")
    return record


def _object_type_id(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _callable_id(value: object) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
        return _object_type_id(value)
    return f"{module}.{qualname}"


def _study_execution_engine_record(
    *,
    detector_id: str,
    subset: NativeCOCODataset,
    adapter: DetectorAdapter,
    runtime_identity_factory: RuntimeIdentityFactory,
    shard_runner: ShardRunner,
    shard_merger: ShardMerger,
) -> dict[str, Any]:
    """Describe the local, resumable execution components used by one run."""

    _identifier(detector_id, label="detector_id")
    exact_streaming_dataset = type(subset) is LazyNativeCOCOSubset and subset.loader_attested
    adapter_class = _object_type_id(adapter)
    exact_adapter = adapter_class == _OFFICIAL_DETECTOR_ADAPTER_CLASS_IDS.get(detector_id)
    exact_runtime = runtime_identity_factory is runtime_reproducibility_identity
    exact_runner = shard_runner is run_coco_ldr_condition_shard
    exact_merger = shard_merger is merge_prediction_shards
    checks = {
        "exact_streaming_loader_dataset": exact_streaming_dataset,
        "exact_detector_adapter_class": exact_adapter,
        "exact_runtime_identity_factory": exact_runtime,
        "exact_condition_shard_runner": exact_runner,
        "exact_prediction_shard_merger": exact_merger,
    }
    return {
        "schema_version": 3,
        "record_type": "phycam_coco_study_execution_engine",
        "implementation_id": _EXECUTION_ENGINE_IMPLEMENTATION_ID,
        "mode": "local_reproduction",
        "publication_stack_match": all(checks.values()),
        "dataset_class": _object_type_id(subset),
        "dataset_loader_attested": bool(subset.loader_attested),
        "adapter_class": adapter_class,
        "runtime_identity_factory": _callable_id(runtime_identity_factory),
        "condition_shard_runner": _callable_id(shard_runner),
        "prediction_shard_merger": _callable_id(shard_merger),
        "component_checks": checks,
    }


def _validate_execution_engine_record(
    value: Mapping[str, Any],
    *,
    detector_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("study execution engine must be a mapping")
    record = json_value(value)
    expected_keys = {
        "schema_version",
        "record_type",
        "implementation_id",
        "mode",
        "publication_stack_match",
        "dataset_class",
        "dataset_loader_attested",
        "adapter_class",
        "runtime_identity_factory",
        "condition_shard_runner",
        "prediction_shard_merger",
        "component_checks",
    }
    if set(record) != expected_keys:
        raise ValueError("study execution engine has missing or unknown fields")
    if (
        record.get("schema_version") != 3
        or record.get("record_type") != "phycam_coco_study_execution_engine"
        or record.get("implementation_id") != _EXECUTION_ENGINE_IMPLEMENTATION_ID
    ):
        raise ValueError("study execution engine identity drifted")
    for key in (
        "dataset_class",
        "adapter_class",
        "runtime_identity_factory",
        "condition_shard_runner",
        "prediction_shard_merger",
    ):
        if not isinstance(record.get(key), str) or not record[key]:
            raise TypeError(f"study execution engine {key} must be a nonempty string")
    if not isinstance(record.get("dataset_loader_attested"), bool):
        raise TypeError("study execution engine dataset_loader_attested must be bool")
    if not isinstance(record.get("publication_stack_match"), bool):
        raise TypeError("study execution engine publication_stack_match must be bool")
    checks = record.get("component_checks")
    expected_check_keys = {
        "exact_streaming_loader_dataset",
        "exact_detector_adapter_class",
        "exact_runtime_identity_factory",
        "exact_condition_shard_runner",
        "exact_prediction_shard_merger",
    }
    if not isinstance(checks, Mapping) or set(checks) != expected_check_keys:
        raise ValueError("study execution engine component checks are noncanonical")
    if any(not isinstance(checks[key], bool) for key in expected_check_keys):
        raise TypeError("study execution engine component checks must be bool")
    _identifier(detector_id, label="detector_id")
    derived_checks = {
        "exact_streaming_loader_dataset": (
            record["dataset_class"] == _OFFICIAL_DATASET_CLASS_ID
            and record["dataset_loader_attested"] is True
        ),
        "exact_detector_adapter_class": (
            record["adapter_class"] == _OFFICIAL_DETECTOR_ADAPTER_CLASS_IDS.get(detector_id)
        ),
        "exact_runtime_identity_factory": (
            record["runtime_identity_factory"] == _OFFICIAL_RUNTIME_FACTORY_ID
        ),
        "exact_condition_shard_runner": (
            record["condition_shard_runner"] == _OFFICIAL_SHARD_RUNNER_ID
        ),
        "exact_prediction_shard_merger": (
            record["prediction_shard_merger"] == _OFFICIAL_SHARD_MERGER_ID
        ),
    }
    if record["component_checks"] != derived_checks:
        raise ValueError("study execution engine component checks do not match their identities")
    if record["publication_stack_match"] is not all(derived_checks.values()):
        raise ValueError("study execution engine publication-stack flag is inconsistent")
    if record.get("mode") != "local_reproduction":
        raise ValueError("study execution engine must describe local reproduction")
    return record


def build_study_run_manifest(
    *,
    plan: CocoStudyPlan,
    allocation: DetectorAllocation,
    model: Mapping[str, Any],
    runtime: Mapping[str, Any],
    preprocessing: LetterboxConfig,
    execution_engine: Mapping[str, Any],
) -> "CocoStudyRunManifest":
    """Bind one detector worker to the frozen study and attested runtime."""

    if not isinstance(plan, CocoStudyPlan):
        raise TypeError("plan must be a CocoStudyPlan")
    if plan.allocation(allocation.detector_id) != allocation:
        raise ValueError("detector allocation differs from the frozen study plan")
    validated_model = validate_detector_model_contract(allocation, model)
    validated_runtime = _runtime_identity_valid(runtime)
    expected_inference_execution = json_value(plan.record["design"]["inference_execution"])
    if validated_runtime.get("inference_execution") != expected_inference_execution:
        raise ValueError(
            "runtime deterministic inference attestation differs from the frozen study plan"
        )
    if validated_runtime["detector_device"]["requested"] != allocation.requested_device:
        raise ValueError("runtime requested device differs from the frozen detector allocation")
    expected_preprocessing = LetterboxConfig(allocation.input_shape, allocation.pad_value)
    if preprocessing != expected_preprocessing:
        raise ValueError("preprocessing differs from the detector allocation")
    cells = plan.cells(allocation.detector_id)
    validated_execution_engine = _validate_execution_engine_record(
        execution_engine,
        detector_id=allocation.detector_id,
    )
    payload = {
        "schema_version": 6,
        "record_type": "phycam_coco_detector_study_run_manifest",
        "study_plan_sha256": plan.study_plan_sha256,
        "dataset_sha256": plan.record["dataset"]["dataset_sha256"],
        "image_selection_sha256": plan.record["image_selection"]["selection_sha256"],
        "detector_allocation": allocation.to_dict(),
        "model": validated_model,
        "runtime": validated_runtime,
        "execution_engine": validated_execution_engine,
        "inference_execution_contract": expected_inference_execution,
        "preprocessing": json_value(preprocessing.identity),
        "ordered_execution_cell_sha256": [cell["cell_sha256"] for cell in cells],
        "output_layout": {
            "path_components": (
                "predictions/{detector_id}/{profile_id}/{arm_id}/shard_{ordinal:05d}.jsonl.gz"
            ),
            "merge_index": "merges/{detector_id}/{profile_id}/{arm_id}.index.json",
            "completion": "completions/{detector_id}.complete.json",
            "path_fields_are_semantic_only": True,
        },
    }
    return CocoStudyRunManifest({**payload, "study_run_sha256": canonical_sha256(payload)})


def _validate_run_manifest_record(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("study run manifest must be a mapping")
    record = json_value(value)
    expected_keys = {
        "schema_version",
        "record_type",
        "study_plan_sha256",
        "dataset_sha256",
        "image_selection_sha256",
        "detector_allocation",
        "model",
        "runtime",
        "execution_engine",
        "inference_execution_contract",
        "preprocessing",
        "ordered_execution_cell_sha256",
        "output_layout",
        "study_run_sha256",
    }
    if set(record) != expected_keys:
        raise ValueError("study run manifest has missing or unknown fields")
    if record.get("schema_version") != 6 or record.get("record_type") != (
        "phycam_coco_detector_study_run_manifest"
    ):
        raise ValueError("study run manifest requires the schema-v6 detector run type")
    for key in (
        "study_plan_sha256",
        "dataset_sha256",
        "image_selection_sha256",
        "study_run_sha256",
    ):
        _sha256(record.get(key), label=key)
    allocation = DetectorAllocation.from_dict(record.get("detector_allocation"))
    validate_detector_model_contract(allocation, record.get("model"))
    runtime = _runtime_identity_valid(record.get("runtime"))
    _validate_execution_engine_record(
        record.get("execution_engine"),
        detector_id=allocation.detector_id,
    )
    inference_execution = record.get("inference_execution_contract")
    if not isinstance(inference_execution, Mapping):
        raise TypeError("study run inference execution contract must be a mapping")
    if runtime.get("inference_execution") != inference_execution:
        raise ValueError("study run runtime and inference execution contract differ")
    if runtime["detector_device"]["requested"] != allocation.requested_device:
        raise ValueError("study run device differs from its detector allocation")
    preprocessing = record.get("preprocessing")
    expected_preprocessing = LetterboxConfig(allocation.input_shape, allocation.pad_value)
    if preprocessing != json_value(expected_preprocessing.identity):
        raise ValueError("study run preprocessing differs from its detector allocation")
    cells = record.get("ordered_execution_cell_sha256")
    if not isinstance(cells, list) or not cells:
        raise ValueError("study run must contain ordered execution cell identities")
    if len(cells) != len(set(cells)):
        raise ValueError("study run execution cell identities must be unique")
    for index, digest in enumerate(cells):
        _sha256(digest, label=f"ordered_execution_cell_sha256[{index}]")
    if record.get("output_layout") != {
        "path_components": (
            "predictions/{detector_id}/{profile_id}/{arm_id}/shard_{ordinal:05d}.jsonl.gz"
        ),
        "merge_index": "merges/{detector_id}/{profile_id}/{arm_id}.index.json",
        "completion": "completions/{detector_id}.complete.json",
        "path_fields_are_semantic_only": True,
    }:
        raise ValueError("study run output layout drifted")
    payload = {key: item for key, item in record.items() if key != "study_run_sha256"}
    if record["study_run_sha256"] != canonical_sha256(payload):
        raise ValueError("study_run_sha256 does not match the run manifest payload")
    return record


@dataclass(frozen=True, slots=True)
class CocoStudyRunManifest:
    """Validated immutable detector-worker run identity."""

    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "record",
            _immutable_mapping(_validate_run_manifest_record(self.record)),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CocoStudyRunManifest":
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        return json_value(self.record)

    @property
    def study_run_sha256(self) -> str:
        return str(self.record["study_run_sha256"])


def materialize_study_run_manifest(path: str | Path, manifest: CocoStudyRunManifest) -> bool:
    """Publish or exactly resume one detector-worker run manifest."""

    if not isinstance(manifest, CocoStudyRunManifest):
        raise TypeError("manifest must be a CocoStudyRunManifest")
    return _publish_or_validate(Path(path), manifest.to_dict())


def load_study_run_manifest(path: str | Path) -> CocoStudyRunManifest:
    """Read and fully validate one canonical detector-worker manifest."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"study run manifest is missing: {manifest_path}")
    try:
        raw = manifest_path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to parse study run manifest: {manifest_path}") from exc
    manifest = CocoStudyRunManifest.from_dict(value)
    if raw != _record_bytes(manifest.to_dict()):
        raise ValueError("study run manifest is not in canonical deterministic JSON form")
    return manifest


class DetectorAdapter(Protocol):
    """Structural boundary used by the study runner and test doubles."""

    identity: Mapping[str, Any]
    device: str

    @property
    def execution_device(self) -> str: ...

    def detect_batch(self, inputs: Sequence[DetectorInput]) -> Iterable[Mapping[str, Any]]: ...


RuntimeIdentityFactory = Callable[..., Mapping[str, Any]]
ShardRunner = Callable[..., Any]
ShardMerger = Callable[..., PredictionShardMerge]
StudyProgressReporter = Callable[[Mapping[str, Any]], None]


def _emit_study_progress(
    reporter: StudyProgressReporter | None,
    phase: str,
    *,
    completed: int,
    total: int,
    **context: Any,
) -> None:
    if reporter is None:
        return
    reporter(
        MappingProxyType(
            {
                "schema_version": 2,
                "record_type": "phycam_coco_study_execution_progress",
                "phase": phase,
                "completed": completed,
                "total": total,
                **context,
            }
        )
    )


def _attest_detector_execution(
    adapter: DetectorAdapter,
    *,
    subset: NativeCOCODataset,
    preprocessing: LetterboxConfig,
    detect_batch: Callable[[Sequence[DetectorInput]], Iterable[Mapping[str, Any]]] | None = None,
) -> str:
    detector = adapter.detect_batch if detect_batch is None else detect_batch
    try:
        actual = adapter.execution_device
    except RuntimeError:
        image_id = subset.image_ids[0]
        frame = make_ldr_input_frame(subset.image(image_id), image_id=str(image_id))
        detector_input = letterbox(frame, preprocessing)
        outputs = tuple(detector((detector_input,)))
        if len(outputs) != 1 or not isinstance(outputs[0], Mapping):
            raise RuntimeError("detector attestation inference returned an invalid output")
        actual = adapter.execution_device
    if not torch_device_attestation_matches(adapter.device, actual):
        raise RuntimeError("detector execution-device attestation differs from requested device")
    return str(actual)


def _partition_image_ids(
    image_ids: Sequence[int], *, shard_size: int
) -> tuple[tuple[int, ...], ...]:
    size = positive_int(shard_size, field_name="shard_size")
    selected = _image_ids(image_ids)
    return tuple(tuple(selected[start : start + size]) for start in range(0, len(selected), size))


def _merge_index_record(
    *,
    plan: CocoStudyPlan,
    manifest: CocoStudyRunManifest,
    cell: Mapping[str, Any],
    merged: PredictionShardMerge,
) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "record_type": "phycam_coco_condition_prediction_merge_index",
        "study_plan_sha256": plan.study_plan_sha256,
        "study_run_sha256": manifest.study_run_sha256,
        "cell_sha256": cell["cell_sha256"],
        "prediction_shard_index": json_value(merged.index),
    }
    return {**payload, "merge_record_sha256": canonical_sha256(payload)}


def _study_run_completion_record(
    *,
    plan: CocoStudyPlan,
    manifest: CocoStudyRunManifest,
    ordered_merge_record_sha256: Sequence[str],
) -> dict[str, Any]:
    merges = [
        _sha256(value, label=f"ordered_merge_record_sha256[{index}]")
        for index, value in enumerate(ordered_merge_record_sha256)
    ]
    if len(merges) != len(manifest.record["ordered_execution_cell_sha256"]):
        raise RuntimeError("completion receipt requires exactly one merge identity per cell")
    if len(merges) != len(set(merges)):
        raise RuntimeError("completion receipt merge identities must be unique")
    runtime_sha256 = manifest.record["runtime"]["runtime_identity_sha256"]
    payload = {
        "schema_version": 2,
        "record_type": "phycam_coco_detector_study_run_completion",
        "status": "complete_after_final_integrity_rechecks",
        "study_plan_sha256": plan.study_plan_sha256,
        "study_run_sha256": manifest.study_run_sha256,
        "detector_id": manifest.record["detector_allocation"]["detector_id"],
        "runtime_identity_sha256_start": runtime_sha256,
        "runtime_identity_sha256_end": runtime_sha256,
        "ordered_merge_record_sha256": merges,
    }
    return {**payload, "completion_sha256": canonical_sha256(payload)}


def validate_study_run_completion_record(
    value: Mapping[str, Any],
    *,
    plan: CocoStudyPlan,
    manifest: CocoStudyRunManifest,
    ordered_merge_record_sha256: Sequence[str],
) -> dict[str, Any]:
    """Validate the durable receipt proving every final worker recheck passed."""

    if not isinstance(value, Mapping):
        raise TypeError("study-run completion receipt must be a mapping")
    record = json_value(value)
    expected = _study_run_completion_record(
        plan=plan,
        manifest=manifest,
        ordered_merge_record_sha256=ordered_merge_record_sha256,
    )
    if record != expected:
        raise ValueError("study-run completion receipt differs from its run and merge identities")
    return record


@dataclass(frozen=True, slots=True)
class StudyExecutionSummary:
    """Compact result for dry-run scheduling or a completed detector worker."""

    study_plan_sha256: str
    detector_id: str
    dry_run: bool
    cell_count: int
    shard_count: int
    image_inferences: int
    study_run_sha256: str | None = None
    resumed_shards: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 2,
            "record_type": "phycam_coco_study_execution_summary",
            "study_plan_sha256": self.study_plan_sha256,
            "detector_id": self.detector_id,
            "dry_run": self.dry_run,
            "cell_count": self.cell_count,
            "shard_count": self.shard_count,
            "image_inferences": self.image_inferences,
            "study_run_sha256": self.study_run_sha256,
            "resumed_shards": self.resumed_shards,
        }
        return {**payload, "summary_sha256": canonical_sha256(payload)}


def _execute_study_detector_run_unlocked(
    *,
    plan: CocoStudyPlan,
    subset: NativeCOCODataset,
    detector_id: str,
    output_root: str | Path,
    adapter: DetectorAdapter | None = None,
    dry_run: bool = False,
    repository_root: str | Path | None = None,
    runtime_identity_factory: RuntimeIdentityFactory = runtime_reproducibility_identity,
    shard_runner: ShardRunner = run_coco_ldr_condition_shard,
    shard_merger: ShardMerger = merge_prediction_shards,
    progress: StudyProgressReporter | None = None,
) -> StudyExecutionSummary:
    """Run or resume every frozen cell allocated to one detector.

    ``dry_run=True`` performs all plan/dataset schedule checks but creates no
    directories, model, runtime identity, or predictions.  Normal execution
    attests the detector on one untouched native image before freezing the run
    manifest, then delegates every shard to the streaming runner and validates
    complete nonoverlapping coverage through the deterministic shard merger.
    """

    if not isinstance(plan, CocoStudyPlan):
        raise TypeError("plan must be a CocoStudyPlan")
    if not isinstance(dry_run, bool):
        raise TypeError("dry_run must be bool")
    selected_detector = _identifier(detector_id, label="detector_id")
    allocation = plan.allocation(selected_detector)
    if tuple(subset.image_ids) != plan.image_ids:
        raise ValueError("native COCO subset order differs from the frozen study plan")
    if json_value(subset.identity) != json_value(plan.record["dataset"]):
        raise ValueError("native COCO dataset identity differs from the frozen study plan")
    cells = plan.cells(selected_detector)
    partitions = _partition_image_ids(plan.image_ids, shard_size=allocation.shard_size)
    shard_count = len(cells) * len(partitions)
    image_inferences = len(cells) * len(plan.image_ids)
    if dry_run:
        return StudyExecutionSummary(
            study_plan_sha256=plan.study_plan_sha256,
            detector_id=selected_detector,
            dry_run=True,
            cell_count=len(cells),
            shard_count=shard_count,
            image_inferences=image_inferences,
        )
    if adapter is None:
        raise ValueError("adapter is required unless dry_run=True")
    if not callable(getattr(adapter, "detect_batch", None)):
        raise TypeError("adapter must expose a callable detect_batch")
    execution_engine = _study_execution_engine_record(
        detector_id=selected_detector,
        subset=subset,
        adapter=adapter,
        runtime_identity_factory=runtime_identity_factory,
        shard_runner=shard_runner,
        shard_merger=shard_merger,
    )
    inference_execution = json_value(plan.record["design"]["inference_execution"])
    configure_deterministic_inference(inference_execution)
    model = validate_detector_model_contract(allocation, adapter.identity)
    preprocessing = LetterboxConfig(allocation.input_shape, allocation.pad_value)
    detect_batch = adapter.detect_batch
    actual_device = _attest_detector_execution(
        adapter,
        subset=subset,
        preprocessing=preprocessing,
        detect_batch=detect_batch,
    )
    runtime = runtime_identity_factory(
        requested_detector_device=adapter.device,
        actual_detector_device=actual_device,
        repository_root=repository_root,
        expected_inference_execution=inference_execution,
    )
    manifest = build_study_run_manifest(
        plan=plan,
        allocation=allocation,
        model=model,
        runtime=runtime,
        preprocessing=preprocessing,
        execution_engine=execution_engine,
    )
    root = Path(output_root)
    manifest_path = root / "manifests" / f"{selected_detector}.run.json"
    materialize_study_run_manifest(manifest_path, manifest)
    _emit_study_progress(
        progress,
        "detector_run_started",
        completed=0,
        total=shard_count,
        detector_id=selected_detector,
        cell_count=len(cells),
        shard_count=shard_count,
    )

    profiles = plan.profiles
    resumed = 0
    completed_shards = 0
    merge_record_sha256: list[str] = []
    for cell_index, cell in enumerate(cells):
        profile_id = str(cell["profile_id"])
        arm_id = str(cell["arm_id"])
        _emit_study_progress(
            progress,
            "cell_started",
            completed=cell_index,
            total=len(cells),
            detector_id=selected_detector,
            profile_id=profile_id,
            arm_id=arm_id,
        )
        profile = profiles[profile_id]
        condition = plan.condition(profile_id, arm_id)
        shard_paths: list[Path] = []
        for ordinal, image_partition in enumerate(partitions):
            shard_path = (
                root
                / "predictions"
                / selected_detector
                / profile_id
                / arm_id
                / f"shard_{ordinal:05d}.jsonl.gz"
            )
            existed = shard_path.exists()
            _emit_study_progress(
                progress,
                "shard_started",
                completed=completed_shards,
                total=shard_count,
                detector_id=selected_detector,
                profile_id=profile_id,
                arm_id=arm_id,
                shard_ordinal=ordinal,
                shard_image_count=len(image_partition),
            )
            shard_runner(
                subset=subset,
                condition=condition,
                image_ids=image_partition,
                profile=profile,
                preprocessing=preprocessing,
                model=model,
                run=manifest.to_dict(),
                detect_batch=detect_batch,
                shard_path=shard_path,
                label_space=allocation.label_space,
                batch_size=allocation.batch_size,
            )
            if existed:
                resumed += 1
            shard_paths.append(shard_path)
            completed_shards += 1
            _emit_study_progress(
                progress,
                "shard_completed",
                completed=completed_shards,
                total=shard_count,
                detector_id=selected_detector,
                profile_id=profile_id,
                arm_id=arm_id,
                shard_ordinal=ordinal,
                resumed=existed,
            )
        merged = shard_merger(shard_paths, expected_image_ids=plan.image_ids)
        merge_record = _merge_index_record(
            plan=plan,
            manifest=manifest,
            cell=cell,
            merged=merged,
        )
        merge_path = root / "merges" / selected_detector / profile_id / f"{arm_id}.index.json"
        _publish_or_validate(merge_path, merge_record)
        merge_record_sha256.append(str(merge_record["merge_record_sha256"]))
        _emit_study_progress(
            progress,
            "cell_completed",
            completed=cell_index + 1,
            total=len(cells),
            detector_id=selected_detector,
            profile_id=profile_id,
            arm_id=arm_id,
        )

    observed = adapter.execution_device
    expected = manifest.record["runtime"]["detector_device"]["resolved_actual"]
    if not torch_device_attestation_matches(expected, observed):
        raise RuntimeError("detector execution device drifted during the study run")
    final_runtime = runtime_identity_factory(
        requested_detector_device=adapter.device,
        actual_detector_device=observed,
        repository_root=repository_root,
        expected_inference_execution=inference_execution,
    )
    if final_runtime != runtime:
        raise RuntimeError(
            "detector runtime, package, hardware, or repository state drifted during execution"
        )
    completion = _study_run_completion_record(
        plan=plan,
        manifest=manifest,
        ordered_merge_record_sha256=merge_record_sha256,
    )
    completion_path = root / "completions" / f"{selected_detector}.complete.json"
    _publish_or_validate(completion_path, completion)
    _emit_study_progress(
        progress,
        "detector_run_completed",
        completed=shard_count,
        total=shard_count,
        detector_id=selected_detector,
        resumed_shards=resumed,
    )
    return StudyExecutionSummary(
        study_plan_sha256=plan.study_plan_sha256,
        detector_id=selected_detector,
        dry_run=False,
        cell_count=len(cells),
        shard_count=shard_count,
        image_inferences=image_inferences,
        study_run_sha256=manifest.study_run_sha256,
        resumed_shards=resumed,
    )


def execute_study_detector_run(
    *,
    plan: CocoStudyPlan,
    subset: NativeCOCODataset,
    detector_id: str,
    output_root: str | Path,
    adapter: DetectorAdapter | None = None,
    dry_run: bool = False,
    repository_root: str | Path | None = None,
    runtime_identity_factory: RuntimeIdentityFactory = runtime_reproducibility_identity,
    shard_runner: ShardRunner = run_coco_ldr_condition_shard,
    shard_merger: ShardMerger = merge_prediction_shards,
    progress: StudyProgressReporter | None = None,
) -> StudyExecutionSummary:
    """Run one detector while excluding analysis from the mutable study tree.

    Detector runs hold a shared layout lock for their complete transaction, so
    different detectors may run concurrently. The
    analyzer takes the same lock exclusively before trusting or promoting the
    completed layout. Dry-run validation remains read-only and lock-free.
    """

    arguments = {
        "plan": plan,
        "subset": subset,
        "detector_id": detector_id,
        "output_root": output_root,
        "adapter": adapter,
        "dry_run": dry_run,
        "repository_root": repository_root,
        "runtime_identity_factory": runtime_identity_factory,
        "shard_runner": shard_runner,
        "shard_merger": shard_merger,
        "progress": progress,
    }
    if dry_run:
        return _execute_study_detector_run_unlocked(**arguments)
    with advisory_target_lock(
        output_root,
        purpose="study-layout",
        exclusive=False,
    ):
        return _execute_study_detector_run_unlocked(**arguments)


__all__ = [
    "COCO_VAL2017_ANNOTATION_SHA256",
    "COCO_CONFIDENCE_FLOOR",
    "COCO_IOU_THRESHOLDS",
    "COCO_MAXIMUM_DETECTIONS",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_SHARD_SIZE",
    "FULL_COCO_VAL2017_DATASET_SHA256",
    "FULL_COCO_VAL2017_SELECTION_SHA256",
    "MAX_BOOTSTRAP_SEED",
    "PUBLICATION_ANCHOR_GRID",
    "PUBLICATION_DEFOCUS_GRID",
    "PUBLICATION_EXECUTION_CELL_COUNT",
    "PUBLICATION_IMAGE_COUNT",
    "CocoStudyPlan",
    "CocoStudyRunManifest",
    "DetectorAllocation",
    "StudyExecutionSummary",
    "build_publication_study_plan",
    "build_study_run_manifest",
    "derive_study_evidence_tier_record",
    "execute_study_detector_run",
    "load_study_plan",
    "load_study_run_manifest",
    "materialize_study_plan",
    "materialize_study_run_manifest",
    "publication_detector_allocations",
    "publication_reproduction_contract",
    "validate_detector_model_contract",
    "validate_study_run_completion_record",
]
