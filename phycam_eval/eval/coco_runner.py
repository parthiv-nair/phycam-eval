"""Concrete native-COCO runner for the deterministic LDR camera tier."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .._canonical import canonical_sha256, freeze_json_value, json_value
from ..boundary import BoundaryContract
from ..capture import make_ldr_input_frame, render_ldr
from ..domains import DataMode
from ..experiments.conditions import (
    BaselineCondition,
    BaselineKind,
    ExperimentCondition,
    MechanismComparatorCondition,
    to_ldr_capture_severity,
)
from ..pipeline import CameraPipeline, StageSpec
from ..profiles import CameraProfile
from .coco import (
    LazyNativeCOCOSubset,
    NativeCOCODataset,
    NativeCOCOSubset,
    detector_output_to_native_prediction,
)
from .harness import CameraSample, EvaluationRun, EvaluationSource, run_detector_evaluation
from .metrics import compute_map, compute_paired_map_bootstrap
from .model_provenance import (
    validate_detector_execution_identity,
    validate_model_identity,
)
from .preprocess import DetectorInput, LetterboxConfig, LetterboxGeometry, letterbox


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = freeze_json_value(value)
    if not isinstance(frozen, MappingProxyType):
        raise TypeError("record must be a mapping")
    return frozen


def _metric_record(value: Mapping[str, Any]) -> dict[str, Any]:
    per_class = value.get("per_class_ap")
    if not isinstance(per_class, Mapping):
        raise ValueError("metric result requires per_class_ap")
    result = {key: json_value(item) for key, item in value.items() if key != "per_class_ap"}
    result["per_class_ap"] = [
        {"category_id": int(category_id), "ap": float(ap)}
        for category_id, ap in sorted(per_class.items())
    ]
    return result


def _validate_benchmark_dataset_binding(
    dataset: Mapping[str, Any],
    evaluation: EvaluationRun,
) -> None:
    """Bind the benchmark dataset envelope to every evaluated source record."""

    if dataset.get("schema_version") != 2 or dataset.get("record_type") != "native_coco_subset":
        raise ValueError("benchmark dataset must be a schema-v2 native COCO subset")
    ordered_ids = dataset.get("ordered_image_ids")
    artifacts = dataset.get("image_artifacts")
    if not isinstance(ordered_ids, list) or not isinstance(artifacts, list):
        raise ValueError("benchmark dataset requires ordered images and artifacts")
    selected = json_value(evaluation.image_selection).get("ordered_values")
    if not isinstance(selected, list) or [item.get("image_id") for item in selected] != ordered_ids:
        raise ValueError("benchmark dataset image order drifted from evaluation sources")
    if len(artifacts) != len(selected):
        raise ValueError("benchmark image artifacts drifted from evaluation sources")
    dataset_sha256 = dataset.get("dataset_sha256")
    for index, record in enumerate(selected):
        source = record.get("source_identity", {}).get("content", {})
        target = record.get("target_identity", {}).get("content", {})
        if source.get("dataset_sha256") != dataset_sha256:
            raise ValueError("evaluation source is bound to a different dataset")
        if target.get("dataset_sha256") != dataset_sha256:
            raise ValueError("evaluation target is bound to a different dataset")
        if source.get("artifact") != artifacts[index]:
            raise ValueError("evaluation source artifact drifted from benchmark dataset")
        if source.get("decode_contract") != dataset.get("decode_contract"):
            raise ValueError("evaluation source decode contract drifted from benchmark dataset")
        if target.get("annotation_artifact") != dataset.get("annotation_artifact"):
            raise ValueError("evaluation target annotation artifact drifted from benchmark dataset")
        if target.get("target_contract") != dataset.get("target_contract"):
            raise ValueError("evaluation target contract drifted from benchmark dataset")


@dataclass(frozen=True, slots=True)
class COCOBenchmarkResult:
    """Portable schema-v3 result for one ordered LDR condition run."""

    evaluation: EvaluationRun
    dataset: Mapping[str, Any]
    model: Mapping[str, Any]
    condition_metrics: tuple[Mapping[str, Any], ...]
    statistics: Mapping[str, Any] | None = None
    execution: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, EvaluationRun):
            raise TypeError("evaluation must be an EvaluationRun")
        dataset = json_value(self.dataset)
        dataset_payload = {key: value for key, value in dataset.items() if key != "dataset_sha256"}
        if dataset.get("dataset_sha256") != canonical_sha256(dataset_payload):
            raise ValueError("dataset_sha256 does not match the embedded dataset")
        _validate_benchmark_dataset_binding(dataset, self.evaluation)
        model = validate_model_identity(self.model)
        if self.execution is None:
            raise ValueError("benchmark requires detector execution attestation")
        execution = validate_detector_execution_identity(self.execution, model=model)
        if execution != json_value(self.evaluation.detector_execution):
            raise ValueError("benchmark execution drifted from evaluation execution")
        if model != json_value(self.evaluation.detector_model):
            raise ValueError("benchmark model drifted from evaluation model")
        metrics = tuple(self.condition_metrics)
        condition_records: dict[str, dict[str, Any]] = {}
        for evaluation_record in self.evaluation.records:
            if evaluation_record.condition_record is None:
                raise ValueError("COCO evaluation records require embedded condition records")
            normalized_condition = json_value(evaluation_record.condition_record)
            prior = condition_records.setdefault(
                evaluation_record.condition_id,
                normalized_condition,
            )
            if prior != normalized_condition:
                raise ValueError("evaluation condition records drifted within one condition")
        for metric in metrics:
            if not isinstance(metric, Mapping):
                raise TypeError("condition metrics must contain mappings")
            condition_sha256 = metric.get("condition_sha256")
            condition = metric.get("condition")
            if not isinstance(condition, Mapping):
                raise ValueError("condition metric requires its embedded condition record")
            normalized_condition = json_value(condition)
            payload = {
                key: value
                for key, value in normalized_condition.items()
                if key != "condition_sha256"
            }
            if (
                normalized_condition.get("condition_sha256") != condition_sha256
                or canonical_sha256(payload) != condition_sha256
                or condition_records.get(condition_sha256) != normalized_condition
            ):
                raise ValueError("condition metric drifted from evaluated condition provenance")
        expected_conditions = [record["condition_sha256"] for record in metrics]
        selection = self.evaluation.condition_selection
        observed_conditions: list[str] = []
        for record in self.evaluation.records:
            if record.condition_id not in observed_conditions:
                observed_conditions.append(record.condition_id)
        if expected_conditions != observed_conditions:
            raise ValueError("condition metrics do not match evaluation condition order")
        if selection["count"] != len(metrics):
            raise ValueError("condition metric count does not match evaluation selection")
        object.__setattr__(self, "dataset", _immutable_mapping(dataset))
        object.__setattr__(self, "model", _immutable_mapping(model))
        object.__setattr__(self, "execution", _immutable_mapping(execution))
        object.__setattr__(
            self,
            "condition_metrics",
            tuple(_immutable_mapping(json_value(record)) for record in metrics),
        )
        for name in ("statistics",):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise TypeError(f"{name} must be a mapping when supplied")
                object.__setattr__(self, name, _immutable_mapping(json_value(value)))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 3,
            "record_type": "native_coco_ldr_detector_benchmark",
            "claim_scope": "synthetic_ldr_redegradation",
            "dataset": json_value(self.dataset),
            "model": json_value(self.model),
            "evaluation": self.evaluation.to_dict(),
            "prediction_coordinate_space": "native_stored_image",
            "camera_to_native_box_mapping": "condition_output_full_window_axis_scale_v2",
            "condition_metrics": [json_value(record) for record in self.condition_metrics],
            "statistics": None if self.statistics is None else json_value(self.statistics),
            "execution": None if self.execution is None else json_value(self.execution),
        }
        return {**payload, "benchmark_sha256": canonical_sha256(payload)}

    @property
    def benchmark_sha256(self) -> str:
        return self.to_dict()["benchmark_sha256"]


def run_coco_ldr_benchmark(
    *,
    subset: NativeCOCODataset,
    profile: CameraProfile,
    conditions: Sequence[ExperimentCondition],
    preprocessing: LetterboxConfig,
    model: Mapping[str, Any],
    detect_batch: Callable[[tuple[DetectorInput, ...]], Iterable[Mapping[str, Any]]],
    label_space: str = "coco_sparse",
    batch_size: int = 1,
    iou_thresholds: Sequence[float] | None = None,
    bootstrap_iterations: int = 0,
    bootstrap_seed: int = 42,
    baseline_condition: str | None = None,
    execution: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
) -> COCOBenchmarkResult:
    """Render, preprocess, detect, and score an ordered COCO LDR study.

    The present target adapter supports full-window, spatially stationary LDR
    conditions. Nonzero rolling motion is rejected until an annotation-time
    rolling-projection adapter is supplied. LDR capture is deterministic, so
    its condition binding must declare exactly realization ``0``.
    """

    if not isinstance(subset, (NativeCOCOSubset, LazyNativeCOCOSubset)):
        raise TypeError("subset must be a native COCO dataset")
    if not subset.loader_attested:
        raise ValueError("COCO benchmark requires a loader-attested native subset")
    if not isinstance(profile, CameraProfile):
        raise TypeError("profile must be a CameraProfile")
    if profile.data_mode is not DataMode.LDR_REDEGRADATION:
        raise ValueError("COCO JPEG evaluation requires an LDR_REDEGRADATION profile")
    if isinstance(conditions, (str, bytes)) or not isinstance(conditions, Sequence):
        raise TypeError("conditions must be an ordered sequence")
    selected_conditions = tuple(conditions)
    if not selected_conditions or not all(
        isinstance(condition, ExperimentCondition) for condition in selected_conditions
    ):
        raise TypeError("conditions must contain ExperimentCondition values")
    if not isinstance(preprocessing, LetterboxConfig):
        raise TypeError("preprocessing must be a LetterboxConfig")
    validated_model = validate_model_identity(model)
    if execution is None:
        raise ValueError("detector execution attestation is required")
    resolved_execution = execution

    binding = selected_conditions[0].binding
    if any(condition.binding != binding for condition in selected_conditions):
        raise ValueError("all conditions must share one profile and realization binding")
    binding.assert_profile(profile)
    if binding.realization_ids != (0,):
        raise ValueError("deterministic LDR evaluation requires realization_ids=(0,)")
    severities: dict[str, Any] = {}
    comparators: dict[str, Any] = {}
    for condition in selected_conditions:
        if isinstance(condition, MechanismComparatorCondition):
            from ..comparators.capture import comparator_config_from_dict

            comparators[condition.condition_hash] = comparator_config_from_dict(
                condition.match["config"]
            )
            continue
        if (
            isinstance(condition, BaselineCondition)
            and condition.kind is BaselineKind.UNTOUCHED_INPUT
        ):
            severities[condition.condition_hash] = None
            continue
        severity = to_ldr_capture_severity(condition, profile=profile)
        if any(value != 0.0 for value in severity.angular_velocity_rad_s):
            raise ValueError(
                "nonzero rolling motion requires an annotation-time target projection adapter"
            )
        severities[condition.condition_hash] = severity

    condition_records = tuple(condition.to_dict() for condition in selected_conditions)

    def load_source(image_id: int) -> EvaluationSource:
        index = subset.image_ids.index(image_id)
        source = subset.image(image_id)
        source_content = {
            "dataset_sha256": subset.identity["dataset_sha256"],
            "artifact": subset.identity["image_artifacts"][index],
            "decoded_shape_hwc": list(source.shape),
            "decode_contract": subset.identity["decode_contract"],
        }
        target_annotations = {
            "dataset_sha256": subset.identity["dataset_sha256"],
            "annotation_artifact": subset.identity["annotation_artifact"],
            "target_contract": subset.identity["target_contract"],
            "target_content_sha256": canonical_sha256(subset.target(image_id)),
        }
        return EvaluationSource.from_payloads(
            image_id=image_id,
            payload=source,
            source_content=source_content,
            target_annotations=target_annotations,
        )

    def bind_selected_condition(
        frame,
        provenance: Mapping[str, Any],
        condition: Mapping[str, Any],
        expected_renderer_condition: Mapping[str, Any],
        realization_id: int,
    ) -> CameraSample:
        selected = json_value(condition)
        selected_sha256 = selected["condition_sha256"]
        selected_payload = {
            key: value for key, value in selected.items() if key != "condition_sha256"
        }
        if canonical_sha256(selected_payload) != selected_sha256:
            raise ValueError("selected condition record changed during camera rendering")
        normalized = json_value(provenance)
        renderer_condition = normalized.get("capture_condition")
        renderer_sha256 = normalized.get("capture_condition_sha256")
        if not isinstance(renderer_condition, Mapping) or (
            canonical_sha256(renderer_condition) != renderer_sha256
        ):
            raise ValueError("renderer returned an invalid physical capture condition")
        expected_renderer = json_value(expected_renderer_condition)
        if json_value(renderer_condition) != expected_renderer:
            raise ValueError("renderer capture condition does not match expected condition")
        renderer_realization = normalized.get("realization_id")
        if renderer_realization is not None and renderer_realization != realization_id:
            raise ValueError("renderer realization_id does not match selected realization")
        normalized["renderer_capture_condition"] = renderer_condition
        normalized["renderer_capture_condition_sha256"] = renderer_sha256
        normalized["capture_condition"] = selected_payload
        normalized["capture_condition_sha256"] = selected_sha256
        normalized["realization_id"] = realization_id
        return CameraSample(frame, normalized)

    def render_camera(
        source: Any,
        image_id: int,
        condition: Mapping[str, Any],
        realization_id: int,
    ) -> CameraSample:
        condition_id = condition.get("condition_sha256")
        if condition_id in comparators:
            from ..comparators.capture import render_ldr_comparator

            result = render_ldr_comparator(
                source,
                profile,
                comparators[condition_id],
                image_id=str(image_id),
            )
            expected_renderer = {
                "common_neutral": {"physical_defocus_edge_waves_ref": 0.0},
                "comparator": comparators[condition_id].to_dict(),
                "comparator_boundary_contract": BoundaryContract(
                    profile.optics.boundary_policy,
                    profile.optics.boundary_constant_value,
                ).to_dict(),
            }
            return bind_selected_condition(
                result.output_frame,
                result.provenance,
                condition,
                expected_renderer,
                realization_id,
            )
        severity = severities[condition_id]
        if severity is None:
            input_frame = make_ldr_input_frame(source, image_id=str(image_id))
            pipeline = CameraPipeline(
                (
                    StageSpec(
                        name="untouched_input_bypass",
                        input_domain=input_frame.domain,
                        output_domain=input_frame.domain,
                        input_units=input_frame.metadata.units,
                        output_units=input_frame.metadata.units,
                        deterministic=True,
                        implementation_id="eval.untouched_input_identity.v1",
                        neutral_condition="identity camera bypass",
                        operation=None,
                    ),
                ),
                profile=profile,
            )
            output_frame, provenance = pipeline.run_with_provenance(input_frame)
            provenance["capture_condition"] = {"baseline_type": "untouched_input"}
            provenance["capture_condition_sha256"] = canonical_sha256(
                provenance["capture_condition"]
            )
            provenance["physical_contract"] = {
                "camera_graph_bypassed": True,
                "claim_scope": "untouched_detector_input_baseline",
            }
            return bind_selected_condition(
                output_frame,
                provenance,
                condition,
                {"baseline_type": "untouched_input"},
                realization_id,
            )
        result = render_ldr(
            source,
            profile,
            severity,
            image_id=str(image_id),
        )
        return bind_selected_condition(
            result.output_frame,
            result.provenance,
            condition,
            severity.to_dict(),
            realization_id,
        )

    evaluation = run_detector_evaluation(
        image_ids=subset.image_ids,
        condition_ids=condition_records,
        realization_ids=binding.realization_ids,
        load_image=load_source,
        render_camera=render_camera,
        preprocess_frame=lambda frame: letterbox(frame, preprocessing),
        detect_batch=detect_batch,
        detector_model=validated_model,
        detector_execution=resolved_execution,
        batch_size=batch_size,
    )

    target_records = list(subset.targets)
    results: list[Mapping[str, Any]] = []
    predictions_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in selected_conditions:
        predictions: list[dict[str, Any]] = []
        for record in evaluation.records:
            if record.condition_id != condition.condition_hash:
                continue
            geometry = LetterboxGeometry.from_dict(record.preprocessing_geometry)
            detector_output = record.detector_output
            if not isinstance(detector_output, Mapping):
                raise TypeError("detector outputs must be mappings")
            prediction = detector_output_to_native_prediction(
                image_id=record.image_id,
                detector_output=detector_output,
                geometry=geometry,
                native_shape=subset.image_shape(record.image_id),
                label_space=label_space,
                realization_id=record.realization_id,
            )
            predictions.append(prediction)
        predictions_by_condition[condition.condition_hash] = predictions
        metrics = compute_map(
            predictions,
            target_records,
            iou_thresholds,
            category_ids=subset.category_ids,
        )
        results.append(
            {
                "condition_sha256": condition.condition_hash,
                "condition": condition.to_dict(),
                "aggregation": "single_deterministic_realization",
                "realization_ids": list(binding.realization_ids),
                "metrics": _metric_record(metrics),
                "uncertainty": None,
            }
        )
    statistics: Mapping[str, Any] | None = None
    if bootstrap_iterations:
        baseline = baseline_condition or selected_conditions[0].condition_hash
        paired = compute_paired_map_bootstrap(
            predictions_by_condition,
            target_records,
            baseline_condition=baseline,
            n_bootstrap=bootstrap_iterations,
            seed=bootstrap_seed,
            iou_thresholds=iou_thresholds,
            category_ids=subset.category_ids,
        )
        normalized_conditions: list[dict[str, Any]] = []
        uncertainty_by_condition: dict[str, dict[str, Any]] = {}
        for record in paired["conditions"]:
            normalized = dict(record)
            normalized["metrics"] = _metric_record(record["metrics"])
            normalized_conditions.append(normalized)
            uncertainty_by_condition[record["condition"]] = {
                key: value
                for key, value in normalized.items()
                if key not in {"condition", "metrics"}
            }
        statistics = {**paired, "conditions": normalized_conditions}
        results = [
            {**record, "uncertainty": uncertainty_by_condition[record["condition_sha256"]]}
            for record in results
        ]
    return COCOBenchmarkResult(
        evaluation=evaluation,
        dataset=subset.identity,
        model=validated_model,
        condition_metrics=tuple(results),
        statistics=statistics,
        execution=evaluation.detector_execution,
    )


__all__ = ["COCOBenchmarkResult", "run_coco_ldr_benchmark"]
