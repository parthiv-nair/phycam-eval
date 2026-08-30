"""Image-model validation and detector-evaluation utilities."""

from .coco import (
    LazyNativeCOCOSubset,
    NativeCOCODataset,
    NativeCOCOSubset,
    decode_native_srgb,
    detector_output_to_native_prediction,
    load_native_coco_subset,
)
from .coco_runner import COCOBenchmarkResult, run_coco_ldr_benchmark
from .coco_stream import run_coco_ldr_condition_shard
from .detectors import (
    HuggingFaceDETRAdapter,
    TorchvisionFasterRCNNAdapter,
    TorchvisionRetinaNetAdapter,
    UltralyticsYOLOAdapter,
)
from .harness import (
    CameraSample,
    EvaluationRecord,
    EvaluationRun,
    EvaluationSource,
    run_detector_evaluation,
)
from .image_quality import (
    EncircledEnergyCurve,
    ImageDifference,
    PointSourceDiagnostics,
    compare_images,
    encircled_energy_curve,
    point_source_diagnostics,
)
from .metrics import (
    COCO80_TO_91,
    compute_hierarchical_map_ci,
    compute_map,
    compute_paired_map_bootstrap,
    compute_paired_map_curve_auc_bootstrap,
    yolo_to_coco_category_ids,
)
from .model_provenance import (
    detector_execution_identity,
    validate_detector_execution_identity,
)
from .mtf import (
    MTFCurve,
    first_downward_crossing,
    kernel_axis_mtf,
    measure_slanted_edge_mtf,
    mtf50,
    otf_axis_mtf,
)
from .protocol import (
    DEFAULT_INFERENCE_SEED,
    DEFAULT_RUNTIME_DISTRIBUTIONS,
    configure_deterministic_inference,
    deterministic_inference_execution_contract,
    evaluation_source_record,
    ordered_source_selection_identity,
    runtime_reproducibility_identity,
    source_content_identity,
    target_annotation_identity,
    validate_camera_provenance,
    validate_ordered_source_selection_identity,
)
from .shards import (
    PredictionShard,
    PredictionShardMerge,
    make_prediction_record,
    make_prediction_shard_header,
    merge_prediction_shards,
    prediction_shard_receipt_path,
    validate_existing_prediction_shard,
    validate_prediction_shard,
    write_prediction_shard,
)


def __getattr__(name: str):
    """Lazily expose study analysis without importing comparators in a cycle."""

    if name == "verify_completed_study_layout":
        from .study_analysis import verify_completed_study_layout

        globals()[name] = verify_completed_study_layout
        return verify_completed_study_layout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "COCO80_TO_91",
    "COCOBenchmarkResult",
    "CameraSample",
    "DEFAULT_RUNTIME_DISTRIBUTIONS",
    "DEFAULT_INFERENCE_SEED",
    "EncircledEnergyCurve",
    "EvaluationRecord",
    "EvaluationRun",
    "EvaluationSource",
    "HuggingFaceDETRAdapter",
    "ImageDifference",
    "MTFCurve",
    "LazyNativeCOCOSubset",
    "NativeCOCODataset",
    "NativeCOCOSubset",
    "PointSourceDiagnostics",
    "PredictionShard",
    "PredictionShardMerge",
    "TorchvisionFasterRCNNAdapter",
    "TorchvisionRetinaNetAdapter",
    "UltralyticsYOLOAdapter",
    "compare_images",
    "configure_deterministic_inference",
    "deterministic_inference_execution_contract",
    "detector_execution_identity",
    "compute_hierarchical_map_ci",
    "compute_map",
    "compute_paired_map_curve_auc_bootstrap",
    "compute_paired_map_bootstrap",
    "decode_native_srgb",
    "detector_output_to_native_prediction",
    "encircled_energy_curve",
    "evaluation_source_record",
    "first_downward_crossing",
    "kernel_axis_mtf",
    "load_native_coco_subset",
    "measure_slanted_edge_mtf",
    "make_prediction_record",
    "make_prediction_shard_header",
    "merge_prediction_shards",
    "mtf50",
    "otf_axis_mtf",
    "ordered_source_selection_identity",
    "point_source_diagnostics",
    "prediction_shard_receipt_path",
    "run_coco_ldr_benchmark",
    "run_coco_ldr_condition_shard",
    "runtime_reproducibility_identity",
    "run_detector_evaluation",
    "source_content_identity",
    "target_annotation_identity",
    "validate_camera_provenance",
    "validate_detector_execution_identity",
    "validate_existing_prediction_shard",
    "validate_ordered_source_selection_identity",
    "validate_prediction_shard",
    "verify_completed_study_layout",
    "write_prediction_shard",
    "yolo_to_coco_category_ids",
]
