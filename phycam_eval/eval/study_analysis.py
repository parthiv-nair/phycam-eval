"""Validate completed native-COCO runs and compute the study analysis.

The runner stores predictions rather than metrics.  Analysis reloads the plan,
validates every resumable shard and merge index, and then evaluates the
prespecified paired-bootstrap and curve-AUC statistics.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import multiprocessing
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .._canonical import canonical_sha256, finite_float, freeze_json_value, json_value
from ..experiments.conditions import MechanismComparatorCondition, PhysicalDefocusCondition
from .coco import NativeCOCODataset, load_native_coco_subset
from .coco_stream import _condition_contract
from .locking import advisory_target_lock
from .metrics import (
    assemble_paired_map_bootstrap,
    assemble_paired_map_curve_auc_bootstrap,
    compute_map_bootstrap_arm,
    compute_paired_map_bootstrap,
    compute_paired_map_curve_auc_bootstrap,
)
from .preprocess import LetterboxConfig
from .protocol import (
    ANALYSIS_RUNTIME_IDENTITY_METHOD,
    DEFAULT_RUNTIME_DISTRIBUTIONS,
    PACKAGE_IDENTITY_METHOD,
    RUNTIME_IDENTITY_METHOD,
    WORKTREE_STATE_METHOD,
    analysis_runtime_reproducibility_identity,
    validate_project_installation_identity,
)
from .shards import (
    PredictionShardMerge,
    make_prediction_shard_header,
    merge_prediction_shards,
    prediction_shard_receipt_path,
    validate_existing_prediction_shard,
)
from .study import (
    CocoStudyPlan,
    CocoStudyRunManifest,
    DetectorAllocation,
    derive_study_evidence_tier_record,
    load_study_plan,
    load_study_run_manifest,
    validate_study_run_completion_record,
)

STUDY_ANALYSIS_IMPLEMENTATION_ID = "phycam.native_coco_study_analysis.v8"
NONOFFICIAL_STUDY_ANALYSIS_IMPLEMENTATION_ID = (
    "phycam.native_coco_study_analysis.injected_statistics.v8"
)
_STATISTICS_ENGINES_CONTRACT_ID = "phycam.study_analysis_statistics_engines.v1"
_EXECUTION_ELIGIBILITY_CONTRACT_ID = "phycam.study_analysis_execution_eligibility.v5"
_CHECKPOINT_EXECUTION_CONTRACT_ID = "phycam.study_analysis_checkpoint_execution.v1"
_CHECKPOINT_PLAN_TYPE = "phycam_study_analysis_checkpoint_plan"
_ARM_CHECKPOINT_TYPE = "phycam_study_analysis_arm_checkpoint"
_CHECKPOINT_MANIFEST_TYPE = "phycam_study_analysis_checkpoint_manifest"
_RESULT_TYPE = "phycam_coco_publication_study_analysis"
_PUBLICATION_TYPE = "phycam_coco_study_analysis_publication_index"
_POINT_METRIC_KEYS = {
    "map50",
    "map50_95",
    "mean_ap",
    "mean_ap_iou_thresholds",
    "map75",
    "map50_95_small",
    "map50_95_medium",
    "map50_95_large",
    "mean_ap_small",
    "mean_ap_medium",
    "mean_ap_large",
    "ar100",
    "mean_ar100",
    "ar100_small",
    "ar100_medium",
    "ar100_large",
    "per_class_ap",
}
_SCALAR_POINT_METRICS = tuple(
    sorted(_POINT_METRIC_KEYS - {"mean_ap_iou_thresholds", "per_class_ap"})
)

ProgressReporter = Callable[[Mapping[str, Any]], None]
PairedBootstrap = Callable[..., Mapping[str, Any]]
CurveBootstrap = Callable[..., Mapping[str, Any]]


def _record_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            json_value(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _emit(
    reporter: ProgressReporter | None,
    phase: str,
    *,
    completed: int,
    total: int,
    **context: Any,
) -> None:
    if reporter is None:
        return
    event = {
        "schema_version": 2,
        "record_type": "phycam_coco_study_analysis_progress",
        "phase": phase,
        "completed": completed,
        "total": total,
        **context,
    }
    reporter(MappingProxyType(event))


def _require_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symbolic link: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")


def _parse_canonical_record_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError(f"{label} payload must be bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to parse {label}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must contain one JSON mapping")
    normalized = json_value(value)
    if payload != _record_bytes(normalized):
        raise ValueError(f"{label} is not canonical deterministic JSON")
    return normalized


def _load_canonical_record(path: Path, *, label: str) -> dict[str, Any]:
    _require_regular_file(path, label=label)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"failed to read {label}: {path}") from exc
    return _parse_canonical_record_bytes(payload, label=f"{label}: {path}")


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_or_validate_bytes(path: Path, payload: bytes) -> bool:
    if path.exists() or path.is_symlink():
        _require_regular_file(path, label="analysis publication artifact")
        if path.read_bytes() != payload:
            raise ValueError(f"existing analysis publication drifted: {path}")
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
            os.link(temporary, path)
        except FileExistsError:
            _require_regular_file(path, label="concurrent analysis publication artifact")
            if path.read_bytes() != payload:
                raise ValueError(f"concurrent analysis publication drifted: {path}") from None
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _json_result_value(value: Any, *, label: str) -> Any:
    """Normalize metric output, including integer per-category mapping keys."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return finite_float(value, field_name=label)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"{label} contains an object array")
        return _json_result_value(value.tolist(), label=label)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            if isinstance(raw_key, (int, np.integer)) and not isinstance(raw_key, (bool, np.bool_)):
                key = str(int(raw_key))
            elif isinstance(raw_key, str):
                key = raw_key
            else:
                raise TypeError(f"{label} has a non-string/non-integer mapping key")
            if key in normalized:
                raise ValueError(f"{label} contains colliding mapping keys")
            normalized[key] = _json_result_value(item, label=f"{label}.{key}")
        return dict(sorted(normalized.items(), key=lambda item: item[0].encode("utf-8")))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_result_value(item, label=f"{label}[{index}]") for index, item in enumerate(value)
        ]
    raise TypeError(f"{label} contains unsupported {type(value).__name__}")


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _analysis_worker_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("analysis_workers must be an integer")
    if value not in (1, 2):
        raise ValueError("analysis_workers must be 1 or 2")
    return value


def _require_separate_scratch_path(
    scratch_work_dir: str | Path,
    *,
    forbidden_roots: Sequence[str | Path],
) -> Path:
    """Require an explicit scratch tree disjoint from immutable/final trees."""

    scratch = Path(scratch_work_dir)
    if not scratch.name:
        raise ValueError("scratch work directory must have a final path component")
    if scratch.is_symlink():
        raise RuntimeError("analysis scratch work directory must not be a symbolic link")
    resolved_scratch = scratch.resolve(strict=False)
    for raw_root in forbidden_roots:
        resolved_root = Path(raw_root).resolve(strict=False)
        if _path_is_within(resolved_scratch, resolved_root) or _path_is_within(
            resolved_root,
            resolved_scratch,
        ):
            raise ValueError(
                "analysis scratch work directory must be outside completed and final outputs"
            )
    return scratch


def _checkpoint_implementation_sha256() -> str:
    """Hash the loaded orchestration and statistics source implementations."""

    modules = {
        inspect.getmodule(compute_map_bootstrap_arm),
        inspect.getmodule(_checkpoint_implementation_sha256),
    }
    if None in modules:
        raise RuntimeError("cannot identify checkpoint implementation modules")
    source_records: list[tuple[str, str]] = []
    for module in sorted(modules, key=lambda value: str(value.__name__)):
        try:
            source = inspect.getsource(module)
        except (OSError, TypeError) as exc:
            raise RuntimeError("cannot read checkpoint implementation source") from exc
        source_records.append((str(module.__name__), source))
    payload = {
        "contract_id": _CHECKPOINT_EXECUTION_CONTRACT_ID,
        "study_analysis_implementation_id": STUDY_ANALYSIS_IMPLEMENTATION_ID,
        "sources": [
            {
                "module": module,
                "sha256": _sha256_bytes(source.encode("utf-8")),
            }
            for module, source in source_records
        ],
    }
    return canonical_sha256(payload)


def _checkpoint_cell_binding(
    cell: _ValidatedCell,
    *,
    ordinal: int,
) -> dict[str, Any]:
    value = cell.cell
    payload = {
        "ordinal": ordinal,
        "detector_id": str(value["detector_id"]),
        "profile_id": str(value["profile_id"]),
        "arm_id": str(value["arm_id"]),
        "cell_sha256": _require_sha256(value["cell_sha256"], label="cell_sha256"),
        "condition_sha256": _require_sha256(
            value["condition_sha256"],
            label="condition_sha256",
        ),
        "merge_record_sha256": _require_sha256(
            cell.merge_record_sha256,
            label="merge_record_sha256",
        ),
        "prediction_shard_index_sha256": _require_sha256(
            cell.prediction_shard_index_sha256,
            label="prediction_shard_index_sha256",
        ),
    }
    return {**payload, "arm_input_sha256": canonical_sha256(payload)}


def _checkpoint_plan_record(
    *,
    plan: CocoStudyPlan,
    cells: Sequence[_ValidatedCell],
    targets: Sequence[Mapping[str, Any]],
    iou_thresholds: Sequence[float],
    category_ids: Sequence[int],
    coordinates: Sequence[float],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    normalized_targets = _json_result_value(targets, label="analysis targets")
    payload = {
        "schema_version": 1,
        "record_type": _CHECKPOINT_PLAN_TYPE,
        "contract_id": _CHECKPOINT_EXECUTION_CONTRACT_ID,
        "implementation_sha256": _checkpoint_implementation_sha256(),
        "study_plan_sha256": plan.study_plan_sha256,
        "dataset_sha256": plan.record["dataset"]["dataset_sha256"],
        "image_selection_sha256": plan.record["image_selection"]["selection_sha256"],
        "ordered_image_ids": list(plan.image_ids),
        "target_input_sha256": canonical_sha256(normalized_targets),
        "bootstrap_replicates": iterations,
        "bootstrap_seed": seed,
        "iou_thresholds": [
            finite_float(value, field_name="checkpoint IoU threshold") for value in iou_thresholds
        ],
        "category_ids": [int(value) for value in category_ids],
        "primary_ordered_coordinates": [
            finite_float(value, field_name="checkpoint primary coordinate") for value in coordinates
        ],
        "ordered_cells": [
            _checkpoint_cell_binding(cell, ordinal=ordinal) for ordinal, cell in enumerate(cells)
        ],
    }
    return {**payload, "checkpoint_plan_sha256": canonical_sha256(payload)}


def _checkpoint_filename(binding: Mapping[str, Any]) -> str:
    return f"{int(binding['ordinal']):05d}.{binding['cell_sha256']}.checkpoint.json"


_ARM_STATISTIC_KEYS = {
    "method",
    "sampling_contract",
    "iterations",
    "seed",
    "image_count",
    "category_ids",
    "iou_thresholds",
    "point_metrics",
    "bootstrap_samples",
    "bootstrap_failures",
    "matching_evaluations",
    "bootstrap_accumulation",
    "memory_strategy",
}


def _validated_checkpoint_arm_statistics(
    value: Mapping[str, Any],
    *,
    checkpoint_plan: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    normalized = _json_result_value(value, label=label)
    if not isinstance(normalized, dict) or set(normalized) != _ARM_STATISTIC_KEYS:
        raise ValueError(f"{label} has missing or unknown fields")
    expected_metadata = {
        "method": "image_cluster_cached_coco_percentile_bootstrap_arm_v1",
        "sampling_contract": "numpy_seedsequence_default_rng_choice_native_image_positions_v1",
        "iterations": checkpoint_plan["bootstrap_replicates"],
        "seed": checkpoint_plan["bootstrap_seed"],
        "image_count": len(checkpoint_plan["ordered_image_ids"]),
        "category_ids": checkpoint_plan["category_ids"],
        "iou_thresholds": checkpoint_plan["iou_thresholds"],
        "bootstrap_failures": 0,
        "matching_evaluations": 1,
        "bootstrap_accumulation": ("repeat_cached_per_image_category_matches_then_coco_pr_v1"),
        "memory_strategy": "single_condition_match_cache_v1",
    }
    for key, expected in expected_metadata.items():
        if normalized[key] != expected:
            raise ValueError(f"{label} field {key!r} differs from the checkpoint plan")
    _point_metrics(
        normalized["point_metrics"],
        iou_thresholds=checkpoint_plan["iou_thresholds"],
        category_ids=checkpoint_plan["category_ids"],
        label=f"{label}.point_metrics",
    )
    samples = normalized["bootstrap_samples"]
    if not isinstance(samples, Mapping) or set(samples) != {"map50", "map50_95"}:
        raise ValueError(f"{label} bootstrap samples are noncanonical")
    iterations = int(checkpoint_plan["bootstrap_replicates"])
    for metric in ("map50", "map50_95"):
        values = samples[metric]
        if not isinstance(values, list) or len(values) != iterations:
            raise ValueError(f"{label} {metric} sample count differs from the plan")
        for index, raw_value in enumerate(values):
            sample = finite_float(raw_value, field_name=f"{label}.{metric}[{index}]")
            if not 0.0 <= sample <= 1.0:
                raise ValueError(f"{label} {metric} samples must lie in [0, 1]")
    return normalized


def _arm_checkpoint_record(
    *,
    checkpoint_plan: Mapping[str, Any],
    binding: Mapping[str, Any],
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_statistics = _validated_checkpoint_arm_statistics(
        statistics,
        checkpoint_plan=checkpoint_plan,
        label="computed bootstrap arm",
    )
    payload = {
        "schema_version": 1,
        "record_type": _ARM_CHECKPOINT_TYPE,
        "checkpoint_plan_sha256": checkpoint_plan["checkpoint_plan_sha256"],
        "arm": json_value(binding),
        "statistics": normalized_statistics,
        "arm_statistics_sha256": canonical_sha256(normalized_statistics),
    }
    return {**payload, "checkpoint_sha256": canonical_sha256(payload)}


def _validate_arm_checkpoint_record(
    value: Mapping[str, Any],
    *,
    checkpoint_plan: Mapping[str, Any],
    binding: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    record = json_value(value)
    expected_keys = {
        "schema_version",
        "record_type",
        "checkpoint_plan_sha256",
        "arm",
        "statistics",
        "arm_statistics_sha256",
        "checkpoint_sha256",
    }
    if set(record) != expected_keys:
        raise ValueError(f"{label} has missing or unknown fields")
    if record["schema_version"] != 1 or record["record_type"] != _ARM_CHECKPOINT_TYPE:
        raise ValueError(f"{label} has an unsupported schema or type")
    if record["checkpoint_plan_sha256"] != checkpoint_plan["checkpoint_plan_sha256"]:
        raise ValueError(f"{label} is bound to a different checkpoint plan")
    if record["arm"] != json_value(binding):
        raise ValueError(f"{label} is bound to a different study cell")
    statistics = _validated_checkpoint_arm_statistics(
        record["statistics"],
        checkpoint_plan=checkpoint_plan,
        label=f"{label}.statistics",
    )
    supplied_statistics = _require_sha256(
        record["arm_statistics_sha256"],
        label=f"{label} arm_statistics_sha256",
    )
    if supplied_statistics != canonical_sha256(statistics):
        raise ValueError(f"{label} statistics hash does not match its payload")
    supplied = _require_sha256(
        record["checkpoint_sha256"],
        label=f"{label} checkpoint_sha256",
    )
    payload = {key: item for key, item in record.items() if key != "checkpoint_sha256"}
    if supplied != canonical_sha256(payload):
        raise ValueError(f"{label} hash does not match its payload")
    return record


def _compute_arm_checkpoint(
    *,
    cell: _ValidatedCell,
    binding: Mapping[str, Any],
    checkpoint_plan: Mapping[str, Any],
    image_ids: Sequence[int],
    targets: Sequence[Mapping[str, Any]],
    progress: ProgressReporter | None,
) -> tuple[dict[str, Any], bytes]:
    detector_id = str(binding["detector_id"])
    profile_id = str(binding["profile_id"])
    arm_id = str(binding["arm_id"])

    def report_arm(completed: int, total: int) -> None:
        _emit(
            progress,
            "secondary_paired_bootstrap_progress",
            completed=completed,
            total=total,
            detector_id=detector_id,
            profile_id=profile_id,
            arm_id=arm_id,
            cell_ordinal=int(binding["ordinal"]),
            cell_count=len(checkpoint_plan["ordered_cells"]),
        )

    predictions = _reload_cell_predictions(cell, image_ids=image_ids)
    statistics = compute_map_bootstrap_arm(
        predictions,
        targets,
        n_bootstrap=int(checkpoint_plan["bootstrap_replicates"]),
        seed=int(checkpoint_plan["bootstrap_seed"]),
        iou_thresholds=checkpoint_plan["iou_thresholds"],
        category_ids=checkpoint_plan["category_ids"],
        progress=report_arm,
    )
    record = _arm_checkpoint_record(
        checkpoint_plan=checkpoint_plan,
        binding=binding,
        statistics=statistics,
    )
    return record, _record_bytes(record)


_PROCESS_ARM_TARGETS: tuple[Mapping[str, Any], ...] | None = None
_PROCESS_ARM_IMAGE_IDS: tuple[int, ...] | None = None
_PROCESS_CHECKPOINT_PLAN: Mapping[str, Any] | None = None


def _initialize_arm_process_worker(
    targets: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int],
    checkpoint_plan: Mapping[str, Any],
) -> None:
    """Install immutable shared task inputs once in each spawned worker."""

    global _PROCESS_ARM_TARGETS, _PROCESS_ARM_IMAGE_IDS, _PROCESS_CHECKPOINT_PLAN
    _PROCESS_ARM_TARGETS = tuple(targets)
    _PROCESS_ARM_IMAGE_IDS = tuple(int(value) for value in image_ids)
    _PROCESS_CHECKPOINT_PLAN = checkpoint_plan


def _compute_arm_checkpoint_in_process(
    task: Mapping[str, Any],
) -> tuple[int, dict[str, Any], bytes]:
    """Spawn-safe worker entry point; children only read validated shard paths."""

    if (
        _PROCESS_ARM_TARGETS is None
        or _PROCESS_ARM_IMAGE_IDS is None
        or _PROCESS_CHECKPOINT_PLAN is None
    ):
        raise RuntimeError("analysis arm worker was not initialized")
    ordinal = int(task["ordinal"])
    cell = _ValidatedCell(
        cell=task["cell"],
        merge_record_sha256=str(task["merge_record_sha256"]),
        prediction_shard_index_sha256=str(task["prediction_shard_index_sha256"]),
        shard_count=int(task["shard_count"]),
        prediction_count=int(task["prediction_count"]),
        shard_paths=tuple(Path(value) for value in task["shard_paths"]),
    )
    record, payload = _compute_arm_checkpoint(
        cell=cell,
        binding=task["binding"],
        checkpoint_plan=_PROCESS_CHECKPOINT_PLAN,
        image_ids=_PROCESS_ARM_IMAGE_IDS,
        targets=_PROCESS_ARM_TARGETS,
        progress=None,
    )
    return ordinal, record, payload


def _arm_process_task(
    ordinal: int,
    cell: _ValidatedCell,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "cell": json_value(cell.cell),
        "merge_record_sha256": cell.merge_record_sha256,
        "prediction_shard_index_sha256": cell.prediction_shard_index_sha256,
        "shard_count": cell.shard_count,
        "prediction_count": cell.prediction_count,
        "shard_paths": [str(value) for value in cell.shard_paths],
        "binding": json_value(binding),
    }


def _two_worker_process_pool(
    *,
    targets: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int],
    checkpoint_plan: Mapping[str, Any],
) -> ProcessPoolExecutor:
    options = {
        "max_workers": 2,
        "mp_context": multiprocessing.get_context("spawn"),
        "initializer": _initialize_arm_process_worker,
        "initargs": (targets, image_ids, checkpoint_plan),
    }
    try:
        return ProcessPoolExecutor(**options)
    except PermissionError:
        # CPython's first process-pool construction probes SC_SEM_NSEMS_MAX.
        # Some macOS application sandboxes deny only that advisory sysconf
        # probe; CPython records the completed probe before propagating the
        # denial. Treat only that unavailable advisory value as indeterminate
        # (the same -1 contract accepted by CPython); real semaphore setup,
        # permission, and resource failures still fail closed.
        original_sysconf = os.sysconf

        def sandbox_safe_sysconf(name: str) -> int:
            if name == "SC_SEM_NSEMS_MAX":
                try:
                    return int(original_sysconf(name))
                except PermissionError:
                    return -1
            return int(original_sysconf(name))

        os.sysconf = sandbox_safe_sysconf
        try:
            return ProcessPoolExecutor(**options)
        finally:
            os.sysconf = original_sysconf


def _checkpoint_manifest_record(
    *,
    checkpoint_plan: Mapping[str, Any],
    ordered_checkpoints: Sequence[tuple[Mapping[str, Any], bytes]],
) -> dict[str, Any]:
    cells = checkpoint_plan["ordered_cells"]
    if len(ordered_checkpoints) != len(cells):
        raise RuntimeError("checkpoint manifest does not cover every study cell")
    entries: list[dict[str, Any]] = []
    for binding, (record, payload) in zip(cells, ordered_checkpoints, strict=True):
        validated = _validate_arm_checkpoint_record(
            record,
            checkpoint_plan=checkpoint_plan,
            binding=binding,
            label="checkpoint manifest arm",
        )
        entries.append(
            {
                "ordinal": binding["ordinal"],
                "cell_sha256": binding["cell_sha256"],
                "filename": _checkpoint_filename(binding),
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "checkpoint_sha256": validated["checkpoint_sha256"],
                "arm_statistics_sha256": validated["arm_statistics_sha256"],
            }
        )
    payload = {
        "schema_version": 1,
        "record_type": _CHECKPOINT_MANIFEST_TYPE,
        "checkpoint_plan_sha256": checkpoint_plan["checkpoint_plan_sha256"],
        "completed_arm_count": len(entries),
        "ordered_arm_checkpoints": entries,
    }
    return {**payload, "checkpoint_manifest_sha256": canonical_sha256(payload)}


def _validate_checkpoint_manifest(
    value: Mapping[str, Any],
    *,
    checkpoint_plan: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint manifest must be a mapping")
    record = json_value(value)
    if set(record) != {
        "schema_version",
        "record_type",
        "checkpoint_plan_sha256",
        "completed_arm_count",
        "ordered_arm_checkpoints",
        "checkpoint_manifest_sha256",
    }:
        raise ValueError("checkpoint manifest has missing or unknown fields")
    if record["schema_version"] != 1 or record["record_type"] != _CHECKPOINT_MANIFEST_TYPE:
        raise ValueError("checkpoint manifest has an unsupported schema or type")
    if record["checkpoint_plan_sha256"] != checkpoint_plan["checkpoint_plan_sha256"]:
        raise ValueError("checkpoint manifest is bound to a different checkpoint plan")
    cells = checkpoint_plan["ordered_cells"]
    entries = record["ordered_arm_checkpoints"]
    if (
        record["completed_arm_count"] != len(cells)
        or not isinstance(entries, list)
        or len(entries) != len(cells)
    ):
        raise ValueError("checkpoint manifest does not cover every study cell")
    for binding, entry in zip(cells, entries, strict=True):
        if not isinstance(entry, Mapping) or set(entry) != {
            "ordinal",
            "cell_sha256",
            "filename",
            "bytes",
            "sha256",
            "checkpoint_sha256",
            "arm_statistics_sha256",
        }:
            raise ValueError("checkpoint manifest arm entry is noncanonical")
        if (
            entry["ordinal"] != binding["ordinal"]
            or entry["cell_sha256"] != binding["cell_sha256"]
            or entry["filename"] != _checkpoint_filename(binding)
        ):
            raise ValueError("checkpoint manifest arm order or identity drifted")
        if (
            not isinstance(entry["bytes"], int)
            or isinstance(entry["bytes"], bool)
            or entry["bytes"] <= 0
        ):
            raise ValueError("checkpoint manifest arm byte count is invalid")
        for key in ("sha256", "checkpoint_sha256", "arm_statistics_sha256"):
            _require_sha256(entry[key], label=f"checkpoint manifest {key}")
    supplied = _require_sha256(
        record["checkpoint_manifest_sha256"],
        label="checkpoint_manifest_sha256",
    )
    payload = {key: item for key, item in record.items() if key != "checkpoint_manifest_sha256"}
    if supplied != canonical_sha256(payload):
        raise ValueError("checkpoint manifest hash does not match its payload")
    return record


def _prepare_checkpoint_workspace(
    scratch: Path,
    *,
    checkpoint_plan: Mapping[str, Any],
) -> tuple[Path, bool]:
    if scratch.is_symlink():
        raise RuntimeError("analysis scratch work directory must not be a symbolic link")
    if scratch.exists() and not scratch.is_dir():
        raise RuntimeError("analysis scratch work path must be a regular directory")
    scratch.mkdir(parents=True, exist_ok=True)
    allowed = {"checkpoint.plan.json", "checkpoint.manifest.json", "arms"}
    observed = set()
    for entry in scratch.iterdir():
        if entry.name not in allowed:
            raise ValueError(f"analysis scratch contains an unmanaged entry: {entry.name}")
        if entry.is_symlink():
            raise RuntimeError(f"analysis scratch contains a symbolic link: {entry}")
        if entry.name == "arms":
            if not entry.is_dir():
                raise RuntimeError("analysis scratch arms path must be a regular directory")
        elif not entry.is_file():
            raise RuntimeError(f"analysis scratch contains a nonregular file: {entry}")
        observed.add(entry.name)
    plan_path = scratch / "checkpoint.plan.json"
    if "checkpoint.plan.json" not in observed and observed:
        raise ValueError("analysis scratch has checkpoints without its binding plan")
    plan_bytes = _record_bytes(checkpoint_plan)
    _publish_or_validate_bytes(plan_path, plan_bytes)
    observed_plan = _load_canonical_record(plan_path, label="analysis checkpoint plan")
    if observed_plan != json_value(checkpoint_plan):
        raise ValueError("analysis checkpoint plan differs from current inputs")
    arms = scratch / "arms"
    arms.mkdir(exist_ok=True)
    if arms.is_symlink() or not arms.is_dir():
        raise RuntimeError("analysis checkpoint arms path must be a regular directory")
    expected_names = {_checkpoint_filename(binding) for binding in checkpoint_plan["ordered_cells"]}
    for entry in arms.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise RuntimeError(f"analysis checkpoint is not a regular file: {entry}")
        if entry.name not in expected_names:
            raise ValueError(f"analysis scratch contains an unmanaged checkpoint: {entry.name}")
    manifest_path = scratch / "checkpoint.manifest.json"
    return arms, manifest_path.exists() or manifest_path.is_symlink()


def _official_arm_statistics(
    *,
    plan: CocoStudyPlan,
    cells: Sequence[_ValidatedCell],
    targets: Sequence[Mapping[str, Any]],
    iou_thresholds: Sequence[float],
    category_ids: Sequence[int],
    coordinates: Sequence[float],
    iterations: int,
    seed: int,
    scratch_work_dir: Path | None,
    analysis_workers: int,
    progress: ProgressReporter | None,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    analysis_workers = _analysis_worker_count(analysis_workers)
    checkpoint_plan = _checkpoint_plan_record(
        plan=plan,
        cells=cells,
        targets=targets,
        iou_thresholds=iou_thresholds,
        category_ids=category_ids,
        coordinates=coordinates,
        iterations=iterations,
        seed=seed,
    )

    def execute(
        arms_directory: Path | None, committed_manifest: bool
    ) -> tuple[
        dict[tuple[str, str, str], dict[str, Any]],
        dict[str, Any],
    ]:
        stored: dict[int, tuple[dict[str, Any], bytes]] = {}
        missing: list[tuple[int, _ValidatedCell, Mapping[str, Any]]] = []
        for ordinal, (cell, binding) in enumerate(
            zip(cells, checkpoint_plan["ordered_cells"], strict=True)
        ):
            path = (
                None if arms_directory is None else arms_directory / _checkpoint_filename(binding)
            )
            if path is not None and (path.exists() or path.is_symlink()):
                record = _load_canonical_record(path, label="analysis arm checkpoint")
                validated = _validate_arm_checkpoint_record(
                    record,
                    checkpoint_plan=checkpoint_plan,
                    binding=binding,
                    label=f"analysis arm checkpoint {ordinal}",
                )
                payload = path.read_bytes()
                if payload != _record_bytes(validated):
                    raise RuntimeError("analysis arm checkpoint drifted while loading")
                stored[ordinal] = (validated, payload)
                _emit(
                    progress,
                    "analysis_arm_checkpoint_resumed",
                    completed=len(stored),
                    total=len(cells),
                    detector_id=binding["detector_id"],
                    profile_id=binding["profile_id"],
                    arm_id=binding["arm_id"],
                )
            else:
                missing.append((ordinal, cell, binding))
        if committed_manifest and missing:
            raise ValueError("committed checkpoint manifest references missing arm checkpoints")

        def compute_one(
            ordinal: int,
            cell: _ValidatedCell,
            binding: Mapping[str, Any],
        ) -> tuple[int, dict[str, Any], bytes]:
            record, payload = _compute_arm_checkpoint(
                cell=cell,
                binding=binding,
                checkpoint_plan=checkpoint_plan,
                image_ids=plan.image_ids,
                targets=targets,
                progress=progress,
            )
            return ordinal, record, payload

        if analysis_workers == 1:
            computed = (compute_one(ordinal, cell, binding) for ordinal, cell, binding in missing)
            for ordinal, record, payload in computed:
                binding = checkpoint_plan["ordered_cells"][ordinal]
                if arms_directory is not None:
                    _publish_or_validate_bytes(
                        arms_directory / _checkpoint_filename(binding),
                        payload,
                    )
                stored[ordinal] = (record, payload)
                _emit(
                    progress,
                    "analysis_arm_checkpoint_completed",
                    completed=len(stored),
                    total=len(cells),
                    detector_id=binding["detector_id"],
                    profile_id=binding["profile_id"],
                    arm_id=binding["arm_id"],
                )
        elif missing:
            worker_targets = _json_result_value(targets, label="analysis process targets")
            with _two_worker_process_pool(
                targets=worker_targets,
                image_ids=tuple(plan.image_ids),
                checkpoint_plan=json_value(checkpoint_plan),
            ) as pool:
                futures = {}
                for ordinal, cell, binding in missing:
                    _emit(
                        progress,
                        "analysis_arm_checkpoint_started",
                        completed=len(stored),
                        total=len(cells),
                        detector_id=binding["detector_id"],
                        profile_id=binding["profile_id"],
                        arm_id=binding["arm_id"],
                    )
                    future = pool.submit(
                        _compute_arm_checkpoint_in_process,
                        _arm_process_task(ordinal, cell, binding),
                    )
                    futures[future] = ordinal
                for future in as_completed(futures):
                    ordinal, record, payload = future.result()
                    binding = checkpoint_plan["ordered_cells"][ordinal]
                    if arms_directory is not None:
                        _publish_or_validate_bytes(
                            arms_directory / _checkpoint_filename(binding),
                            payload,
                        )
                    stored[ordinal] = (record, payload)
                    _emit(
                        progress,
                        "analysis_arm_checkpoint_completed",
                        completed=len(stored),
                        total=len(cells),
                        detector_id=binding["detector_id"],
                        profile_id=binding["profile_id"],
                        arm_id=binding["arm_id"],
                    )
        if set(stored) != set(range(len(cells))):
            raise RuntimeError("arm checkpoint execution did not cover every study cell")
        ordered = tuple(stored[index] for index in range(len(cells)))
        manifest = _checkpoint_manifest_record(
            checkpoint_plan=checkpoint_plan,
            ordered_checkpoints=ordered,
        )
        if arms_directory is not None:
            manifest_path = arms_directory.parent / "checkpoint.manifest.json"
            _publish_or_validate_bytes(manifest_path, _record_bytes(manifest))
            observed_manifest = _load_canonical_record(
                manifest_path,
                label="analysis checkpoint manifest",
            )
            if observed_manifest != manifest:
                raise ValueError("analysis checkpoint manifest differs from completed arms")
        arm_by_key = {
            (
                str(binding["detector_id"]),
                str(binding["profile_id"]),
                str(binding["arm_id"]),
            ): record["statistics"]
            for binding, (record, _) in zip(
                checkpoint_plan["ordered_cells"],
                ordered,
                strict=True,
            )
        }
        execution = {
            "contract_id": _CHECKPOINT_EXECUTION_CONTRACT_ID,
            "mode": "checkpointed_equivalent_official_builtins",
            "deterministic_worker_counts": [1, 2],
            "selected_worker_count": analysis_workers,
            "checkpoint_plan": checkpoint_plan,
            "checkpoint_manifest": manifest,
        }
        return arm_by_key, execution

    if scratch_work_dir is None:
        return execute(None, False)
    with advisory_target_lock(
        scratch_work_dir,
        purpose="analysis-checkpoints",
        exclusive=True,
    ):
        arms_directory, committed_manifest = _prepare_checkpoint_workspace(
            scratch_work_dir,
            checkpoint_plan=checkpoint_plan,
        )
        return execute(arms_directory, committed_manifest)


def _injected_checkpoint_execution_record() -> dict[str, Any]:
    return {
        "contract_id": _CHECKPOINT_EXECUTION_CONTRACT_ID,
        "mode": "injected_statistics_legacy",
        "deterministic_worker_counts": [1],
        "selected_worker_count": 1,
        "checkpoint_plan": None,
        "checkpoint_manifest": None,
    }


def derive_study_evidence_tier(plan: CocoStudyPlan) -> dict[str, Any]:
    """Classify a frozen plan without upgrading protocol variants to confirmatory."""

    if not isinstance(plan, CocoStudyPlan):
        raise TypeError("plan must be a CocoStudyPlan")
    return derive_study_evidence_tier_record(plan.record)


def _statistics_engines_record(
    paired_bootstrap: PairedBootstrap,
    curve_bootstrap: CurveBootstrap,
) -> dict[str, Any]:
    paired_mode = (
        "official_builtin"
        if paired_bootstrap is compute_paired_map_bootstrap
        else "injected_nonofficial"
    )
    curve_mode = (
        "official_builtin"
        if curve_bootstrap is compute_paired_map_curve_auc_bootstrap
        else "injected_nonofficial"
    )
    return {
        "contract_id": _STATISTICS_ENGINES_CONTRACT_ID,
        "official_exact_builtins": (
            paired_mode == "official_builtin" and curve_mode == "official_builtin"
        ),
        "paired_bootstrap": paired_mode,
        "curve_bootstrap": curve_mode,
    }


def _validate_statistics_engines(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("study analysis statistics_engines must be a mapping")
    record = json_value(value)
    if set(record) != {
        "contract_id",
        "official_exact_builtins",
        "paired_bootstrap",
        "curve_bootstrap",
    }:
        raise ValueError("study analysis statistics_engines is noncanonical")
    if record["contract_id"] != _STATISTICS_ENGINES_CONTRACT_ID:
        raise ValueError("study analysis statistics engine contract is unsupported")
    modes = (record["paired_bootstrap"], record["curve_bootstrap"])
    if any(mode not in {"official_builtin", "injected_nonofficial"} for mode in modes):
        raise ValueError("study analysis statistics engine mode is unsupported")
    official = all(mode == "official_builtin" for mode in modes)
    if not isinstance(record["official_exact_builtins"], bool) or (
        record["official_exact_builtins"] is not official
    ):
        raise ValueError("study analysis official statistics-engine flag is inconsistent")
    return record


def _validate_runtime_environment_identity(
    value: Any,
    *,
    analysis: bool,
) -> dict[str, Any]:
    label = "analysis runtime" if analysis else "detector runtime"
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} identity must be a mapping")
    record = json_value(value)
    common_keys = {
        "schema_version",
        "record_type",
        "method",
        "run_identity",
        "python",
        "operating_system",
        "hardware",
        "packages",
        "git",
        "runtime_identity_sha256",
    }
    expected_keys = (
        common_keys
        if analysis
        else common_keys
        | {
            "detector_device",
            "inference_execution",
        }
    )
    if set(record) != expected_keys:
        raise ValueError(f"{label} identity has missing or unknown fields")
    expected_type = (
        "analysis_runtime_reproducibility_identity"
        if analysis
        else "runtime_reproducibility_identity"
    )
    expected_method = ANALYSIS_RUNTIME_IDENTITY_METHOD if analysis else RUNTIME_IDENTITY_METHOD
    if (
        record["schema_version"] != 2
        or record["record_type"] != expected_type
        or record["method"] != expected_method
    ):
        raise ValueError(f"{label} identity has an unsupported schema or method")
    supplied = _require_sha256(
        record["runtime_identity_sha256"],
        label=f"{label} runtime_identity_sha256",
    )
    payload = {key: item for key, item in record.items() if key != "runtime_identity_sha256"}
    if supplied != canonical_sha256(payload):
        raise ValueError(f"{label} identity hash does not match its payload")

    run_identity = record["run_identity"]
    if not isinstance(run_identity, Mapping) or set(run_identity) != {
        "run_profile",
        "run_id",
        "archival_eligible",
    }:
        raise ValueError(f"{label} run identity is noncanonical")
    profile = run_identity["run_profile"]
    run_id = run_identity["run_id"]
    if not isinstance(profile, str) or not profile:
        raise ValueError(f"{label} run profile must be a nonempty string")
    if run_id is not None and (not isinstance(run_id, str) or not run_id):
        raise ValueError(f"{label} run_id must be null or a nonempty string")
    expected_archival = profile == "archival" and bool(run_id)
    if run_identity["archival_eligible"] is not expected_archival:
        raise ValueError(f"{label} archival eligibility is inconsistent")

    python = record["python"]
    if not isinstance(python, Mapping) or set(python) != {"implementation", "version"}:
        raise ValueError(f"{label} Python identity is noncanonical")
    if any(not isinstance(python[key], str) or not python[key] for key in python):
        raise ValueError(f"{label} Python identity contains an empty value")
    packages = record["packages"]
    if not isinstance(packages, Mapping) or set(packages) != {
        "method",
        "versions",
        "installed_distribution_universe",
        "project_installation",
    }:
        raise ValueError(f"{label} package identity is noncanonical")
    if packages["method"] != PACKAGE_IDENTITY_METHOD or not isinstance(
        packages["versions"], Mapping
    ):
        raise ValueError(f"{label} package-version method is unsupported")
    if set(packages["versions"]) != set(DEFAULT_RUNTIME_DISTRIBUTIONS):
        raise ValueError(f"{label} package universe differs from the frozen runtime contract")
    if any(
        version is not None and (not isinstance(version, str) or not version)
        for version in packages["versions"].values()
    ):
        raise ValueError(f"{label} package versions must be null or nonempty strings")
    installed_universe = packages["installed_distribution_universe"]
    if not isinstance(installed_universe, Mapping) or not installed_universe:
        raise ValueError(f"{label} installed distribution universe must be nonempty")
    for distribution, version in installed_universe.items():
        if (
            not isinstance(distribution, str)
            or not distribution
            or re.sub(r"[-_.]+", "-", distribution).lower() != distribution
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", distribution) is None
            or not isinstance(version, str)
            or not version
        ):
            raise ValueError(
                f"{label} installed distribution universe is not normalized name-to-version"
            )
    validate_project_installation_identity(packages["project_installation"])
    if not isinstance(record["operating_system"], Mapping) or not isinstance(
        record["hardware"], Mapping
    ):
        raise ValueError(f"{label} operating-system and hardware identities must be mappings")
    git = record["git"]
    expected_git_keys = {
        "available",
        "unavailable_reason",
        "commit",
        "branch",
        "detached_head",
        "dirty",
        "untracked_file_count",
        "ignored_files_included",
        "worktree_state_method",
        "worktree_state_sha256",
    }
    if not isinstance(git, Mapping) or set(git) != expected_git_keys:
        raise ValueError(f"{label} Git identity is noncanonical")
    if not isinstance(git["available"], bool):
        raise ValueError(f"{label} Git availability flag must be bool")
    if git["worktree_state_method"] != WORKTREE_STATE_METHOD:
        raise ValueError(f"{label} Git worktree identity method is unsupported")
    if git["ignored_files_included"] is not False:
        raise ValueError(f"{label} Git identity must exclude ignored files")
    if git["available"]:
        if git["unavailable_reason"] is not None:
            raise ValueError(f"{label} available Git identity has an unavailable reason")
        if (
            not isinstance(git["commit"], str)
            or len(git["commit"]) != 40
            or any(character not in "0123456789abcdef" for character in git["commit"])
        ):
            raise ValueError(f"{label} available Git identity requires a lowercase SHA-1 commit")
        if git["branch"] is not None and (not isinstance(git["branch"], str) or not git["branch"]):
            raise ValueError(f"{label} Git branch must be null or a nonempty string")
        if not isinstance(git["detached_head"], bool) or git["detached_head"] is not (
            git["branch"] is None
        ):
            raise ValueError(f"{label} Git detached-head status is inconsistent")
        _require_sha256(git["worktree_state_sha256"], label=f"{label} worktree identity")
        if not isinstance(git["dirty"], bool):
            raise ValueError(f"{label} Git dirty flag must be bool")
        if (
            not isinstance(git["untracked_file_count"], int)
            or isinstance(git["untracked_file_count"], bool)
            or git["untracked_file_count"] < 0
        ):
            raise ValueError(f"{label} Git untracked-file count is invalid")
    else:
        if not isinstance(git["unavailable_reason"], str) or not git["unavailable_reason"]:
            raise ValueError(f"{label} unavailable Git identity requires a reason")
        if any(
            git[key] is not None
            for key in (
                "commit",
                "branch",
                "detached_head",
                "dirty",
                "untracked_file_count",
                "worktree_state_sha256",
            )
        ):
            raise ValueError(f"{label} unavailable Git identity contains observed state")
    return record


def _derive_execution_eligibility(
    *,
    plan: CocoStudyPlan,
    manifests: Mapping[str, CocoStudyRunManifest],
    analysis_runtime: Mapping[str, Any],
    statistics_engines: Mapping[str, Any],
) -> dict[str, Any]:
    validated_analysis = _validate_runtime_environment_identity(
        analysis_runtime,
        analysis=True,
    )
    worker_runtimes = [
        _validate_runtime_environment_identity(manifest.record["runtime"], analysis=False)
        for manifest in manifests.values()
    ]
    runtimes = [*worker_runtimes, validated_analysis]
    run_identities = [runtime["run_identity"] for runtime in runtimes]
    run_ids = [identity["run_id"] for identity in run_identities]
    git_identities = [runtime["git"] for runtime in runtimes]
    all_git_clean = all(
        git["available"] is True and git["dirty"] is False for git in git_identities
    )
    git_bindings = {
        (
            git["commit"],
            git["worktree_state_method"],
            git["worktree_state_sha256"],
        )
        for git in git_identities
    }
    checks = {
        "official_exact_statistics_engines": bool(statistics_engines["official_exact_builtins"]),
        "all_detector_execution_stacks_match_publication": all(
            manifest.record["execution_engine"]["publication_stack_match"] is True
            for manifest in manifests.values()
        ),
        "all_runs_archival": all(
            identity["run_profile"] == "archival" and identity["archival_eligible"] is True
            for identity in run_identities
        ),
        "shared_nonempty_run_id": (
            all(isinstance(run_id, str) and bool(run_id) for run_id in run_ids)
            and len(set(run_ids)) == 1
        ),
        "all_git_available_and_clean": all_git_clean,
        "same_python_implementation_and_version": (
            len({json.dumps(runtime["python"], sort_keys=True) for runtime in runtimes}) == 1
        ),
        "same_hardware_identity": (
            len({json.dumps(runtime["hardware"], sort_keys=True) for runtime in runtimes}) == 1
        ),
        "same_git_commit_and_worktree_identity": (all_git_clean and len(git_bindings) == 1),
        "same_package_versions": (
            len({json.dumps(runtime["packages"], sort_keys=True) for runtime in runtimes}) == 1
        ),
    }
    return {
        "contract_id": _EXECUTION_ELIGIBILITY_CONTRACT_ID,
        "mode": "local_reproduction",
        "confirmatory_eligible": False,
        "tier": "reproduction_execution",
        "checks": checks,
    }


def _partitions(image_ids: Sequence[int], shard_size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(value) for value in image_ids[start : start + shard_size])
        for start in range(0, len(image_ids), shard_size)
    )


def _expected_merge_record(
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


def _manifest_matches_plan(
    manifest: CocoStudyRunManifest,
    *,
    plan: CocoStudyPlan,
    allocation: DetectorAllocation,
) -> None:
    record = manifest.record
    expected_cells = [cell["cell_sha256"] for cell in plan.cells(allocation.detector_id)]
    expected = {
        "study_plan_sha256": plan.study_plan_sha256,
        "dataset_sha256": plan.record["dataset"]["dataset_sha256"],
        "image_selection_sha256": plan.record["image_selection"]["selection_sha256"],
        "detector_allocation": allocation.to_dict(),
        "ordered_execution_cell_sha256": expected_cells,
        "preprocessing": json_value(
            LetterboxConfig(allocation.input_shape, allocation.pad_value).identity
        ),
    }
    for key, value in expected.items():
        if json_value(record[key]) != json_value(value):
            raise ValueError(
                f"detector run {allocation.detector_id!r} field {key!r} differs from the plan"
            )


def _prediction_records(merged: PredictionShardMerge) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for row in merged.predictions:
        prediction = json_value(row["prediction"])
        if not isinstance(prediction, Mapping) or set(prediction) != {"boxes", "labels", "scores"}:
            raise ValueError("study prediction rows require exactly boxes, labels, and scores")
        result.append({"image_id": int(row["image_id"]), **prediction})
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _ValidatedCell:
    cell: Mapping[str, Any]
    merge_record_sha256: str
    prediction_shard_index_sha256: str
    shard_count: int
    prediction_count: int
    shard_paths: tuple[Path, ...]


def _reload_cell_predictions(
    cell: _ValidatedCell,
    *,
    image_ids: Sequence[int],
) -> tuple[Mapping[str, Any], ...]:
    merged = merge_prediction_shards(cell.shard_paths, expected_image_ids=image_ids)
    if merged.index["index_sha256"] != cell.prediction_shard_index_sha256:
        raise RuntimeError("prediction shards drifted after completed-layout validation")
    records = _prediction_records(merged)
    if len(records) != cell.prediction_count:
        raise RuntimeError("prediction count drifted after completed-layout validation")
    return records


class _LazyArmPredictions(Mapping[str, Sequence[Mapping[str, Any]]]):
    """Reload exactly one detector/profile arm when the statistic requests it."""

    def __init__(self, cells: Sequence[_ValidatedCell], image_ids: Sequence[int]):
        self._cells = {str(value.cell["arm_id"]): value for value in cells}
        self._order = tuple(str(value.cell["arm_id"]) for value in cells)
        self._image_ids = tuple(image_ids)

    def __getitem__(self, arm_id: str) -> Sequence[Mapping[str, Any]]:
        try:
            cell = self._cells[arm_id]
        except KeyError as exc:
            raise KeyError(arm_id) from exc
        return _reload_cell_predictions(cell, image_ids=self._image_ids)

    def __iter__(self):
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._order)


class _LazyConditionAxis(Sequence[Sequence[Mapping[str, Any]]]):
    """Reload one coordinate arm at a time for the primary curve statistic."""

    def __init__(self, cells: Sequence[_ValidatedCell], image_ids: Sequence[int]):
        self._cells = tuple(cells)
        self._image_ids = tuple(image_ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(
                _reload_cell_predictions(cell, image_ids=self._image_ids)
                for cell in self._cells[index]
            )
        return _reload_cell_predictions(self._cells[index], image_ids=self._image_ids)

    def __len__(self) -> int:
        return len(self._cells)


def _select_primary_curve_cells(
    plan: CocoStudyPlan,
    cells_by_key: Mapping[tuple[str, str, str], _ValidatedCell],
) -> tuple[tuple[float, ...], tuple[_ValidatedCell, ...], tuple[_ValidatedCell, ...]]:
    """Select the prespecified physical and Gaussian cells in coordinate order."""

    primary = plan.record["analysis_protocol"]["primary_estimand"]
    detector_id = str(primary["primary_detector_id"])
    profile_id = str(primary["primary_profile_id"])
    coordinates = tuple(float(value) for value in primary["ordered_edge_waves_ref"])
    if len(coordinates) < 2:
        raise ValueError("publication AP-curve AUC requires at least two defocus coordinates")

    physical: list[_ValidatedCell] = []
    gaussian: list[_ValidatedCell] = []
    for coordinate in coordinates:
        physical_candidates: list[_ValidatedCell] = []
        gaussian_candidates: list[_ValidatedCell] = []
        for key, value in cells_by_key.items():
            if key[:2] != (detector_id, profile_id):
                continue
            condition = plan.condition(profile_id, key[2])
            if (
                isinstance(condition, PhysicalDefocusCondition)
                and condition.edge_waves_ref == coordinate
            ):
                physical_candidates.append(value)
            if (
                isinstance(condition, MechanismComparatorCondition)
                and condition.comparator_family == "gaussian"
                and condition.target_edge_waves_ref == coordinate
            ):
                gaussian_candidates.append(value)
        if len(physical_candidates) != 1 or len(gaussian_candidates) != 1:
            raise ValueError("primary physical/Gaussian curve cells are incomplete or duplicated")
        physical.append(physical_candidates[0])
        gaussian.append(gaussian_candidates[0])
    return coordinates, tuple(physical), tuple(gaussian)


def _managed_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"managed study layout is not a regular directory: {root}")
    files: set[Path] = set()
    for value in root.rglob("*"):
        if value.is_symlink():
            raise RuntimeError(f"managed study layout contains a symbolic link: {value}")
        if value.is_file():
            files.add(value.relative_to(root))
        elif not value.is_dir():
            raise RuntimeError(f"managed study layout contains a special file: {value}")
    return files


def _validate_completed_layout(
    *,
    plan: CocoStudyPlan,
    subset: NativeCOCODataset,
    output_root: Path,
    progress: ProgressReporter | None,
) -> tuple[
    dict[str, CocoStudyRunManifest],
    tuple[_ValidatedCell, ...],
    dict[str, dict[str, Any]],
]:
    if tuple(subset.image_ids) != plan.image_ids:
        raise ValueError("analysis COCO image order differs from the frozen study plan")
    if json_value(subset.identity) != json_value(plan.record["dataset"]):
        raise ValueError("analysis COCO dataset identity differs from the frozen study plan")
    if output_root.is_symlink() or not output_root.is_dir():
        raise FileNotFoundError(f"completed study output root is missing or unsafe: {output_root}")

    allocations = plan.allocations
    expected_manifests: set[Path] = set()
    expected_predictions: set[Path] = set()
    expected_merges: set[Path] = set()
    expected_completions: set[Path] = set()
    manifests: dict[str, CocoStudyRunManifest] = {}
    completions: dict[str, dict[str, Any]] = {}
    validated_cells: list[_ValidatedCell] = []
    total_cells = len(plan.cells())

    for allocation in allocations:
        detector_id = allocation.detector_id
        manifest_relative = Path(f"{detector_id}.run.json")
        manifest_path = output_root / "manifests" / manifest_relative
        expected_manifests.add(manifest_relative)
        _require_regular_file(manifest_path, label="study detector run manifest")
        manifest = load_study_run_manifest(manifest_path)
        _manifest_matches_plan(manifest, plan=plan, allocation=allocation)
        manifests[detector_id] = manifest
        preprocessing = LetterboxConfig(allocation.input_shape, allocation.pad_value)
        partitions = _partitions(plan.image_ids, allocation.shard_size)
        detector_merge_sha256: list[str] = []

        for cell in plan.cells(detector_id):
            profile_id = str(cell["profile_id"])
            arm_id = str(cell["arm_id"])
            profile = plan.profiles[profile_id]
            condition = plan.condition(profile_id, arm_id)
            condition_binding, _, _ = _condition_contract(
                condition=condition,
                profile=profile,
                preprocessing=preprocessing,
                label_space=allocation.label_space,
            )
            shard_paths: list[Path] = []
            for ordinal, selected in enumerate(partitions):
                relative = Path(detector_id) / profile_id / arm_id / f"shard_{ordinal:05d}.jsonl.gz"
                shard_path = output_root / "predictions" / relative
                receipt_path = prediction_shard_receipt_path(shard_path)
                expected_predictions.add(relative)
                expected_predictions.add(receipt_path.relative_to(output_root / "predictions"))
                expected_header = make_prediction_shard_header(
                    run=manifest.to_dict(),
                    dataset=plan.record["dataset"],
                    model=manifest.record["model"],
                    camera_profile_sha256=profile.profile_hash,
                    condition=condition_binding,
                    image_ids=selected,
                )
                shard = validate_existing_prediction_shard(
                    shard_path,
                    expected_header=expected_header,
                    receipt_path=receipt_path,
                )
                if shard is None:
                    raise FileNotFoundError(f"completed prediction shard is missing: {shard_path}")
                shard_paths.append(shard_path)

            merged = merge_prediction_shards(shard_paths, expected_image_ids=plan.image_ids)
            merge_relative = Path(detector_id) / profile_id / f"{arm_id}.index.json"
            merge_path = output_root / "merges" / merge_relative
            expected_merges.add(merge_relative)
            observed_merge = _load_canonical_record(
                merge_path,
                label="condition prediction merge index",
            )
            expected_merge = _expected_merge_record(
                plan=plan,
                manifest=manifest,
                cell=cell,
                merged=merged,
            )
            if observed_merge != expected_merge:
                raise ValueError(
                    f"published merge index differs from independently merged shards: "
                    f"{detector_id}/{profile_id}/{arm_id}"
                )
            detector_merge_sha256.append(str(expected_merge["merge_record_sha256"]))
            validated_cells.append(
                _ValidatedCell(
                    cell=cell,
                    merge_record_sha256=str(expected_merge["merge_record_sha256"]),
                    prediction_shard_index_sha256=str(merged.index["index_sha256"]),
                    shard_count=len(shard_paths),
                    prediction_count=len(merged.predictions),
                    shard_paths=tuple(shard_paths),
                )
            )
            _emit(
                progress,
                "validate_completed_cells",
                completed=len(validated_cells),
                total=total_cells,
                detector_id=detector_id,
                profile_id=profile_id,
                arm_id=arm_id,
            )

        completion_relative = Path(f"{detector_id}.complete.json")
        completion_path = output_root / "completions" / completion_relative
        expected_completions.add(completion_relative)
        completion = _load_canonical_record(
            completion_path,
            label="detector study-run completion receipt",
        )
        validate_study_run_completion_record(
            completion,
            plan=plan,
            manifest=manifest,
            ordered_merge_record_sha256=detector_merge_sha256,
        )
        completions[detector_id] = completion

    observed_manifests = _managed_files(output_root / "manifests")
    observed_predictions = _managed_files(output_root / "predictions")
    observed_merges = _managed_files(output_root / "merges")
    observed_completions = _managed_files(output_root / "completions")
    for label, observed, expected in (
        ("manifest", observed_manifests, expected_manifests),
        ("prediction", observed_predictions, expected_predictions),
        ("merge", observed_merges, expected_merges),
        ("completion", observed_completions, expected_completions),
    ):
        if observed != expected:
            missing = sorted(str(value) for value in expected - observed)
            extra = sorted(str(value) for value in observed - expected)
            raise ValueError(
                f"completed study {label} layout differs from the frozen semantic layout; "
                f"missing={missing}, extra={extra}"
            )
    if len(validated_cells) != total_cells:
        raise RuntimeError("validated cell cardinality differs from the frozen study plan")
    return manifests, tuple(validated_cells), completions


def _point_metrics(
    value: Mapping[str, Any],
    *,
    iou_thresholds: Sequence[float],
    category_ids: Sequence[int],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _POINT_METRIC_KEYS:
        raise ValueError(f"{label} does not contain the complete standard COCO metric record")
    thresholds = tuple(
        finite_float(item, field_name=f"{label}.mean_ap_iou_thresholds")
        for item in value["mean_ap_iou_thresholds"]
    )
    if thresholds != tuple(iou_thresholds):
        raise ValueError(f"{label} IoU threshold grid differs from the frozen analysis protocol")
    result: dict[str, Any] = {}
    for metric in _SCALAR_POINT_METRICS:
        scalar = finite_float(value[metric], field_name=f"{label}.{metric}")
        if scalar != -1.0 and not 0.0 <= scalar <= 1.0:
            raise ValueError(f"{label}.{metric} must be -1 or lie in [0, 1]")
        result[metric] = scalar
    raw_per_class = value["per_class_ap"]
    if isinstance(raw_per_class, list):
        if any(
            not isinstance(item, Mapping) or set(item) != {"category_id", "ap"}
            for item in raw_per_class
        ):
            raise ValueError(f"{label}.per_class_ap list is noncanonical")
        if [item["category_id"] for item in raw_per_class] != list(category_ids):
            raise ValueError(f"{label}.per_class_ap list differs from canonical category order")
        per_class_items = [(item["category_id"], item["ap"]) for item in raw_per_class]
    elif not isinstance(raw_per_class, Mapping):
        raise TypeError(f"{label}.per_class_ap must be a mapping or canonical list")
    else:
        per_class_items = list(raw_per_class.items())
    normalized_per_class: dict[int, float] = {}
    for raw_category, raw_ap in per_class_items:
        if isinstance(raw_category, bool):
            raise TypeError(f"{label}.per_class_ap category IDs must be integers")
        try:
            category = int(raw_category)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label}.per_class_ap category IDs must be integers") from exc
        if str(category) != str(raw_category):
            raise ValueError(f"{label}.per_class_ap category IDs are noncanonical")
        if category in normalized_per_class:
            raise ValueError(f"{label}.per_class_ap contains duplicate categories")
        ap = finite_float(raw_ap, field_name=f"{label}.per_class_ap[{category}]")
        if ap != -1.0 and not 0.0 <= ap <= 1.0:
            raise ValueError(f"{label}.per_class_ap values must be -1 or lie in [0, 1]")
        normalized_per_class[category] = ap
    if set(normalized_per_class) != set(category_ids):
        raise ValueError(f"{label}.per_class_ap category universe differs from COCO")
    result["mean_ap_iou_thresholds"] = list(thresholds)
    result["per_class_ap"] = [
        {"category_id": category, "ap": normalized_per_class[category]} for category in category_ids
    ]
    aliases = (
        ("map50_95", "mean_ap"),
        ("map50_95_small", "mean_ap_small"),
        ("map50_95_medium", "mean_ap_medium"),
        ("map50_95_large", "mean_ap_large"),
        ("ar100", "mean_ar100"),
    )
    for published, generic in aliases:
        if result[published] != result[generic]:
            raise ValueError(f"{label}.{published} differs from its {generic} alias")
    return result


def _secondary_result(
    raw: Mapping[str, Any],
    *,
    condition_order: Sequence[str],
    iterations: int,
    seed: int,
    image_count: int,
    iou_thresholds: Sequence[float],
    category_ids: Sequence[int],
    label: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    normalized = _json_result_value(raw, label=label)
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must return a mapping")
    expected_keys = {
        "method",
        "iterations",
        "seed",
        "image_count",
        "category_ids",
        "baseline_condition",
        "condition_order",
        "conditions",
        "bootstrap_failures",
        "matching_evaluations_per_condition",
        "bootstrap_accumulation",
        "memory_strategy",
    }
    if set(normalized) != expected_keys:
        raise ValueError(f"{label} has missing or unknown fields")
    expected_metadata = {
        "method": "paired_image_cluster_cached_coco_percentile_bootstrap_v2",
        "iterations": iterations,
        "seed": seed,
        "image_count": image_count,
        "category_ids": list(category_ids),
        "baseline_condition": "modeled_neutral",
        "condition_order": list(condition_order),
        "bootstrap_failures": 0,
        "matching_evaluations_per_condition": 1,
        "bootstrap_accumulation": ("repeat_cached_per_image_category_matches_then_coco_pr_v1"),
        "memory_strategy": "one_condition_match_cache_at_a_time_v1",
    }
    for key, expected in expected_metadata.items():
        if normalized.get(key) != expected:
            raise ValueError(f"{label} field {key!r} differs from the frozen protocol")
    conditions = normalized.get("conditions")
    if not isinstance(conditions, list) or [item.get("condition") for item in conditions] != list(
        condition_order
    ):
        raise ValueError(f"{label} condition results are incomplete or out of order")
    points: dict[str, dict[str, Any]] = {}
    for item in conditions:
        if not isinstance(item, dict) or set(item) != {
            "condition",
            "metrics",
            "marginal_percentile_95",
            "paired_difference_to_baseline",
            "paired_ratio_to_baseline",
        }:
            raise ValueError(f"{label} contains a noncanonical condition result")
        arm_id = str(item["condition"])
        points[arm_id] = _point_metrics(
            item.get("metrics"),
            iou_thresholds=iou_thresholds,
            category_ids=category_ids,
            label=f"{label}.{arm_id}.metrics",
        )
        item["metrics"] = points[arm_id]
    baseline = points["modeled_neutral"]
    if any(baseline[metric] <= 0.0 for metric in ("map50", "map50_95")):
        raise ValueError(f"{label} modeled-neutral AP must be positive for paired ratios")
    for item in conditions:
        arm_id = str(item["condition"])
        for section_name in (
            "marginal_percentile_95",
            "paired_difference_to_baseline",
            "paired_ratio_to_baseline",
        ):
            section = item[section_name]
            if not isinstance(section, Mapping) or set(section) != {"map50", "map50_95"}:
                raise ValueError(f"{label}.{arm_id}.{section_name} is incomplete")
            for metric in ("map50", "map50_95"):
                value = section[metric]
                if section_name == "marginal_percentile_95":
                    _validated_interval(
                        value,
                        label=f"{label}.{arm_id}.{section_name}.{metric}",
                        minimum=0.0,
                        maximum=1.0,
                    )
                    continue
                if not isinstance(value, Mapping) or set(value) != {
                    "estimate",
                    "percentile_95",
                }:
                    raise ValueError(f"{label}.{arm_id}.{section_name}.{metric} is noncanonical")
                estimate = finite_float(
                    value["estimate"],
                    field_name=f"{label}.{arm_id}.{section_name}.{metric}.estimate",
                )
                minimum, maximum = (
                    (-1.0, 1.0) if section_name == "paired_difference_to_baseline" else (0.0, None)
                )
                if estimate < minimum or (maximum is not None and estimate > maximum):
                    raise ValueError(
                        f"{label}.{arm_id}.{section_name}.{metric} estimate is out of range"
                    )
                _validated_interval(
                    value["percentile_95"],
                    label=f"{label}.{arm_id}.{section_name}.{metric}.percentile_95",
                    minimum=minimum,
                    maximum=maximum,
                )
                expected = (
                    points[arm_id][metric] - baseline[metric]
                    if section_name == "paired_difference_to_baseline"
                    else points[arm_id][metric] / baseline[metric]
                )
                if not np.isclose(estimate, expected, rtol=0.0, atol=2e-14):
                    raise ValueError(
                        f"{label}.{arm_id}.{section_name}.{metric} estimate is inconsistent"
                    )
    neutral = next(item for item in conditions if item["condition"] == "modeled_neutral")
    for metric in ("map50", "map50_95"):
        difference = neutral["paired_difference_to_baseline"][metric]
        ratio = neutral["paired_ratio_to_baseline"][metric]
        if difference["estimate"] != 0.0 or _validated_interval(
            difference["percentile_95"],
            label=f"{label}.modeled_neutral difference self interval",
            minimum=-1.0,
            maximum=1.0,
        ) != (0.0, 0.0):
            raise ValueError(f"{label} modeled-neutral self difference must be exactly zero")
        if ratio["estimate"] != 1.0 or _validated_interval(
            ratio["percentile_95"],
            label=f"{label}.modeled_neutral ratio self interval",
            minimum=0.0,
        ) != (1.0, 1.0):
            raise ValueError(f"{label} modeled-neutral self ratio must be exactly one")
    return normalized, points


def _validated_interval(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float]:
    if not isinstance(value, Mapping) or set(value) != {"lower", "upper"}:
        raise ValueError(f"{label} must contain exactly lower and upper")
    lower = finite_float(value["lower"], field_name=f"{label}.lower")
    upper = finite_float(value["upper"], field_name=f"{label}.upper")
    if lower > upper:
        raise ValueError(f"{label} lower bound exceeds its upper bound")
    if minimum is not None and lower < minimum:
        raise ValueError(f"{label} lower bound is below {minimum}")
    if maximum is not None and upper > maximum:
        raise ValueError(f"{label} upper bound exceeds {maximum}")
    return lower, upper


def _primary_result(
    raw: Mapping[str, Any],
    *,
    coordinates: Sequence[float],
    iterations: int,
    seed: int,
    image_count: int,
    category_ids: Sequence[int],
) -> dict[str, Any]:
    normalized_coordinates = tuple(
        finite_float(value, field_name="primary curve coordinate") for value in coordinates
    )
    if len(normalized_coordinates) < 2 or any(
        right <= left
        for left, right in zip(
            normalized_coordinates,
            normalized_coordinates[1:],
        )
    ):
        raise ValueError("primary curve coordinates must be strictly increasing")
    coordinate_span = normalized_coordinates[-1] - normalized_coordinates[0]
    normalized = _json_result_value(raw, label="primary curve-AUC bootstrap")
    if not isinstance(normalized, dict):
        raise TypeError("primary curve-AUC bootstrap must return a mapping")
    expected_keys = {
        "method",
        "iterations",
        "seed",
        "image_count",
        "category_ids",
        "coordinate_name",
        "coordinate_unit",
        "ordered_coordinates",
        "integration",
        "contrast_orientation",
        "comparator_name",
        "metrics",
        "primary_metric",
        "bootstrap_failures",
        "matching_evaluations_per_condition",
        "bootstrap_accumulation",
        "memory_strategy",
    }
    if set(normalized) != expected_keys:
        raise ValueError("primary curve-AUC result has missing or unknown fields")
    expected = {
        "method": "paired_image_cluster_cached_coco_curve_auc_percentile_bootstrap_v1",
        "iterations": iterations,
        "seed": seed,
        "image_count": image_count,
        "category_ids": list(category_ids),
        "coordinate_name": "edge_waves_ref",
        "coordinate_unit": "waves_at_reference_wavelength",
        "ordered_coordinates": list(normalized_coordinates),
        "integration": "trapezoidal_over_declared_coordinate_grid",
        "contrast_orientation": "physical_minus_comparator",
        "comparator_name": "gaussian",
        "primary_metric": "map50_95",
        "bootstrap_failures": 0,
        "matching_evaluations_per_condition": 1,
        "bootstrap_accumulation": ("repeat_cached_per_image_category_matches_then_coco_pr_v1"),
        "memory_strategy": "one_condition_match_cache_at_a_time_v1",
    }
    for key, value in expected.items():
        if normalized.get(key) != value:
            raise ValueError(f"primary curve-AUC field {key!r} differs from the frozen protocol")
    metrics = normalized.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {"map50", "map50_95"}:
        raise ValueError("primary curve-AUC result is missing the prespecified map50_95 estimand")
    for metric, metric_result in metrics.items():
        if not isinstance(metric_result, Mapping) or set(metric_result) != {
            "physical_curve",
            "comparator_curve",
            "paired_difference_curve",
            "paired_physical_minus_comparator_auc",
        }:
            raise ValueError(f"primary curve-AUC {metric} result is noncanonical")
        curves: dict[str, list[float]] = {}
        for curve_name, interval_name in (
            ("physical_curve", "marginal_percentile_95"),
            ("comparator_curve", "marginal_percentile_95"),
            ("paired_difference_curve", "percentile_95"),
        ):
            curve = metric_result[curve_name]
            if not isinstance(curve, list) or len(curve) != len(normalized_coordinates):
                raise ValueError(f"primary curve-AUC {metric}.{curve_name} is incomplete")
            estimates: list[float] = []
            for index, point in enumerate(curve):
                if not isinstance(point, Mapping) or set(point) != {
                    "coordinate",
                    "estimate",
                    interval_name,
                }:
                    raise ValueError(
                        f"primary curve-AUC {metric}.{curve_name}[{index}] is noncanonical"
                    )
                coordinate = finite_float(
                    point["coordinate"],
                    field_name=f"primary {metric}.{curve_name}[{index}].coordinate",
                )
                if coordinate != normalized_coordinates[index]:
                    raise ValueError("primary curve-AUC point coordinate drifted")
                estimate = finite_float(
                    point["estimate"],
                    field_name=f"primary {metric}.{curve_name}[{index}].estimate",
                )
                minimum, maximum = (
                    (-1.0, 1.0) if curve_name == "paired_difference_curve" else (0.0, 1.0)
                )
                if estimate < minimum or estimate > maximum:
                    raise ValueError(
                        f"primary curve-AUC {metric}.{curve_name}[{index}] is out of range"
                    )
                estimates.append(estimate)
                _validated_interval(
                    point[interval_name],
                    label=f"primary {metric}.{curve_name}[{index}].{interval_name}",
                    minimum=minimum,
                    maximum=maximum,
                )
            curves[curve_name] = estimates
        expected_difference = [
            physical - comparator
            for physical, comparator in zip(curves["physical_curve"], curves["comparator_curve"])
        ]
        if not np.allclose(
            curves["paired_difference_curve"], expected_difference, rtol=0.0, atol=2e-14
        ):
            raise ValueError(f"primary curve-AUC {metric} paired difference is inconsistent")
        auc = metric_result["paired_physical_minus_comparator_auc"]
        if not isinstance(auc, Mapping) or set(auc) != {"estimate", "percentile_95", "unit"}:
            raise ValueError(f"primary curve-AUC {metric} AUC record is noncanonical")
        if auc["unit"] != "AP_times_coordinate_unit":
            raise ValueError(f"primary curve-AUC {metric} unit drifted")
        estimate = finite_float(auc["estimate"], field_name=f"primary {metric} AUC estimate")
        expected_auc = sum(
            0.5
            * (expected_difference[index] + expected_difference[index + 1])
            * (normalized_coordinates[index + 1] - normalized_coordinates[index])
            for index in range(len(normalized_coordinates) - 1)
        )
        if estimate < -coordinate_span or estimate > coordinate_span:
            raise ValueError(f"primary curve-AUC {metric} AUC estimate is out of range")
        if not np.isclose(estimate, expected_auc, rtol=0.0, atol=2e-14):
            raise ValueError(f"primary curve-AUC {metric} AUC estimate is inconsistent")
        _validated_interval(
            auc["percentile_95"],
            label=f"primary {metric} AUC percentile_95",
            minimum=-coordinate_span,
            maximum=coordinate_span,
        )
    return normalized


def _validate_checkpoint_execution(
    value: Mapping[str, Any],
    *,
    plan: CocoStudyPlan,
    coverage_by_key: Mapping[tuple[str, str, str], Mapping[str, Any]],
    official_statistics: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("study analysis checkpoint execution record must be a mapping")
    record = json_value(value)
    if set(record) != {
        "contract_id",
        "mode",
        "deterministic_worker_counts",
        "selected_worker_count",
        "checkpoint_plan",
        "checkpoint_manifest",
    }:
        raise ValueError("study analysis checkpoint execution record is noncanonical")
    if record["contract_id"] != _CHECKPOINT_EXECUTION_CONTRACT_ID:
        raise ValueError("study analysis checkpoint execution contract is unsupported")
    if not official_statistics:
        if record != _injected_checkpoint_execution_record():
            raise ValueError("injected study statistics require the legacy execution record")
        return record
    if record["mode"] != "checkpointed_equivalent_official_builtins" or record[
        "deterministic_worker_counts"
    ] != [1, 2]:
        raise ValueError("official study analysis execution mode is unsupported")
    _analysis_worker_count(record["selected_worker_count"])
    checkpoint_plan = record["checkpoint_plan"]
    if not isinstance(checkpoint_plan, Mapping):
        raise TypeError("official analysis requires a checkpoint plan")
    checkpoint_plan = json_value(checkpoint_plan)
    if set(checkpoint_plan) != {
        "schema_version",
        "record_type",
        "contract_id",
        "implementation_sha256",
        "study_plan_sha256",
        "dataset_sha256",
        "image_selection_sha256",
        "ordered_image_ids",
        "target_input_sha256",
        "bootstrap_replicates",
        "bootstrap_seed",
        "iou_thresholds",
        "category_ids",
        "primary_ordered_coordinates",
        "ordered_cells",
        "checkpoint_plan_sha256",
    }:
        raise ValueError("study analysis checkpoint plan is noncanonical")
    if (
        checkpoint_plan["schema_version"] != 1
        or checkpoint_plan["record_type"] != _CHECKPOINT_PLAN_TYPE
        or checkpoint_plan["contract_id"] != _CHECKPOINT_EXECUTION_CONTRACT_ID
    ):
        raise ValueError("study analysis checkpoint plan has an unsupported schema")
    _require_sha256(
        checkpoint_plan["implementation_sha256"],
        label="checkpoint implementation_sha256",
    )
    _require_sha256(
        checkpoint_plan["target_input_sha256"],
        label="checkpoint target_input_sha256",
    )
    analysis = plan.record["analysis_protocol"]
    expected_global = {
        "study_plan_sha256": plan.study_plan_sha256,
        "dataset_sha256": plan.record["dataset"]["dataset_sha256"],
        "image_selection_sha256": plan.record["image_selection"]["selection_sha256"],
        "ordered_image_ids": list(plan.image_ids),
        "bootstrap_replicates": int(analysis["uncertainty"]["replicates"]),
        "bootstrap_seed": int(analysis["uncertainty"]["seed"]),
        "iou_thresholds": [float(item) for item in analysis["primary_metric"]["iou_thresholds"]],
        "category_ids": [int(item) for item in plan.record["dataset"]["category_ids"]],
        "primary_ordered_coordinates": [
            float(item) for item in analysis["primary_estimand"]["ordered_edge_waves_ref"]
        ],
    }
    for key, expected in expected_global.items():
        if checkpoint_plan[key] != expected:
            raise ValueError(f"study analysis checkpoint plan field {key!r} drifted")
    plan_cells = tuple(plan.cells())
    ordered_cells = checkpoint_plan["ordered_cells"]
    if not isinstance(ordered_cells, list) or len(ordered_cells) != len(plan_cells):
        raise ValueError("study analysis checkpoint cell bindings are incomplete")
    for ordinal, (plan_cell, binding) in enumerate(zip(plan_cells, ordered_cells, strict=True)):
        if not isinstance(binding, Mapping) or set(binding) != {
            "ordinal",
            "detector_id",
            "profile_id",
            "arm_id",
            "cell_sha256",
            "condition_sha256",
            "merge_record_sha256",
            "prediction_shard_index_sha256",
            "arm_input_sha256",
        }:
            raise ValueError("study analysis checkpoint cell binding is noncanonical")
        key = (
            str(plan_cell["detector_id"]),
            str(plan_cell["profile_id"]),
            str(plan_cell["arm_id"]),
        )
        coverage = coverage_by_key.get(key)
        if coverage is None:
            raise ValueError("study analysis checkpoint cell is absent from coverage")
        expected_binding_payload = {
            "ordinal": ordinal,
            "detector_id": key[0],
            "profile_id": key[1],
            "arm_id": key[2],
            "cell_sha256": plan_cell["cell_sha256"],
            "condition_sha256": plan_cell["condition_sha256"],
            "merge_record_sha256": coverage["merge_record_sha256"],
            "prediction_shard_index_sha256": coverage["prediction_shard_index_sha256"],
        }
        expected_binding = {
            **expected_binding_payload,
            "arm_input_sha256": canonical_sha256(expected_binding_payload),
        }
        if binding != expected_binding:
            raise ValueError("study analysis checkpoint cell binding drifted")
    supplied_plan = _require_sha256(
        checkpoint_plan["checkpoint_plan_sha256"],
        label="checkpoint_plan_sha256",
    )
    plan_payload = {
        key: item for key, item in checkpoint_plan.items() if key != "checkpoint_plan_sha256"
    }
    if supplied_plan != canonical_sha256(plan_payload):
        raise ValueError("checkpoint plan hash does not match its payload")
    manifest = _validate_checkpoint_manifest(
        record["checkpoint_manifest"],
        checkpoint_plan=checkpoint_plan,
    )
    record["checkpoint_plan"] = checkpoint_plan
    record["checkpoint_manifest"] = manifest
    return record


def _validate_study_analysis_record(
    value: Mapping[str, Any],
    *,
    plan: CocoStudyPlan,
) -> dict[str, Any]:
    if not isinstance(plan, CocoStudyPlan):
        raise TypeError("study analysis validation requires a CocoStudyPlan")
    if not isinstance(value, Mapping):
        raise TypeError("study analysis result must be a mapping")
    record = json_value(value)
    expected_keys = {
        "schema_version",
        "record_type",
        "implementation_id",
        "statistics_engines",
        "checkpoint_execution",
        "analysis_runtime",
        "execution_eligibility",
        "study_plan_sha256",
        "dataset_sha256",
        "image_selection_sha256",
        "evidence_tier",
        "analysis_protocol",
        "detector_runs",
        "coverage",
        "point_metrics",
        "primary_estimand",
        "secondary_analyses",
        "study_analysis_sha256",
    }
    if set(record) != expected_keys:
        raise ValueError("study analysis result has missing or unknown fields")
    if record["schema_version"] != 6 or record["record_type"] != _RESULT_TYPE:
        raise ValueError("study analysis result has an unsupported schema or record type")
    supplied = _require_sha256(
        record["study_analysis_sha256"],
        label="study_analysis_sha256",
    )
    payload = {key: item for key, item in record.items() if key != "study_analysis_sha256"}
    if supplied != canonical_sha256(payload):
        raise ValueError("study_analysis_sha256 does not match the result payload")

    statistics_engines = _validate_statistics_engines(record["statistics_engines"])
    expected_implementation = (
        STUDY_ANALYSIS_IMPLEMENTATION_ID
        if statistics_engines["official_exact_builtins"]
        else NONOFFICIAL_STUDY_ANALYSIS_IMPLEMENTATION_ID
    )
    if record["implementation_id"] != expected_implementation:
        raise ValueError("study analysis implementation ID differs from its statistics engines")
    if record["study_plan_sha256"] != plan.study_plan_sha256:
        raise ValueError("study analysis is bound to a different study plan")
    if record["dataset_sha256"] != plan.record["dataset"]["dataset_sha256"]:
        raise ValueError("study analysis is bound to a different dataset")
    if record["image_selection_sha256"] != plan.record["image_selection"]["selection_sha256"]:
        raise ValueError("study analysis is bound to a different image selection")
    expected_evidence = derive_study_evidence_tier(plan)
    if record["evidence_tier"] != expected_evidence:
        raise ValueError("study analysis evidence tier differs from its frozen plan")
    analysis = json_value(plan.record["analysis_protocol"])
    if record["analysis_protocol"] != analysis:
        raise ValueError("study analysis protocol differs from its frozen plan")

    analysis_runtime = _validate_runtime_environment_identity(
        record["analysis_runtime"],
        analysis=True,
    )
    detector_runs = record["detector_runs"]
    if not isinstance(detector_runs, list) or len(detector_runs) != len(plan.allocations):
        raise ValueError("study analysis detector runs are incomplete")
    manifests: dict[str, CocoStudyRunManifest] = {}
    completion_records: dict[str, dict[str, Any]] = {}
    for allocation, item in zip(plan.allocations, detector_runs, strict=True):
        if not isinstance(item, Mapping) or set(item) != {
            "detector_id",
            "manifest",
            "completion",
        }:
            raise ValueError("study analysis detector run binding is noncanonical")
        if item["detector_id"] != allocation.detector_id:
            raise ValueError("study analysis detector run order differs from the plan")
        manifest = CocoStudyRunManifest(item["manifest"])
        _manifest_matches_plan(manifest, plan=plan, allocation=allocation)
        if not isinstance(item["completion"], Mapping):
            raise TypeError("study analysis detector completion receipt must be a mapping")
        manifests[allocation.detector_id] = manifest
        completion_records[allocation.detector_id] = json_value(item["completion"])
    expected_execution = _derive_execution_eligibility(
        plan=plan,
        manifests=manifests,
        analysis_runtime=analysis_runtime,
        statistics_engines=statistics_engines,
    )
    if record["execution_eligibility"] != expected_execution:
        raise ValueError("study analysis execution eligibility is inconsistent")
    plan_cells = tuple(plan.cells())
    coverage = record["coverage"]
    if not isinstance(coverage, Mapping) or set(coverage) != {
        "status",
        "ordered_image_count",
        "executed_cell_count",
        "prediction_record_count",
        "cells",
    }:
        raise ValueError("study analysis coverage record is noncanonical")
    if coverage["status"] != "complete_exact_semantic_layout":
        raise ValueError("study analysis coverage is not complete")
    if coverage["ordered_image_count"] != len(plan.image_ids):
        raise ValueError("study analysis coverage image count differs from the plan")
    if coverage["executed_cell_count"] != len(plan_cells):
        raise ValueError("study analysis coverage cell count differs from the plan")
    coverage_cells = coverage["cells"]
    if not isinstance(coverage_cells, list) or len(coverage_cells) != len(plan_cells):
        raise ValueError("study analysis coverage cells are incomplete")
    coverage_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    prediction_count = 0
    for plan_cell, observed in zip(plan_cells, coverage_cells, strict=True):
        if not isinstance(observed, Mapping) or set(observed) != {
            "detector_id",
            "profile_id",
            "arm_id",
            "cell_sha256",
            "prediction_count",
            "shard_count",
            "prediction_shard_index_sha256",
            "merge_record_sha256",
        }:
            raise ValueError("study analysis coverage cell is noncanonical")
        for key in ("detector_id", "profile_id", "arm_id", "cell_sha256"):
            if observed[key] != plan_cell[key]:
                raise ValueError(f"study analysis coverage cell {key} differs from the plan")
        allocation = plan.allocation(str(plan_cell["detector_id"]))
        expected_shards = len(_partitions(plan.image_ids, allocation.shard_size))
        if observed["shard_count"] != expected_shards:
            raise ValueError("study analysis coverage shard count differs from the plan")
        if observed["prediction_count"] != len(plan.image_ids):
            raise ValueError("study analysis coverage prediction count differs from the plan")
        _require_sha256(
            observed["prediction_shard_index_sha256"],
            label="prediction_shard_index_sha256",
        )
        _require_sha256(observed["merge_record_sha256"], label="merge_record_sha256")
        key = (
            str(observed["detector_id"]),
            str(observed["profile_id"]),
            str(observed["arm_id"]),
        )
        if key in coverage_by_key:
            raise ValueError("study analysis coverage contains duplicate cells")
        coverage_by_key[key] = observed
        prediction_count += int(observed["prediction_count"])
    if coverage["prediction_record_count"] != prediction_count:
        raise ValueError("study analysis aggregate prediction count is inconsistent")
    for allocation in plan.allocations:
        merge_sha256 = [
            str(
                coverage_by_key[
                    (
                        allocation.detector_id,
                        str(cell["profile_id"]),
                        str(cell["arm_id"]),
                    )
                ]["merge_record_sha256"]
            )
            for cell in plan.cells(allocation.detector_id)
        ]
        validate_study_run_completion_record(
            completion_records[allocation.detector_id],
            plan=plan,
            manifest=manifests[allocation.detector_id],
            ordered_merge_record_sha256=merge_sha256,
        )

    checkpoint_execution = _validate_checkpoint_execution(
        record["checkpoint_execution"],
        plan=plan,
        coverage_by_key=coverage_by_key,
        official_statistics=bool(statistics_engines["official_exact_builtins"]),
    )
    if checkpoint_execution != record["checkpoint_execution"]:
        raise ValueError("study analysis checkpoint execution record is not canonical")
    iou_thresholds = tuple(float(item) for item in analysis["primary_metric"]["iou_thresholds"])
    category_ids = tuple(int(item) for item in plan.record["dataset"]["category_ids"])
    point_records = record["point_metrics"]
    if not isinstance(point_records, list) or len(point_records) != len(plan_cells):
        raise ValueError("study analysis point metrics are incomplete")
    points: dict[tuple[str, str, str], dict[str, Any]] = {}
    for plan_cell, point in zip(plan_cells, point_records, strict=True):
        if not isinstance(point, Mapping) or set(point) != {
            "detector_id",
            "profile_id",
            "arm_id",
            "cell_sha256",
            "condition_sha256",
            "metrics",
        }:
            raise ValueError("study analysis point metric record is noncanonical")
        for key in (
            "detector_id",
            "profile_id",
            "arm_id",
            "cell_sha256",
            "condition_sha256",
        ):
            if point[key] != plan_cell[key]:
                raise ValueError(f"study analysis point metric {key} differs from the plan")
        identity = (
            str(point["detector_id"]),
            str(point["profile_id"]),
            str(point["arm_id"]),
        )
        if identity not in coverage_by_key or identity in points:
            raise ValueError("study analysis point/coverage cell binding is inconsistent")
        normalized_metrics = _point_metrics(
            point["metrics"],
            iou_thresholds=iou_thresholds,
            category_ids=category_ids,
            label=f"point metrics {'/'.join(identity)}",
        )
        if normalized_metrics != point["metrics"]:
            raise ValueError("study analysis point metrics are not canonical")
        points[identity] = normalized_metrics

    iterations = int(analysis["uncertainty"]["replicates"])
    seed = int(analysis["uncertainty"]["seed"])
    expected_groups: list[tuple[str, str, tuple[Mapping[str, Any], ...]]] = []
    for allocation in plan.allocations:
        for profile_id in allocation.profile_ids:
            selected = tuple(
                cell
                for cell in plan_cells
                if cell["detector_id"] == allocation.detector_id
                and cell["profile_id"] == profile_id
            )
            if selected:
                expected_groups.append((allocation.detector_id, profile_id, selected))
    secondary = record["secondary_analyses"]
    if not isinstance(secondary, list) or len(secondary) != len(expected_groups):
        raise ValueError("study analysis secondary groups are incomplete")
    for observed, (detector_id, profile_id, selected) in zip(
        secondary,
        expected_groups,
        strict=True,
    ):
        if not isinstance(observed, Mapping) or set(observed) != {
            "detector_id",
            "profile_id",
            "confirmatory",
            "interpretation",
            "paired_against_arm_id",
            "bootstrap",
        }:
            raise ValueError("study analysis secondary group is noncanonical")
        if (
            observed["detector_id"] != detector_id
            or observed["profile_id"] != profile_id
            or observed["confirmatory"] is not False
            or observed["interpretation"] != "descriptive_with_interval_no_hypothesis_test"
            or observed["paired_against_arm_id"] != "modeled_neutral"
        ):
            raise ValueError("study analysis secondary group metadata differs from the plan")
        condition_order = tuple(str(cell["arm_id"]) for cell in selected)
        normalized_bootstrap, bootstrap_points = _secondary_result(
            observed["bootstrap"],
            condition_order=condition_order,
            iterations=iterations,
            seed=seed,
            image_count=len(plan.image_ids),
            iou_thresholds=iou_thresholds,
            category_ids=category_ids,
            label=f"secondary bootstrap {detector_id}/{profile_id}",
        )
        if normalized_bootstrap != observed["bootstrap"]:
            raise ValueError("study analysis secondary bootstrap is not canonical")
        for arm_id, metrics in bootstrap_points.items():
            if metrics != points[(detector_id, profile_id, arm_id)]:
                raise ValueError("study analysis secondary metrics differ from point metrics")

    primary_protocol = analysis["primary_estimand"]
    primary_detector = str(primary_protocol["primary_detector_id"])
    primary_profile = str(primary_protocol["primary_profile_id"])
    coordinates = tuple(float(item) for item in primary_protocol["ordered_edge_waves_ref"])
    primary = record["primary_estimand"]
    if not isinstance(primary, Mapping) or set(primary) != {
        "confirmatory",
        "detector_id",
        "profile_id",
        "bootstrap",
    }:
        raise ValueError("study analysis primary estimand is noncanonical")
    expected_confirmatory = bool(
        expected_evidence["confirmatory_eligible"] and expected_execution["confirmatory_eligible"]
    )
    if (
        primary["confirmatory"] is not expected_confirmatory
        or primary["detector_id"] != primary_detector
        or primary["profile_id"] != primary_profile
    ):
        raise ValueError("study analysis primary metadata differs from the frozen contracts")
    normalized_primary = _primary_result(
        primary["bootstrap"],
        coordinates=coordinates,
        iterations=iterations,
        seed=seed,
        image_count=len(plan.image_ids),
        category_ids=category_ids,
    )
    if normalized_primary != primary["bootstrap"]:
        raise ValueError("study analysis primary bootstrap is not canonical")
    for metric, metric_result in normalized_primary["metrics"].items():
        for index, coordinate in enumerate(coordinates):
            physical = []
            comparator = []
            for cell in plan_cells:
                if cell["detector_id"] != primary_detector or cell["profile_id"] != primary_profile:
                    continue
                condition = plan.condition(primary_profile, str(cell["arm_id"]))
                if isinstance(condition, PhysicalDefocusCondition) and (
                    condition.edge_waves_ref == coordinate
                ):
                    physical.append(cell)
                if (
                    isinstance(condition, MechanismComparatorCondition)
                    and condition.comparator_family == "gaussian"
                    and condition.target_edge_waves_ref == coordinate
                ):
                    comparator.append(cell)
            if len(physical) != 1 or len(comparator) != 1:
                raise ValueError("study analysis primary curve cells differ from the plan")
            physical_key = (primary_detector, primary_profile, str(physical[0]["arm_id"]))
            comparator_key = (primary_detector, primary_profile, str(comparator[0]["arm_id"]))
            if not np.isclose(
                metric_result["physical_curve"][index]["estimate"],
                points[physical_key][metric],
                rtol=0.0,
                atol=2e-14,
            ):
                raise ValueError("study analysis primary physical curve differs from point metrics")
            if not np.isclose(
                metric_result["comparator_curve"][index]["estimate"],
                points[comparator_key][metric],
                rtol=0.0,
                atol=2e-14,
            ):
                raise ValueError(
                    "study analysis primary comparator curve differs from point metrics"
                )
    return record


@dataclass(frozen=True, slots=True)
class StudyAnalysisResult:
    """Immutable, plan-bound, content-addressed completed-study analysis."""

    record: Mapping[str, Any]
    plan: InitVar[CocoStudyPlan]
    _validation_plan: CocoStudyPlan = field(init=False, repr=False, compare=False)

    def __post_init__(self, plan: CocoStudyPlan) -> None:
        normalized = _validate_study_analysis_record(self.record, plan=plan)
        frozen = freeze_json_value(normalized)
        if not isinstance(frozen, MappingProxyType):  # pragma: no cover - narrowed above
            raise TypeError("study analysis result must freeze to a mapping")
        object.__setattr__(self, "record", frozen)
        object.__setattr__(self, "_validation_plan", plan)

    @property
    def study_analysis_sha256(self) -> str:
        return str(self.record["study_analysis_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return json_value(self.record)


def load_study_analysis(
    path: str | Path,
    *,
    plan: CocoStudyPlan,
) -> StudyAnalysisResult:
    """Load canonical analysis JSON and fully revalidate it against its plan."""

    record = _load_canonical_record(Path(path), label="study analysis result")
    return StudyAnalysisResult(record, plan)


def _analyze_completed_study_under_lock(
    *,
    plan: CocoStudyPlan,
    subset: NativeCOCODataset,
    output_root: str | Path,
    paired_bootstrap: PairedBootstrap = compute_paired_map_bootstrap,
    curve_bootstrap: CurveBootstrap = compute_paired_map_curve_auc_bootstrap,
    repository_root: str | Path | None = None,
    progress: ProgressReporter | None = None,
    scratch_work_dir: Path | None = None,
    analysis_workers: int = 1,
    observed_analysis_runtime: Mapping[str, Any] | None = None,
    canonical_analysis_runtime: Mapping[str, Any] | None = None,
) -> StudyAnalysisResult:
    """Validate all completed runs and execute the frozen publication analysis.

    Point metrics are obtained from each detector/profile paired bootstrap,
    whose internal COCO point pass evaluates every executed arm once.  This
    avoids an otherwise redundant full COCO matching pass over every cell.
    """

    if not isinstance(plan, CocoStudyPlan):
        raise TypeError("plan must be a CocoStudyPlan")
    if not callable(paired_bootstrap) or not callable(curve_bootstrap):
        raise TypeError("bootstrap implementations must be callable")
    analysis_workers = _analysis_worker_count(analysis_workers)
    root = Path(output_root)
    statistics_engines = _statistics_engines_record(paired_bootstrap, curve_bootstrap)
    analysis_implementation = (
        STUDY_ANALYSIS_IMPLEMENTATION_ID
        if statistics_engines["official_exact_builtins"]
        else NONOFFICIAL_STUDY_ANALYSIS_IMPLEMENTATION_ID
    )
    observed_runtime = _validate_runtime_environment_identity(
        analysis_runtime_reproducibility_identity(repository_root=repository_root)
        if observed_analysis_runtime is None
        else observed_analysis_runtime,
        analysis=True,
    )
    evidence_tier = derive_study_evidence_tier(plan)
    frozen_evidence_tier = plan.record.get("evidence_tier")
    if frozen_evidence_tier is not None and json_value(frozen_evidence_tier) != evidence_tier:
        raise ValueError("study plan evidence tier differs from recomputed eligibility")
    _emit(progress, "validate_completed_layout", completed=0, total=len(plan.cells()))
    manifests, cells, completions = _validate_completed_layout(
        plan=plan,
        subset=subset,
        output_root=root,
        progress=progress,
    )
    observed_execution_eligibility = _derive_execution_eligibility(
        plan=plan,
        manifests=manifests,
        analysis_runtime=observed_runtime,
        statistics_engines=statistics_engines,
    )
    if canonical_analysis_runtime is None:
        analysis_runtime = observed_runtime
        execution_eligibility = observed_execution_eligibility
    else:
        analysis_runtime = _validate_runtime_environment_identity(
            canonical_analysis_runtime,
            analysis=True,
        )
        execution_eligibility = _derive_execution_eligibility(
            plan=plan,
            manifests=manifests,
            analysis_runtime=analysis_runtime,
            statistics_engines=statistics_engines,
        )
    by_key = {
        (
            str(value.cell["detector_id"]),
            str(value.cell["profile_id"]),
            str(value.cell["arm_id"]),
        ): value
        for value in cells
    }
    targets = tuple(subset.target(image_id) for image_id in plan.image_ids)
    analysis = plan.record["analysis_protocol"]
    iou_thresholds = tuple(float(value) for value in analysis["primary_metric"]["iou_thresholds"])
    category_ids = tuple(int(value) for value in subset.category_ids)
    iterations = int(analysis["uncertainty"]["replicates"])
    seed = int(analysis["uncertainty"]["seed"])

    primary_protocol = analysis["primary_estimand"]
    primary_detector = str(primary_protocol["primary_detector_id"])
    primary_profile = str(primary_protocol["primary_profile_id"])
    coordinates, physical, gaussian = _select_primary_curve_cells(plan, by_key)

    groups: list[tuple[str, str, tuple[_ValidatedCell, ...]]] = []
    for allocation in plan.allocations:
        for profile_id in allocation.profile_ids:
            selected = tuple(
                value
                for value in cells
                if value.cell["detector_id"] == allocation.detector_id
                and value.cell["profile_id"] == profile_id
            )
            if selected:
                groups.append((allocation.detector_id, profile_id, selected))
    point_by_cell: dict[tuple[str, str, str], dict[str, Any]] = {}
    secondary: list[dict[str, Any]] = []
    if statistics_engines["official_exact_builtins"]:
        arm_by_cell, checkpoint_execution = _official_arm_statistics(
            plan=plan,
            cells=cells,
            targets=targets,
            iou_thresholds=iou_thresholds,
            category_ids=category_ids,
            coordinates=coordinates,
            iterations=iterations,
            seed=seed,
            scratch_work_dir=scratch_work_dir,
            analysis_workers=analysis_workers,
            progress=progress,
        )
    else:
        if scratch_work_dir is not None:
            raise ValueError("checkpointed analysis requires the official built-in statistics")
        if analysis_workers != 1:
            raise ValueError("parallel analysis requires the official built-in statistics")
        arm_by_cell = {}
        checkpoint_execution = _injected_checkpoint_execution_record()
    for group_index, (detector_id, profile_id, selected) in enumerate(groups):
        condition_order = tuple(str(value.cell["arm_id"]) for value in selected)
        if condition_order.count("modeled_neutral") != 1:
            raise ValueError(
                f"detector/profile group {detector_id}/{profile_id} lacks modeled-neutral"
            )
        _emit(
            progress,
            "secondary_paired_bootstrap_started",
            completed=group_index,
            total=len(groups),
            detector_id=detector_id,
            profile_id=profile_id,
            iterations=iterations,
        )

        def report_secondary(condition: str, completed: int, total: int) -> None:
            _emit(
                progress,
                "secondary_paired_bootstrap_progress",
                completed=completed,
                total=total,
                detector_id=detector_id,
                profile_id=profile_id,
                arm_id=condition,
                group_ordinal=group_index,
                group_count=len(groups),
            )

        if statistics_engines["official_exact_builtins"]:
            raw = assemble_paired_map_bootstrap(
                {
                    str(value.cell["arm_id"]): arm_by_cell[
                        (detector_id, profile_id, str(value.cell["arm_id"]))
                    ]
                    for value in selected
                },
                baseline_condition="modeled_neutral",
            )
        else:
            raw = paired_bootstrap(
                _LazyArmPredictions(selected, plan.image_ids),
                targets,
                baseline_condition="modeled_neutral",
                n_bootstrap=iterations,
                seed=seed,
                iou_thresholds=iou_thresholds,
                category_ids=category_ids,
                progress=report_secondary,
            )
        result, points = _secondary_result(
            raw,
            condition_order=condition_order,
            iterations=iterations,
            seed=seed,
            image_count=len(plan.image_ids),
            iou_thresholds=iou_thresholds,
            category_ids=category_ids,
            label=f"secondary bootstrap {detector_id}/{profile_id}",
        )
        if result.get("image_count") != len(plan.image_ids):
            raise ValueError("secondary bootstrap image count differs from the study plan")
        for arm_id, metrics in points.items():
            point_by_cell[(detector_id, profile_id, arm_id)] = metrics
        secondary.append(
            {
                "detector_id": detector_id,
                "profile_id": profile_id,
                "confirmatory": False,
                "interpretation": ("descriptive_with_interval_no_hypothesis_test"),
                "paired_against_arm_id": "modeled_neutral",
                "bootstrap": result,
            }
        )
        _emit(
            progress,
            "secondary_paired_bootstrap_completed",
            completed=group_index + 1,
            total=len(groups),
            detector_id=detector_id,
            profile_id=profile_id,
            iterations=iterations,
        )
    if set(point_by_cell) != set(by_key):
        raise RuntimeError("secondary analysis did not produce point metrics for every study cell")

    _emit(
        progress,
        "primary_curve_auc_bootstrap_started",
        completed=0,
        total=1,
        detector_id=primary_detector,
        profile_id=primary_profile,
        iterations=iterations,
    )

    def report_primary(condition: str, completed: int, total: int) -> None:
        _emit(
            progress,
            "primary_curve_auc_bootstrap_progress",
            completed=completed,
            total=total,
            detector_id=primary_detector,
            profile_id=primary_profile,
            curve_condition=condition,
        )

    if statistics_engines["official_exact_builtins"]:
        primary_raw = assemble_paired_map_curve_auc_bootstrap(
            [
                arm_by_cell[(primary_detector, primary_profile, str(value.cell["arm_id"]))]
                for value in physical
            ],
            [
                arm_by_cell[(primary_detector, primary_profile, str(value.cell["arm_id"]))]
                for value in gaussian
            ],
            coordinates=coordinates,
            comparator_name="gaussian",
        )
    else:
        primary_raw = curve_bootstrap(
            _LazyConditionAxis(physical, plan.image_ids),
            _LazyConditionAxis(gaussian, plan.image_ids),
            targets,
            coordinates=coordinates,
            comparator_name="gaussian",
            n_bootstrap=iterations,
            seed=seed,
            iou_thresholds=iou_thresholds,
            category_ids=category_ids,
            progress=report_primary,
        )
    primary = _primary_result(
        primary_raw,
        coordinates=coordinates,
        iterations=iterations,
        seed=seed,
        image_count=len(plan.image_ids),
        category_ids=category_ids,
    )
    for metric, metric_result in primary["metrics"].items():
        for index, (physical_cell, gaussian_cell) in enumerate(
            zip(physical, gaussian, strict=True)
        ):
            physical_arm = str(physical_cell.cell["arm_id"])
            gaussian_arm = str(gaussian_cell.cell["arm_id"])
            expected_physical = point_by_cell[(primary_detector, primary_profile, physical_arm)][
                metric
            ]
            expected_gaussian = point_by_cell[(primary_detector, primary_profile, gaussian_arm)][
                metric
            ]
            observed_physical = metric_result["physical_curve"][index]["estimate"]
            observed_gaussian = metric_result["comparator_curve"][index]["estimate"]
            if not np.isclose(
                observed_physical,
                expected_physical,
                rtol=0.0,
                atol=2e-14,
            ):
                raise ValueError(
                    f"primary {metric} physical curve disagrees with exact point metrics"
                )
            if not np.isclose(
                observed_gaussian,
                expected_gaussian,
                rtol=0.0,
                atol=2e-14,
            ):
                raise ValueError(
                    f"primary {metric} Gaussian curve disagrees with exact point metrics"
                )
    _emit(
        progress,
        "primary_curve_auc_bootstrap_completed",
        completed=1,
        total=1,
        detector_id=primary_detector,
        profile_id=primary_profile,
        iterations=iterations,
    )

    # No user callback runs after this event. Any mutation performed by the
    # callback is therefore covered by the final layout and runtime snapshots.
    _emit(
        progress,
        "final_completed_layout_revalidation_started",
        completed=len(cells),
        total=len(cells),
    )
    (
        final_manifests,
        final_cells,
        final_completions,
    ) = _validate_completed_layout(
        plan=plan,
        subset=subset,
        output_root=root,
        progress=None,
    )
    if {detector_id: manifest.to_dict() for detector_id, manifest in final_manifests.items()} != {
        detector_id: manifest.to_dict() for detector_id, manifest in manifests.items()
    }:
        raise RuntimeError("detector run manifests drifted during study analysis")
    if final_completions != completions:
        raise RuntimeError("detector completion receipts drifted during study analysis")

    def cell_snapshot(values: Sequence[_ValidatedCell]) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                json_value(value.cell),
                value.merge_record_sha256,
                value.prediction_shard_index_sha256,
                value.shard_count,
                value.prediction_count,
                tuple(str(path) for path in value.shard_paths),
            )
            for value in values
        )

    if cell_snapshot(final_cells) != cell_snapshot(cells):
        raise RuntimeError("prediction artifacts drifted during study analysis")
    final_observed_runtime = _validate_runtime_environment_identity(
        analysis_runtime_reproducibility_identity(
            repository_root=repository_root,
        ),
        analysis=True,
    )
    if final_observed_runtime != observed_runtime:
        raise RuntimeError("analysis runtime, package, or repository state drifted during analysis")
    final_observed_execution = _derive_execution_eligibility(
        plan=plan,
        manifests=final_manifests,
        analysis_runtime=final_observed_runtime,
        statistics_engines=statistics_engines,
    )
    if final_observed_execution != observed_execution_eligibility:
        raise RuntimeError("analysis execution eligibility drifted during analysis")
    final_analysis_runtime = analysis_runtime
    final_execution_eligibility = _derive_execution_eligibility(
        plan=plan,
        manifests=final_manifests,
        analysis_runtime=final_analysis_runtime,
        statistics_engines=statistics_engines,
    )
    if final_execution_eligibility != execution_eligibility:
        raise RuntimeError("canonical analysis execution eligibility drifted during analysis")
    point_records = [
        {
            "detector_id": str(value.cell["detector_id"]),
            "profile_id": str(value.cell["profile_id"]),
            "arm_id": str(value.cell["arm_id"]),
            "cell_sha256": str(value.cell["cell_sha256"]),
            "condition_sha256": str(value.cell["condition_sha256"]),
            "metrics": point_by_cell[
                (
                    str(value.cell["detector_id"]),
                    str(value.cell["profile_id"]),
                    str(value.cell["arm_id"]),
                )
            ],
        }
        for value in cells
    ]
    coverage_cells = [
        {
            "detector_id": str(value.cell["detector_id"]),
            "profile_id": str(value.cell["profile_id"]),
            "arm_id": str(value.cell["arm_id"]),
            "cell_sha256": str(value.cell["cell_sha256"]),
            "prediction_count": value.prediction_count,
            "shard_count": value.shard_count,
            "prediction_shard_index_sha256": value.prediction_shard_index_sha256,
            "merge_record_sha256": value.merge_record_sha256,
        }
        for value in cells
    ]
    payload = {
        "schema_version": 6,
        "record_type": _RESULT_TYPE,
        "implementation_id": analysis_implementation,
        "statistics_engines": statistics_engines,
        "checkpoint_execution": checkpoint_execution,
        "analysis_runtime": analysis_runtime,
        "execution_eligibility": execution_eligibility,
        "study_plan_sha256": plan.study_plan_sha256,
        "dataset_sha256": plan.record["dataset"]["dataset_sha256"],
        "image_selection_sha256": plan.record["image_selection"]["selection_sha256"],
        "evidence_tier": evidence_tier,
        "analysis_protocol": json_value(analysis),
        "detector_runs": [
            {
                "detector_id": allocation.detector_id,
                "manifest": manifests[allocation.detector_id].to_dict(),
                "completion": completions[allocation.detector_id],
            }
            for allocation in plan.allocations
        ],
        "coverage": {
            "status": "complete_exact_semantic_layout",
            "ordered_image_count": len(plan.image_ids),
            "executed_cell_count": len(cells),
            "prediction_record_count": sum(value.prediction_count for value in cells),
            "cells": coverage_cells,
        },
        "point_metrics": point_records,
        "primary_estimand": {
            "confirmatory": (
                evidence_tier["confirmatory_eligible"]
                and execution_eligibility["confirmatory_eligible"]
            ),
            "detector_id": primary_detector,
            "profile_id": primary_profile,
            "bootstrap": primary,
        },
        "secondary_analyses": secondary,
    }
    return StudyAnalysisResult(
        {**payload, "study_analysis_sha256": canonical_sha256(payload)},
        plan,
    )


def analyze_completed_study(
    *,
    plan: CocoStudyPlan,
    subset: NativeCOCODataset,
    output_root: str | Path,
    paired_bootstrap: PairedBootstrap = compute_paired_map_bootstrap,
    curve_bootstrap: CurveBootstrap = compute_paired_map_curve_auc_bootstrap,
    repository_root: str | Path | None = None,
    progress: ProgressReporter | None = None,
    scratch_work_dir: str | Path | None = None,
    analysis_workers: int = 1,
) -> StudyAnalysisResult:
    """Analyze one immutable completed-study snapshot under an exclusive lock."""

    analysis_workers = _analysis_worker_count(analysis_workers)
    if analysis_workers == 2 and scratch_work_dir is None:
        raise ValueError("two-worker analysis requires an explicit scratch_work_dir")
    scratch = (
        None
        if scratch_work_dir is None
        else _require_separate_scratch_path(
            scratch_work_dir,
            forbidden_roots=(output_root,)
            + (() if repository_root is None else (repository_root,)),
        )
    )

    with advisory_target_lock(
        output_root,
        purpose="study-layout",
        exclusive=True,
    ):
        return _analyze_completed_study_under_lock(
            plan=plan,
            subset=subset,
            output_root=output_root,
            paired_bootstrap=paired_bootstrap,
            curve_bootstrap=curve_bootstrap,
            repository_root=repository_root,
            progress=progress,
            scratch_work_dir=scratch,
            analysis_workers=analysis_workers,
        )


def verify_rederived_completed_study_analysis(
    *,
    plan: CocoStudyPlan,
    subset: NativeCOCODataset,
    output_root: str | Path,
    published_analysis: StudyAnalysisResult,
    repository_root: str | Path,
    progress: ProgressReporter | None = None,
) -> Mapping[str, Any]:
    """Rerun the built-in statistics and exact-compare a publication."""

    if not isinstance(plan, CocoStudyPlan):
        raise TypeError("plan must be a CocoStudyPlan")
    if not isinstance(published_analysis, StudyAnalysisResult):
        raise TypeError("published_analysis must be a StudyAnalysisResult")
    published = StudyAnalysisResult(published_analysis.record, plan)
    if published.record["statistics_engines"]["official_exact_builtins"] is not True:
        raise ValueError("published study analysis did not use the built-in statistics")
    root = Path(output_root)
    with advisory_target_lock(
        root,
        purpose="study-layout",
        exclusive=True,
    ):
        observed_runtime = _validate_runtime_environment_identity(
            analysis_runtime_reproducibility_identity(
                repository_root=repository_root,
            ),
            analysis=True,
        )
        rederived = _analyze_completed_study_under_lock(
            plan=plan,
            subset=subset,
            output_root=root,
            repository_root=repository_root,
            progress=progress,
            analysis_workers=int(published.record["checkpoint_execution"]["selected_worker_count"]),
            observed_analysis_runtime=observed_runtime,
            canonical_analysis_runtime=published.record["analysis_runtime"],
        )
    if rederived.to_dict() != published.to_dict():
        raise ValueError(
            "published study analysis does not exactly rederive from completed prediction shards"
        )
    frozen = freeze_json_value(observed_runtime)
    if not isinstance(frozen, MappingProxyType):  # pragma: no cover - narrowed above
        raise TypeError("study reanalysis runtime must freeze to a mapping")
    return frozen


def _completed_layout_verification_record(
    *,
    plan: CocoStudyPlan,
    manifests: Mapping[str, CocoStudyRunManifest],
    cells: Sequence[_ValidatedCell],
    completions: Mapping[str, Mapping[str, Any]],
    published_analysis: StudyAnalysisResult | None,
) -> dict[str, Any]:
    """Build the compact identity returned by the public layout verifier."""

    detector_runs = [
        {
            "detector_id": allocation.detector_id,
            "study_run_sha256": manifests[allocation.detector_id].study_run_sha256,
            "runtime_identity_sha256": manifests[allocation.detector_id].record["runtime"][
                "runtime_identity_sha256"
            ],
            "completion_sha256": completions[allocation.detector_id]["completion_sha256"],
        }
        for allocation in plan.allocations
    ]
    coverage_cells = [
        {
            "detector_id": str(value.cell["detector_id"]),
            "profile_id": str(value.cell["profile_id"]),
            "arm_id": str(value.cell["arm_id"]),
            "cell_sha256": str(value.cell["cell_sha256"]),
            "prediction_count": value.prediction_count,
            "shard_count": value.shard_count,
            "prediction_shard_index_sha256": value.prediction_shard_index_sha256,
            "merge_record_sha256": value.merge_record_sha256,
        }
        for value in cells
    ]
    coverage = {
        "status": "complete_exact_semantic_layout",
        "ordered_image_count": len(plan.image_ids),
        "executed_cell_count": len(cells),
        "prediction_record_count": sum(value.prediction_count for value in cells),
        "cells": coverage_cells,
    }
    published_sha256: str | None = None
    if published_analysis is not None:
        if not isinstance(published_analysis, StudyAnalysisResult):
            raise TypeError("published_analysis must be a StudyAnalysisResult or None")
        # Reconstructing the result under its validation plan catches callers
        # that somehow obtained an object without going through the public
        # plan-bound loader.
        published = StudyAnalysisResult(published_analysis.record, plan)
        expected_detector_runs = [
            {
                "detector_id": allocation.detector_id,
                "manifest": manifests[allocation.detector_id].to_dict(),
                "completion": json_value(completions[allocation.detector_id]),
            }
            for allocation in plan.allocations
        ]
        if json_value(published.record["detector_runs"]) != expected_detector_runs:
            raise ValueError(
                "published analysis detector runs differ from the completed study layout"
            )
        if json_value(published.record["coverage"]) != coverage:
            raise ValueError("published analysis coverage differs from the completed study layout")
        published_sha256 = published.study_analysis_sha256

    payload = {
        "schema_version": 2,
        "record_type": "phycam_completed_study_layout_verification",
        "study_plan_sha256": plan.study_plan_sha256,
        "dataset_sha256": plan.record["dataset"]["dataset_sha256"],
        "image_selection_sha256": plan.record["image_selection"]["selection_sha256"],
        "detector_runs": detector_runs,
        "coverage": coverage,
        "published_analysis_sha256": published_sha256,
    }
    return {**payload, "layout_verification_sha256": canonical_sha256(payload)}


def verify_completed_study_layout(
    *,
    plan: CocoStudyPlan,
    subset: NativeCOCODataset,
    output_root: str | Path,
    published_analysis: StudyAnalysisResult | None = None,
    progress: ProgressReporter | None = None,
) -> Mapping[str, Any]:
    """Verify a completed run without recomputing any statistical estimates.

    The exclusive study-layout lock prevents shared-lock writers from
    changing shards between validation and construction of the returned
    content-addressed verification record.  When a published analysis is
    supplied, its exact detector bindings and coverage must equal the live
    completed-study snapshot.
    """

    if not isinstance(plan, CocoStudyPlan):
        raise TypeError("plan must be a CocoStudyPlan")
    if published_analysis is not None and not isinstance(
        published_analysis,
        StudyAnalysisResult,
    ):
        raise TypeError("published_analysis must be a StudyAnalysisResult or None")
    root = Path(output_root)
    with advisory_target_lock(
        root,
        purpose="study-layout",
        exclusive=True,
    ):
        manifests, cells, completions = _validate_completed_layout(
            plan=plan,
            subset=subset,
            output_root=root,
            progress=progress,
        )
        record = _completed_layout_verification_record(
            plan=plan,
            manifests=manifests,
            cells=cells,
            completions=completions,
            published_analysis=published_analysis,
        )
    frozen = freeze_json_value(record)
    if not isinstance(frozen, MappingProxyType):  # pragma: no cover - narrowed above
        raise TypeError("completed-study layout verification must freeze to a mapping")
    return frozen


def _csv_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _float_text(value: Any) -> str:
    return format(finite_float(value, field_name="CSV numeric value"), ".17g")


def _point_csv(result: StudyAnalysisResult) -> bytes:
    plan_hash = result.record["study_plan_sha256"]
    rows: list[dict[str, Any]] = []
    for cell in result.record["point_metrics"]:
        common = {
            "study_plan_sha256": plan_hash,
            "detector_id": cell["detector_id"],
            "profile_id": cell["profile_id"],
            "arm_id": cell["arm_id"],
            "cell_sha256": cell["cell_sha256"],
        }
        for metric in _SCALAR_POINT_METRICS:
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "category_id": "",
                    "value": _float_text(cell["metrics"][metric]),
                }
            )
        for category in cell["metrics"]["per_class_ap"]:
            rows.append(
                {
                    **common,
                    "metric": "per_class_ap50",
                    "category_id": category["category_id"],
                    "value": _float_text(category["ap"]),
                }
            )
    return _csv_bytes(
        (
            "study_plan_sha256",
            "detector_id",
            "profile_id",
            "arm_id",
            "cell_sha256",
            "metric",
            "category_id",
            "value",
        ),
        rows,
    )


def _interval(value: Mapping[str, Any]) -> tuple[str, str]:
    return _float_text(value["lower"]), _float_text(value["upper"])


def _secondary_csv(result: StudyAnalysisResult) -> bytes:
    rows: list[dict[str, Any]] = []
    plan_hash = result.record["study_plan_sha256"]
    for group in result.record["secondary_analyses"]:
        for condition in group["bootstrap"]["conditions"]:
            arm_id = condition["condition"]
            for metric in ("map50", "map50_95"):
                point = condition["metrics"][metric]
                marginal = condition["marginal_percentile_95"][metric]
                difference = condition["paired_difference_to_baseline"][metric]
                ratio = condition["paired_ratio_to_baseline"][metric]
                for estimand, estimate, interval in (
                    ("marginal", point, marginal),
                    (
                        "difference_to_modeled_neutral",
                        difference["estimate"],
                        difference["percentile_95"],
                    ),
                    ("ratio_to_modeled_neutral", ratio["estimate"], ratio["percentile_95"]),
                ):
                    lower, upper = _interval(interval)
                    rows.append(
                        {
                            "study_plan_sha256": plan_hash,
                            "detector_id": group["detector_id"],
                            "profile_id": group["profile_id"],
                            "arm_id": arm_id,
                            "baseline_arm_id": group["paired_against_arm_id"],
                            "metric": metric,
                            "estimand": estimand,
                            "estimate": _float_text(estimate),
                            "lower_95": lower,
                            "upper_95": upper,
                            "confirmatory": "false",
                        }
                    )
    return _csv_bytes(
        (
            "study_plan_sha256",
            "detector_id",
            "profile_id",
            "arm_id",
            "baseline_arm_id",
            "metric",
            "estimand",
            "estimate",
            "lower_95",
            "upper_95",
            "confirmatory",
        ),
        rows,
    )


def _primary_csv(result: StudyAnalysisResult) -> bytes:
    primary = result.record["primary_estimand"]
    bootstrap = primary["bootstrap"]
    rows: list[dict[str, Any]] = []
    for metric, metric_result in bootstrap["metrics"].items():
        curves = (
            ("physical", metric_result["physical_curve"]),
            ("gaussian", metric_result["comparator_curve"]),
            ("physical_minus_gaussian", metric_result["paired_difference_curve"]),
        )
        for estimand, curve in curves:
            for point in curve:
                interval = point.get("percentile_95", point.get("marginal_percentile_95"))
                lower, upper = _interval(interval)
                rows.append(
                    {
                        "study_plan_sha256": result.record["study_plan_sha256"],
                        "detector_id": primary["detector_id"],
                        "profile_id": primary["profile_id"],
                        "metric": metric,
                        "estimand": estimand,
                        "edge_waves_ref": _float_text(point["coordinate"]),
                        "estimate": _float_text(point["estimate"]),
                        "lower_95": lower,
                        "upper_95": upper,
                        "unit": "AP",
                        "confirmatory": "false",
                    }
                )
        auc = metric_result["paired_physical_minus_comparator_auc"]
        lower, upper = _interval(auc["percentile_95"])
        rows.append(
            {
                "study_plan_sha256": result.record["study_plan_sha256"],
                "detector_id": primary["detector_id"],
                "profile_id": primary["profile_id"],
                "metric": metric,
                "estimand": "physical_minus_gaussian_curve_auc",
                "edge_waves_ref": "",
                "estimate": _float_text(auc["estimate"]),
                "lower_95": lower,
                "upper_95": upper,
                "unit": auc["unit"],
                "confirmatory": (
                    "true" if primary["confirmatory"] and metric == "map50_95" else "false"
                ),
            }
        )
    return _csv_bytes(
        (
            "study_plan_sha256",
            "detector_id",
            "profile_id",
            "metric",
            "estimand",
            "edge_waves_ref",
            "estimate",
            "lower_95",
            "upper_95",
            "unit",
            "confirmatory",
        ),
        rows,
    )


_PUBLICATION_ARTIFACT_SPECS = {
    "canonical_analysis_json": (
        "study_analysis",
        "json",
        "application/json",
    ),
    "tidy_point_metrics_csv": (
        "coco_point_metrics",
        "csv",
        "text/csv; charset=utf-8",
    ),
    "tidy_primary_estimand_csv": (
        "primary_estimand",
        "csv",
        "text/csv; charset=utf-8",
    ),
    "tidy_secondary_intervals_csv": (
        "secondary_intervals",
        "csv",
        "text/csv; charset=utf-8",
    ),
}
_PUBLICATION_STEMS = tuple(
    (stem, extension) for stem, extension, _ in _PUBLICATION_ARTIFACT_SPECS.values()
)


def _publication_flat_files(directory: Path) -> set[str]:
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("analysis publication is not a regular directory")
    names: set[str] = set()
    for entry in directory.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise RuntimeError(f"analysis publication contains a nonregular entry: {entry}")
        names.add(entry.name)
    return names


def _is_owned_uncommitted_publication_file(name: str) -> bool:
    for stem, extension in _PUBLICATION_STEMS:
        committed = rf"{re.escape(stem)}\.[0-9a-f]{{64}}\.{re.escape(extension)}"
        if re.fullmatch(committed, name) or re.fullmatch(
            rf"\.{committed}\..+\.tmp",
            name,
        ):
            return True
    return bool(re.fullmatch(r"\.publication\.index\.json\..+\.tmp", name))


def _clean_uncommitted_publication(directory: Path) -> None:
    """Remove only owned crash residue when no committed index exists."""

    if not directory.exists():
        directory.mkdir(parents=True)
        return
    names = _publication_flat_files(directory)
    unexpected = sorted(name for name in names if not _is_owned_uncommitted_publication_file(name))
    if unexpected:
        raise ValueError(f"uncommitted analysis output contains non-owned artifacts: {unexpected}")
    for name in sorted(names):
        (directory / name).unlink()
    _fsync_directory(directory)


def _load_verified_study_analysis_publication(
    output_directory: str | Path,
    *,
    plan: CocoStudyPlan,
) -> tuple[StudyAnalysisResult, dict[str, Any]]:
    if not isinstance(plan, CocoStudyPlan):
        raise TypeError("publication verification requires a CocoStudyPlan")
    directory = Path(output_directory)
    if directory.is_symlink() or not directory.is_dir():
        raise FileNotFoundError("study analysis publication directory is missing or unsafe")
    index_path = directory / "publication.index.json"
    index = _load_canonical_record(index_path, label="analysis publication index")
    if set(index) != {
        "schema_version",
        "record_type",
        "study_plan_sha256",
        "study_analysis_sha256",
        "artifacts",
        "publication_index_sha256",
    }:
        raise ValueError("analysis publication index has missing or unknown fields")
    if index["schema_version"] != 2 or index["record_type"] != _PUBLICATION_TYPE:
        raise ValueError("analysis publication index has an unsupported schema or type")
    if index["study_plan_sha256"] != plan.study_plan_sha256:
        raise ValueError("analysis publication index is bound to a different plan")
    _require_sha256(index["study_analysis_sha256"], label="study_analysis_sha256")
    supplied_index_hash = _require_sha256(
        index["publication_index_sha256"],
        label="publication_index_sha256",
    )
    index_payload = {key: item for key, item in index.items() if key != "publication_index_sha256"}
    if supplied_index_hash != canonical_sha256(index_payload):
        raise ValueError("publication_index_sha256 does not match the index payload")

    artifacts = index["artifacts"]
    expected_roles = list(_PUBLICATION_ARTIFACT_SPECS)
    if not isinstance(artifacts, list) or [item.get("role") for item in artifacts] != (
        expected_roles
    ):
        raise ValueError("analysis publication artifact roles are incomplete or out of order")
    filenames: set[str] = set()
    artifact_by_role: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "role",
            "filename",
            "media_type",
            "bytes",
            "sha256",
        }:
            raise ValueError("analysis publication artifact record is noncanonical")
        role = str(artifact["role"])
        stem, extension, media_type = _PUBLICATION_ARTIFACT_SPECS[role]
        digest = _require_sha256(artifact["sha256"], label=f"{role} sha256")
        expected_filename = f"{stem}.{digest}.{extension}"
        if artifact["filename"] != expected_filename or Path(expected_filename).name != (
            expected_filename
        ):
            raise ValueError(f"analysis publication {role} filename differs from its digest")
        if artifact["media_type"] != media_type:
            raise ValueError(f"analysis publication {role} media type drifted")
        if (
            not isinstance(artifact["bytes"], int)
            or isinstance(artifact["bytes"], bool)
            or artifact["bytes"] < 0
        ):
            raise ValueError(f"analysis publication {role} byte count is invalid")
        if expected_filename in filenames:
            raise ValueError("analysis publication contains duplicate filenames")
        filenames.add(expected_filename)
        artifact_by_role[role] = artifact

    expected_files = filenames | {index_path.name}
    observed_files = _publication_flat_files(directory)
    if observed_files != expected_files:
        missing = sorted(expected_files.difference(observed_files))
        extra = sorted(observed_files.difference(expected_files))
        raise ValueError(
            f"analysis publication directory differs from its index; "
            f"missing={missing}, extra={extra}"
        )
    payload_by_role: dict[str, bytes] = {}
    for role, artifact in artifact_by_role.items():
        path = directory / str(artifact["filename"])
        _require_regular_file(path, label=f"analysis publication {role}")
        payload = path.read_bytes()
        if len(payload) != artifact["bytes"] or _sha256_bytes(payload) != artifact["sha256"]:
            raise ValueError(f"analysis publication {role} bytes differ from the index")
        payload_by_role[role] = payload

    analysis_record = _parse_canonical_record_bytes(
        payload_by_role["canonical_analysis_json"],
        label="canonical study analysis",
    )
    result = StudyAnalysisResult(analysis_record, plan)
    if result.study_analysis_sha256 != index["study_analysis_sha256"]:
        raise ValueError("analysis publication index and canonical JSON identities differ")
    regenerated = {
        "canonical_analysis_json": _record_bytes(result.to_dict()),
        "tidy_point_metrics_csv": _point_csv(result),
        "tidy_primary_estimand_csv": _primary_csv(result),
        "tidy_secondary_intervals_csv": _secondary_csv(result),
    }
    for role, expected_payload in regenerated.items():
        if payload_by_role[role] != expected_payload:
            raise ValueError(f"analysis publication {role} is not regenerated from canonical JSON")
    return result, index


def load_verified_study_analysis_publication(
    output_directory: str | Path,
    *,
    plan: CocoStudyPlan,
) -> tuple[StudyAnalysisResult, Mapping[str, Any]]:
    """Return the plan-bound result and immutable index from one locked snapshot."""

    with advisory_target_lock(
        output_directory,
        purpose="publication",
        exclusive=False,
    ):
        result, index = _load_verified_study_analysis_publication(output_directory, plan=plan)
    frozen = freeze_json_value(index)
    if not isinstance(frozen, MappingProxyType):  # pragma: no cover - narrowed above
        raise TypeError("analysis publication index must freeze to a mapping")
    return result, frozen


def verify_study_analysis_publication(
    output_directory: str | Path,
    *,
    plan: CocoStudyPlan,
) -> Mapping[str, Any]:
    """Fully verify one publication directory and return its immutable index."""

    _, index = load_verified_study_analysis_publication(output_directory, plan=plan)
    return index


def load_study_analysis_publication(
    output_directory: str | Path,
    *,
    plan: CocoStudyPlan,
) -> StudyAnalysisResult:
    """Load a fully verified publication and return its plan-bound analysis."""

    result, _ = load_verified_study_analysis_publication(output_directory, plan=plan)
    return result


def _publish_study_analysis_under_lock(
    output_directory: str | Path,
    result: StudyAnalysisResult,
) -> Mapping[str, Any]:
    """Publish with index-last commit while the caller holds the directory lock."""

    if not isinstance(result, StudyAnalysisResult):
        raise TypeError("result must be a StudyAnalysisResult")
    result = StudyAnalysisResult(result.record, result._validation_plan)
    directory = Path(output_directory)
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise RuntimeError("analysis output must be a regular directory")
    result_payload = _record_bytes(result.to_dict())
    payloads = {
        "canonical_analysis_json": ("json", result_payload),
        "tidy_point_metrics_csv": ("csv", _point_csv(result)),
        "tidy_primary_estimand_csv": ("csv", _primary_csv(result)),
        "tidy_secondary_intervals_csv": ("csv", _secondary_csv(result)),
    }
    artifacts: list[dict[str, Any]] = []
    artifact_payloads: dict[str, bytes] = {}
    for role, (extension, payload) in payloads.items():
        digest = _sha256_bytes(payload)
        stem, expected_extension, media_type = _PUBLICATION_ARTIFACT_SPECS[role]
        if extension != expected_extension:  # pragma: no cover - fixed table above
            raise RuntimeError("analysis publication extension table is inconsistent")
        filename = f"{stem}.{digest}.{extension}"
        artifacts.append(
            {
                "role": role,
                "filename": filename,
                "media_type": media_type,
                "bytes": len(payload),
                "sha256": digest,
            }
        )
        artifact_payloads[filename] = payload
    index_payload = {
        "schema_version": 2,
        "record_type": _PUBLICATION_TYPE,
        "study_plan_sha256": result.record["study_plan_sha256"],
        "study_analysis_sha256": result.study_analysis_sha256,
        "artifacts": artifacts,
    }
    index = {
        **index_payload,
        "publication_index_sha256": canonical_sha256(index_payload),
    }
    index_bytes = _record_bytes(index)
    index_path = directory / "publication.index.json"
    expected_names = set(artifact_payloads) | {index_path.name}
    if index_path.exists() or index_path.is_symlink():
        observed_index = _load_canonical_record(index_path, label="analysis publication index")
        if observed_index != index:
            raise ValueError("existing analysis publication index drifted")
        _, verified_index = _load_verified_study_analysis_publication(
            directory,
            plan=result._validation_plan,
        )
        frozen_existing = freeze_json_value(verified_index)
        if not isinstance(frozen_existing, MappingProxyType):  # pragma: no cover
            raise TypeError("analysis publication index must freeze to a mapping")
        return frozen_existing

    # Without an index there is no committed publication. Remove only files
    # owned by this publisher (including interrupted temporary files), then
    # restart the deterministic transaction from an empty directory.
    _clean_uncommitted_publication(directory)
    for filename, payload in artifact_payloads.items():
        _publish_or_validate_bytes(directory / filename, payload)
    # The index is the commit record and is always published last.
    _publish_or_validate_bytes(index_path, index_bytes)
    observed_names = _publication_flat_files(directory)
    if observed_names != expected_names:
        raise RuntimeError("analysis publication is not complete after atomic publication")
    _, verified_index = _load_verified_study_analysis_publication(
        directory,
        plan=result._validation_plan,
    )
    if verified_index != index:
        raise RuntimeError("analysis publication verification returned a different index")
    frozen = freeze_json_value(verified_index)
    if not isinstance(frozen, MappingProxyType):  # pragma: no cover - narrowed above
        raise TypeError("analysis publication index must freeze to a mapping")
    return frozen


def publish_study_analysis(
    output_directory: str | Path,
    result: StudyAnalysisResult,
) -> Mapping[str, Any]:
    """Transactionally publish or exactly resume content-addressed results."""

    with advisory_target_lock(
        output_directory,
        purpose="publication",
        exclusive=True,
    ):
        return _publish_study_analysis_under_lock(output_directory, result)


def analyze_and_publish_completed_study(
    *,
    plan_path: str | Path,
    coco_root: str | Path,
    output_root: str | Path,
    analysis_output: str | Path,
    paired_bootstrap: PairedBootstrap = compute_paired_map_bootstrap,
    curve_bootstrap: CurveBootstrap = compute_paired_map_curve_auc_bootstrap,
    repository_root: str | Path | None = None,
    progress: ProgressReporter | None = None,
    scratch_work_dir: str | Path | None = None,
    analysis_workers: int = 1,
) -> Mapping[str, Any]:
    """Load, fully validate, analyze, and publish one frozen completed study."""

    analysis_workers = _analysis_worker_count(analysis_workers)
    if analysis_workers == 2 and scratch_work_dir is None:
        raise ValueError("two-worker analysis requires an explicit scratch_work_dir")
    scratch = (
        None
        if scratch_work_dir is None
        else _require_separate_scratch_path(
            scratch_work_dir,
            forbidden_roots=(output_root, analysis_output)
            + (() if repository_root is None else (repository_root,)),
        )
    )

    plan = load_study_plan(plan_path)
    subset = load_native_coco_subset(
        coco_root,
        split="val2017",
        ordered_image_ids=plan.image_ids,
        max_images=None,
        eager=False,
    )
    # Hold the completed-run tree exclusively through both analysis and the
    # index-last publication commit. Detector runners use the same
    # lock in shared mode, so the promoted result refers to one immutable
    # source snapshot rather than a check-then-use interval.
    with advisory_target_lock(
        output_root,
        purpose="study-layout",
        exclusive=True,
    ):
        result = _analyze_completed_study_under_lock(
            plan=plan,
            subset=subset,
            output_root=output_root,
            paired_bootstrap=paired_bootstrap,
            curve_bootstrap=curve_bootstrap,
            repository_root=repository_root,
            progress=progress,
            scratch_work_dir=scratch,
            analysis_workers=analysis_workers,
        )
        return publish_study_analysis(analysis_output, result)


__all__ = [
    "NONOFFICIAL_STUDY_ANALYSIS_IMPLEMENTATION_ID",
    "STUDY_ANALYSIS_IMPLEMENTATION_ID",
    "StudyAnalysisResult",
    "analyze_and_publish_completed_study",
    "analyze_completed_study",
    "derive_study_evidence_tier",
    "load_study_analysis",
    "load_study_analysis_publication",
    "load_verified_study_analysis_publication",
    "publish_study_analysis",
    "verify_completed_study_layout",
    "verify_rederived_completed_study_analysis",
    "verify_study_analysis_publication",
]
