"""COCO detection metrics with explicit stochastic-capture uncertainty.

The deterministic entry point evaluates at most one prediction record per
image. The hierarchical entry point requires a complete
``image_id x realization_id`` grid and reports image, realization, and joint
cluster-bootstrap intervals separately.
"""

from __future__ import annotations

import contextlib
import io
import math
import numbers
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

COCO80_TO_91 = np.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ],
    dtype=np.int64,
)

_SCALAR_METRICS = (
    "map50",
    "map50_95",
    "mean_ap",
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
)
_POINT_METRIC_KEYS = frozenset((*_SCALAR_METRICS, "mean_ap_iou_thresholds", "per_class_ap"))

_BOOTSTRAP_METRICS = ("map50", "map50_95")
_ARM_BOOTSTRAP_METHOD = "image_cluster_cached_coco_percentile_bootstrap_arm_v1"
_ARM_SAMPLING_CONTRACT = "numpy_seedsequence_default_rng_choice_native_image_positions_v1"
_BOOTSTRAP_ACCUMULATION = "repeat_cached_per_image_category_matches_then_coco_pr_v1"
_ARM_MEMORY_STRATEGY = "single_condition_match_cache_v1"
_PAIRED_MEMORY_STRATEGY = "one_condition_match_cache_at_a_time_v1"


def _as_numpy(value: object, *, name: str, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    for method in ("detach", "cpu"):
        operation = getattr(value, method, None)
        if callable(operation):
            value = operation()
    operation = getattr(value, "numpy", None)
    if callable(operation):
        value = operation()
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc


def _integer(value: object, *, name: str, minimum: int | None = None) -> int:
    operation = getattr(value, "item", None)
    if callable(operation):
        value = operation()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, numbers.Integral):
        result = int(value)
    else:
        number = float(value)
        if not math.isfinite(number) or number != int(number):
            raise ValueError(f"{name} must be a finite integer")
        result = int(number)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def yolo_to_coco_category_ids(class_ids: object) -> np.ndarray:
    """Map contiguous COCO80 detector indices to sparse COCO category IDs."""

    raw = _as_numpy(class_ids, name="class_ids")
    if raw.size == 0:
        return np.empty(raw.shape, dtype=np.int64)
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("class_ids must contain integers") from exc
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError("class_ids must contain integers")
    integer = numeric.astype(np.int64)
    if np.any(integer < 0) or np.any(integer >= len(COCO80_TO_91)):
        raise ValueError("COCO80 class IDs must lie in [0, 79]")
    return COCO80_TO_91[integer]


def _record(record: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{kind} records must be mappings")
    required = {"image_id", "boxes", "labels"}
    if kind == "prediction":
        required.add("scores")
    missing = required.difference(record)
    if missing:
        raise ValueError(f"{kind} record is missing keys: {sorted(missing)}")

    image_id = _integer(record["image_id"], name=f"{kind} image_id")
    boxes = _as_numpy(record["boxes"], name=f"{kind} boxes", dtype=np.dtype(np.float64))
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{kind} boxes must have shape (N, 4)")
    if not np.all(np.isfinite(boxes)):
        raise ValueError(f"{kind} boxes must be finite")
    if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
        raise ValueError(f"{kind} boxes must have positive width and height")

    raw_labels = _as_numpy(record["labels"], name=f"{kind} labels")
    if raw_labels.ndim != 1 or len(raw_labels) != len(boxes):
        raise ValueError(f"{kind} labels must have one entry per box")
    try:
        numeric_labels = raw_labels.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{kind} labels must contain positive integers") from exc
    if (
        not np.all(np.isfinite(numeric_labels))
        or np.any(numeric_labels != np.floor(numeric_labels))
        or np.any(numeric_labels <= 0)
    ):
        raise ValueError(f"{kind} labels must contain positive integers")
    labels = numeric_labels.astype(np.int64)
    normalized: dict[str, Any] = {"image_id": image_id, "boxes": boxes, "labels": labels}

    if kind == "prediction":
        scores = _as_numpy(record["scores"], name="prediction scores", dtype=np.dtype(np.float64))
        if scores.ndim != 1 or len(scores) != len(boxes):
            raise ValueError("prediction scores must have one entry per box")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("prediction scores must be finite and lie in [0, 1]")
        normalized["scores"] = scores
    else:
        if "area" in record:
            area = _as_numpy(record["area"], name="target area", dtype=np.dtype(np.float64))
            if area.ndim != 1 or len(area) != len(boxes):
                raise ValueError("target area must have one entry per box")
            if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
                raise ValueError("target area must be finite and positive")
            normalized["area"] = area
        if "iscrowd" in record:
            raw_crowd = _as_numpy(record["iscrowd"], name="target iscrowd")
            if raw_crowd.ndim != 1 or len(raw_crowd) != len(boxes):
                raise ValueError("target iscrowd must have one entry per box")
            try:
                crowd_numeric = raw_crowd.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("target iscrowd must contain only 0 or 1") from exc
            if not np.all(np.isfinite(crowd_numeric)) or np.any(
                ~np.isin(crowd_numeric, (0.0, 1.0))
            ):
                raise ValueError("target iscrowd must contain only 0 or 1")
            normalized["iscrowd"] = crowd_numeric.astype(np.int64)
    return normalized


def _thresholds(values: Sequence[float] | None) -> np.ndarray:
    if values is None:
        return np.linspace(0.50, 0.95, 10, dtype=np.float64)
    result = _as_numpy(values, name="iou_thresholds", dtype=np.dtype(np.float64))
    if result.ndim != 1 or not len(result):
        raise ValueError("iou_thresholds must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0) or np.any(result > 1.0):
        raise ValueError("iou_thresholds must be finite and lie in (0, 1]")
    if np.any(np.diff(result) <= 0.0):
        raise ValueError("iou_thresholds must be strictly increasing and unique")
    return result


def _is_standard_coco_thresholds(values: np.ndarray) -> bool:
    standard = np.linspace(0.50, 0.95, 10, dtype=np.float64)
    return values.shape == standard.shape and bool(
        np.allclose(values, standard, rtol=0.0, atol=1e-12)
    )


def _categories(
    values: Sequence[int] | None,
    *,
    observed: Iterable[int],
) -> list[int]:
    if values is None:
        result = sorted(set(int(value) for value in observed))
    else:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError("category_ids must be an ordered sequence")
        result = [_integer(value, name="category_id", minimum=1) for value in values]
        if not result:
            raise ValueError("category_ids must not be empty")
        if len(result) != len(set(result)):
            raise ValueError("category_ids must be unique")
        result.sort()
    if not result:
        raise ValueError("at least one category ID is required")
    return result


def _empty_result(
    annotations: list[dict[str, Any]],
    categories: list[int],
    thresholds: np.ndarray,
) -> dict[str, Any]:
    non_crowd = [annotation for annotation in annotations if not annotation["iscrowd"]]

    def available(*, minimum: float = 0.0, maximum: float = 1e10) -> float:
        return 0.0 if any(minimum <= item["area"] <= maximum for item in non_crowd) else -1.0

    def class_value(category_id: int) -> float:
        return 0.0 if any(item["category_id"] == category_id for item in non_crowd) else -1.0

    has_50 = bool(np.any(np.isclose(thresholds, 0.50, rtol=0.0, atol=1e-12)))
    has_75 = bool(np.any(np.isclose(thresholds, 0.75, rtol=0.0, atol=1e-12)))
    standard = _is_standard_coco_thresholds(thresholds)
    overall = available()
    small = available(maximum=32**2)
    medium = available(minimum=32**2, maximum=96**2)
    large = available(minimum=96**2)
    return {
        "map50": overall if has_50 else -1.0,
        "map50_95": overall if standard else -1.0,
        "mean_ap": overall,
        "mean_ap_iou_thresholds": thresholds.tolist(),
        "map75": overall if has_75 else -1.0,
        "map50_95_small": small if standard else -1.0,
        "map50_95_medium": medium if standard else -1.0,
        "map50_95_large": large if standard else -1.0,
        "mean_ap_small": small,
        "mean_ap_medium": medium,
        "mean_ap_large": large,
        "ar100": overall if standard else -1.0,
        "mean_ar100": overall,
        "ar100_small": small if standard else -1.0,
        "ar100_medium": medium if standard else -1.0,
        "ar100_large": large if standard else -1.0,
        "per_class_ap": {
            category_id: class_value(category_id) if has_50 else -1.0 for category_id in categories
        },
    }


def compute_map(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    iou_thresholds: Sequence[float] | None = None,
    *,
    category_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compute COCO AP from native-coordinate ``xyxy`` records.

    ``category_ids`` fixes the evaluated category universe. This matters for
    image bootstrap samples, where a valid class can be absent from one
    resample. When omitted, the union of target and prediction labels is used.
    """

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("compute_map requires the optional pycocotools dependency") from exc
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("predictions must be a sequence of mappings")
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence of mappings")
    if not targets:
        raise ValueError("targets must contain at least one image record")
    normalized_targets = [_record(item, kind="target") for item in targets]
    normalized_predictions = [_record(item, kind="prediction") for item in predictions]
    thresholds = _thresholds(iou_thresholds)
    target_ids = [item["image_id"] for item in normalized_targets]
    prediction_ids = [item["image_id"] for item in normalized_predictions]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("targets must contain unique image_id values")
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("predictions must contain unique image_id values")
    unknown_images = sorted(set(prediction_ids).difference(target_ids))
    if unknown_images:
        raise ValueError(f"predictions reference unknown image IDs: {unknown_images}")

    images = [{"id": image_id} for image_id in target_ids]
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for target in normalized_targets:
        for index, (box, label) in enumerate(zip(target["boxes"], target["labels"])):
            x_min, y_min, x_max, y_max = (float(value) for value in box)
            area = (
                float(target["area"][index])
                if "area" in target
                else (x_max - x_min) * (y_max - y_min)
            )
            crowd = int(target["iscrowd"][index]) if "iscrowd" in target else 0
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": target["image_id"],
                    "category_id": int(label),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": area,
                    "iscrowd": crowd,
                }
            )
            annotation_id += 1
    if not annotations:
        raise ValueError("COCO AP is undefined because targets contain no annotations")
    observed_categories = [item["category_id"] for item in annotations]
    observed_categories.extend(
        int(label) for prediction in normalized_predictions for label in prediction["labels"]
    )
    categories = _categories(category_ids, observed=observed_categories)
    category_set = set(categories)
    unknown_target_labels = sorted(
        {item["category_id"] for item in annotations}.difference(category_set)
    )
    if unknown_target_labels:
        raise ValueError(f"targets reference undeclared categories: {unknown_target_labels}")

    detections: list[dict[str, Any]] = []
    for prediction in normalized_predictions:
        unknown_labels = sorted(set(int(value) for value in prediction["labels"]) - category_set)
        if unknown_labels:
            raise ValueError(f"predictions reference undeclared categories: {unknown_labels}")
        for box, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            x_min, y_min, x_max, y_max = (float(value) for value in box)
            detections.append(
                {
                    "image_id": prediction["image_id"],
                    "category_id": int(label),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "score": float(score),
                }
            )
    if not detections:
        return _empty_result(annotations, categories, thresholds)

    ground_truth = COCO()
    ground_truth.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": value, "name": str(value)} for value in categories],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth.createIndex()
        detected = ground_truth.loadRes(detections)
        evaluator = COCOeval(ground_truth, detected, iouType="bbox")
        evaluator.params.iouThrs = thresholds
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]

    def mean_valid(values: np.ndarray) -> float:
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else -1.0

    def at_threshold(value: float, *, area_index: int = 0) -> float:
        matches = np.flatnonzero(np.isclose(thresholds, value, rtol=0.0, atol=1e-12))
        return -1.0 if not len(matches) else mean_valid(precision[matches[0], :, :, area_index, 2])

    match_50 = np.flatnonzero(np.isclose(thresholds, 0.50, rtol=0.0, atol=1e-12))
    standard = _is_standard_coco_thresholds(thresholds)
    mean_ap = mean_valid(precision[:, :, :, 0, 2])
    mean_ap_small = mean_valid(precision[:, :, :, 1, 2])
    mean_ap_medium = mean_valid(precision[:, :, :, 2, 2])
    mean_ap_large = mean_valid(precision[:, :, :, 3, 2])
    mean_ar100 = mean_valid(recall[:, :, 0, 2])
    ar100_small = mean_valid(recall[:, :, 1, 2])
    ar100_medium = mean_valid(recall[:, :, 2, 2])
    ar100_large = mean_valid(recall[:, :, 3, 2])
    per_class: dict[int, float] = {}
    for category_index, category_id in enumerate(categories):
        per_class[category_id] = (
            -1.0
            if not len(match_50)
            else mean_valid(precision[match_50[0], :, category_index, 0, 2])
        )
    return {
        "map50": at_threshold(0.50),
        "map50_95": mean_ap if standard else -1.0,
        "mean_ap": mean_ap,
        "mean_ap_iou_thresholds": thresholds.tolist(),
        "map75": at_threshold(0.75),
        "map50_95_small": mean_ap_small if standard else -1.0,
        "map50_95_medium": mean_ap_medium if standard else -1.0,
        "map50_95_large": mean_ap_large if standard else -1.0,
        "mean_ap_small": mean_ap_small,
        "mean_ap_medium": mean_ap_medium,
        "mean_ap_large": mean_ap_large,
        "ar100": mean_ar100 if standard else -1.0,
        "mean_ar100": mean_ar100,
        "ar100_small": ar100_small if standard else -1.0,
        "ar100_medium": ar100_medium if standard else -1.0,
        "ar100_large": ar100_large if standard else -1.0,
        "per_class_ap": per_class,
    }


def _mean_metrics(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not values:
        raise ValueError("at least one metric record is required")

    def mean_valid(items: Iterable[float]) -> float:
        valid = [float(item) for item in items if float(item) >= 0.0]
        return float(np.mean(valid)) if valid else -1.0

    categories = sorted({int(category) for value in values for category in value["per_class_ap"]})
    return {
        **{metric: mean_valid(value[metric] for value in values) for metric in _SCALAR_METRICS},
        "per_class_ap": {
            category: mean_valid(value["per_class_ap"].get(category, -1.0) for value in values)
            for category in categories
        },
    }


def _half_width(samples: Sequence[float]) -> float:
    if len(samples) < 2:
        raise ValueError("an uncertainty interval requires at least two samples")
    return 1.96 * float(np.std(np.asarray(samples, dtype=np.float64), ddof=1))


def _percentile_interval(samples: Sequence[float]) -> dict[str, float]:
    if len(samples) < 2:
        raise ValueError("an uncertainty interval requires at least two samples")
    values = np.asarray(samples, dtype=np.float64)
    lower, upper = np.quantile(values, (0.025, 0.975))
    return {"lower": float(lower), "upper": float(upper)}


def _trapezoidal_integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Integrate one aligned curve without depending on NumPy 2-only APIs."""

    return float(
        np.sum(
            0.5 * (values[:-1] + values[1:]) * np.diff(coordinates),
            dtype=np.float64,
        )
    )


def _report_bootstrap_progress(
    callback: Callable[[str, int, int], None] | None,
    *,
    condition: str,
    completed: int,
    total: int,
) -> None:
    if callback is not None:
        callback(condition, completed, total)


@dataclass(frozen=True, slots=True)
class _COCOImageMatchCache:
    """Per-image COCO matching records reusable across image bootstraps."""

    image_ids: tuple[int, ...]
    category_ids: tuple[int, ...]
    thresholds: np.ndarray
    recall_thresholds: np.ndarray
    eval_images: tuple[Mapping[str, Any] | None, ...]
    area_count: int
    max_detections: int

    def accumulate(self, sampled_positions: Sequence[int]) -> dict[str, float]:
        """Repeat cached image matches and perform the standard COCO accumulation."""

        positions = tuple(int(value) for value in sampled_positions)
        image_count = len(self.image_ids)
        if len(positions) != image_count or any(
            value < 0 or value >= image_count for value in positions
        ):
            raise ValueError("sampled_positions must select one valid image per bootstrap slot")
        threshold_count = len(self.thresholds)
        recall_count = len(self.recall_thresholds)
        category_count = len(self.category_ids)
        precision = -np.ones(
            (threshold_count, recall_count, category_count),
            dtype=np.float64,
        )
        for category_index in range(category_count):
            offset = category_index * self.area_count * image_count
            selected = [self.eval_images[offset + position] for position in positions]
            evaluations = [value for value in selected if value is not None]
            if not evaluations:
                continue
            detection_scores = np.concatenate(
                [value["dtScores"][: self.max_detections] for value in evaluations]
            )
            order = np.argsort(-detection_scores, kind="mergesort")
            matches = np.concatenate(
                [value["dtMatches"][:, : self.max_detections] for value in evaluations],
                axis=1,
            )[:, order]
            detection_ignore = np.concatenate(
                [value["dtIgnore"][:, : self.max_detections] for value in evaluations],
                axis=1,
            )[:, order]
            ground_truth_ignore = np.concatenate([value["gtIgnore"] for value in evaluations])
            positive_ground_truth = int(np.count_nonzero(ground_truth_ignore == 0))
            if positive_ground_truth == 0:
                continue
            precision[:, :, category_index] = 0.0
            if detection_scores.size == 0:
                continue
            true_positive = np.logical_and(matches, np.logical_not(detection_ignore))
            false_positive = np.logical_and(
                np.logical_not(matches),
                np.logical_not(detection_ignore),
            )
            tp_sum = np.cumsum(true_positive, axis=1, dtype=np.float64)
            fp_sum = np.cumsum(false_positive, axis=1, dtype=np.float64)
            for threshold_index, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                recall = tp / positive_ground_truth
                curve = tp / (fp + tp + np.spacing(1))
                for index in range(len(curve) - 1, 0, -1):
                    if curve[index] > curve[index - 1]:
                        curve[index - 1] = curve[index]
                recall_indices = np.searchsorted(recall, self.recall_thresholds, side="left")
                valid = recall_indices < len(curve)
                precision[threshold_index, valid, category_index] = curve[recall_indices[valid]]

        def mean_valid(values: np.ndarray) -> float:
            valid = values[values > -1]
            return float(np.mean(valid)) if valid.size else -1.0

        match_50 = np.flatnonzero(np.isclose(self.thresholds, 0.50, rtol=0.0, atol=1e-12))
        return {
            "map50": -1.0 if not len(match_50) else mean_valid(precision[match_50[0]]),
            "map50_95": mean_valid(precision),
        }


def _build_coco_image_match_cache(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    thresholds: np.ndarray,
    category_ids: Sequence[int],
) -> _COCOImageMatchCache:
    """Run COCO's expensive IoU matching exactly once for one condition."""

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("paired COCO bootstrap requires pycocotools") from exc
    normalized_targets = [_record(value, kind="target") for value in targets]
    normalized_predictions = [_record(value, kind="prediction") for value in predictions]
    image_ids = tuple(value["image_id"] for value in normalized_targets)
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for target in normalized_targets:
        for index, (box, label) in enumerate(zip(target["boxes"], target["labels"])):
            x_min, y_min, x_max, y_max = (float(value) for value in box)
            area = (
                float(target["area"][index])
                if "area" in target
                else (x_max - x_min) * (y_max - y_min)
            )
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": target["image_id"],
                    "category_id": int(label),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": area,
                    "iscrowd": int(target["iscrowd"][index]) if "iscrowd" in target else 0,
                }
            )
            annotation_id += 1
    detections: list[dict[str, Any]] = []
    for prediction in normalized_predictions:
        for box, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            x_min, y_min, x_max, y_max = (float(value) for value in box)
            detections.append(
                {
                    "image_id": prediction["image_id"],
                    "category_id": int(label),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "score": float(score),
                }
            )
    if not detections:
        # pycocotools cannot load an empty result list. A remote, zero-score
        # false positive is metrically equivalent to no detections while still
        # allowing the standard evaluator to construct per-image match caches.
        detections.append(
            {
                "image_id": image_ids[0],
                "category_id": int(category_ids[0]),
                "bbox": [-1e12, -1e12, 1.0, 1.0],
                "score": 0.0,
            }
        )
    ground_truth = COCO()
    ground_truth.dataset = {
        "images": [{"id": value} for value in image_ids],
        "annotations": annotations,
        "categories": [{"id": int(value), "name": str(value)} for value in category_ids],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth.createIndex()
        detected = ground_truth.loadRes(detections)
        evaluator = COCOeval(ground_truth, detected, iouType="bbox")
        evaluator.params.imgIds = list(image_ids)
        evaluator.params.catIds = [int(value) for value in category_ids]
        evaluator.params.iouThrs = np.array(thresholds, dtype=np.float64, copy=True)
        evaluator.evaluate()
    parameters = evaluator._paramsEval
    if parameters is None:
        raise RuntimeError("pycocotools did not retain evaluated parameters")
    if tuple(parameters.imgIds) != image_ids:
        raise RuntimeError("pycocotools changed the ordered bootstrap image axis")
    if tuple(parameters.catIds) != tuple(category_ids):
        raise RuntimeError("pycocotools changed the ordered category axis")
    return _COCOImageMatchCache(
        image_ids=image_ids,
        category_ids=tuple(int(value) for value in category_ids),
        thresholds=np.array(parameters.iouThrs, dtype=np.float64, copy=True),
        recall_thresholds=np.array(parameters.recThrs, dtype=np.float64, copy=True),
        eval_images=tuple(evaluator.evalImgs),
        area_count=len(parameters.areaRng),
        max_detections=int(max(parameters.maxDets)),
    )


def _finite_float(value: object, *, name: str) -> float:
    operation = getattr(value, "item", None)
    if callable(operation):
        value = operation()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _decimal_category_id(value: object, *, name: str) -> int:
    if isinstance(value, str):
        try:
            result = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a canonical positive decimal integer") from exc
        if str(result) != value or result < 1:
            raise ValueError(f"{name} must be a canonical positive decimal integer")
        return result
    return _integer(value, name=name, minimum=1)


def _arm_point_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-round-trip-safe copy of one canonical ``compute_map`` result."""

    if not isinstance(metrics, Mapping):
        raise TypeError("point_metrics must be a mapping")
    if set(metrics) != _POINT_METRIC_KEYS:
        raise ValueError("point_metrics must contain the complete canonical COCO metric record")
    result = dict(metrics)
    for metric in _SCALAR_METRICS:
        value = _finite_float(result[metric], name=f"point_metrics {metric}")
        if metric in _BOOTSTRAP_METRICS and not 0.0 <= value <= 1.0:
            raise ValueError(f"point_metrics {metric} must lie in [0, 1]")
        if metric not in _BOOTSTRAP_METRICS and value != -1.0 and not 0.0 <= value <= 1.0:
            raise ValueError(f"point_metrics {metric} must be -1 or lie in [0, 1]")
        result[metric] = value
    raw_thresholds = result["mean_ap_iou_thresholds"]
    if isinstance(raw_thresholds, (str, bytes)) or not isinstance(raw_thresholds, Sequence):
        raise ValueError("point_metrics mean_ap_iou_thresholds must be a sequence")
    result["mean_ap_iou_thresholds"] = [
        _finite_float(value, name="point_metrics IoU threshold") for value in raw_thresholds
    ]
    raw_per_class = result["per_class_ap"]
    if not isinstance(raw_per_class, Mapping):
        raise ValueError("point_metrics per_class_ap must be a mapping")
    per_class: dict[str, float] = {}
    for raw_category, value in raw_per_class.items():
        category = _decimal_category_id(
            raw_category,
            name="point_metrics per_class_ap category",
        )
        key = str(category)
        if key in per_class:
            raise ValueError("point_metrics per_class_ap categories must be unique")
        ap = _finite_float(value, name=f"point_metrics per_class_ap[{key}]")
        if ap != -1.0 and not 0.0 <= ap <= 1.0:
            raise ValueError("point_metrics per_class_ap values must be -1 or lie in [0, 1]")
        per_class[key] = ap
    result["per_class_ap"] = per_class
    for published, generic in (
        ("map50_95", "mean_ap"),
        ("map50_95_small", "mean_ap_small"),
        ("map50_95_medium", "mean_ap_medium"),
        ("map50_95_large", "mean_ap_large"),
        ("ar100", "mean_ar100"),
    ):
        if result[published] != result[generic]:
            raise ValueError(f"point_metrics {published} differs from {generic}")
    return result


def _public_point_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Restore integer per-class keys used by the legacy public result schema."""

    result = _arm_point_metrics(metrics)
    if "per_class_ap" in result:
        result["per_class_ap"] = {
            int(category): value for category, value in result["per_class_ap"].items()
        }
    return result


def _normalize_bootstrap_arm_records(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, ...]]:
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("predictions must be a sequence")
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence")
    normalized_targets = [_record(item, kind="target") for item in targets]
    image_ids = tuple(item["image_id"] for item in normalized_targets)
    if len(image_ids) < 2:
        raise ValueError("paired image bootstrap requires at least two target images")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("targets must contain unique image_id values")
    normalized_predictions = [_record(item, kind="prediction") for item in predictions]
    prediction_ids = [item["image_id"] for item in normalized_predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("predictions contain duplicate image IDs")
    if set(prediction_ids) != set(image_ids):
        raise ValueError("predictions must contain one prediction record per target image")
    by_image = {item["image_id"]: item for item in normalized_predictions}
    return (
        [by_image[image_id] for image_id in image_ids],
        normalized_targets,
        image_ids,
    )


def _compute_map_bootstrap_arm(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
    iou_thresholds: Sequence[float] | None = None,
    category_ids: Sequence[int] | None = None,
    progress: Callable[[int, int], None] | None = None,
    failure_context: str | None,
    require_positive_samples: bool,
    require_nonnegative_samples: bool,
) -> dict[str, Any]:
    """Implement one arm with optional legacy-wrapper sample validation."""

    iterations = _integer(n_bootstrap, name="n_bootstrap", minimum=2)
    random_seed = _integer(seed, name="seed", minimum=0)
    if random_seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must not exceed 2**64 - 1")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable when supplied")
    normalized_predictions, normalized_targets, image_ids = _normalize_bootstrap_arm_records(
        predictions,
        targets,
    )
    observed_categories = [int(label) for item in normalized_targets for label in item["labels"]]
    observed_categories.extend(
        int(label) for item in normalized_predictions for label in item["labels"]
    )
    categories = _categories(category_ids, observed=observed_categories)
    thresholds = _thresholds(iou_thresholds)
    if not _is_standard_coco_thresholds(thresholds):
        raise ValueError("map bootstrap arm requires the standard COCO IoU grid 0.50:0.05:0.95")

    point = compute_map(
        normalized_predictions,
        normalized_targets,
        thresholds,
        category_ids=categories,
    )
    cache = _build_coco_image_match_cache(
        normalized_predictions,
        normalized_targets,
        thresholds=thresholds,
        category_ids=categories,
    )
    original_positions = tuple(range(len(image_ids)))
    cached_point = cache.accumulate(original_positions)
    for metric in _BOOTSTRAP_METRICS:
        if not np.isclose(
            cached_point[metric],
            point[metric],
            rtol=0.0,
            atol=2e-14,
        ):
            raise RuntimeError("cached COCO accumulation failed point-estimate validation")

    if progress is not None:
        progress(0, iterations)
    samples = {metric: [] for metric in _BOOTSTRAP_METRICS}
    report_interval = max(1, iterations // 20)
    rng = np.random.default_rng(np.random.SeedSequence(random_seed))
    for bootstrap_index in range(iterations):
        chosen = tuple(
            int(value) for value in rng.choice(len(image_ids), size=len(image_ids), replace=True)
        )
        try:
            iteration = cache.accumulate(chosen)
            for metric in _BOOTSTRAP_METRICS:
                value = _finite_float(
                    iteration[metric],
                    name=f"bootstrap sample {metric}",
                )
                if require_positive_samples and value <= 0.0:
                    raise ValueError("bootstrap baseline metric is not positive")
                if require_nonnegative_samples and value < 0.0:
                    raise ValueError("bootstrap draw contains undefined AP")
                if failure_context is None and not 0.0 <= value <= 1.0:
                    raise ValueError("bootstrap AP samples must lie in [0, 1]")
                samples[metric].append(value)
        except Exception as exc:
            message = (
                f"map bootstrap arm iteration {bootstrap_index + 1} of {iterations} failed"
                if failure_context is None
                else f"{failure_context}, iteration {bootstrap_index + 1} of {iterations} failed"
            )
            raise RuntimeError(message) from exc
        completed = bootstrap_index + 1
        if progress is not None and (completed == iterations or completed % report_interval == 0):
            progress(completed, iterations)
    del cache, normalized_predictions

    return {
        "method": _ARM_BOOTSTRAP_METHOD,
        "sampling_contract": _ARM_SAMPLING_CONTRACT,
        "iterations": iterations,
        "seed": random_seed,
        "image_count": len(image_ids),
        "category_ids": categories,
        "iou_thresholds": thresholds.tolist(),
        "point_metrics": _arm_point_metrics(point),
        "bootstrap_samples": samples,
        "bootstrap_failures": 0,
        "matching_evaluations": 1,
        "bootstrap_accumulation": _BOOTSTRAP_ACCUMULATION,
        "memory_strategy": _ARM_MEMORY_STRATEGY,
    }


def compute_map_bootstrap_arm(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
    iou_thresholds: Sequence[float] | None = None,
    category_ids: Sequence[int] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Compute one reusable deterministic-condition bootstrap arm.

    The ordered sample lists replay the historical paired-bootstrap RNG stream:
    a fresh ``SeedSequence(seed)`` drives the same native-image cluster draw at
    every sample index.  An arm is therefore independently checkpointable and
    can later be combined with any other arm computed with identical metadata.
    The payload uses decimal-string ``per_class_ap`` keys so canonical JSON
    serialization and loading do not alter it.
    """

    arm = _compute_map_bootstrap_arm(
        predictions,
        targets,
        n_bootstrap=n_bootstrap,
        seed=seed,
        iou_thresholds=iou_thresholds,
        category_ids=category_ids,
        progress=progress,
        failure_context=None,
        require_positive_samples=False,
        require_nonnegative_samples=False,
    )
    _validated_bootstrap_arm(arm, name="computed bootstrap arm")
    return arm


def _validated_bootstrap_arm(
    arm: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(arm, Mapping):
        raise TypeError(f"{name} must be a bootstrap-arm mapping")
    required = {
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
    missing = required.difference(arm)
    unknown = set(arm).difference(required)
    if missing or unknown:
        raise ValueError(
            f"{name} has missing or unknown keys; "
            f"missing={sorted(missing)}, unknown={sorted(unknown, key=repr)}"
        )
    if arm["method"] != _ARM_BOOTSTRAP_METHOD:
        raise ValueError(f"{name} uses an unsupported bootstrap-arm method")
    if arm["sampling_contract"] != _ARM_SAMPLING_CONTRACT:
        raise ValueError(f"{name} uses an unsupported sampling contract")
    iterations = _integer(arm["iterations"], name=f"{name} iterations", minimum=2)
    random_seed = _integer(arm["seed"], name=f"{name} seed", minimum=0)
    if random_seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{name} seed must not exceed 2**64 - 1")
    image_count = _integer(arm["image_count"], name=f"{name} image_count", minimum=2)
    raw_categories = arm["category_ids"]
    if isinstance(raw_categories, (str, bytes)) or not isinstance(raw_categories, Sequence):
        raise TypeError(f"{name} category_ids must be an ordered sequence")
    declared_categories = [
        _integer(value, name=f"{name} category_id", minimum=1) for value in raw_categories
    ]
    categories = _categories(declared_categories, observed=())
    if declared_categories != categories:
        raise ValueError(f"{name} category_ids must be sorted and unique")
    raw_thresholds = arm["iou_thresholds"]
    if isinstance(raw_thresholds, (str, bytes)) or not isinstance(raw_thresholds, Sequence):
        raise TypeError(f"{name} iou_thresholds must be an explicit ordered sequence")
    thresholds = _thresholds(raw_thresholds)
    if not _is_standard_coco_thresholds(thresholds):
        raise ValueError(f"{name} must use the standard COCO IoU grid")
    if (
        _integer(
            arm["bootstrap_failures"],
            name=f"{name} bootstrap_failures",
            minimum=0,
        )
        != 0
    ):
        raise ValueError(f"{name} must not contain bootstrap failures")
    if (
        _integer(
            arm["matching_evaluations"],
            name=f"{name} matching_evaluations",
            minimum=1,
        )
        != 1
    ):
        raise ValueError(f"{name} must contain exactly one matching evaluation")
    if arm["bootstrap_accumulation"] != _BOOTSTRAP_ACCUMULATION:
        raise ValueError(f"{name} uses an unsupported bootstrap accumulation")
    if arm["memory_strategy"] != _ARM_MEMORY_STRATEGY:
        raise ValueError(f"{name} uses an unsupported memory strategy")
    point_metrics = _public_point_metrics(arm["point_metrics"])
    if point_metrics["mean_ap_iou_thresholds"] != thresholds.tolist():
        raise ValueError(f"{name} point_metrics uses a different IoU threshold grid")
    if "per_class_ap" in point_metrics and set(point_metrics["per_class_ap"]) != set(categories):
        raise ValueError(f"{name} per_class_ap does not match category_ids")
    raw_samples = arm["bootstrap_samples"]
    if not isinstance(raw_samples, Mapping):
        raise TypeError(f"{name} bootstrap_samples must be a mapping")
    if set(raw_samples) != set(_BOOTSTRAP_METRICS):
        raise ValueError(f"{name} bootstrap_samples must contain exactly map50 and map50_95")
    samples: dict[str, list[float]] = {}
    for metric in _BOOTSTRAP_METRICS:
        if metric not in raw_samples:
            raise ValueError(f"{name} bootstrap_samples is missing {metric!r}")
        values = raw_samples[metric]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{name} {metric} bootstrap samples must be a sequence")
        if len(values) != iterations:
            raise ValueError(f"{name} {metric} bootstrap sample count must equal iterations")
        samples[metric] = []
        for value in values:
            sample = _finite_float(value, name=f"{name} {metric} bootstrap sample")
            if not 0.0 <= sample <= 1.0:
                raise ValueError(f"{name} {metric} bootstrap samples must lie in [0, 1]")
            samples[metric].append(sample)
    return {
        "iterations": iterations,
        "seed": random_seed,
        "image_count": image_count,
        "category_ids": categories,
        "iou_thresholds": thresholds.tolist(),
        "point_metrics": point_metrics,
        "bootstrap_samples": samples,
    }


def _validated_bootstrap_arms(
    named_arms: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    if not named_arms:
        raise ValueError("at least one bootstrap arm is required")
    validated = {name: _validated_bootstrap_arm(arm, name=name) for name, arm in named_arms}
    reference = validated[named_arms[0][0]]
    for name, arm in validated.items():
        for field in ("iterations", "seed", "image_count", "category_ids", "iou_thresholds"):
            if arm[field] != reference[field]:
                raise ValueError(f"bootstrap arm {name!r} has incompatible {field}")
    return validated


def assemble_paired_map_bootstrap(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    baseline_condition: str,
) -> dict[str, Any]:
    """Assemble reusable arms into the legacy paired-condition result.

    Callers persisting arms separately must verify that every arm is bound to
    the same ordered target-image axis before calling this statistics-only
    assembler.  The study checkpoint protocol provides that external binding.
    """

    if not isinstance(arms, Mapping) or not arms:
        raise TypeError("arms must be a nonempty ordered mapping")
    if not isinstance(baseline_condition, str) or not baseline_condition:
        raise TypeError("baseline_condition must be a nonempty string")
    condition_order = tuple(arms)
    if any(not isinstance(condition, str) or not condition for condition in condition_order):
        raise TypeError("condition keys must be nonempty strings")
    if baseline_condition not in condition_order:
        raise ValueError("baseline_condition is absent from arms")
    validated = _validated_bootstrap_arms(
        [(condition, arms[condition]) for condition in condition_order]
    )
    metadata = validated[condition_order[0]]
    points = {condition: validated[condition]["point_metrics"] for condition in condition_order}
    samples = {
        condition: validated[condition]["bootstrap_samples"] for condition in condition_order
    }
    for metric in _BOOTSTRAP_METRICS:
        if points[baseline_condition][metric] <= 0.0:
            raise ValueError(f"baseline {metric} must be positive for paired ratios")
        if any(value <= 0.0 for value in samples[baseline_condition][metric]):
            raise ValueError(f"baseline bootstrap {metric} must be positive for paired ratios")

    differences: dict[str, dict[str, list[float]]] = {}
    ratios: dict[str, dict[str, list[float]]] = {}
    for condition in condition_order:
        differences[condition] = {}
        ratios[condition] = {}
        for metric in _BOOTSTRAP_METRICS:
            baseline_samples = samples[baseline_condition][metric]
            condition_samples = samples[condition][metric]
            differences[condition][metric] = [
                value - baseline_value
                for value, baseline_value in zip(condition_samples, baseline_samples)
            ]
            ratios[condition][metric] = [
                value / baseline_value
                for value, baseline_value in zip(condition_samples, baseline_samples)
            ]

    condition_results: list[dict[str, Any]] = []
    for condition in condition_order:
        baseline_metrics = points[baseline_condition]
        condition_results.append(
            {
                "condition": condition,
                "metrics": points[condition],
                "marginal_percentile_95": {
                    metric: _percentile_interval(samples[condition][metric])
                    for metric in _BOOTSTRAP_METRICS
                },
                "paired_difference_to_baseline": {
                    metric: {
                        "estimate": float(points[condition][metric])
                        - float(baseline_metrics[metric]),
                        "percentile_95": _percentile_interval(differences[condition][metric]),
                    }
                    for metric in _BOOTSTRAP_METRICS
                },
                "paired_ratio_to_baseline": {
                    metric: {
                        "estimate": float(points[condition][metric])
                        / float(baseline_metrics[metric]),
                        "percentile_95": _percentile_interval(ratios[condition][metric]),
                    }
                    for metric in _BOOTSTRAP_METRICS
                },
            }
        )
    return {
        "method": "paired_image_cluster_cached_coco_percentile_bootstrap_v2",
        "iterations": metadata["iterations"],
        "seed": metadata["seed"],
        "image_count": metadata["image_count"],
        "category_ids": metadata["category_ids"],
        "baseline_condition": baseline_condition,
        "condition_order": list(condition_order),
        "conditions": condition_results,
        "bootstrap_failures": 0,
        "matching_evaluations_per_condition": 1,
        "bootstrap_accumulation": _BOOTSTRAP_ACCUMULATION,
        "memory_strategy": _PAIRED_MEMORY_STRATEGY,
    }


def assemble_paired_map_curve_auc_bootstrap(
    physical_arms: Sequence[Mapping[str, Any]],
    comparator_arms: Sequence[Mapping[str, Any]],
    *,
    coordinates: Sequence[float],
    comparator_name: str = "gaussian",
) -> dict[str, Any]:
    """Assemble reusable arms into the legacy paired curve-AUC result.

    Callers persisting arms separately must verify that every arm is bound to
    the same ordered target-image axis before calling this statistics-only
    assembler.  The study checkpoint protocol provides that external binding.
    """

    if not isinstance(comparator_name, str) or not comparator_name.strip():
        raise TypeError("comparator_name must be a nonempty string")
    resolved_comparator_name = comparator_name.strip()
    if isinstance(coordinates, (str, bytes)) or not isinstance(coordinates, Sequence):
        raise TypeError("coordinates must be an ordered sequence")
    coordinate_values = np.asarray(coordinates, dtype=np.float64)
    if coordinate_values.ndim != 1 or len(coordinate_values) < 2:
        raise ValueError("coordinates must contain at least two values")
    if not np.all(np.isfinite(coordinate_values)) or np.any(np.diff(coordinate_values) <= 0.0):
        raise ValueError("coordinates must be finite and strictly increasing")
    for name, value in (("physical_arms", physical_arms), ("comparator_arms", comparator_arms)):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{name} must be an ordered sequence of bootstrap arms")
        if len(value) != len(coordinate_values):
            raise ValueError(f"{name} must align one-to-one with coordinates")

    named_arms = [
        item
        for index in range(len(coordinate_values))
        for item in (
            (f"physical_{index}", physical_arms[index]),
            (f"comparator_{index}", comparator_arms[index]),
        )
    ]
    validated = _validated_bootstrap_arms(named_arms)
    metadata = validated[named_arms[0][0]]
    iterations = metadata["iterations"]
    points = {name: validated[name]["point_metrics"] for name, _ in named_arms}
    samples = {name: validated[name]["bootstrap_samples"] for name, _ in named_arms}
    point_curves = {
        metric: {
            "physical": np.asarray(
                [points[f"physical_{index}"][metric] for index in range(len(coordinate_values))],
                dtype=np.float64,
            ),
            "comparator": np.asarray(
                [points[f"comparator_{index}"][metric] for index in range(len(coordinate_values))],
                dtype=np.float64,
            ),
        }
        for metric in _BOOTSTRAP_METRICS
    }
    if any(np.any(curves[name] < 0.0) for curves in point_curves.values() for name in curves):
        raise ValueError("curve AUC requires defined nonnegative AP at every coordinate")
    for name, arm_samples in samples.items():
        for metric in _BOOTSTRAP_METRICS:
            if any(value < 0.0 for value in arm_samples[metric]):
                raise ValueError(f"curve AUC bootstrap arm {name!r} contains undefined AP")

    physical_samples = {metric: [[] for _ in coordinate_values] for metric in _BOOTSTRAP_METRICS}
    comparator_samples = {metric: [[] for _ in coordinate_values] for metric in _BOOTSTRAP_METRICS}
    difference_samples = {metric: [[] for _ in coordinate_values] for metric in _BOOTSTRAP_METRICS}
    auc_samples = {metric: [] for metric in _BOOTSTRAP_METRICS}
    for bootstrap_index in range(iterations):
        for metric in _BOOTSTRAP_METRICS:
            physical_curve = np.asarray(
                [
                    samples[f"physical_{index}"][metric][bootstrap_index]
                    for index in range(len(coordinate_values))
                ],
                dtype=np.float64,
            )
            comparator_curve = np.asarray(
                [
                    samples[f"comparator_{index}"][metric][bootstrap_index]
                    for index in range(len(coordinate_values))
                ],
                dtype=np.float64,
            )
            differences = physical_curve - comparator_curve
            for index in range(len(coordinate_values)):
                physical_samples[metric][index].append(float(physical_curve[index]))
                comparator_samples[metric][index].append(float(comparator_curve[index]))
                difference_samples[metric][index].append(float(differences[index]))
            auc_samples[metric].append(_trapezoidal_integral(differences, coordinate_values))

    results: dict[str, Any] = {}
    for metric in _BOOTSTRAP_METRICS:
        physical_curve = point_curves[metric]["physical"]
        comparator_curve = point_curves[metric]["comparator"]
        difference_curve = physical_curve - comparator_curve
        results[metric] = {
            "physical_curve": [
                {
                    "coordinate": float(coordinate_values[index]),
                    "estimate": float(physical_curve[index]),
                    "marginal_percentile_95": _percentile_interval(physical_samples[metric][index]),
                }
                for index in range(len(coordinate_values))
            ],
            "comparator_curve": [
                {
                    "coordinate": float(coordinate_values[index]),
                    "estimate": float(comparator_curve[index]),
                    "marginal_percentile_95": _percentile_interval(
                        comparator_samples[metric][index]
                    ),
                }
                for index in range(len(coordinate_values))
            ],
            "paired_difference_curve": [
                {
                    "coordinate": float(coordinate_values[index]),
                    "estimate": float(difference_curve[index]),
                    "percentile_95": _percentile_interval(difference_samples[metric][index]),
                }
                for index in range(len(coordinate_values))
            ],
            "paired_physical_minus_comparator_auc": {
                "estimate": _trapezoidal_integral(difference_curve, coordinate_values),
                "percentile_95": _percentile_interval(auc_samples[metric]),
                "unit": "AP_times_coordinate_unit",
            },
        }
    return {
        "method": "paired_image_cluster_cached_coco_curve_auc_percentile_bootstrap_v1",
        "iterations": iterations,
        "seed": metadata["seed"],
        "image_count": metadata["image_count"],
        "category_ids": metadata["category_ids"],
        "coordinate_name": "edge_waves_ref",
        "coordinate_unit": "waves_at_reference_wavelength",
        "ordered_coordinates": coordinate_values.tolist(),
        "integration": "trapezoidal_over_declared_coordinate_grid",
        "contrast_orientation": "physical_minus_comparator",
        "comparator_name": resolved_comparator_name,
        "metrics": results,
        "primary_metric": "map50_95",
        "bootstrap_failures": 0,
        "matching_evaluations_per_condition": 1,
        "bootstrap_accumulation": _BOOTSTRAP_ACCUMULATION,
        "memory_strategy": _PAIRED_MEMORY_STRATEGY,
    }


def compute_paired_map_bootstrap(
    predictions_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    targets: Sequence[Mapping[str, Any]],
    *,
    baseline_condition: str,
    n_bootstrap: int = 200,
    seed: int = 42,
    iou_thresholds: Sequence[float] | None = None,
    category_ids: Sequence[int] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Compute paired deterministic-condition COCO intervals and contrasts.

    This compatibility wrapper computes one independently checkpointable arm
    per condition, then delegates all contrasts and intervals to
    :func:`assemble_paired_map_bootstrap`.  Reinitializing the seeded draw
    stream in each arm preserves the historical paired RNG semantics exactly.
    """

    iterations = _integer(n_bootstrap, name="n_bootstrap", minimum=2)
    random_seed = _integer(seed, name="seed", minimum=0)
    if random_seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must not exceed 2**64 - 1")
    if not isinstance(predictions_by_condition, Mapping) or not predictions_by_condition:
        raise TypeError("predictions_by_condition must be a nonempty ordered mapping")
    if not isinstance(baseline_condition, str) or not baseline_condition:
        raise TypeError("baseline_condition must be a nonempty string")

    normalized_targets = [_record(item, kind="target") for item in targets]
    image_ids = tuple(item["image_id"] for item in normalized_targets)
    if len(image_ids) < 2:
        raise ValueError("paired image bootstrap requires at least two target images")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("targets must contain unique image_id values")
    condition_order = tuple(predictions_by_condition)
    if any(not isinstance(condition, str) or not condition for condition in condition_order):
        raise TypeError("condition keys must be nonempty strings")
    if baseline_condition not in condition_order:
        raise ValueError("baseline_condition is absent from predictions_by_condition")
    observed_categories = [int(label) for item in normalized_targets for label in item["labels"]]

    def normalized_condition(condition: str) -> list[dict[str, Any]]:
        raw_predictions = predictions_by_condition[condition]
        if isinstance(raw_predictions, (str, bytes)) or not isinstance(raw_predictions, Sequence):
            raise TypeError(f"condition {condition!r} predictions must be a sequence")
        normalized = [_record(item, kind="prediction") for item in raw_predictions]
        ids = [item["image_id"] for item in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError(f"condition {condition!r} contains duplicate image IDs")
        if set(ids) != set(image_ids):
            raise ValueError(
                f"condition {condition!r} must contain one prediction record per target image"
            )
        by_image = {item["image_id"]: item for item in normalized}
        return [by_image[image_id] for image_id in image_ids]

    if category_ids is None:
        for condition in condition_order:
            normalized = normalized_condition(condition)
            observed_categories.extend(
                int(label) for item in normalized for label in item["labels"]
            )
    categories = _categories(category_ids, observed=observed_categories)
    thresholds = _thresholds(iou_thresholds)
    if not _is_standard_coco_thresholds(thresholds):
        raise ValueError(
            "paired map50_95 bootstrap requires the standard COCO IoU grid 0.50:0.05:0.95"
        )
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable when supplied")

    arms_by_condition: dict[str, dict[str, Any]] = {}
    processing_order = (baseline_condition,) + tuple(
        condition for condition in condition_order if condition != baseline_condition
    )
    for condition in processing_order:
        predictions = normalized_condition(condition)

        def arm_progress(
            completed: int,
            total: int,
            *,
            _condition: str = condition,
        ) -> None:
            _report_bootstrap_progress(
                progress,
                condition=_condition,
                completed=completed,
                total=total,
            )

        arm = _compute_map_bootstrap_arm(
            predictions,
            normalized_targets,
            n_bootstrap=iterations,
            seed=random_seed,
            iou_thresholds=thresholds,
            category_ids=categories,
            progress=arm_progress if progress is not None else None,
            failure_context=f"paired bootstrap condition {condition!r}",
            require_positive_samples=condition == baseline_condition,
            require_nonnegative_samples=False,
        )
        arms_by_condition[condition] = arm

    ordered_arms = {condition: arms_by_condition[condition] for condition in condition_order}
    return assemble_paired_map_bootstrap(
        ordered_arms,
        baseline_condition=baseline_condition,
    )


def compute_paired_map_curve_auc_bootstrap(
    physical_predictions: Sequence[Sequence[Mapping[str, Any]]],
    comparator_predictions: Sequence[Sequence[Mapping[str, Any]]],
    targets: Sequence[Mapping[str, Any]],
    *,
    coordinates: Sequence[float],
    comparator_name: str = "gaussian",
    n_bootstrap: int = 2_000,
    seed: int = 20_260_715,
    iou_thresholds: Sequence[float] | None = None,
    category_ids: Sequence[int] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Bootstrap a paired physical-minus-comparator AP-curve AUC.

    This compatibility wrapper computes each coordinate/curve arm independently
    and delegates the cross-coordinate covariance calculation to
    :func:`assemble_paired_map_curve_auc_bootstrap`.
    """

    iterations = _integer(n_bootstrap, name="n_bootstrap", minimum=2)
    random_seed = _integer(seed, name="seed", minimum=0)
    if random_seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must not exceed 2**64 - 1")
    if not isinstance(comparator_name, str) or not comparator_name.strip():
        raise TypeError("comparator_name must be a nonempty string")
    resolved_comparator_name = comparator_name.strip()
    if isinstance(coordinates, (str, bytes)) or not isinstance(coordinates, Sequence):
        raise TypeError("coordinates must be an ordered sequence")
    coordinate_values = np.asarray(coordinates, dtype=np.float64)
    if coordinate_values.ndim != 1 or len(coordinate_values) < 2:
        raise ValueError("coordinates must contain at least two values")
    if not np.all(np.isfinite(coordinate_values)) or np.any(np.diff(coordinate_values) <= 0.0):
        raise ValueError("coordinates must be finite and strictly increasing")
    for name, value in (
        ("physical_predictions", physical_predictions),
        ("comparator_predictions", comparator_predictions),
    ):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{name} must be an ordered sequence of prediction sequences")
        if len(value) != len(coordinate_values):
            raise ValueError(f"{name} must align one-to-one with coordinates")

    normalized_targets = [_record(item, kind="target") for item in targets]
    image_ids = tuple(item["image_id"] for item in normalized_targets)
    if len(image_ids) < 2:
        raise ValueError("paired image bootstrap requires at least two target images")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("targets must contain unique image_id values")
    condition_order = tuple(
        name
        for index in range(len(coordinate_values))
        for name in (f"physical_{index}", f"comparator_{index}")
    )
    observed_categories = [int(label) for item in normalized_targets for label in item["labels"]]

    def normalized_condition(condition: str) -> list[dict[str, Any]]:
        kind, raw_index = condition.rsplit("_", 1)
        index = int(raw_index)
        raw_predictions = (
            physical_predictions[index] if kind == "physical" else comparator_predictions[index]
        )
        if isinstance(raw_predictions, (str, bytes)) or not isinstance(raw_predictions, Sequence):
            raise TypeError(f"{condition} predictions must be a sequence")
        normalized = [_record(item, kind="prediction") for item in raw_predictions]
        ids = [item["image_id"] for item in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{condition} contains duplicate image IDs")
        if set(ids) != set(image_ids):
            raise ValueError(f"{condition} must contain one prediction per target image")
        by_image = {item["image_id"]: item for item in normalized}
        return [by_image[image_id] for image_id in image_ids]

    if category_ids is None:
        for condition in condition_order:
            normalized = normalized_condition(condition)
            observed_categories.extend(
                int(label) for item in normalized for label in item["labels"]
            )
    categories = _categories(category_ids, observed=observed_categories)
    thresholds = _thresholds(iou_thresholds)
    if not _is_standard_coco_thresholds(thresholds):
        raise ValueError(
            "paired curve-AUC bootstrap requires the standard COCO IoU grid 0.50:0.05:0.95"
        )
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable when supplied")

    physical_arms: dict[int, dict[str, Any]] = {}
    comparator_arms: dict[int, dict[str, Any]] = {}
    for condition in condition_order:
        predictions = normalized_condition(condition)

        def arm_progress(
            completed: int,
            total: int,
            *,
            _condition: str = condition,
        ) -> None:
            _report_bootstrap_progress(
                progress,
                condition=_condition,
                completed=completed,
                total=total,
            )

        arm = _compute_map_bootstrap_arm(
            predictions,
            normalized_targets,
            n_bootstrap=iterations,
            seed=random_seed,
            iou_thresholds=thresholds,
            category_ids=categories,
            progress=arm_progress if progress is not None else None,
            failure_context=f"paired curve-AUC bootstrap condition {condition!r}",
            require_positive_samples=False,
            require_nonnegative_samples=True,
        )
        kind, raw_index = condition.rsplit("_", 1)
        destination = physical_arms if kind == "physical" else comparator_arms
        destination[int(raw_index)] = arm

    return assemble_paired_map_curve_auc_bootstrap(
        [physical_arms[index] for index in range(len(coordinate_values))],
        [comparator_arms[index] for index in range(len(coordinate_values))],
        coordinates=coordinate_values.tolist(),
        comparator_name=resolved_comparator_name,
    )


def compute_hierarchical_map_ci(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    realization_ids: Sequence[int],
    n_bootstrap: int = 200,
    seed: int = 42,
    iou_thresholds: Sequence[float] | None = None,
    category_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate a complete image-realization grid with declared uncertainty."""

    iterations = _integer(n_bootstrap, name="n_bootstrap", minimum=2)
    random_seed = _integer(seed, name="seed", minimum=0)
    if random_seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must not exceed 2**64 - 1")
    if isinstance(realization_ids, (str, bytes)) or not isinstance(realization_ids, Sequence):
        raise TypeError("realization_ids must be an ordered sequence")
    realizations = tuple(
        _integer(value, name="realization_id", minimum=0) for value in realization_ids
    )
    if len(realizations) < 2:
        raise ValueError("hierarchical uncertainty requires at least two realizations")
    if len(set(realizations)) != len(realizations):
        raise ValueError("realization_ids must be unique")

    normalized_targets = [_record(item, kind="target") for item in targets]
    image_ids = tuple(item["image_id"] for item in normalized_targets)
    if len(image_ids) < 2:
        raise ValueError("hierarchical uncertainty requires at least two images")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("targets must contain unique image_id values")
    target_by_image = {item["image_id"]: item for item in normalized_targets}
    observed_categories = [int(label) for item in normalized_targets for label in item["labels"]]
    for raw in predictions:
        if not isinstance(raw, Mapping):
            raise TypeError("prediction records must be mappings")
        if "labels" in raw:
            labels = _as_numpy(raw["labels"], name="prediction labels")
            try:
                observed_categories.extend(int(value) for value in labels)
            except (TypeError, ValueError) as exc:
                raise ValueError("prediction labels must contain integers") from exc
    categories = _categories(category_ids, observed=observed_categories)

    grid: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in predictions:
        prediction = _record(raw, kind="prediction")
        if "realization_id" not in raw:
            raise ValueError("hierarchical predictions require realization_id")
        realization = _integer(raw["realization_id"], name="realization_id", minimum=0)
        key = (prediction["image_id"], realization)
        if key in grid:
            raise ValueError("predictions contain a duplicate image-realization record")
        grid[key] = prediction
    expected = {(image, realization) for image in image_ids for realization in realizations}
    missing = expected.difference(grid)
    extra = set(grid).difference(expected)
    if missing or extra:
        raise ValueError(
            "predictions must form the complete declared image-realization grid; "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    def evaluate(images: Sequence[int], realization: int) -> dict[str, Any]:
        sample_predictions: list[dict[str, Any]] = []
        sample_targets: list[dict[str, Any]] = []
        for synthetic_id, image_id in enumerate(images, start=1):
            prediction = dict(grid[(image_id, realization)])
            target = dict(target_by_image[image_id])
            prediction["image_id"] = synthetic_id
            target["image_id"] = synthetic_id
            sample_predictions.append(prediction)
            sample_targets.append(target)
        return compute_map(
            sample_predictions,
            sample_targets,
            iou_thresholds,
            category_ids=categories,
        )

    per_realization = {
        realization: evaluate(image_ids, realization) for realization in realizations
    }
    point = _mean_metrics([per_realization[value] for value in realizations])
    image_seed, realization_seed, joint_seed = np.random.SeedSequence(random_seed).spawn(3)
    image_rng = np.random.default_rng(image_seed)
    realization_rng = np.random.default_rng(realization_seed)
    joint_rng = np.random.default_rng(joint_seed)
    image_samples = {metric: [] for metric in ("map50", "map50_95")}
    realization_samples = {metric: [] for metric in ("map50", "map50_95")}
    joint_samples = {metric: [] for metric in ("map50", "map50_95")}

    for bootstrap_index in range(iterations):
        sampled_images = tuple(
            int(value) for value in image_rng.choice(image_ids, size=len(image_ids), replace=True)
        )
        sampled_realizations = tuple(
            int(value)
            for value in realization_rng.choice(realizations, size=len(realizations), replace=True)
        )
        joint_images = tuple(
            int(value) for value in joint_rng.choice(image_ids, size=len(image_ids), replace=True)
        )
        joint_realizations = tuple(
            int(value)
            for value in joint_rng.choice(realizations, size=len(realizations), replace=True)
        )
        try:
            image_metric = _mean_metrics(
                [evaluate(sampled_images, realization) for realization in realizations]
            )
            realization_metric = _mean_metrics(
                [per_realization[realization] for realization in sampled_realizations]
            )
            joint_metric = _mean_metrics(
                [evaluate(joint_images, realization) for realization in joint_realizations]
            )
        except Exception as exc:
            raise RuntimeError(
                f"hierarchical bootstrap iteration {bootstrap_index + 1} of {iterations} failed"
            ) from exc
        for metric in image_samples:
            image_samples[metric].append(float(image_metric[metric]))
            realization_samples[metric].append(float(realization_metric[metric]))
            joint_samples[metric].append(float(joint_metric[metric]))

    return {
        **point,
        "aggregation": "arithmetic_mean_of_complete_per_realization_coco_metrics",
        "image_count": len(image_ids),
        "category_ids": categories,
        "realization_ids": list(realizations),
        "per_realization": [
            {"realization_id": realization, "metrics": per_realization[realization]}
            for realization in realizations
        ],
        "uncertainty": {
            "method": "separate_and_joint_image_realization_cluster_bootstrap_v1",
            "iterations": iterations,
            "seed": random_seed,
            "normal_half_width_95": {
                "image_cluster": {
                    metric: _half_width(samples) for metric, samples in image_samples.items()
                },
                "realization": {
                    metric: _half_width(samples) for metric, samples in realization_samples.items()
                },
                "hierarchical_joint": {
                    metric: _half_width(samples) for metric, samples in joint_samples.items()
                },
            },
            "bootstrap_failures": 0,
        },
    }


__all__ = [
    "COCO80_TO_91",
    "compute_hierarchical_map_ci",
    "compute_map",
    "compute_paired_map_bootstrap",
    "yolo_to_coco_category_ids",
]
