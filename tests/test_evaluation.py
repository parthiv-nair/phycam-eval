"""Core COCO estimators and one native-coordinate benchmark smoke path."""

from __future__ import annotations

import json

import numpy as np
import pytest

from phycam_eval.eval.coco import load_native_coco_subset
from phycam_eval.eval.coco_runner import run_coco_ldr_benchmark
from phycam_eval.eval.metrics import (
    COCO80_TO_91,
    compute_map,
    compute_paired_map_bootstrap,
    compute_paired_map_curve_auc_bootstrap,
    yolo_to_coco_category_ids,
)
from phycam_eval.eval.model_provenance import detector_execution_identity, model_identity
from phycam_eval.eval.preprocess import LetterboxConfig
from phycam_eval.experiments import BaselineCondition, BaselineKind, ConditionBinding
from phycam_eval.reference_profiles import synthetic_ldr_profile

pytest.importorskip("PIL")
pytest.importorskip("pycocotools")


def _target(image_id: int) -> dict[str, object]:
    return {
        "image_id": image_id,
        "boxes": np.array([[1.0, 2.0, 11.0, 12.0]]),
        "labels": np.array([1]),
        "area": np.array([100.0]),
        "iscrowd": np.array([0]),
    }


def _prediction(image_id: int, *, detected: bool = True) -> dict[str, object]:
    return {
        "image_id": image_id,
        "boxes": np.array([[1.0, 2.0, 11.0, 12.0]]) if detected else np.empty((0, 4)),
        "labels": np.array([1]) if detected else np.empty(0, dtype=np.int64),
        "scores": np.array([0.9]) if detected else np.empty(0),
    }


def _fixture_model():
    return model_identity(
        backend="test",
        model_id="perfect-box-fixture",
        revision="v1",
        artifacts=[{"name": "fixture.bin", "bytes": 0, "sha256": "0" * 64}],
    )


def _fixture_execution():
    model = _fixture_model()
    return detector_execution_identity(
        model=model,
        implementation_id="fixture.perfect_box.v1",
        requested_device="cpu",
        actual_device="cpu",
        device_attestation={"method": "fixture.device_match.v1", "matches": True},
        inference_execution={"method": "fixture.deterministic.v1", "attested": True, "seed": 0},
    )


def test_coco_map_handles_perfect_empty_and_yolo_category_mapping() -> None:
    target = [_target(7)]
    perfect = compute_map([_prediction(7)], target)
    empty = compute_map([_prediction(7, detected=False)], target)
    assert perfect["map50"] == pytest.approx(1.0)
    assert perfect["map50_95"] == pytest.approx(1.0)
    assert perfect["ar100"] == pytest.approx(1.0)
    assert empty["map50"] == empty["map50_95"] == 0.0
    np.testing.assert_array_equal(
        yolo_to_coco_category_ids(np.array([0, 11, 79])),
        COCO80_TO_91[[0, 11, 79]],
    )


def test_paired_image_cluster_bootstrap_is_reproducible_and_preserves_pairing() -> None:
    targets = [_target(value) for value in (1, 2, 3)]
    baseline = [_prediction(value) for value in (1, 2, 3)]
    degraded = [_prediction(value, detected=value != 3) for value in (1, 2, 3)]
    kwargs = {
        "baseline_condition": "baseline",
        "n_bootstrap": 8,
        "seed": 17,
        "category_ids": [1],
    }
    first = compute_paired_map_bootstrap(
        {"baseline": baseline, "degraded": degraded}, targets, **kwargs
    )
    second = compute_paired_map_bootstrap(
        {"baseline": baseline, "degraded": degraded}, targets, **kwargs
    )
    assert first == second
    baseline_result, degraded_result = first["conditions"]
    assert baseline_result["paired_difference_to_baseline"]["map50"]["estimate"] == 0.0
    assert degraded_result["paired_difference_to_baseline"]["map50"]["estimate"] < 0.0
    assert degraded_result["paired_ratio_to_baseline"]["map50"]["estimate"] < 1.0


def test_paired_curve_auc_uses_common_draws_and_the_trapezoid_rule() -> None:
    targets = [_target(value) for value in (1, 2, 3)]
    perfect = [_prediction(value) for value in (1, 2, 3)]
    one_miss = [_prediction(value, detected=value != 3) for value in (1, 2, 3)]
    two_miss = [_prediction(value, detected=value == 1) for value in (1, 2, 3)]
    empty = [_prediction(value, detected=False) for value in (1, 2, 3)]
    result = compute_paired_map_curve_auc_bootstrap(
        [perfect, one_miss],
        [two_miss, empty],
        targets,
        coordinates=[0.5, 1.5],
        comparator_name="gaussian",
        n_bootstrap=12,
        seed=23,
        category_ids=[1],
    )
    metric = result["metrics"]["map50_95"]
    difference = [record["estimate"] for record in metric["paired_difference_curve"]]
    expected_auc = 0.5 * (difference[0] + difference[1])
    assert metric["paired_physical_minus_comparator_auc"]["estimate"] == pytest.approx(expected_auc)

    identical = compute_paired_map_curve_auc_bootstrap(
        [perfect, one_miss],
        [perfect, one_miss],
        targets,
        coordinates=[0.5, 1.5],
        n_bootstrap=4,
        seed=3,
        category_ids=[1],
    )
    assert (
        identical["metrics"]["map50_95"]["paired_physical_minus_comparator_auc"]["estimate"] == 0.0
    )


def test_one_image_runner_preserves_native_coordinates_and_neutral_distinction(tmp_path) -> None:
    from PIL import Image

    root = tmp_path / "coco"
    image_dir = root / "images" / "val2017"
    annotation_dir = root / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    image = np.zeros((16, 20, 3), dtype=np.uint8)
    image[2:6, 2:10] = 255
    Image.fromarray(image).save(image_dir / "one.png")
    annotation = {
        "images": [{"id": 9, "file_name": "one.png", "height": 16, "width": 20}],
        "annotations": [
            {"id": 1, "image_id": 9, "category_id": 3, "bbox": [2, 2, 8, 4], "area": 32}
        ],
        "categories": [{"id": 3, "name": "object"}],
    }
    (annotation_dir / "instances_val2017.json").write_text(json.dumps(annotation), encoding="utf-8")
    subset = load_native_coco_subset(root, max_images=1)
    profile = synthetic_ldr_profile()
    binding = ConditionBinding.from_profile(profile, (0,))
    conditions = [
        BaselineCondition(binding, BaselineKind.UNTOUCHED_INPUT),
        BaselineCondition(binding, BaselineKind.MODELED_NEUTRAL),
    ]

    def detect_batch(inputs):
        outputs = []
        for detector_input in inputs:
            height, width = detector_input.geometry.input_shape
            native_box = np.array([[2.0, 2.0, 10.0, 6.0]])
            camera_box = native_box * np.array([width / 20, height / 16] * 2)
            outputs.append(
                {
                    "boxes": detector_input.geometry.forward_boxes(camera_box).tolist(),
                    "labels": [3],
                    "scores": [0.99],
                }
            )
        return outputs

    result = run_coco_ldr_benchmark(
        subset=subset,
        profile=profile,
        conditions=conditions,
        preprocessing=LetterboxConfig((16, 16), 0.5),
        model=_fixture_model(),
        detect_batch=detect_batch,
        execution=_fixture_execution(),
    )
    assert [record["metrics"]["map50"] for record in result.condition_metrics] == pytest.approx(
        [1.0, 1.0]
    )
    assert result.evaluation.records[0].camera_provenance["stage_graph"][0]["name"] == (
        "untouched_input_bypass"
    )
    assert result.evaluation.records[1].camera_provenance["stage_graph"][0]["name"] == (
        "inverse_srgb"
    )
