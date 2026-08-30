"""Command smoke test for the retained local study workflow."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from phycam_eval._canonical import canonical_sha256
from phycam_eval.cli import coco_study
from phycam_eval.eval.study import (
    _study_execution_engine_record,
    _validate_execution_engine_record,
    load_study_plan,
)
from phycam_eval.eval.study_analysis import _select_primary_curve_cells

pytest.importorskip("PIL")


def test_study_cli_describes_the_publication_reproduction_contract(capsys) -> None:
    assert coco_study.main(["protocol"]) == 0
    contract = json.loads(capsys.readouterr().out)
    supplied_digest = contract.pop("protocol_contract_sha256")

    assert supplied_digest == canonical_sha256(contract)
    assert contract["dataset"]["expected_image_count"] == 5_000
    assert contract["design"]["expected_execution_cell_count"] == 67
    assert contract["design"]["ordered_edge_waves_ref"] == [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    assert [
        allocation["detector_id"] for allocation in contract["design"]["detector_allocations"]
    ] == ["yolov8n", "fasterrcnn_r50_fpn", "retinanet_r50_fpn_v2", "detr_r50"]
    assert contract["analysis_protocol"]["uncertainty"]["replicates"] == 2_000
    assert contract["analysis_protocol"]["uncertainty"]["seed"] == 20_260_715
    assert contract["evidence_scope"] == {
        "matching_rerun_tier": "publication_protocol_reproduction",
        "confirmatory_eligible": False,
        "historical_run_attested_here": False,
    }


def test_local_provenance_does_not_overclaim_adapter_or_repository(tmp_path, monkeypatch) -> None:
    class FakeSubset:
        loader_attested = False

    class FakeAdapter:
        pass

    def fake_runtime(**_kwargs):
        return {}

    def fake_runner(**_kwargs):
        return None

    def fake_merger(*_args, **_kwargs):
        return None

    record = _study_execution_engine_record(
        detector_id="yolov8n",
        subset=FakeSubset(),
        adapter=FakeAdapter(),
        runtime_identity_factory=fake_runtime,
        shard_runner=fake_runner,
        shard_merger=fake_merger,
    )
    assert record["component_checks"]["exact_detector_adapter_class"] is False
    assert _validate_execution_engine_record(record, detector_id="yolov8n") == record

    wheel_module = tmp_path / "site-packages" / "phycam_eval" / "cli" / "coco_study.py"
    monkeypatch.setattr(coco_study, "__file__", str(wheel_module))
    assert coco_study._repository_root(None) is None


def test_study_cli_builds_the_complete_local_publication_plan(tmp_path, capsys) -> None:
    from PIL import Image

    root = tmp_path / "coco"
    image_dir = root / "images" / "val2017"
    annotation_dir = root / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(image_dir / "one.png")
    annotation = {
        "images": [{"id": 1, "file_name": "one.png", "height": 8, "width": 10}],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }
    (annotation_dir / "instances_val2017.json").write_text(json.dumps(annotation), encoding="utf-8")

    output_plan = tmp_path / "study-plan.json"
    assert (
        coco_study.main(
            [
                "plan",
                "--coco-root",
                str(root),
                "--output-plan",
                str(output_plan),
                "--max-images",
                "1",
                "--bootstrap-replicates",
                "2",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_run"] is False
    assert summary["images"] == 1
    assert summary["profiles"] == 2
    assert summary["arms"] == 52
    assert summary["execution_cells"] == 67

    plan = load_study_plan(output_plan)
    assert plan.record["evidence_tier"]["publication_protocol_match"] is False
    assert plan.confirmatory_eligible is False
    checks = plan.record["evidence_tier"]["checks"]
    assert checks["exact_primary_and_replication_profiles"] is True
    assert checks["exact_default_detector_allocations"] is True
    assert checks["exact_canonical_condition_arms"] is True
    assert checks["complete_default_execution_cell_allocation"] is True
    cells = plan.cells()
    counts = Counter(str(cell["detector_id"]) for cell in cells)
    assert counts == {
        "yolov8n": 52,
        "fasterrcnn_r50_fpn": 5,
        "retinanet_r50_fpn_v2": 5,
        "detr_r50": 5,
    }
    yolo_profiles = Counter(str(cell["profile_id"]) for cell in plan.cells("yolov8n"))
    assert yolo_profiles == {"primary": 26, "replication": 26}
    expected_secondary_arms = {
        "untouched_input",
        "modeled_neutral",
        "physical_w_0p5",
        "physical_w_1p5",
        "physical_w_3",
    }
    for detector_id in (
        "fasterrcnn_r50_fpn",
        "retinanet_r50_fpn_v2",
        "detr_r50",
    ):
        assert {str(cell["arm_id"]) for cell in plan.cells(detector_id)} == (
            expected_secondary_arms
        )

    cells_by_key = {
        (
            str(cell["detector_id"]),
            str(cell["profile_id"]),
            str(cell["arm_id"]),
        ): cell
        for cell in cells
    }
    coordinates, physical, gaussian = _select_primary_curve_cells(plan, cells_by_key)
    assert coordinates == (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    assert [cell["arm_id"] for cell in physical] == [
        "physical_w_0p5",
        "physical_w_0p75",
        "physical_w_1",
        "physical_w_1p5",
        "physical_w_2",
        "physical_w_3",
    ]
    assert [cell["arm_id"] for cell in gaussian] == [
        "comparator_gaussian_w_0p5",
        "comparator_gaussian_w_0p75",
        "comparator_gaussian_w_1",
        "comparator_gaussian_w_1p5",
        "comparator_gaussian_w_2",
        "comparator_gaussian_w_3",
    ]

    output_root = tmp_path / "runs"
    assert (
        coco_study.main(
            [
                "run",
                "--plan",
                str(output_plan),
                "--coco-root",
                str(root),
                "--detector",
                "yolov8n",
                "--output-root",
                str(output_root),
                "--dry-run",
            ]
        )
        == 0
    )
    run_summary = json.loads(capsys.readouterr().out)
    assert run_summary["dry_run"] is True
    assert run_summary["cell_count"] == 52
    assert run_summary["shard_count"] == 52
    assert run_summary["image_inferences"] == 52
    assert not output_root.exists()
