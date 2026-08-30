"""Inspect, plan, run, and analyze the frozen native-COCO publication study.

The ``protocol`` command prints the dataset-free reproduction contract. The
``plan`` command is the plan-only path: it materializes every camera arm,
mechanism match, detector allocation, analysis choice, and execution cell
without loading a detector. The ``run`` command executes one allocated
detector worker and automatically resumes only byte-identical manifests and
validated prediction shards. Add ``--dry-run`` to either command to perform
read-only validation and scheduling.
The ``analyze`` command independently revalidates the complete semantic run
layout before publishing the frozen statistics as content-addressed JSON and
tidy CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..eval.coco import load_native_coco_subset
from ..eval.protocol import (
    configure_deterministic_inference,
    parse_float_grid,
)
from ..eval.study import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_SHARD_SIZE,
    MAX_BOOTSTRAP_SEED,
    PUBLICATION_DEFOCUS_GRID,
    CocoStudyPlan,
    DetectorAllocation,
    build_publication_study_plan,
    execute_study_detector_run,
    load_study_plan,
    materialize_study_plan,
    publication_detector_allocations,
    publication_reproduction_contract,
)
from ..eval.study_analysis import analyze_and_publish_completed_study
from ..reference_profiles import (
    synthetic_coco_ldr_native_profile,
    synthetic_coco_ldr_native_replication_profile,
)


def _comma_grid(values: Sequence[float]) -> str:
    return ",".join(format(value, "g") for value in values)


def _bootstrap_replicates_argument(value: str) -> int:
    try:
        replicates = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bootstrap replicates must be an integer") from exc
    if replicates < 2:
        raise argparse.ArgumentTypeError("bootstrap replicates must be at least two")
    return replicates


def _bootstrap_seed_argument(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bootstrap seed must be an integer") from exc
    if seed < 0 or seed > MAX_BOOTSTRAP_SEED:
        raise argparse.ArgumentTypeError(
            f"bootstrap seed must be between 0 and {MAX_BOOTSTRAP_SEED}"
        )
    return seed


def build_parser() -> argparse.ArgumentParser:
    """Build the importable study command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "protocol",
        help="print the dataset-free publication reproduction contract",
    )

    plan = commands.add_parser(
        "plan",
        help="materialize the frozen study plan only; no detector is loaded",
    )
    plan.add_argument("--coco-root", type=Path, required=True)
    plan.add_argument("--output-plan", type=Path, required=True)
    plan.add_argument(
        "--profiles",
        default="primary,replication",
        help="ordered set drawn from: primary, replication",
    )
    plan.add_argument(
        "--defocus-waves",
        default=_comma_grid(PUBLICATION_DEFOCUS_GRID),
        help="strictly increasing positive reference-edge wave grid",
    )
    plan.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="default is the complete COCO val2017 split",
    )
    plan.add_argument("--image-offset", type=int, default=0)
    plan.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    plan.add_argument(
        "--bootstrap-replicates",
        type=_bootstrap_replicates_argument,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    plan.add_argument(
        "--bootstrap-seed",
        type=_bootstrap_seed_argument,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    plan.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and validate the plan without writing it",
    )

    run = commands.add_parser(
        "run",
        help="execute or resume one detector allocation from a frozen plan",
    )
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--coco-root", type=Path, required=True)
    run.add_argument("--detector", required=True)
    run.add_argument(
        "--artifact",
        type=Path,
        help="checkpoint file, or the pinned local model directory for DETR",
    )
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument(
        "--repository-root",
        type=Path,
        help="source checkout for Git provenance; omit to auto-detect a Git checkout",
    )
    run.add_argument(
        "--device",
        default=None,
        help="must equal the frozen allocation device; defaults to that device",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="validate dataset identity and print the frozen schedule without inference or writes",
    )
    run.add_argument(
        "--quiet-progress",
        action="store_true",
        help="suppress per-cell and per-shard JSON progress records on stderr",
    )

    analyze = commands.add_parser(
        "analyze",
        help="validate every completed study artifact and publish the frozen analysis",
    )
    analyze.add_argument("--plan", type=Path, required=True)
    analyze.add_argument("--coco-root", type=Path, required=True)
    analyze.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="completed detector-run output root",
    )
    analyze.add_argument(
        "--analysis-output",
        type=Path,
        help="default is OUTPUT_ROOT/analysis",
    )
    analyze.add_argument(
        "--repository-root",
        type=Path,
        help="source checkout for Git provenance; omit to auto-detect a Git checkout",
    )
    analyze.add_argument(
        "--quiet-progress",
        action="store_true",
        help="suppress JSON progress records on stderr",
    )
    analyze.add_argument(
        "--scratch-work-dir",
        type=Path,
        help=("external arm-checkpoint workspace; must be outside detector and analysis outputs"),
    )
    analyze.add_argument(
        "--analysis-workers",
        type=int,
        choices=(1, 2),
        default=1,
        help="deterministic arm workers; two requires --scratch-work-dir",
    )
    return parser


def _protocol_command() -> int:
    print(json.dumps(publication_reproduction_contract(), allow_nan=False, sort_keys=True))
    return 0


def _profile_selection(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("profiles must be a comma-separated string")
    names = tuple(token.strip() for token in text.split(","))
    if not names or any(not name for name in names):
        raise ValueError("profiles must be a nonempty comma-separated selection")
    if len(set(names)) != len(names):
        raise ValueError("profiles must not contain duplicates")
    factories = {
        "primary": synthetic_coco_ldr_native_profile,
        "replication": synthetic_coco_ldr_native_replication_profile,
    }
    unknown = set(names).difference(factories)
    if unknown:
        raise ValueError(f"unknown publication profiles: {sorted(unknown)}")
    return {name: factories[name]() for name in names}


def _repository_root(value: Path | None) -> Path | None:
    """Use the package checkout when identifiable, otherwise leave Git unavailable."""

    if value is not None:
        return value.resolve()
    candidate = Path(__file__).resolve().parents[2]
    package_root = Path(__file__).resolve().parents[1]
    if (
        (candidate / ".git").exists()
        and (candidate / "pyproject.toml").is_file()
        and (candidate / "phycam_eval").resolve() == package_root
    ):
        return candidate
    return None


def _plan_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive when supplied")
    if args.image_offset < 0:
        parser.error("--image-offset must be nonnegative")
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")
    if args.bootstrap_replicates < 2:
        parser.error("--bootstrap-replicates must be at least two")
    if args.bootstrap_seed < 0 or args.bootstrap_seed > MAX_BOOTSTRAP_SEED:
        parser.error(f"--bootstrap-seed must be between 0 and {MAX_BOOTSTRAP_SEED}")
    try:
        profiles = _profile_selection(args.profiles)
        defocus = parse_float_grid(
            args.defocus_waves,
            label="defocus_waves",
            minimum=0.0,
            minimum_inclusive=False,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if tuple(sorted(defocus)) != defocus:
        parser.error("--defocus-waves must be strictly increasing")
    subset = load_native_coco_subset(
        args.coco_root,
        split="val2017",
        max_images=args.max_images,
        image_offset=args.image_offset,
        eager=False,
    )
    allocations = publication_detector_allocations(
        tuple(profiles),
        primary_profile_id="primary" if "primary" in profiles else tuple(profiles)[0],
        shard_size=args.shard_size,
    )
    try:
        plan = build_publication_study_plan(
            dataset=subset.identity,
            image_ids=subset.image_ids,
            profiles=profiles,
            primary_profile_id="primary" if "primary" in profiles else tuple(profiles)[0],
            detector_allocations=allocations,
            defocus_grid=defocus,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if not args.dry_run:
        materialize_study_plan(args.output_plan, plan)
    summary = {
        "study_plan_sha256": plan.study_plan_sha256,
        "dry_run": bool(args.dry_run),
        "images": len(plan.image_ids),
        "profiles": len(plan.profiles),
        "arms": len(plan.record["arms"]),
        "detectors": len(plan.allocations),
        "execution_cells": len(plan.cells()),
        "output_plan": None if args.dry_run else str(args.output_plan),
    }
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


def _adapter(
    allocation: DetectorAllocation,
    artifact: Path,
    *,
    device: str,
) -> Any:
    from ..eval.detectors import (
        HuggingFaceDETRAdapter,
        TorchvisionFasterRCNNAdapter,
        TorchvisionRetinaNetAdapter,
        UltralyticsYOLOAdapter,
    )

    if allocation.detector_id == "yolov8n":
        if allocation.nms_iou_threshold is None:
            raise RuntimeError("YOLO allocation is missing its NMS threshold")
        return UltralyticsYOLOAdapter(
            artifact,
            device=device,
            confidence=allocation.confidence_threshold,
            iou=allocation.nms_iou_threshold,
            maximum_detections=int(
                allocation.backend_settings["maximum_detections_before_coco_limit"]
            ),
            evaluation_maximum_detections=int(allocation.backend_settings["maximum_detections"]),
        )
    if allocation.detector_id == "fasterrcnn_r50_fpn":
        if allocation.nms_iou_threshold is None:
            raise RuntimeError("Faster R-CNN allocation is missing its NMS threshold")
        return TorchvisionFasterRCNNAdapter(
            artifact,
            input_shape=allocation.input_shape,
            device=device,
            confidence=allocation.confidence_threshold,
            nms_iou=allocation.nms_iou_threshold,
        )
    if allocation.detector_id == "retinanet_r50_fpn_v2":
        if allocation.nms_iou_threshold is None:
            raise RuntimeError("RetinaNet allocation is missing its NMS threshold")
        return TorchvisionRetinaNetAdapter(
            artifact,
            input_shape=allocation.input_shape,
            device=device,
            confidence=allocation.confidence_threshold,
            nms_iou=allocation.nms_iou_threshold,
        )
    if allocation.detector_id == "detr_r50":
        if allocation.nms_iou_threshold is not None:
            raise RuntimeError("DETR allocation must not declare NMS")
        return HuggingFaceDETRAdapter(
            artifact,
            input_shape=allocation.input_shape,
            device=device,
            confidence=allocation.confidence_threshold,
        )
    raise ValueError(
        f"the CLI has no adapter factory for custom detector allocation {allocation.detector_id!r}"
    )


def _run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    plan: CocoStudyPlan = load_study_plan(args.plan)
    try:
        allocation = plan.allocation(args.detector)
    except (TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.artifact is None and not args.dry_run:
        parser.error("--artifact is required unless --dry-run is used")
    selected_device = allocation.requested_device if args.device is None else args.device
    if selected_device != allocation.requested_device:
        parser.error(
            f"--device must equal the frozen {allocation.detector_id} allocation device "
            f"{allocation.requested_device!r}"
        )

    subset = load_native_coco_subset(
        args.coco_root,
        split="val2017",
        ordered_image_ids=plan.image_ids,
        max_images=None,
        eager=False,
    )
    adapter = None
    if not args.dry_run:
        try:
            configure_deterministic_inference(plan.record["design"]["inference_execution"])
            assert args.artifact is not None
            adapter = _adapter(allocation, args.artifact, device=selected_device)
        except (ImportError, FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            parser.error(str(exc))

    def report(event: Any) -> None:
        print(json.dumps(dict(event), allow_nan=False, sort_keys=True), file=sys.stderr, flush=True)

    summary = execute_study_detector_run(
        plan=plan,
        subset=subset,
        detector_id=allocation.detector_id,
        output_root=args.output_root,
        adapter=adapter,
        dry_run=args.dry_run,
        repository_root=_repository_root(args.repository_root),
        progress=None if args.quiet_progress or args.dry_run else report,
    )
    print(json.dumps(summary.to_dict(), allow_nan=False, sort_keys=True))
    return 0


def _analyze_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    destination = (
        args.output_root / "analysis" if args.analysis_output is None else args.analysis_output
    )

    def report(event: Any) -> None:
        print(json.dumps(dict(event), allow_nan=False, sort_keys=True), file=sys.stderr, flush=True)

    if args.analysis_workers == 2 and args.scratch_work_dir is None:
        parser.error("--analysis-workers 2 requires --scratch-work-dir")
    try:
        publication = analyze_and_publish_completed_study(
            plan_path=args.plan,
            coco_root=args.coco_root,
            output_root=args.output_root,
            analysis_output=destination,
            repository_root=_repository_root(args.repository_root),
            progress=None if args.quiet_progress else report,
            scratch_work_dir=args.scratch_work_dir,
            analysis_workers=args.analysis_workers,
        )
    except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    summary = {
        "study_plan_sha256": publication["study_plan_sha256"],
        "study_analysis_sha256": publication["study_analysis_sha256"],
        "publication_index_sha256": publication["publication_index_sha256"],
        "analysis_output": str(destination),
        "artifact_count": len(publication["artifacts"]),
    }
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a plan-only or resumable detector-worker command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "protocol":
        return _protocol_command()
    if args.command == "plan":
        return _plan_command(args, parser)
    if args.command == "run":
        return _run_command(args, parser)
    if args.command == "analyze":
        return _analyze_command(args, parser)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through the importable main
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
