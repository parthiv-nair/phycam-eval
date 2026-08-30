#!/usr/bin/env python3
"""Generate numerical-validation JSON and CSVs for the synthetic COCO study."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from phycam_eval.eval.coco import load_native_coco_subset
from phycam_eval.profiles import CameraProfile
from phycam_eval.reference_profiles import (
    synthetic_coco_ldr_native_profile,
    synthetic_coco_ldr_native_replication_profile,
)
from phycam_eval.validation_evidence import (
    build_validation_evidence,
    verify_validation_evidence,
    write_validation_evidence,
)

DEFAULT_OUTPUT_ROOT = Path("results") / "scientific_validation"
DEFAULT_EDGE_WAVES = (0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _edge_waves(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("edge waves must be comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one edge-wave value is required")
    return result


def _selected_profile(name: str) -> CameraProfile:
    if name == "primary":
        return synthetic_coco_ldr_native_profile()
    if name == "replication":
        return synthetic_coco_ldr_native_replication_profile()
    raise ValueError("unsupported validation profile")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco-root",
        type=Path,
        required=True,
        help="COCO root containing annotations/ and images/val2017/",
    )
    parser.add_argument(
        "--profile",
        choices=("primary", "replication"),
        default="primary",
        help="publication camera profile to validate (default: primary)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(f"artifact directory (default: {DEFAULT_OUTPUT_ROOT}/<profile>)"),
    )
    parser.add_argument(
        "--edge-waves",
        type=_edge_waves,
        default=DEFAULT_EDGE_WAVES,
        help="ordered physical severities in reference waves (comma-separated)",
    )
    parser.add_argument(
        "--frequency-samples",
        type=int,
        default=2049,
        help="samples from DC through image Nyquist (default: 2049)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the three generated files in an existing output directory",
    )
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="return success after writing evidence even when a numerical criterion fails",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    output = arguments.output or DEFAULT_OUTPUT_ROOT / arguments.profile
    dataset = load_native_coco_subset(
        arguments.coco_root,
        split="val2017",
        max_images=None,
        eager=False,
    )
    report = build_validation_evidence(
        _selected_profile(arguments.profile),
        arguments.edge_waves,
        dataset=dataset,
        frequency_sample_count=arguments.frequency_samples,
    )
    artifacts = write_validation_evidence(
        report,
        output,
        overwrite=arguments.overwrite,
    )
    verified = verify_validation_evidence(
        artifacts.root,
        expected_evidence_sha256=artifacts.evidence_sha256,
        expected_dataset=dataset,
    )
    print(
        f"wrote validation evidence to {artifacts.root} "
        f"for profile={arguments.profile} "
        f"dataset_sha256={dataset.identity['dataset_sha256']} "
        f"with evidence_sha256={verified['evidence_sha256']}; "
        f"all_passed={verified['summary']['all_passed']}"
    )
    return 0 if report["summary"]["all_passed"] or arguments.allow_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
