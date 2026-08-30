"""Deterministic numerical-validation evidence for the synthetic camera study.

The report produced here validates implementation properties of the declared
numerical model.  It is deliberately not a hardware-calibration certificate.
Every acceptance threshold is serialized with the measurements, and full
transfer curves are retained for paper figures and independent inspection.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray
from scipy.special import j1 as scipy_bessel_j1

from ._canonical import canonical_sha256, json_value
from .capture import LDRCaptureSeverity, build_ldr_pipeline
from .comparators.capture import comparator_config_from_dict
from .comparators.gaussian import GaussianComparatorConfig, gaussian_kernel
from .comparators.matching import (
    MechanismMatch,
    luminance_weighted_kernel,
    match_common_neutral_comparators,
)
from .comparators.transfer_families import (
    QuadraticCosineComparatorConfig,
    SampledIncoherentComparatorConfig,
    quadratic_cosine_response,
    sampled_incoherent_kernel,
    sampled_incoherent_retained_energy,
)
from .eval.coco import LazyNativeCOCOSubset
from .eval.mtf import first_downward_crossing
from .formation import render_joint_photosite_exposure
from .optics.defocus import DefocusConfig, DefocusModel, build_defocus_model
from .optics.psf import ContinuousPSF
from .optics.pupil import PupilSampling
from .optics.sampling import CellAverageTransferKernel, collapse_cell_average_transfer
from .profiles import CameraProfile
from .readout.timing import ReadoutTiming
from .source_grid import GridGeometry

FloatArray = NDArray[np.float64]

_REPORT_SCHEMA_VERSION = 5
_IMPLEMENTATION_ID = "phycam-scientific-validation-evidence-v5"
_DCT_SHAPE_CONTRACT_SCHEMA_VERSION = 1
_DCT_SHAPE_CONTRACT_KEYS = {
    "schema_version",
    "scope",
    "loader_class",
    "loader_attested",
    "dataset",
    "split",
    "dataset_sha256",
    "dataset_image_selection_sha256",
    "ordered_image_count",
    "ordered_image_ids_sha256",
    "annotation_sha256",
    "image_shape_records",
    "image_shape_records_sha256",
    "unique_axis_dimensions",
    "unique_axis_dimensions_sha256",
    "axis_dimension_count",
    "axis_dimension_min",
    "axis_dimension_max",
    "dimension_source",
    "record_sha256",
}
_ATTESTED_DCT_SHAPE_SCOPE = "loader_attested_native_coco_subset"
_DECLARED_DCT_SHAPE_SCOPE = "declared_axis_dimensions_not_dataset_attested"
_LAZY_COCO_CLASS_ID = "phycam_eval.eval.coco.LazyNativeCOCOSubset"


def _sha256_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _axis_dimensions(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be an ordered sequence of integers")
    normalized: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name}[{index}] must be an integer")
        dimension = int(value)
        if dimension < 2:
            raise ValueError(f"{name}[{index}] must be at least two")
        normalized.append(dimension)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if tuple(sorted(set(normalized))) != tuple(normalized):
        raise ValueError(f"{name} must be strictly increasing and duplicate-free")
    return tuple(normalized)


def _shape_contract_payload(
    *,
    scope: str,
    dimensions: Sequence[int],
    loader_class: str | None,
    loader_attested: bool,
    dataset: str | None,
    split: str | None,
    dataset_sha256: str | None,
    dataset_image_selection_sha256: str | None,
    ordered_image_count: int | None,
    ordered_image_ids_sha256: str | None,
    annotation_sha256: str | None,
    image_shape_records: list[dict[str, int]] | None,
    image_shape_records_sha256: str | None,
    dimension_source: str,
) -> dict[str, Any]:
    axis_dimensions = _axis_dimensions(dimensions, name="unique_axis_dimensions")
    payload = {
        "schema_version": _DCT_SHAPE_CONTRACT_SCHEMA_VERSION,
        "scope": scope,
        "loader_class": loader_class,
        "loader_attested": loader_attested,
        "dataset": dataset,
        "split": split,
        "dataset_sha256": dataset_sha256,
        "dataset_image_selection_sha256": dataset_image_selection_sha256,
        "ordered_image_count": ordered_image_count,
        "ordered_image_ids_sha256": ordered_image_ids_sha256,
        "annotation_sha256": annotation_sha256,
        "image_shape_records": image_shape_records,
        "image_shape_records_sha256": image_shape_records_sha256,
        "unique_axis_dimensions": list(axis_dimensions),
        "unique_axis_dimensions_sha256": canonical_sha256(list(axis_dimensions)),
        "axis_dimension_count": len(axis_dimensions),
        "axis_dimension_min": axis_dimensions[0],
        "axis_dimension_max": axis_dimensions[-1],
        "dimension_source": dimension_source,
    }
    return {**payload, "record_sha256": canonical_sha256(payload)}


def _dataset_shape_contract(dataset: LazyNativeCOCOSubset) -> dict[str, Any]:
    """Bind executed DCT axis dimensions to the exact byte-verifying loader."""

    if type(dataset) is not LazyNativeCOCOSubset or not dataset.loader_attested:
        raise ValueError(
            "dataset-bound validation requires the exact loader-attested LazyNativeCOCOSubset"
        )
    identity = json_value(dataset.identity)
    if identity.get("dataset") != "COCO" or identity.get("split") != "val2017":
        raise ValueError("dataset-bound validation requires native COCO val2017")
    if tuple(identity.get("ordered_image_ids", ())) != dataset.image_ids:
        raise ValueError("dataset identity and loaded image order disagree")
    selection = identity.get("image_selection")
    annotation = identity.get("annotation_artifact")
    if not isinstance(selection, Mapping) or not isinstance(annotation, Mapping):
        raise ValueError("dataset identity is missing selection or annotation provenance")
    records = [
        {"image_id": image_id, "height": shape[0], "width": shape[1]}
        for image_id, shape in zip(dataset.image_ids, dataset.image_shapes, strict=True)
    ]
    dimensions = sorted(
        {value for shape in dataset.image_shapes for value in (int(shape[0]), int(shape[1]))}
    )
    return _shape_contract_payload(
        scope=_ATTESTED_DCT_SHAPE_SCOPE,
        dimensions=dimensions,
        loader_class=_LAZY_COCO_CLASS_ID,
        loader_attested=True,
        dataset="COCO",
        split="val2017",
        dataset_sha256=_sha256_digest(
            identity.get("dataset_sha256"), name="dataset dataset_sha256"
        ),
        dataset_image_selection_sha256=_sha256_digest(
            selection.get("selection_sha256"),
            name="dataset image selection_sha256",
        ),
        ordered_image_count=len(dataset.image_ids),
        ordered_image_ids_sha256=canonical_sha256({"ordered_image_ids": list(dataset.image_ids)}),
        annotation_sha256=_sha256_digest(
            annotation.get("sha256"), name="dataset annotation sha256"
        ),
        image_shape_records=records,
        image_shape_records_sha256=canonical_sha256(records),
        dimension_source=(
            "COCO metadata dimensions crosschecked against the exact selected encoded "
            "image bytes by load_native_coco_subset"
        ),
    )


def _declared_shape_contract(dimensions: Sequence[int]) -> dict[str, Any]:
    return _shape_contract_payload(
        scope=_DECLARED_DCT_SHAPE_SCOPE,
        dimensions=dimensions,
        loader_class=None,
        loader_attested=False,
        dataset=None,
        split=None,
        dataset_sha256=None,
        dataset_image_selection_sha256=None,
        ordered_image_count=None,
        ordered_image_ids_sha256=None,
        annotation_sha256=None,
        image_shape_records=None,
        image_shape_records_sha256=None,
        dimension_source="caller-declared dimensions; no dataset identity is claimed",
    )


def _normalize_shape_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("DCT-I axis shape contract must be a mapping")
    record = json_value(value)
    if set(record) != _DCT_SHAPE_CONTRACT_KEYS:
        raise ValueError("DCT-I axis shape contract has missing or unknown fields")
    if record.get("schema_version") != _DCT_SHAPE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported DCT-I axis shape contract schema")
    supplied_hash = _sha256_digest(
        record.get("record_sha256"), name="DCT-I shape contract record_sha256"
    )
    payload = {key: item for key, item in record.items() if key != "record_sha256"}
    if canonical_sha256(payload) != supplied_hash:
        raise ValueError("DCT-I axis shape contract identity mismatch")
    dimensions = _axis_dimensions(
        record.get("unique_axis_dimensions"), name="unique_axis_dimensions"
    )
    if record.get("unique_axis_dimensions_sha256") != canonical_sha256(list(dimensions)):
        raise ValueError("DCT-I unique axis dimension identity mismatch")
    if (
        record.get("axis_dimension_count") != len(dimensions)
        or record.get("axis_dimension_min") != dimensions[0]
        or record.get("axis_dimension_max") != dimensions[-1]
    ):
        raise ValueError("DCT-I axis dimension summary is inconsistent")
    scope = record.get("scope")
    if scope == _DECLARED_DCT_SHAPE_SCOPE:
        expected = _declared_shape_contract(dimensions)
    elif scope == _ATTESTED_DCT_SHAPE_SCOPE:
        if (
            record.get("loader_class") != _LAZY_COCO_CLASS_ID
            or record.get("loader_attested") is not True
        ):
            raise ValueError("dataset-bound DCT-I contract lacks exact-loader attestation")
        if record.get("dataset") != "COCO" or record.get("split") != "val2017":
            raise ValueError("dataset-bound DCT-I contract must identify COCO val2017")
        for name in (
            "dataset_sha256",
            "dataset_image_selection_sha256",
            "ordered_image_ids_sha256",
            "annotation_sha256",
            "image_shape_records_sha256",
        ):
            _sha256_digest(record.get(name), name=f"DCT-I shape contract {name}")
        shape_records = record.get("image_shape_records")
        count = record.get("ordered_image_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("dataset-bound DCT-I ordered image count must be positive")
        if not isinstance(shape_records, list) or len(shape_records) != count:
            raise ValueError("dataset-bound DCT-I image shapes do not align with image count")
        image_ids: list[int] = []
        observed_dimensions: set[int] = set()
        for index, shape in enumerate(shape_records):
            if not isinstance(shape, Mapping) or set(shape) != {"image_id", "height", "width"}:
                raise ValueError(f"DCT-I image shape record {index} is noncanonical")
            image_id = shape.get("image_id")
            height = shape.get("height")
            width = shape.get("width")
            if any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in (image_id, height, width)
            ):
                raise TypeError("DCT-I image shape fields must be integers")
            if image_id < 0 or height < 2 or width < 2:
                raise ValueError("DCT-I image shape fields are outside their valid range")
            image_ids.append(image_id)
            observed_dimensions.update((height, width))
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("DCT-I image shape records contain duplicate image IDs")
        if sorted(observed_dimensions) != list(dimensions):
            raise ValueError("DCT-I unique dimensions do not derive from the image shapes")
        if record.get("image_shape_records_sha256") != canonical_sha256(shape_records):
            raise ValueError("DCT-I image shape record identity mismatch")
        if record.get("ordered_image_ids_sha256") != canonical_sha256(
            {"ordered_image_ids": image_ids}
        ):
            raise ValueError("DCT-I ordered image identity mismatch")
        expected = _shape_contract_payload(
            scope=_ATTESTED_DCT_SHAPE_SCOPE,
            dimensions=dimensions,
            loader_class=_LAZY_COCO_CLASS_ID,
            loader_attested=True,
            dataset="COCO",
            split="val2017",
            dataset_sha256=record["dataset_sha256"],
            dataset_image_selection_sha256=record["dataset_image_selection_sha256"],
            ordered_image_count=count,
            ordered_image_ids_sha256=record["ordered_image_ids_sha256"],
            annotation_sha256=record["annotation_sha256"],
            image_shape_records=shape_records,
            image_shape_records_sha256=record["image_shape_records_sha256"],
            dimension_source=(
                "COCO metadata dimensions crosschecked against the exact selected encoded "
                "image bytes by load_native_coco_subset"
            ),
        )
    else:
        raise ValueError("unsupported DCT-I axis shape contract scope")
    if record != expected:
        raise ValueError("DCT-I axis shape contract is noncanonical")
    return record


@dataclass(frozen=True, slots=True)
class ValidationCriteria:
    """Predeclared numerical acceptance thresholds carried by every report."""

    flux_absolute_error_max: float = 5e-12
    symmetry_relative_linf_max: float = 5e-12
    convergence_declared_refined_mtf_linf_max: float = 0.005
    convergence_declared_refined_mtf50_relative_error_max: float = 0.01
    convergence_error_contraction_ratio_max: float = 0.75
    implemented_match_mtf50_relative_error_max: float = 0.005
    independent_tent_quadrature_linf_max: float = 2e-8
    independent_tent_retained_energy_absolute_error_max: float = 5e-9
    independent_equal_grid_formation_linf_max: float = 5e-13
    analytic_airy_center_relative_error_max: float = 1e-3
    analytic_airy_normalized_intensity_linf_max: float = 5e-4
    analytic_circular_aperture_mtf_linf_max: float = 1.2e-3

    def __post_init__(self) -> None:
        for name in (
            "flux_absolute_error_max",
            "symmetry_relative_linf_max",
            "convergence_declared_refined_mtf_linf_max",
            "convergence_declared_refined_mtf50_relative_error_max",
            "convergence_error_contraction_ratio_max",
            "implemented_match_mtf50_relative_error_max",
            "independent_tent_quadrature_linf_max",
            "independent_tent_retained_energy_absolute_error_max",
            "independent_equal_grid_formation_linf_max",
            "analytic_airy_center_relative_error_max",
            "analytic_airy_normalized_intensity_linf_max",
            "analytic_circular_aperture_mtf_linf_max",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.convergence_error_contraction_ratio_max >= 1.0:
            raise ValueError("convergence_error_contraction_ratio_max must be below one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "acceptance_scope": "numerical_implementation_only",
            "flux_absolute_error_max": self.flux_absolute_error_max,
            "symmetry_relative_linf_max": self.symmetry_relative_linf_max,
            "convergence_declared_refined_mtf_linf_max": (
                self.convergence_declared_refined_mtf_linf_max
            ),
            "convergence_declared_refined_mtf50_relative_error_max": (
                self.convergence_declared_refined_mtf50_relative_error_max
            ),
            "convergence_error_contraction_ratio_max": (
                self.convergence_error_contraction_ratio_max
            ),
            "implemented_match_mtf50_relative_error_max": (
                self.implemented_match_mtf50_relative_error_max
            ),
            "independent_tent_quadrature_linf_max": (self.independent_tent_quadrature_linf_max),
            "independent_tent_retained_energy_absolute_error_max": (
                self.independent_tent_retained_energy_absolute_error_max
            ),
            "independent_equal_grid_formation_linf_max": (
                self.independent_equal_grid_formation_linf_max
            ),
            "analytic_airy_center_relative_error_max": (
                self.analytic_airy_center_relative_error_max
            ),
            "analytic_airy_normalized_intensity_linf_max": (
                self.analytic_airy_normalized_intensity_linf_max
            ),
            "analytic_circular_aperture_mtf_linf_max": (
                self.analytic_circular_aperture_mtf_linf_max
            ),
            "full_curve_comparator_equivalence_required": False,
            "full_curve_diagnostics_role": (
                "descriptive mechanism diagnostics; comparators are matched only at MTF50"
            ),
        }


@dataclass(frozen=True, slots=True)
class SamplingLevel:
    """One pupil quadrature/FFT level in the numerical convergence ladder."""

    name: str
    pupil_grid_size: int
    pupil_fft_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        for field_name in ("pupil_grid_size", "pupil_fft_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{field_name} must be an integer")
            value = int(value)
            if value < 3 or value % 2 != 1:
                raise ValueError(f"{field_name} must be an odd integer of at least three")
            object.__setattr__(self, field_name, value)
        if self.pupil_fft_size < self.pupil_grid_size:
            raise ValueError("pupil_fft_size cannot be smaller than pupil_grid_size")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pupil_grid_size": self.pupil_grid_size,
            "pupil_fft_size": self.pupil_fft_size,
        }


@dataclass(frozen=True, slots=True)
class ValidationArtifactSet:
    """Paths for one written JSON/CSV numerical-evidence set."""

    root: Path
    evidence_sha256: str
    evidence_json: Path
    metrics_csv: Path
    curves_csv: Path


def default_sampling_ladder(profile: CameraProfile) -> tuple[SamplingLevel, ...]:
    """Return the declared sampling bracketed by one coarser/finer level."""

    if not isinstance(profile, CameraProfile):
        raise TypeError("profile must be a CameraProfile")
    grid = profile.optics.pupil_grid_size
    fft = profile.optics.pupil_fft_size
    coarse_grid = (grid + 1) // 2
    coarse_fft = (fft + 1) // 2
    if coarse_grid < 3 or coarse_fft < coarse_grid:
        raise ValueError("profile sampling is too small for the default convergence ladder")
    return (
        SamplingLevel("coarse", coarse_grid, coarse_fft),
        SamplingLevel("declared", grid, fft),
        SamplingLevel("refined", 2 * grid - 1, 2 * fft - 1),
    )


def _float_sequence(
    values: Sequence[float],
    *,
    name: str,
    positive: bool,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be an ordered sequence")
    result = tuple(float(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be nonempty and unique")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    if positive and not all(value > 0.0 for value in result):
        raise ValueError(f"{name} must contain only positive values")
    if not positive and not all(value >= 0.0 for value in result):
        raise ValueError(f"{name} must contain only nonnegative values")
    return result


def _array_sha256(values: ArrayLike) -> str:
    """Hash a numeric array under an endian-independent binary64 contract."""

    array = np.asarray(values)
    if np.iscomplexobj(array):
        canonical = np.ascontiguousarray(array, dtype=np.dtype("<c16"))
        dtype = "little-endian-complex128"
    else:
        canonical = np.ascontiguousarray(array, dtype=np.dtype("<f8"))
        dtype = "little-endian-float64"
    digest = hashlib.sha256()
    digest.update(b"phycam-array-sha256-v1\0")
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(value) for value in canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _symmetry_diagnostics(values: ArrayLike) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    scale = max(float(np.max(np.abs(array))), np.finfo(np.float64).tiny)
    return {
        "horizontal_relative_linf": float(np.max(np.abs(array - array[:, ::-1])) / scale),
        "vertical_relative_linf": float(np.max(np.abs(array - array[::-1, :])) / scale),
        "transpose_relative_linf": float(np.max(np.abs(array - array.T)) / scale),
    }


def _signed_axis_response(kernel: ArrayLike, frequency: FloatArray) -> tuple[FloatArray, float]:
    values = np.asarray(kernel, dtype=np.float64)
    if values.ndim != 2 or any(size <= 0 for size in values.shape):
        raise ValueError("kernel must be a nonempty two-dimensional array")
    line_spread = values.sum(axis=0, dtype=np.float64)
    positions = np.arange(line_spread.size, dtype=np.float64) - (line_spread.size - 1) / 2.0
    response = np.exp(-2j * np.pi * frequency[:, None] * positions[None, :]) @ line_spread
    dc = response[0]
    if abs(dc) <= 0.0:
        raise ValueError("kernel has zero DC response")
    response = response / dc
    imaginary_residual = float(np.max(np.abs(response.imag)))
    signed = np.asarray(response.real, dtype=np.float64)
    signed[0] = 1.0
    return signed, imaginary_residual


def _crossing_record(frequency: FloatArray, mtf: FloatArray, level: float) -> dict[str, Any]:
    value = first_downward_crossing(frequency, mtf, level=level)
    if math.isnan(value):
        return {
            "level": level,
            "value_cycles_per_pixel": None,
            "censoring": "right_at_nyquist",
            "nyquist_cycles_per_pixel": float(frequency[-1]),
            "mtf_at_nyquist": float(mtf[-1]),
        }
    return {
        "level": level,
        "value_cycles_per_pixel": value,
        "censoring": None,
        "nyquist_cycles_per_pixel": float(frequency[-1]),
        "mtf_at_nyquist": float(mtf[-1]),
    }


def _relative_crossing_error(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    left_value = left["value_cycles_per_pixel"]
    right_value = right["value_cycles_per_pixel"]
    if left_value is None or right_value is None:
        return None
    return abs(float(left_value) - float(right_value)) / float(right_value)


def _trapezoid(values: FloatArray, frequency: FloatArray) -> float:
    return float(
        np.sum(
            0.5 * (values[:-1] + values[1:]) * np.diff(frequency),
            dtype=np.float64,
        )
    )


def _curve_record(frequency: FloatArray, values: FloatArray, *, name: str) -> dict[str, Any]:
    payload = {
        "name": name,
        "frequency_unit": "cycles/pixel",
        "frequency": frequency.tolist(),
        "values": values.tolist(),
    }
    return {**payload, "curve_sha256": canonical_sha256(payload)}


def _independent_airy_normalized_intensity(
    dimensionless_radius: FloatArray,
) -> FloatArray:
    """Evaluate the clear-aperture Airy law without the production PSF helper."""

    radius = np.asarray(dimensionless_radius, dtype=np.float64)
    if radius.ndim != 1 or np.any(radius < 0.0) or not np.all(np.isfinite(radius)):
        raise ValueError("dimensionless Airy radii must be a finite nonnegative vector")
    argument = np.pi * radius
    amplitude = np.ones_like(argument)
    nonzero = argument != 0.0
    amplitude[nonzero] = 2.0 * scipy_bessel_j1(argument[nonzero]) / argument[nonzero]
    return np.asarray(np.square(amplitude), dtype=np.float64)


def _independent_circular_aperture_mtf(
    dimensionless_frequency: FloatArray,
) -> FloatArray:
    """Evaluate the analytic incoherent clear-circular-aperture MTF."""

    frequency = np.asarray(dimensionless_frequency, dtype=np.float64)
    if (
        frequency.ndim != 1
        or np.any(frequency < 0.0)
        or np.any(frequency > 1.0)
        or not np.all(np.isfinite(frequency))
    ):
        raise ValueError("dimensionless circular-aperture frequencies must lie in [0, 1]")
    return np.asarray(
        2.0
        / np.pi
        * (np.arccos(frequency) - frequency * np.sqrt(np.maximum(1.0 - np.square(frequency), 0.0))),
        dtype=np.float64,
    )


def _direct_continuous_psf_axis_mtf(
    psf: ContinuousPSF,
    frequency_cpm: FloatArray,
) -> tuple[FloatArray, FloatArray, float]:
    """Directly integrate horizontal/vertical line spreads at requested frequencies."""

    frequency = np.asarray(frequency_cpm, dtype=np.float64)
    if frequency.ndim != 1 or np.any(frequency < 0.0) or not np.all(np.isfinite(frequency)):
        raise ValueError("direct-DFT frequencies must be a finite nonnegative vector")
    spacing = psf.sample_spacing_m
    horizontal_line_spread = psf.density.sum(axis=0, dtype=np.float64) * spacing
    vertical_line_spread = psf.density.sum(axis=1, dtype=np.float64) * spacing
    phase = np.exp(-2j * np.pi * frequency[:, None] * psf.axis_m[None, :])
    horizontal = phase @ horizontal_line_spread * spacing
    vertical = phase @ vertical_line_spread * spacing
    if abs(horizontal[0]) <= 0.0 or abs(vertical[0]) <= 0.0:
        raise RuntimeError("direct continuous-PSF DFT produced invalid DC")
    horizontal = horizontal / horizontal[0]
    vertical = vertical / vertical[0]
    imaginary_residual = float(max(np.max(np.abs(horizontal.imag)), np.max(np.abs(vertical.imag))))
    return (
        np.asarray(np.abs(horizontal), dtype=np.float64),
        np.asarray(np.abs(vertical), dtype=np.float64),
        imaginary_residual,
    )


def _analytic_zero_defocus_channel_record(
    channel_name: str,
    psf: ContinuousPSF,
    *,
    pixel_pitch_m: float,
    criteria: ValidationCriteria,
) -> dict[str, Any]:
    """Compare one declared W=0 PSF with independent circular-aperture formulae."""

    if psf.edge_waves != 0.0:
        raise ValueError("analytic Airy validation requires an exact W=0 PSF")
    center = psf.density.shape[0] // 2
    optical_scale_m = psf.wavelength_m * psf.f_number
    dimensionless_radius = psf.axis_m[center:] / optical_scale_m
    radial_selection = dimensionless_radius <= 3.0
    dimensionless_radius = np.asarray(dimensionless_radius[radial_selection], dtype=np.float64)
    offsets = np.arange(dimensionless_radius.size)
    center_density = float(psf.density[center, center])
    analytic_center_density = float(np.pi / (4.0 * optical_scale_m**2))
    analytic_intensity = _independent_airy_normalized_intensity(dimensionless_radius)
    directions = {
        "positive_x": np.asarray(
            psf.density[center, center + offsets] / center_density,
            dtype=np.float64,
        ),
        "negative_x": np.asarray(
            psf.density[center, center - offsets] / center_density,
            dtype=np.float64,
        ),
        "positive_y": np.asarray(
            psf.density[center + offsets, center] / center_density,
            dtype=np.float64,
        ),
        "negative_y": np.asarray(
            psf.density[center - offsets, center] / center_density,
            dtype=np.float64,
        ),
    }
    directional_errors = {
        name: float(np.max(np.abs(values - analytic_intensity)))
        for name, values in directions.items()
    }
    intensity_linf_error = max(directional_errors.values())
    center_relative_error = abs(center_density / analytic_center_density - 1.0)
    intensity_curve_payload = {
        "radius_unit": "r/(lambda*N)",
        "maximum_dimensionless_radius_inclusive": 3.0,
        "dimensionless_radius": dimensionless_radius.tolist(),
        "analytic_normalized_intensity": analytic_intensity.tolist(),
        "numerical_normalized_intensity_by_direction": {
            name: values.tolist() for name, values in directions.items()
        },
    }
    intensity_curve_set = {
        **intensity_curve_payload,
        "curve_set_sha256": canonical_sha256(intensity_curve_payload),
    }

    sample_count = psf.density.shape[0]
    positive_frequency_cpm = np.arange(center + 1, dtype=np.float64) / (
        sample_count * psf.sample_spacing_m
    )
    dimensionless_frequency = positive_frequency_cpm * optical_scale_m
    frequency_selection = dimensionless_frequency <= 0.9
    positive_frequency_cpm = np.asarray(
        positive_frequency_cpm[frequency_selection], dtype=np.float64
    )
    dimensionless_frequency = np.asarray(
        dimensionless_frequency[frequency_selection], dtype=np.float64
    )
    horizontal_mtf, vertical_mtf, imaginary_residual = _direct_continuous_psf_axis_mtf(
        psf,
        positive_frequency_cpm,
    )
    analytic_mtf = _independent_circular_aperture_mtf(dimensionless_frequency)
    horizontal_mtf_error = float(np.max(np.abs(horizontal_mtf - analytic_mtf)))
    vertical_mtf_error = float(np.max(np.abs(vertical_mtf - analytic_mtf)))
    mtf_linf_error = max(horizontal_mtf_error, vertical_mtf_error)
    frequency_cycles_per_pixel = positive_frequency_cpm * pixel_pitch_m
    mtf_curves = {
        "analytic": _curve_record(
            frequency_cycles_per_pixel,
            analytic_mtf,
            name=f"analytic_clear_circular_aperture_mtf_{channel_name}",
        ),
        "numerical_horizontal": _curve_record(
            frequency_cycles_per_pixel,
            horizontal_mtf,
            name=f"declared_w0_continuous_psf_horizontal_mtf_{channel_name}",
        ),
        "numerical_vertical": _curve_record(
            frequency_cycles_per_pixel,
            vertical_mtf,
            name=f"declared_w0_continuous_psf_vertical_mtf_{channel_name}",
        ),
    }
    passed = bool(
        center_relative_error <= criteria.analytic_airy_center_relative_error_max
        and intensity_linf_error <= criteria.analytic_airy_normalized_intensity_linf_max
        and mtf_linf_error <= criteria.analytic_circular_aperture_mtf_linf_max
    )
    payload = {
        "channel": channel_name,
        "wavelength_m": psf.wavelength_m,
        "f_number": psf.f_number,
        "edge_waves": psf.edge_waves,
        "optical_scale_lambda_times_f_number_m": optical_scale_m,
        "continuous_psf_sha256": _array_sha256(psf.density),
        "continuous_psf_model_identity": psf.model_identity,
        "airy_intensity": {
            "numerical_center_density_per_square_m": center_density,
            "analytic_center_density_per_square_m": analytic_center_density,
            "center_relative_error": center_relative_error,
            "center_relative_error_tolerance": (criteria.analytic_airy_center_relative_error_max),
            "directional_normalized_intensity_linf_errors": directional_errors,
            "maximum_normalized_intensity_linf_error": intensity_linf_error,
            "normalized_intensity_linf_tolerance": (
                criteria.analytic_airy_normalized_intensity_linf_max
            ),
            "curves": intensity_curve_set,
        },
        "incoherent_mtf": {
            "frequency_unit": "cycles/pixel",
            "dimensionless_frequency_unit": "lambda*N*cycles/m",
            "maximum_dimensionless_frequency_inclusive": 0.9,
            "dimensionless_frequency": dimensionless_frequency.tolist(),
            "horizontal_linf_error": horizontal_mtf_error,
            "vertical_linf_error": vertical_mtf_error,
            "maximum_linf_error": mtf_linf_error,
            "linf_tolerance": criteria.analytic_circular_aperture_mtf_linf_max,
            "maximum_direct_dft_imaginary_residual": imaginary_residual,
            "curves": mtf_curves,
        },
        "passed": passed,
    }
    return {**payload, "record_sha256": canonical_sha256(payload)}


def _analytic_zero_defocus_record(
    profile: CameraProfile,
    neutral_model: DefocusModel,
    criteria: ValidationCriteria,
) -> dict[str, Any]:
    """Archive profile-bound independent analytic diffraction evidence at W=0."""

    if neutral_model.config.edge_waves_ref != 0.0:
        raise ValueError("analytic zero-defocus validation requires a neutral model")
    implementation_contract = {
        "schema_version": 1,
        "implementation_id": "independent_analytic_circular_aperture_airy_v1",
        "claim": "declared_profile_continuous_psf_at_exact_w0_matches_clear_circular_aperture",
        "production_helpers_forbidden": [
            "phycam_eval.optics.psf.airy_psf_density",
            "phycam_eval.optics.psf.psf_to_otf",
        ],
        "production_helpers_used_by_oracle": False,
        "airy_density_formula": (
            "pi/(4*(lambda*N)^2) * [2*J1(pi*r/(lambda*N))/(pi*r/(lambda*N))]^2"
        ),
        "airy_center_limit": "pi/(4*(lambda*N)^2)",
        "incoherent_mtf_formula": "2/pi * (acos(v)-v*sqrt(1-v^2)); v=lambda*N*f",
        "bessel_implementation": "scipy.special.j1",
        "psf_comparison": (
            "four declared-grid cardinal radial axes through r/(lambda*N)<=3, "
            "normalized by the numerical and analytic center values"
        ),
        "mtf_numerical_evaluation": (
            "explicit complex-exponential Riemann integration of horizontal and vertical "
            "line spreads; no production OTF helper"
        ),
        "mtf_comparison_domain": "0<=lambda*N*f<=0.9",
    }
    channel_records = [
        _analytic_zero_defocus_channel_record(
            channel,
            neutral_model.channels[index].psf,
            pixel_pitch_m=profile.sensor.pixel_pitch_m,
            criteria=criteria,
        )
        for index, channel in enumerate(("R", "G", "B"))
    ]
    payload = {
        "implementation_contract": implementation_contract,
        "implementation_sha256": canonical_sha256(implementation_contract),
        "camera_profile_sha256": profile.profile_hash,
        "optics_profile_sha256": canonical_sha256(profile.optics.to_dict()),
        "neutral_model_sha256": neutral_model.cache_key,
        "declared_sampling": {
            "pupil_grid_size": profile.optics.pupil_grid_size,
            "pupil_q_max": profile.optics.pupil_q_max,
            "pupil_fft_size": profile.optics.pupil_fft_size,
        },
        "edge_waves_ref": neutral_model.config.edge_waves_ref,
        "channel_count": len(channel_records),
        "channels": channel_records,
        "passed": all(record["passed"] for record in channel_records),
    }
    return {**payload, "record_sha256": canonical_sha256(payload)}


def _model_config(profile: CameraProfile, waves: float) -> DefocusConfig:
    _, model = build_ldr_pipeline(
        profile,
        LDRCaptureSeverity(edge_waves_ref=waves),
    )
    return model.config


def _model_at_level(config: DefocusConfig, level: SamplingLevel) -> DefocusModel:
    return build_defocus_model(
        replace(
            config,
            pupil_sampling=PupilSampling(
                level.pupil_grid_size,
                config.pupil_sampling.q_max,
                level.pupil_fft_size,
            ),
        ),
        use_cache=False,
    )


def _independent_midpoint_tent_transfer(
    psf: ContinuousPSF,
    pitch_m: float,
    *,
    subdivisions: int,
) -> tuple[FloatArray, float]:
    """Numerically integrate the source/photosite tent without production helpers."""

    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions <= 0:
        raise ValueError("independent tent subdivisions must be a positive integer")
    spacing_m = float(psf.sample_spacing_m)
    support_half_extent_m = 0.5 * psf.axis_m.size * spacing_m
    maximum_offset = max(
        0,
        int(math.ceil((support_half_extent_m + pitch_m) / pitch_m) - 1),
    )
    centers_m = np.arange(-maximum_offset, maximum_offset + 1, dtype=np.float64) * pitch_m
    midpoint_offsets_m = (
        (np.arange(subdivisions, dtype=np.float64) + 0.5) / subdivisions - 0.5
    ) * spacing_m
    quadrature_nodes_m = psf.axis_m[:, None] + midpoint_offsets_m[None, :]
    tent_samples = np.maximum(
        pitch_m - np.abs(centers_m[:, None, None] - quadrature_nodes_m[None, :, :]),
        0.0,
    )
    integrated_tent = tent_samples.sum(axis=2, dtype=np.float64) * spacing_m / subdivisions
    mass = np.asarray(
        integrated_tent @ psf.density @ integrated_tent.T / pitch_m**2,
        dtype=np.float64,
    )
    represented_energy = float(mass.sum(dtype=np.float64))
    if not math.isfinite(represented_energy) or represented_energy <= 0.0:
        raise RuntimeError("independent tent quadrature produced invalid energy")
    return mass, represented_energy


def _independent_transfer_oracle_record(
    psf: ContinuousPSF,
    transfer: CellAverageTransferKernel,
    criteria: ValidationCriteria,
) -> dict[str, Any]:
    """Compare production transfer weights with a midpoint-quadrature oracle."""

    implementation_contract = {
        "implementation_id": "independent_midpoint_tent_quadrature_v1",
        "production_helpers_forbidden": [
            "_tent_antiderivative",
            "_cell_average_tent_matrix",
            "_crop_encircled_energy",
        ],
        "integrand": "psf_density_times_separable_equal_cell_overlap_tent",
        "quadrature": "uniform_midpoints_inside_each_finite_psf_density_cell",
        "levels": [16, 256],
        "comparison_support": "centered_support_selected_by_executed_transfer",
        "normalization": "independent_mass_sum_on_selected_support",
    }
    implementation_sha256 = canonical_sha256(implementation_contract)
    level_records: list[dict[str, Any]] = []
    for subdivisions in implementation_contract["levels"]:
        mass, represented_energy = _independent_midpoint_tent_transfer(
            psf,
            transfer.sample_spacing_m[0],
            subdivisions=subdivisions,
        )
        center_y = mass.shape[0] // 2
        center_x = mass.shape[1] // 2
        half_y = transfer.values.shape[0] // 2
        half_x = transfer.values.shape[1] // 2
        if half_y > center_y or half_x > center_x:
            raise RuntimeError("executed transfer support exceeds the independent oracle support")
        selected_mass = np.array(
            mass[
                center_y - half_y : center_y + half_y + 1,
                center_x - half_x : center_x + half_x + 1,
            ],
            dtype=np.float64,
            copy=True,
        )
        selected_energy = float(selected_mass.sum(dtype=np.float64))
        normalized = selected_mass / selected_energy
        level_records.append(
            {
                "subdivisions_per_psf_axis_cell": subdivisions,
                "full_support_shape": list(mass.shape),
                "represented_energy": represented_energy,
                "represented_energy_absolute_error": abs(represented_energy - psf.energy),
                "selected_retained_energy": selected_energy / represented_energy,
                "selected_retained_energy_absolute_error": abs(
                    selected_energy / represented_energy - transfer.retained_energy
                ),
                "normalized_transfer_linf_error": float(
                    np.max(np.abs(normalized - transfer.values))
                ),
                "normalized_transfer_sha256": _array_sha256(normalized),
            }
        )
    coarse, fine = level_records
    contraction_passed = (
        fine["normalized_transfer_linf_error"] < coarse["normalized_transfer_linf_error"]
    )
    passed = bool(
        contraction_passed
        and fine["normalized_transfer_linf_error"] <= criteria.independent_tent_quadrature_linf_max
        and fine["selected_retained_energy_absolute_error"]
        <= criteria.independent_tent_retained_energy_absolute_error_max
    )
    payload = {
        "implementation_contract": implementation_contract,
        "implementation_sha256": implementation_sha256,
        "levels": level_records,
        "error_contraction_passed": contraction_passed,
        "fine_normalized_transfer_linf_tolerance": (criteria.independent_tent_quadrature_linf_max),
        "fine_retained_energy_absolute_error_tolerance": (
            criteria.independent_tent_retained_energy_absolute_error_max
        ),
        "passed": passed,
    }
    return {**payload, "record_sha256": canonical_sha256(payload)}


def _channel_record(
    model: DefocusModel,
    channel_index: int,
    channel_name: str,
    criteria: ValidationCriteria,
) -> dict[str, Any]:
    channel = model.channels[channel_index]
    psf = channel.psf
    diagnostic_kernel = channel.kernel
    transfer = collapse_cell_average_transfer(
        psf,
        model.config.pixel_pitch_m,
        encircled_energy=model.config.encircled_energy,
    )
    psf_flux_error = abs(psf.energy - 1.0)
    transfer_flux_error = abs(float(transfer.values.sum(dtype=np.float64)) - 1.0)
    psf_symmetry = _symmetry_diagnostics(psf.density)
    transfer_symmetry = _symmetry_diagnostics(transfer.values)
    independent_oracle = _independent_transfer_oracle_record(psf, transfer, criteria)
    maximum_symmetry_error = max(*psf_symmetry.values(), *transfer_symmetry.values())
    retained_energy_passed = (
        transfer.retained_energy + criteria.flux_absolute_error_max
        >= transfer.requested_encircled_energy
    )
    passed = (
        psf_flux_error <= criteria.flux_absolute_error_max
        and transfer_flux_error <= criteria.flux_absolute_error_max
        and maximum_symmetry_error <= criteria.symmetry_relative_linf_max
        and retained_energy_passed
        and independent_oracle["passed"]
    )
    payload = {
        "channel": channel_name,
        "wavelength_m": channel.wavelength_m,
        "edge_waves": channel.edge_waves,
        "continuous_psf": {
            "shape": list(psf.density.shape),
            "sample_spacing_m": psf.sample_spacing_m,
            "energy": psf.energy,
            "flux_absolute_error": psf_flux_error,
            "symmetry": psf_symmetry,
            "array_sha256": _array_sha256(psf.density),
        },
        "equal_grid_cell_average_transfer": {
            "representation": "exact_equal_grid_cell_average_transfer_v1",
            "integration_rule": transfer.integration_rule,
            "shape": list(transfer.values.shape),
            "sample_spacing_m": list(transfer.sample_spacing_m),
            "sum": float(transfer.values.sum(dtype=np.float64)),
            "flux_absolute_error": transfer_flux_error,
            "retained_energy_before_renormalization": transfer.retained_energy,
            "requested_encircled_energy": transfer.requested_encircled_energy,
            "retained_energy_passed": retained_energy_passed,
            "symmetry": transfer_symmetry,
            "array_sha256": _array_sha256(transfer.values),
        },
        "single_aperture_point_sample_diagnostic": {
            "acceptance_role": "diagnostic_only_not_the_native_formation_operator",
            "shape": list(diagnostic_kernel.values.shape),
            "array_sha256": _array_sha256(diagnostic_kernel.values),
        },
        "independent_cell_average_transfer_oracle": independent_oracle,
        "maximum_symmetry_relative_linf": maximum_symmetry_error,
        "passed": passed,
    }
    return {**payload, "record_sha256": canonical_sha256(payload)}


def _manual_reflect_convolution(plane: FloatArray, kernel: FloatArray) -> FloatArray:
    """Small direct whole-sample-reflection oracle independent of SciPy convolution."""

    half_y = kernel.shape[0] // 2
    half_x = kernel.shape[1] // 2
    padded = np.pad(plane, ((half_y, half_y), (half_x, half_x)), mode="reflect")
    flipped = kernel[::-1, ::-1]
    result = np.empty_like(plane, dtype=np.float64)
    for row in range(plane.shape[0]):
        for column in range(plane.shape[1]):
            result[row, column] = np.sum(
                padded[
                    row : row + kernel.shape[0],
                    column : column + kernel.shape[1],
                ]
                * flipped,
                dtype=np.float64,
            )
    return result


def _independent_equal_grid_formation_record(
    profile: CameraProfile,
    model: DefocusModel,
    criteria: ValidationCriteria,
) -> dict[str, Any]:
    """Prove that equal-grid rendering applies the validated transfer exactly once."""

    if profile.optics.boundary_policy != "reflect":
        raise ValueError("independent publication formation oracle requires reflect boundary")
    kernels = tuple(
        collapse_cell_average_transfer(
            channel.psf,
            model.config.pixel_pitch_m,
            encircled_energy=model.config.encircled_energy,
        )
        for channel in model.channels
    )
    height, width = 11, 13
    indices = np.arange(height * width * 3, dtype=np.float64).reshape(height, width, 3)
    scene = np.mod(indices * 17.0 + 11.0, 251.0) / 251.0
    grid = GridGeometry.square_pixels(height, width, model.config.pixel_pitch_m)
    timing = ReadoutTiming(height, line_time_s=0.0, exposure_s=0.01)
    executed = render_joint_photosite_exposure(
        scene,
        source=grid,
        sensor=grid,
        timing=timing,
        optical_kernels=kernels,
        optics_boundary="reflect",
        warp_boundary="reflect",
        convolution_method="direct",
    )
    oracle = np.stack(
        [
            _manual_reflect_convolution(scene[:, :, channel], kernels[channel].values)
            for channel in range(3)
        ],
        axis=2,
    )
    channel_errors = [
        float(np.max(np.abs(executed[:, :, channel] - oracle[:, :, channel])))
        for channel in range(3)
    ]
    maximum_error = max(channel_errors)
    implementation_contract = {
        "implementation_id": "independent_direct_reflect_equal_grid_formation_v1",
        "production_convolution_helpers_used_by_oracle": False,
        "boundary_extension": "numpy_reflect_whole_sample",
        "convolution": "explicit_output_pixel_and_kernel_loops_with_flipped_kernel",
        "source_grid": "exactly_equal_to_sensor_grid",
        "acceptance_claim": "executed_equal_grid_branch_is_one_transfer_convolution_only",
        "scene_shape": [height, width, 3],
    }
    payload = {
        "implementation_contract": implementation_contract,
        "implementation_sha256": canonical_sha256(implementation_contract),
        "model_sha256": model.cache_key,
        "kernel_sha256_by_channel": [_array_sha256(kernel.values) for kernel in kernels],
        "scene_sha256": _array_sha256(scene),
        "executed_output_sha256": _array_sha256(executed),
        "oracle_output_sha256": _array_sha256(oracle),
        "channel_linf_errors": channel_errors,
        "maximum_linf_error": maximum_error,
        "tolerance": criteria.independent_equal_grid_formation_linf_max,
        "passed": maximum_error <= criteria.independent_equal_grid_formation_linf_max,
    }
    return {**payload, "record_sha256": canonical_sha256(payload)}


def _physical_severity_record(
    profile: CameraProfile,
    waves: float,
    ladder: tuple[SamplingLevel, SamplingLevel, SamplingLevel],
    frequency: FloatArray,
    criteria: ValidationCriteria,
) -> dict[str, Any]:
    base_config = _model_config(profile, waves)
    models = tuple(_model_at_level(base_config, level) for level in ladder)
    weights = profile.isp.output_luminance_coefficients
    signed_curves = tuple(
        _signed_axis_response(luminance_weighted_kernel(model, weights), frequency)[0]
        for model in models
    )
    mtf_curves = tuple(np.abs(curve) for curve in signed_curves)
    curve_records = []
    for level, model, mtf in zip(ladder, models, mtf_curves):
        curve = _curve_record(frequency, mtf, name=f"physical_luminance_mtf_{level.name}")
        curve_records.append(
            {
                "sampling": level.to_dict(),
                "model_sha256": model.cache_key,
                "mtf50": _crossing_record(frequency, mtf, 0.5),
                "mtf10": _crossing_record(frequency, mtf, 0.1),
                "curve": curve,
            }
        )
    coarse_declared_linf = float(np.max(np.abs(mtf_curves[0] - mtf_curves[1])))
    declared_refined_linf = float(np.max(np.abs(mtf_curves[1] - mtf_curves[2])))
    if coarse_declared_linf == 0.0:
        contraction_ratio = 0.0 if declared_refined_linf == 0.0 else None
    else:
        contraction_ratio = declared_refined_linf / coarse_declared_linf
    contraction_passed = (
        contraction_ratio is not None
        and contraction_ratio <= criteria.convergence_error_contraction_ratio_max
    )
    declared_refined_mtf50_error = _relative_crossing_error(
        curve_records[1]["mtf50"],
        curve_records[2]["mtf50"],
    )
    mtf50_censoring_agrees = (
        curve_records[1]["mtf50"]["censoring"] == curve_records[2]["mtf50"]["censoring"]
    )
    mtf50_passed = (
        mtf50_censoring_agrees
        if declared_refined_mtf50_error is None
        else declared_refined_mtf50_error
        <= criteria.convergence_declared_refined_mtf50_relative_error_max
    )
    convergence_passed = (
        declared_refined_linf <= criteria.convergence_declared_refined_mtf_linf_max
        and contraction_passed
        and mtf50_passed
    )

    repeated = _model_at_level(base_config, ladder[1])

    def channel_hashes(model: DefocusModel) -> list[tuple[str, str, str]]:
        return [
            (
                _array_sha256(channel.psf.density),
                _array_sha256(channel.kernel.values),
                _array_sha256(
                    collapse_cell_average_transfer(
                        channel.psf,
                        model.config.pixel_pitch_m,
                        encircled_energy=model.config.encircled_energy,
                    ).values
                ),
            )
            for channel in model.channels
        ]

    declared_array_hashes = channel_hashes(models[1])
    repeated_array_hashes = channel_hashes(repeated)
    identity = {
        "config_cache_key": models[1].config.cache_key,
        "model_cache_key": models[1].cache_key,
        "repeated_model_cache_key": repeated.cache_key,
        "cache_key_consistent": (
            models[1].config.cache_key == models[1].cache_key == repeated.cache_key
        ),
        "repeated_array_hashes_exact": declared_array_hashes == repeated_array_hashes,
        "channel_array_hashes": [
            {
                "channel": channel,
                "continuous_psf_sha256": hashes[0],
                "single_aperture_diagnostic_sha256": hashes[1],
                "equal_grid_cell_average_transfer_sha256": hashes[2],
            }
            for channel, hashes in zip(("R", "G", "B"), declared_array_hashes)
        ],
    }
    identity["passed"] = bool(
        identity["cache_key_consistent"] and identity["repeated_array_hashes_exact"]
    )
    channel_records = [
        _channel_record(models[1], index, channel, criteria)
        for index, channel in enumerate(("R", "G", "B"))
    ]
    payload = {
        "edge_waves_ref": waves,
        "declared_model_sha256": models[1].cache_key,
        "identity": identity,
        "channels": channel_records,
        "convergence": {
            "levels": curve_records,
            "coarse_to_declared_mtf_linf": coarse_declared_linf,
            "declared_to_refined_mtf_linf": declared_refined_linf,
            "error_contraction_ratio": contraction_ratio,
            "error_contraction_passed": contraction_passed,
            "declared_refined_mtf50_relative_error": declared_refined_mtf50_error,
            "declared_refined_mtf50_censoring_agrees": mtf50_censoring_agrees,
            "declared_refined_mtf50_passed": mtf50_passed,
            "passed": convergence_passed,
        },
        "passed": bool(
            identity["passed"]
            and all(record["passed"] for record in channel_records)
            and convergence_passed
        ),
    }
    return {**payload, "record_sha256": canonical_sha256(payload)}


def _transfer_integrals(response: FloatArray, frequency: FloatArray) -> dict[str, float]:
    span = float(frequency[-1] - frequency[0])
    energy = _trapezoid(np.square(response), frequency)
    negative_energy = _trapezoid(
        np.square(response) * (response < 0.0).astype(np.float64),
        frequency,
    )
    return {
        "normalized_signed_response_area": _trapezoid(response, frequency) / span,
        "normalized_magnitude_response_area": _trapezoid(np.abs(response), frequency) / span,
        "normalized_response_energy": energy / span,
        "negative_axis_frequency_fraction": _trapezoid(
            (response < 0.0).astype(np.float64), frequency
        )
        / span,
        "negative_response_energy_fraction": 0.0 if energy == 0.0 else negative_energy / energy,
        "minimum_signed_response": float(np.min(response)),
        "maximum_signed_response": float(np.max(response)),
        "response_at_nyquist": float(response[-1]),
    }


def _impulse_diagnostics(
    config: GaussianComparatorConfig
    | QuadraticCosineComparatorConfig
    | SampledIncoherentComparatorConfig,
    criteria: ValidationCriteria,
) -> tuple[dict[str, Any], FloatArray | None]:
    if isinstance(config, QuadraticCosineComparatorConfig):
        return (
            {
                "available": False,
                "reason": (
                    "the DCT-even operator is boundary-dependent and has no single "
                    "shift-invariant finite impulse response"
                ),
                "passed": True,
            },
            None,
        )
    if isinstance(config, GaussianComparatorConfig):
        kernel = gaussian_kernel(config)
        retained_energy = None
        retained_energy_contract = "not_exposed_by_finite_gaussian_implementation"
        retained_energy_passed = True
    else:
        kernel = sampled_incoherent_kernel(config)
        retained_energy = sampled_incoherent_retained_energy(config)
        retained_energy_contract = "declared_radial_encircled_energy_crop"
        retained_energy_passed = (
            retained_energy + criteria.flux_absolute_error_max >= config.encircled_energy
        )
    symmetry = _symmetry_diagnostics(kernel)
    flux_error = abs(float(kernel.sum(dtype=np.float64)) - 1.0)
    negative_mass = float(np.abs(kernel[kernel < 0.0]).sum(dtype=np.float64))
    passed = (
        flux_error <= criteria.flux_absolute_error_max
        and max(symmetry.values()) <= criteria.symmetry_relative_linf_max
        and negative_mass == 0.0
        and retained_energy_passed
    )
    return (
        {
            "available": True,
            "shape": list(kernel.shape),
            "sum": float(kernel.sum(dtype=np.float64)),
            "flux_absolute_error": flux_error,
            "spatial_l2_energy": float(np.square(kernel).sum(dtype=np.float64)),
            "negative_mass": negative_mass,
            "symmetry": symmetry,
            "retained_energy_before_renormalization": retained_energy,
            "retained_energy_contract": retained_energy_contract,
            "retained_energy_passed": retained_energy_passed,
            "array_sha256": _array_sha256(kernel),
            "passed": passed,
        },
        kernel,
    )


def _comparator_response(
    config: GaussianComparatorConfig
    | QuadraticCosineComparatorConfig
    | SampledIncoherentComparatorConfig,
    frequency: FloatArray,
    impulse_kernel: FloatArray | None,
) -> tuple[FloatArray, float | None, str]:
    if isinstance(config, QuadraticCosineComparatorConfig):
        response = np.cos(2.0 * config.alpha * np.square(frequency))
        return response, None, "continuous_axis_response_of_dct_basis_operator"
    if impulse_kernel is None:
        raise RuntimeError("kernel comparator is missing its impulse response")
    response, imaginary_residual = _signed_axis_response(impulse_kernel, frequency)
    return response, imaginary_residual, "exact_dtft_of_executed_finite_kernel"


def _dct_finite_shape_envelope(
    config: QuadraticCosineComparatorConfig,
    *,
    shape_contract: Mapping[str, Any],
    neutral_frequency: FloatArray,
    neutral_mtf: FloatArray,
    target_mtf50: Mapping[str, Any],
    criteria: ValidationCriteria,
) -> dict[str, Any]:
    """Execute every bound DCT-I axis response and bound the MTF50 error."""

    contract = _normalize_shape_contract(shape_contract)
    target_value = target_mtf50["value_cycles_per_pixel"]
    if target_value is None:
        raise ValueError("DCT finite-shape validation requires an observed target MTF50")
    target_value = float(target_value)
    records: list[dict[str, Any]] = []
    for dimension in contract["unique_axis_dimensions"]:
        frequency = np.arange(dimension, dtype=np.float64) / (2.0 * (dimension - 1))
        # This is the exact response array multiplied by
        # apply_quadratic_cosine_comparator, not a separately retyped formula.
        comparator_response = quadratic_cosine_response((1, dimension), config)[0]
        achieved_mtf = np.interp(frequency, neutral_frequency, neutral_mtf) * np.abs(
            comparator_response
        )
        crossing = _crossing_record(frequency, achieved_mtf, 0.5)
        crossing_value = crossing["value_cycles_per_pixel"]
        relative_error = (
            None
            if crossing_value is None
            else abs(float(crossing_value) - target_value) / target_value
        )
        records.append(
            {
                "axis_dimension": dimension,
                "mtf50": crossing,
                "relative_error": relative_error,
            }
        )
    observed = [record for record in records if record["relative_error"] is not None]
    if len(observed) != len(records):
        worst = None
        maximum_error = None
        passed = False
    else:
        worst = max(observed, key=lambda record: float(record["relative_error"]))
        maximum_error = float(worst["relative_error"])
        passed = maximum_error <= criteria.implemented_match_mtf50_relative_error_max
    payload = {
        "method": "execute_exact_dct_i_axis_eigenresponse_for_every_bound_dimension_v2",
        "shape_contract_sha256": contract["record_sha256"],
        "shape_scope": contract["scope"],
        "dataset_sha256": contract["dataset_sha256"],
        "axis_dimension_min_inclusive": contract["axis_dimension_min"],
        "axis_dimension_max_inclusive": contract["axis_dimension_max"],
        "axis_dimension_count": len(records),
        "rectangular_coverage": (
            "each rectangular axis uses the enumerated response for its own dimension"
        ),
        "records": records,
        "records_sha256": canonical_sha256(records),
        "worst_axis_dimension": None if worst is None else worst["axis_dimension"],
        "maximum_relative_mtf50_error": maximum_error,
        "passed": passed,
    }
    return {**payload, "record_sha256": canonical_sha256(payload)}


def _comparator_record(
    match: MechanismMatch,
    frequency: FloatArray,
    neutral_response: FloatArray,
    target_response: FloatArray,
    criteria: ValidationCriteria,
    shape_contract: Mapping[str, Any],
) -> dict[str, Any]:
    config = comparator_config_from_dict(match.config)
    impulse, impulse_kernel = _impulse_diagnostics(config, criteria)
    comparator_response, imaginary_residual, response_method = _comparator_response(
        config,
        frequency,
        impulse_kernel,
    )
    achieved_response = neutral_response * comparator_response
    target_mtf = np.abs(target_response)
    achieved_mtf = np.abs(achieved_response)
    target_mtf50 = _crossing_record(frequency, target_mtf, 0.5)
    design_or_kernel_mtf50 = _crossing_record(frequency, achieved_mtf, 0.5)
    continuous_or_kernel_error = _relative_crossing_error(
        design_or_kernel_mtf50,
        target_mtf50,
    )
    dct_shape_envelope = (
        _dct_finite_shape_envelope(
            config,
            shape_contract=shape_contract,
            neutral_frequency=frequency,
            neutral_mtf=np.abs(neutral_response),
            target_mtf50=target_mtf50,
            criteria=criteria,
        )
        if isinstance(config, QuadraticCosineComparatorConfig)
        else None
    )
    implemented_error = (
        dct_shape_envelope["maximum_relative_mtf50_error"]
        if dct_shape_envelope is not None
        else continuous_or_kernel_error
    )
    implemented_match_passed = (
        implemented_error is not None
        and implemented_error <= criteria.implemented_match_mtf50_relative_error_max
    )
    record_match_passed = (
        match.relative_match_error <= criteria.implemented_match_mtf50_relative_error_max
    )
    config_identity_passed = canonical_sha256(config.to_dict()) == match.config_sha256
    difference = achieved_mtf - target_mtf
    target_area = _trapezoid(target_mtf, frequency)
    achieved_area = _trapezoid(achieved_mtf, frequency)
    transfer = _transfer_integrals(comparator_response, frequency)
    full_curve = {
        "acceptance_role": "diagnostic_only_not_an_equivalence_criterion",
        "physical_target_mtf10": _crossing_record(frequency, target_mtf, 0.1),
        "implemented_achieved_mtf10": _crossing_record(frequency, achieved_mtf, 0.1),
        "max_absolute_mtf_difference": float(np.max(np.abs(difference))),
        "rms_mtf_difference": float(np.sqrt(np.mean(np.square(difference)))),
        "physical_target_normalized_mtf_area": target_area / 0.5,
        "implemented_achieved_normalized_mtf_area": achieved_area / 0.5,
        "normalized_mtf_area_difference": (achieved_area - target_area) / 0.5,
    }
    curves = {
        "physical_target_mtf": _curve_record(
            frequency,
            target_mtf,
            name="physical_target_luminance_mtf",
        ),
        "implemented_comparator_signed_response": _curve_record(
            frequency,
            comparator_response,
            name="implemented_comparator_signed_axis_response",
        ),
        "implemented_common_neutral_mtf": _curve_record(
            frequency,
            achieved_mtf,
            name="implemented_common_neutral_comparator_mtf",
        ),
    }
    payload = {
        "comparator_family": match.comparator_family,
        "target_edge_waves_ref": match.target_edge_waves_ref,
        "match": match.to_dict(),
        "config_identity_passed": config_identity_passed,
        "response_method": response_method,
        "axis_response_maximum_imaginary_residual": imaginary_residual,
        "impulse_response": impulse,
        "transfer_diagnostics": transfer,
        "mtf50_acceptance": {
            "criterion": "implemented_common_neutral_first_downward_mtf50",
            "target": target_mtf50,
            "design_or_finite_kernel": design_or_kernel_mtf50,
            "validation_scope": (
                (
                    "loader_attested_native_coco_axis_dimensions_exact_dct_i_eigenresponse"
                    if shape_contract["scope"] == _ATTESTED_DCT_SHAPE_SCOPE
                    else "declared_axis_dimensions_exact_dct_i_eigenresponse_not_dataset_bound"
                )
                if dct_shape_envelope is not None
                else "exact_executed_finite_shift_invariant_kernel"
            ),
            "dct_finite_shape_envelope": dct_shape_envelope,
            "continuous_design_or_kernel_relative_error": continuous_or_kernel_error,
            "implemented_relative_error": implemented_error,
            "recorded_match_relative_error": match.relative_match_error,
            "recorded_match_passed": record_match_passed,
            "implemented_match_passed": implemented_match_passed,
        },
        "full_curve_diagnostics": full_curve,
        "curves": curves,
        "passed": bool(
            config_identity_passed
            and impulse["passed"]
            and record_match_passed
            and implemented_match_passed
        ),
    }
    return {**payload, "record_sha256": canonical_sha256(payload)}


def _build_validation_evidence(
    profile: CameraProfile,
    physical_edge_waves: Sequence[float],
    *,
    shape_contract: Mapping[str, Any],
    comparator_edge_waves: Sequence[float] | None = None,
    criteria: ValidationCriteria = ValidationCriteria(),
    sampling_ladder: Sequence[SamplingLevel] | None = None,
    frequency_sample_count: int = 2049,
) -> dict[str, Any]:
    """Build from one already-normalized DCT-I axis shape contract."""

    if not isinstance(profile, CameraProfile):
        raise TypeError("profile must be a CameraProfile")
    if not isinstance(criteria, ValidationCriteria):
        raise TypeError("criteria must be a ValidationCriteria")
    shape_contract = _normalize_shape_contract(shape_contract)
    physical_waves = _float_sequence(
        physical_edge_waves,
        name="physical_edge_waves",
        positive=False,
    )
    if comparator_edge_waves is None:
        comparator_waves = tuple(value for value in physical_waves if value > 0.0)
    else:
        comparator_waves = _float_sequence(
            comparator_edge_waves,
            name="comparator_edge_waves",
            positive=True,
        )
    if not comparator_waves:
        raise ValueError("at least one positive comparator severity is required")
    if isinstance(frequency_sample_count, bool) or not isinstance(
        frequency_sample_count, (int, np.integer)
    ):
        raise TypeError("frequency_sample_count must be an integer")
    frequency_sample_count = int(frequency_sample_count)
    if frequency_sample_count < 129:
        raise ValueError("frequency_sample_count must be at least 129")
    if sampling_ladder is None:
        ladder = default_sampling_ladder(profile)
    else:
        ladder = tuple(sampling_ladder)
    if len(ladder) != 3 or not all(isinstance(level, SamplingLevel) for level in ladder):
        raise ValueError("sampling_ladder must contain exactly three SamplingLevel values")
    if tuple(level.name for level in ladder) != ("coarse", "declared", "refined"):
        raise ValueError("sampling_ladder names must be coarse, declared, refined in that order")
    if (
        ladder[1].pupil_grid_size != profile.optics.pupil_grid_size
        or ladder[1].pupil_fft_size != profile.optics.pupil_fft_size
    ):
        raise ValueError("the declared sampling level must equal the camera profile")

    frequency = np.linspace(0.0, 0.5, frequency_sample_count, dtype=np.float64)
    physical_records = [
        _physical_severity_record(profile, waves, ladder, frequency, criteria)
        for waves in physical_waves
    ]
    _, neutral_model = build_ldr_pipeline(profile, LDRCaptureSeverity())
    analytic_zero_defocus = _analytic_zero_defocus_record(
        profile,
        neutral_model,
        criteria,
    )
    independent_formation = _independent_equal_grid_formation_record(
        profile,
        neutral_model,
        criteria,
    )
    neutral_kernel = luminance_weighted_kernel(
        neutral_model,
        profile.isp.output_luminance_coefficients,
    )
    neutral_response, neutral_imaginary_residual = _signed_axis_response(
        neutral_kernel,
        frequency,
    )
    comparator_records = []
    for waves in comparator_waves:
        _, target_model = build_ldr_pipeline(
            profile,
            LDRCaptureSeverity(edge_waves_ref=waves),
        )
        target_response, _ = _signed_axis_response(
            luminance_weighted_kernel(
                target_model,
                profile.isp.output_luminance_coefficients,
            ),
            frequency,
        )
        matches = match_common_neutral_comparators(
            profile,
            waves,
            relative_tolerance=criteria.implemented_match_mtf50_relative_error_max,
        )
        comparator_records.extend(
            _comparator_record(
                match,
                frequency,
                neutral_response,
                target_response,
                criteria,
                shape_contract,
            )
            for match in matches
        )

    round_trip = CameraProfile.from_dict(profile.to_dict())
    criteria_record = criteria.to_dict()
    profile_identity = {
        "profile_sha256": profile.profile_hash,
        "round_trip_profile_sha256": round_trip.profile_hash,
        "round_trip_exact": round_trip.to_dict() == profile.to_dict(),
        "passed": round_trip.profile_hash == profile.profile_hash,
    }
    physical_passed = (
        profile_identity["passed"]
        and analytic_zero_defocus["passed"]
        and independent_formation["passed"]
        and all(record["passed"] for record in physical_records)
    )
    comparator_passed = all(record["passed"] for record in comparator_records)
    dataset_shape_attested = shape_contract["scope"] == _ATTESTED_DCT_SHAPE_SCOPE
    payload = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "record_type": "phycam_scientific_validation_evidence",
        "implementation_id": _IMPLEMENTATION_ID,
        "claim_scope": {
            "validated": (
                "deterministic numerical implementation, physical-optics discretization, "
                "declared-profile W=0 diffraction against independent analytic "
                "clear-circular-aperture Airy intensity and incoherent MTF formulae, "
                "equal-grid cell-average formation against independent midpoint-tent and "
                "direct spatial oracles, executed finite-kernel matching, "
                + (
                    "and the exact loader-attested COCO native-axis DCT-I eigenresponse set"
                    if dataset_shape_attested
                    else "and the caller-declared DCT-I axis eigenresponse set"
                )
            ),
            "not_validated": (
                "hardware calibration, real-camera fidelity, detector-performance claims"
                + ("" if dataset_shape_attested else ", and correspondence to any external dataset")
            ),
            "calibration_reference": profile.calibration_reference,
        },
        "provenance": {
            "camera_profile": profile.to_dict(),
            "camera_profile_sha256": profile.profile_hash,
            "criteria": criteria_record,
            "criteria_sha256": canonical_sha256(criteria_record),
            "sampling_ladder": [level.to_dict() for level in ladder],
            "frequency_grid": {
                "unit": "cycles/pixel",
                "minimum": 0.0,
                "maximum": 0.5,
                "sample_count": frequency_sample_count,
                "array_sha256": _array_sha256(frequency),
            },
            "dct_i_axis_shape_contract": shape_contract,
            "numerical_runtime": {
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
            },
            "generated_timestamp": None,
            "timestamp_omission_reason": "byte-reproducible scientific artifact",
        },
        "profile_identity": profile_identity,
        "analytic_zero_defocus_validation": analytic_zero_defocus,
        "independent_equal_grid_formation_validation": independent_formation,
        "common_neutral": {
            "edge_waves_ref": 0.0,
            "model_sha256": neutral_model.cache_key,
            "formation_operator": "exact_equal_grid_cell_average_transfer_v1",
            "luminance_kernel_sha256": _array_sha256(neutral_kernel),
            "axis_response_maximum_imaginary_residual": neutral_imaginary_residual,
            "signed_axis_response": _curve_record(
                frequency,
                neutral_response,
                name="common_neutral_physical_signed_axis_response",
            ),
        },
        "physical_validation": {
            "severities": physical_records,
            "passed": physical_passed,
        },
        "comparator_validation": {
            "records": comparator_records,
            "passed": comparator_passed,
        },
        "summary": {
            "physical_severity_count": len(physical_records),
            "comparator_record_count": len(comparator_records),
            "physical_passed_count": sum(record["passed"] for record in physical_records),
            "comparator_passed_count": sum(record["passed"] for record in comparator_records),
            "analytic_zero_defocus_channel_count": len(analytic_zero_defocus["channels"]),
            "analytic_zero_defocus_passed_count": sum(
                record["passed"] for record in analytic_zero_defocus["channels"]
            ),
            "dataset_shape_attested": dataset_shape_attested,
            "publication_dataset_bound": dataset_shape_attested,
            "all_passed": bool(physical_passed and comparator_passed),
        },
    }
    return {**payload, "evidence_sha256": canonical_sha256(payload)}


def build_validation_evidence(
    profile: CameraProfile,
    physical_edge_waves: Sequence[float],
    *,
    dataset: LazyNativeCOCOSubset | None = None,
    dct_axis_dimensions: Sequence[int] | None = None,
    comparator_edge_waves: Sequence[float] | None = None,
    criteria: ValidationCriteria = ValidationCriteria(),
    sampling_ladder: Sequence[SamplingLevel] | None = None,
    frequency_sample_count: int = 2049,
) -> dict[str, Any]:
    """Build deterministic evidence with an explicit DCT-I shape trust scope.

    Publication evidence must supply the exact loader-attested lazy COCO
    subset.  ``dct_axis_dimensions`` exists for unit and exploratory evidence;
    reports built from it state that they are not bound to an external dataset.
    Exactly one source is required so a caller can never inherit an implicit
    dataset claim from hard-coded dimensions.
    """

    if (dataset is None) == (dct_axis_dimensions is None):
        raise ValueError("supply exactly one of dataset or dct_axis_dimensions")
    shape_contract = (
        _dataset_shape_contract(dataset)
        if dataset is not None
        else _declared_shape_contract(dct_axis_dimensions)
    )
    return _build_validation_evidence(
        profile,
        physical_edge_waves,
        shape_contract=shape_contract,
        comparator_edge_waves=comparator_edge_waves,
        criteria=criteria,
        sampling_ladder=sampling_ladder,
        frequency_sample_count=frequency_sample_count,
    )


_METRIC_FIELDS = (
    "section",
    "severity_edge_waves_ref",
    "family",
    "channel",
    "sampling_level",
    "metric",
    "value",
    "unit",
    "criterion",
    "tolerance",
    "passed",
    "censoring",
    "record_sha256",
)

_CURVE_FIELDS = (
    "severity_edge_waves_ref",
    "family",
    "curve_name",
    "frequency_cycles_per_pixel",
    "value",
    "curve_sha256",
)


def _metric_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    criteria = report["provenance"]["criteria"]
    for physical in report["physical_validation"]["severities"]:
        waves = physical["edge_waves_ref"]
        for channel in physical["channels"]:
            common = {
                "section": "physical_channel",
                "severity_edge_waves_ref": waves,
                "family": "physical_defocus",
                "channel": channel["channel"],
                "sampling_level": "declared",
                "censoring": None,
                "record_sha256": channel["record_sha256"],
            }
            for metric, value, tolerance, unit in (
                (
                    "continuous_psf_flux_absolute_error",
                    channel["continuous_psf"]["flux_absolute_error"],
                    criteria["flux_absolute_error_max"],
                    "dimensionless",
                ),
                (
                    "equal_grid_transfer_flux_absolute_error",
                    channel["equal_grid_cell_average_transfer"]["flux_absolute_error"],
                    criteria["flux_absolute_error_max"],
                    "dimensionless",
                ),
                (
                    "maximum_symmetry_relative_linf",
                    channel["maximum_symmetry_relative_linf"],
                    criteria["symmetry_relative_linf_max"],
                    "relative",
                ),
            ):
                rows.append(
                    {
                        **common,
                        "metric": metric,
                        "value": value,
                        "unit": unit,
                        "criterion": "less_than_or_equal",
                        "tolerance": tolerance,
                        "passed": value <= tolerance,
                    }
                )
            oracle = channel["independent_cell_average_transfer_oracle"]
            fine = oracle["levels"][-1]
            rows.extend(
                [
                    {
                        **common,
                        "metric": "independent_tent_quadrature_linf_error",
                        "value": fine["normalized_transfer_linf_error"],
                        "unit": "absolute_kernel_weight",
                        "criterion": "less_than_or_equal",
                        "tolerance": criteria["independent_tent_quadrature_linf_max"],
                        "passed": fine["normalized_transfer_linf_error"]
                        <= criteria["independent_tent_quadrature_linf_max"],
                    },
                    {
                        **common,
                        "metric": "independent_tent_retained_energy_absolute_error",
                        "value": fine["selected_retained_energy_absolute_error"],
                        "unit": "absolute_energy",
                        "criterion": "less_than_or_equal",
                        "tolerance": criteria[
                            "independent_tent_retained_energy_absolute_error_max"
                        ],
                        "passed": fine["selected_retained_energy_absolute_error"]
                        <= criteria["independent_tent_retained_energy_absolute_error_max"],
                    },
                ]
            )
        convergence = physical["convergence"]
        common = {
            "section": "physical_convergence",
            "severity_edge_waves_ref": waves,
            "family": "physical_defocus",
            "channel": "Rec709_luminance",
            "sampling_level": "declared_to_refined",
            "censoring": None,
            "record_sha256": physical["record_sha256"],
        }
        rows.extend(
            [
                {
                    **common,
                    "metric": "mtf_linf",
                    "value": convergence["declared_to_refined_mtf_linf"],
                    "unit": "absolute_mtf",
                    "criterion": "less_than_or_equal",
                    "tolerance": criteria["convergence_declared_refined_mtf_linf_max"],
                    "passed": convergence["declared_to_refined_mtf_linf"]
                    <= criteria["convergence_declared_refined_mtf_linf_max"],
                },
                {
                    **common,
                    "metric": "error_contraction_ratio",
                    "value": convergence["error_contraction_ratio"],
                    "unit": "ratio",
                    "criterion": "less_than_or_equal",
                    "tolerance": criteria["convergence_error_contraction_ratio_max"],
                    "passed": convergence["error_contraction_passed"],
                },
                {
                    **common,
                    "metric": "mtf50_relative_error",
                    "value": convergence["declared_refined_mtf50_relative_error"],
                    "unit": "relative",
                    "criterion": "less_than_or_equal_when_observed",
                    "tolerance": criteria["convergence_declared_refined_mtf50_relative_error_max"],
                    "passed": convergence["declared_refined_mtf50_passed"],
                    "censoring": (
                        "both_right_at_nyquist"
                        if convergence["declared_refined_mtf50_relative_error"] is None
                        else None
                    ),
                },
            ]
        )
    analytic = report["analytic_zero_defocus_validation"]
    for channel in analytic["channels"]:
        common = {
            "section": "analytic_zero_defocus",
            "severity_edge_waves_ref": 0.0,
            "family": "clear_circular_aperture_airy",
            "channel": channel["channel"],
            "sampling_level": "declared",
            "censoring": None,
            "record_sha256": channel["record_sha256"],
        }
        for metric, value, tolerance, unit in (
            (
                "analytic_airy_center_relative_error",
                channel["airy_intensity"]["center_relative_error"],
                criteria["analytic_airy_center_relative_error_max"],
                "relative",
            ),
            (
                "analytic_airy_normalized_intensity_linf_error",
                channel["airy_intensity"]["maximum_normalized_intensity_linf_error"],
                criteria["analytic_airy_normalized_intensity_linf_max"],
                "absolute_normalized_intensity",
            ),
            (
                "analytic_circular_aperture_mtf_linf_error",
                channel["incoherent_mtf"]["maximum_linf_error"],
                criteria["analytic_circular_aperture_mtf_linf_max"],
                "absolute_mtf",
            ),
        ):
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "criterion": "less_than_or_equal",
                    "tolerance": tolerance,
                    "passed": value <= tolerance,
                }
            )
    formation = report["independent_equal_grid_formation_validation"]
    rows.append(
        {
            "section": "independent_equal_grid_formation",
            "severity_edge_waves_ref": 0.0,
            "family": "physical_neutral",
            "channel": "RGB",
            "sampling_level": "declared",
            "metric": "independent_direct_formation_linf_error",
            "value": formation["maximum_linf_error"],
            "unit": "absolute_linear_rgb",
            "criterion": "less_than_or_equal",
            "tolerance": criteria["independent_equal_grid_formation_linf_max"],
            "passed": formation["passed"],
            "censoring": None,
            "record_sha256": formation["record_sha256"],
        }
    )
    for comparator in report["comparator_validation"]["records"]:
        acceptance = comparator["mtf50_acceptance"]
        common = {
            "section": "comparator",
            "severity_edge_waves_ref": comparator["target_edge_waves_ref"],
            "family": comparator["comparator_family"],
            "channel": "Rec709_luminance",
            "sampling_level": "declared",
            "censoring": None,
            "record_sha256": comparator["record_sha256"],
        }
        rows.append(
            {
                **common,
                "metric": "implemented_mtf50_relative_match_error",
                "value": acceptance["implemented_relative_error"],
                "unit": "relative",
                "criterion": "less_than_or_equal",
                "tolerance": criteria["implemented_match_mtf50_relative_error_max"],
                "passed": acceptance["implemented_match_passed"],
            }
        )
        for metric, value in comparator["transfer_diagnostics"].items():
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "value": value,
                    "unit": "dimensionless",
                    "criterion": "diagnostic_only",
                    "tolerance": None,
                    "passed": None,
                }
            )
        for metric, value in comparator["full_curve_diagnostics"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rows.append(
                    {
                        **common,
                        "metric": metric,
                        "value": value,
                        "unit": "dimensionless",
                        "criterion": "diagnostic_only",
                        "tolerance": None,
                        "passed": None,
                    }
                )
        for name in ("physical_target_mtf10", "implemented_achieved_mtf10"):
            crossing = comparator["full_curve_diagnostics"][name]
            rows.append(
                {
                    **common,
                    "metric": name,
                    "value": crossing["value_cycles_per_pixel"],
                    "unit": "cycles/pixel",
                    "criterion": "diagnostic_only",
                    "tolerance": None,
                    "passed": None,
                    "censoring": crossing["censoring"],
                }
            )
    return rows


def _curve_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def append_curve(
        severity: float,
        family: str,
        curve: Mapping[str, Any],
    ) -> None:
        rows.extend(
            {
                "severity_edge_waves_ref": severity,
                "family": family,
                "curve_name": curve["name"],
                "frequency_cycles_per_pixel": frequency,
                "value": value,
                "curve_sha256": curve["curve_sha256"],
            }
            for frequency, value in zip(curve["frequency"], curve["values"])
        )

    append_curve(
        0.0,
        "physical_common_neutral",
        report["common_neutral"]["signed_axis_response"],
    )
    for channel in report["analytic_zero_defocus_validation"]["channels"]:
        for curve in channel["incoherent_mtf"]["curves"].values():
            append_curve(
                0.0,
                f"analytic_zero_defocus_{channel['channel']}",
                curve,
            )
    for physical in report["physical_validation"]["severities"]:
        for level in physical["convergence"]["levels"]:
            append_curve(
                physical["edge_waves_ref"],
                f"physical_{level['sampling']['name']}",
                level["curve"],
            )
    for comparator in report["comparator_validation"]["records"]:
        for curve in comparator["curves"].values():
            append_curve(
                comparator["target_edge_waves_ref"],
                comparator["comparator_family"],
                curve,
            )
    return rows


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        serialized = {
            key: (
                ""
                if row.get(key) is None
                else (
                    "true"
                    if row.get(key) is True
                    else "false"
                    if row.get(key) is False
                    else row[key]
                )
            )
            for key in fieldnames
        }
        writer.writerow(serialized)
    return stream.getvalue().encode("utf-8")


def _report_bytes(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def validate_validation_evidence_report(
    report: Mapping[str, Any],
    *,
    expected_dataset: LazyNativeCOCOSubset | None = None,
) -> dict[str, Any]:
    """Rebuild and exactly validate every semantic field in an evidence report."""

    if not isinstance(report, Mapping):
        raise TypeError("validation evidence report must be a mapping")
    normalized = json_value(report)
    if not isinstance(normalized, dict):  # pragma: no cover - narrowed by Mapping
        raise TypeError("validation evidence report must normalize to a mapping")
    if (
        type(normalized.get("schema_version")) is not int
        or normalized.get("schema_version") != _REPORT_SCHEMA_VERSION
        or normalized.get("record_type") != "phycam_scientific_validation_evidence"
        or normalized.get("implementation_id") != _IMPLEMENTATION_ID
    ):
        raise ValueError("unsupported validation evidence report schema or implementation")
    evidence_sha256 = _sha256_digest(
        normalized.get("evidence_sha256"),
        name="report evidence_sha256",
    )
    payload = dict(normalized)
    payload.pop("evidence_sha256", None)
    if canonical_sha256(payload) != evidence_sha256:
        raise ValueError("validation evidence report identity mismatch")
    try:
        provenance = normalized["provenance"]
        criteria_record = provenance["criteria"]
        profile = CameraProfile.from_dict(provenance["camera_profile"])
        ladder = tuple(
            SamplingLevel(
                value["name"],
                value["pupil_grid_size"],
                value["pupil_fft_size"],
            )
            for value in provenance["sampling_ladder"]
        )
        frequency_sample_count = provenance["frequency_grid"]["sample_count"]
        shape_contract = _normalize_shape_contract(provenance["dct_i_axis_shape_contract"])
        physical_waves = tuple(
            value["edge_waves_ref"] for value in normalized["physical_validation"]["severities"]
        )
        comparator_records = normalized["comparator_validation"]["records"]
    except (KeyError, TypeError) as exc:
        raise ValueError("validation evidence report is structurally incomplete") from exc
    if not isinstance(criteria_record, Mapping):
        raise ValueError("validation evidence criteria must be a mapping")
    criteria = ValidationCriteria(
        flux_absolute_error_max=criteria_record.get("flux_absolute_error_max"),
        symmetry_relative_linf_max=criteria_record.get("symmetry_relative_linf_max"),
        convergence_declared_refined_mtf_linf_max=criteria_record.get(
            "convergence_declared_refined_mtf_linf_max"
        ),
        convergence_declared_refined_mtf50_relative_error_max=criteria_record.get(
            "convergence_declared_refined_mtf50_relative_error_max"
        ),
        convergence_error_contraction_ratio_max=criteria_record.get(
            "convergence_error_contraction_ratio_max"
        ),
        implemented_match_mtf50_relative_error_max=criteria_record.get(
            "implemented_match_mtf50_relative_error_max"
        ),
        independent_tent_quadrature_linf_max=criteria_record.get(
            "independent_tent_quadrature_linf_max"
        ),
        independent_tent_retained_energy_absolute_error_max=criteria_record.get(
            "independent_tent_retained_energy_absolute_error_max"
        ),
        independent_equal_grid_formation_linf_max=criteria_record.get(
            "independent_equal_grid_formation_linf_max"
        ),
        analytic_airy_center_relative_error_max=criteria_record.get(
            "analytic_airy_center_relative_error_max"
        ),
        analytic_airy_normalized_intensity_linf_max=criteria_record.get(
            "analytic_airy_normalized_intensity_linf_max"
        ),
        analytic_circular_aperture_mtf_linf_max=criteria_record.get(
            "analytic_circular_aperture_mtf_linf_max"
        ),
    )
    if criteria.to_dict() != criteria_record:
        raise ValueError("validation evidence criteria record is noncanonical")
    if expected_dataset is not None:
        expected_shape_contract = _dataset_shape_contract(expected_dataset)
        if shape_contract != expected_shape_contract:
            raise ValueError(
                "validation evidence DCT-I shape contract does not match the expected dataset"
            )
    comparator_waves: list[float] = []
    for value in comparator_records:
        if not isinstance(value, Mapping):
            raise ValueError("validation evidence comparator record must be a mapping")
        waves = float(value["target_edge_waves_ref"])
        if waves not in comparator_waves:
            comparator_waves.append(waves)
    rebuilt = _build_validation_evidence(
        profile,
        physical_waves,
        shape_contract=shape_contract,
        comparator_edge_waves=tuple(comparator_waves),
        criteria=criteria,
        sampling_ladder=ladder,
        frequency_sample_count=frequency_sample_count,
    )
    if normalized != rebuilt:
        raise ValueError("validation evidence report is semantically inconsistent")
    return rebuilt


def write_validation_evidence(
    report: Mapping[str, Any],
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> ValidationArtifactSet:
    """Write a validated report and its two derived CSV tables.

    Existing output directories require ``overwrite=True``.  Overwriting only
    replaces these three science outputs; unrelated files are left alone.
    """

    report_value = validate_validation_evidence_report(report)
    target = Path(output_directory).resolve()
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"validation evidence target is not a directory: {target}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"validation evidence directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=True)

    evidence_json = target / "validation_evidence.json"
    metrics_csv = target / "validation_metrics.csv"
    curves_csv = target / "validation_curves.csv"
    evidence_json.write_bytes(_report_bytes(report_value))
    metrics_csv.write_bytes(_csv_bytes(_metric_rows(report_value), _METRIC_FIELDS))
    curves_csv.write_bytes(_csv_bytes(_curve_rows(report_value), _CURVE_FIELDS))

    # Clean up the one file produced by the former manifest-based writer when
    # reusing an old output directory.  No new manifest is needed because both
    # CSV files can be regenerated directly from the validated JSON report.
    legacy_manifest = target / "manifest.json"
    if overwrite and legacy_manifest.is_file():
        legacy_manifest.unlink()

    verify_validation_evidence(
        target,
        expected_evidence_sha256=report_value["evidence_sha256"],
    )
    return ValidationArtifactSet(
        root=target,
        evidence_sha256=report_value["evidence_sha256"],
        evidence_json=evidence_json,
        metrics_csv=metrics_csv,
        curves_csv=curves_csv,
    )


def verify_validation_evidence(
    directory: str | Path,
    *,
    expected_evidence_sha256: str | None = None,
    expected_dataset: LazyNativeCOCOSubset | None = None,
) -> dict[str, Any]:
    """Rebuild a saved report and verify that both CSVs derive from it."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"validation evidence directory is missing: {root}")
    try:
        report = json.loads((root / "validation_evidence.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("validation evidence report cannot be parsed") from exc

    rebuilt = validate_validation_evidence_report(report, expected_dataset=expected_dataset)
    if expected_evidence_sha256 is not None and rebuilt["evidence_sha256"] != _sha256_digest(
        expected_evidence_sha256,
        name="expected_evidence_sha256",
    ):
        raise ValueError("validation evidence identity does not match the expected digest")

    expected_metrics = _csv_bytes(_metric_rows(rebuilt), _METRIC_FIELDS)
    expected_curves = _csv_bytes(_curve_rows(rebuilt), _CURVE_FIELDS)
    if (root / "validation_metrics.csv").read_bytes() != expected_metrics:
        raise ValueError("validation metrics CSV does not derive from the evidence report")
    if (root / "validation_curves.csv").read_bytes() != expected_curves:
        raise ValueError("validation curves CSV does not derive from the evidence report")
    return rebuilt


def load_validation_evidence_report(
    directory: str | Path,
    *,
    expected_evidence_sha256: str | None = None,
    expected_dataset: LazyNativeCOCOSubset | None = None,
) -> dict[str, Any]:
    """Load and semantically verify a saved numerical-evidence report."""

    return verify_validation_evidence(
        directory,
        expected_evidence_sha256=expected_evidence_sha256,
        expected_dataset=expected_dataset,
    )


__all__ = [
    "SamplingLevel",
    "ValidationArtifactSet",
    "ValidationCriteria",
    "build_validation_evidence",
    "default_sampling_ladder",
    "load_validation_evidence_report",
    "validate_validation_evidence_report",
    "verify_validation_evidence",
    "write_validation_evidence",
]
