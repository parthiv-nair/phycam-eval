"""Mechanism-level image and point-source quality diagnostics."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..optics.psf import ContinuousPSF, PixelIntegratedKernel

FloatArray = NDArray[np.float64]


def _readonly_vector(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class EncircledEnergyCurve:
    """Cumulative PSF energy grouped by physical radius."""

    radius: FloatArray
    energy: FloatArray
    radius_unit: str

    def __post_init__(self) -> None:
        radius = _readonly_vector(self.radius, name="radius")
        energy = _readonly_vector(self.energy, name="energy")
        if radius.shape != energy.shape:
            raise ValueError("radius and energy must have equal shape")
        if radius[0] < 0.0 or np.any(np.diff(radius) <= 0.0):
            raise ValueError("radius must be nonnegative and strictly increasing")
        if energy[0] < 0.0 or np.any(np.diff(energy) < -1e-14):
            raise ValueError("energy must be nonnegative and nondecreasing")
        if not np.isclose(energy[-1], 1.0, rtol=0.0, atol=2e-13):
            raise ValueError("encircled energy must end at one")
        if not isinstance(self.radius_unit, str) or not self.radius_unit:
            raise ValueError("radius_unit must be a non-empty string")
        energy_copy = np.array(energy, copy=True)
        energy_copy[-1] = 1.0
        energy_copy.setflags(write=False)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "energy", energy_copy)

    def radius_at(self, fraction: float, *, interpolate: bool = False) -> float:
        """Return the first radius enclosing ``fraction`` of total energy.

        Optional interpolation is between sampled radial rings and is therefore
        a numerical estimate, not a claim of sub-grid PSF knowledge.
        """

        if isinstance(fraction, bool) or not isinstance(fraction, numbers.Real):
            raise TypeError("fraction must be a real number")
        target = float(fraction)
        if not math.isfinite(target) or not 0.0 < target <= 1.0:
            raise ValueError("fraction must be finite and in (0, 1]")
        index = min(int(np.searchsorted(self.energy, target, side="left")), len(self.energy) - 1)
        if not interpolate or index == 0 or self.energy[index] == self.energy[index - 1]:
            return float(self.radius[index])
        fraction_between = (target - self.energy[index - 1]) / (
            self.energy[index] - self.energy[index - 1]
        )
        return float(
            self.radius[index - 1]
            + fraction_between * (self.radius[index] - self.radius[index - 1])
        )


@dataclass(frozen=True, slots=True)
class PointSourceDiagnostics:
    """Flux, centroid, spread, and declared encircled-energy radii."""

    total_energy: float
    centroid_x: float
    centroid_y: float
    rms_radius: float
    peak_mass: float
    radius_unit: str
    encircled_radii: MappingProxyType

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_energy": self.total_energy,
            "centroid": [self.centroid_x, self.centroid_y],
            "rms_radius": self.rms_radius,
            "peak_mass": self.peak_mass,
            "radius_unit": self.radius_unit,
            "encircled_radii": dict(self.encircled_radii),
        }


@dataclass(frozen=True, slots=True)
class ImageDifference:
    """Absolute error statistics between aligned linear-light images."""

    mse: float
    rmse: float
    mae: float
    mean_error: float
    max_abs_error: float
    psnr_db: float
    data_range: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mean_error": self.mean_error,
            "max_abs_error": self.max_abs_error,
            "psnr_db": self.psnr_db,
            "data_range": self.data_range,
            "sample_count": self.sample_count,
        }


def _psf_mass_and_grid(
    psf: ContinuousPSF | PixelIntegratedKernel | ArrayLike,
    *,
    sample_spacing: float | None,
) -> tuple[FloatArray, FloatArray, FloatArray, str, float]:
    if isinstance(psf, ContinuousPSF):
        mass = np.asarray(psf.density * psf.sample_spacing_m**2, dtype=np.float64)
        axis_x = axis_y = np.asarray(psf.axis_m, dtype=np.float64)
        unit = "m"
        unnormalized_total = float(mass.sum(dtype=np.float64))
    elif isinstance(psf, PixelIntegratedKernel):
        mass = np.asarray(psf.values, dtype=np.float64)
        center = psf.values.shape[0] // 2
        axis_x = axis_y = (
            np.arange(psf.values.shape[0], dtype=np.float64) - center
        ) * psf.pixel_pitch_m
        unit = "m"
        unnormalized_total = float(mass.sum(dtype=np.float64))
    else:
        mass = np.asarray(psf, dtype=np.float64)
        if mass.ndim != 2 or any(size == 0 for size in mass.shape):
            raise ValueError("array PSF must be a non-empty 2-D array")
        if sample_spacing is None:
            spacing = 1.0
            unit = "pixel"
        else:
            spacing = float(sample_spacing)
            if not math.isfinite(spacing) or spacing <= 0.0:
                raise ValueError("sample_spacing must be finite and positive")
            unit = "sample_unit"
        axis_y = (np.arange(mass.shape[0], dtype=np.float64) - (mass.shape[0] - 1) / 2.0) * spacing
        axis_x = (np.arange(mass.shape[1], dtype=np.float64) - (mass.shape[1] - 1) / 2.0) * spacing
        unnormalized_total = float(mass.sum(dtype=np.float64))
    if not np.all(np.isfinite(mass)) or np.any(mass < 0.0):
        raise ValueError("PSF mass must be finite and nonnegative")
    if not math.isfinite(unnormalized_total) or unnormalized_total <= 0.0:
        raise ValueError("PSF must have positive total energy")
    return mass / unnormalized_total, axis_x, axis_y, unit, unnormalized_total


def encircled_energy_curve(
    psf: ContinuousPSF | PixelIntegratedKernel | ArrayLike,
    *,
    sample_spacing: float | None = None,
) -> EncircledEnergyCurve:
    """Compute a normalized curve without splitting equal-radius rings."""

    mass, axis_x, axis_y, unit, _ = _psf_mass_and_grid(psf, sample_spacing=sample_spacing)
    x, y = np.meshgrid(axis_x, axis_y, indexing="xy")
    radii = np.hypot(x, y).ravel()
    weights = mass.ravel()
    order = np.argsort(radii, kind="stable")
    radii = radii[order]
    weights = weights[order]
    unique_radius, starts = np.unique(radii, return_index=True)
    ring_mass = np.add.reduceat(weights, starts)
    energy = np.cumsum(ring_mass, dtype=np.float64)
    energy /= energy[-1]
    return EncircledEnergyCurve(unique_radius, energy, unit)


def point_source_diagnostics(
    psf: ContinuousPSF | PixelIntegratedKernel | ArrayLike,
    *,
    fractions: tuple[float, ...] = (0.5, 0.8),
    sample_spacing: float | None = None,
    interpolate_radii: bool = False,
) -> PointSourceDiagnostics:
    """Measure point-source invariants and EE radii in declared grid units."""

    mass, axis_x, axis_y, unit, original_total = _psf_mass_and_grid(
        psf, sample_spacing=sample_spacing
    )
    x, y = np.meshgrid(axis_x, axis_y, indexing="xy")
    centroid_x = float(np.sum(mass * x, dtype=np.float64))
    centroid_y = float(np.sum(mass * y, dtype=np.float64))
    rms_radius = math.sqrt(
        float(
            np.sum(
                mass * ((x - centroid_x) ** 2 + (y - centroid_y) ** 2),
                dtype=np.float64,
            )
        )
    )
    curve = encircled_energy_curve(psf, sample_spacing=sample_spacing)
    radii: dict[str, float] = {}
    for fraction in fractions:
        if not math.isfinite(float(fraction)) or not 0.0 < float(fraction) <= 1.0:
            raise ValueError("fractions must contain finite values in (0, 1]")
        label = f"EE{100.0 * float(fraction):g}"
        if label in radii:
            raise ValueError("fractions must not contain duplicates")
        radii[label] = curve.radius_at(float(fraction), interpolate=interpolate_radii)
    return PointSourceDiagnostics(
        total_energy=original_total,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        rms_radius=rms_radius,
        peak_mass=float(np.max(mass)),
        radius_unit=unit,
        encircled_radii=MappingProxyType(radii),
    )


def compare_images(
    reference: ArrayLike,
    candidate: ArrayLike,
    *,
    data_range: float | None = None,
) -> ImageDifference:
    """Compare aligned arrays without clipping, encoding, or hidden resizing."""

    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if reference_array.shape != candidate_array.shape:
        raise ValueError("reference and candidate must have identical shapes")
    if reference_array.size == 0:
        raise ValueError("images must not be empty")
    if not np.all(np.isfinite(reference_array)) or not np.all(np.isfinite(candidate_array)):
        raise ValueError("images must contain only finite values")
    if data_range is None:
        span = float(np.ptp(reference_array))
        if span <= 0.0:
            raise ValueError("data_range is required for a constant reference")
    else:
        span = float(data_range)
        if not math.isfinite(span) or span <= 0.0:
            raise ValueError("data_range must be finite and positive")
    error = candidate_array - reference_array
    squared = np.square(error, dtype=np.float64)
    mse = float(np.mean(squared, dtype=np.float64))
    rmse = math.sqrt(mse)
    psnr = math.inf if mse == 0.0 else 20.0 * math.log10(span / rmse)
    return ImageDifference(
        mse=mse,
        rmse=rmse,
        mae=float(np.mean(np.abs(error), dtype=np.float64)),
        mean_error=float(np.mean(error, dtype=np.float64)),
        max_abs_error=float(np.max(np.abs(error))),
        psnr_db=psnr,
        data_range=span,
        sample_count=reference_array.size,
    )


__all__ = [
    "EncircledEnergyCurve",
    "ImageDifference",
    "PointSourceDiagnostics",
    "compare_images",
    "encircled_energy_curve",
    "point_source_diagnostics",
]
