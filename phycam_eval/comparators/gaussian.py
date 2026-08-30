"""Explicitly nonphysical Gaussian comparator for mechanism-matched studies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._canonical import canonical_sha256
from ..eval.image_quality import encircled_energy_curve
from ..eval.mtf import MTFCurve, kernel_axis_mtf, mtf50
from ..optics.convolution import BoundaryPolicy, convolve_spatially_invariant
from ..optics.psf import PixelIntegratedKernel

FloatArray = NDArray[np.float64]


class GaussianMatchCriterion(str, Enum):
    MTF50 = "mtf50"
    EE50 = "ee50"
    EE80 = "ee80"


@dataclass(frozen=True, slots=True)
class GaussianComparatorConfig:
    """One-knob Gaussian definition; ``sigma_pixels=0`` is exact identity."""

    sigma_pixels: float
    boundary: BoundaryPolicy
    truncate_sigma: float = 4.0
    implementation_id: str = "nonphysical-gaussian-linear-light-v1"

    def __post_init__(self) -> None:
        sigma = float(self.sigma_pixels)
        truncate = float(self.truncate_sigma)
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma_pixels must be finite and nonnegative")
        if not math.isfinite(truncate) or truncate <= 0.0:
            raise ValueError("truncate_sigma must be finite and positive")
        if self.boundary not in {"reflect", "constant", "zero", "valid"}:
            raise ValueError("boundary must be reflect, constant, zero, or valid")
        if not isinstance(self.implementation_id, str) or not self.implementation_id:
            raise ValueError("implementation_id must be a non-empty string")
        object.__setattr__(self, "sigma_pixels", 0.0 if sigma == 0.0 else sigma)
        object.__setattr__(self, "truncate_sigma", truncate)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "operator": "gaussian_comparator",
            "physical_model": "none",
            "input_light": "linear",
            "sigma_pixels": self.sigma_pixels,
            "boundary": self.boundary,
            "truncate_sigma": self.truncate_sigma,
            "implementation_id": self.implementation_id,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class MatchedGaussian:
    """A comparator fixed by a pre-detector mechanism-level target."""

    config: GaussianComparatorConfig
    criterion: GaussianMatchCriterion
    target_value: float
    target_unit: str
    achieved_value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "config_sha256": self.config.sha256,
            "criterion": self.criterion.value,
            "target_value": self.target_value,
            "target_unit": self.target_unit,
            "achieved_value": self.achieved_value,
        }


def gaussian_kernel(config: GaussianComparatorConfig) -> FloatArray:
    """Return the finite, normalized sampled kernel declared by ``config``."""

    if not isinstance(config, GaussianComparatorConfig):
        raise TypeError("config must be a GaussianComparatorConfig")
    if config.sigma_pixels == 0.0:
        kernel = np.ones((1, 1), dtype=np.float64)
    else:
        radius = max(1, int(math.ceil(config.truncate_sigma * config.sigma_pixels)))
        coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
        one_d = np.exp(-0.5 * np.square(coordinate / config.sigma_pixels))
        kernel = np.outer(one_d, one_d)
        kernel /= kernel.sum(dtype=np.float64)
    kernel.setflags(write=False)
    return kernel


def apply_gaussian_comparator(
    image: ArrayLike,
    config: GaussianComparatorConfig,
    *,
    constant_value: float = 0.0,
    method: str = "fft",
) -> FloatArray:
    """Apply the comparator without encoding, clipping, or color conversion."""

    array = np.asarray(image, dtype=np.float64)
    if array.ndim not in {2, 3} or any(size == 0 for size in array.shape):
        raise ValueError("image must have shape (H,W) or (H,W,C)")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must contain only finite values")
    if config.sigma_pixels == 0.0:
        return np.array(array, dtype=np.float64, copy=True)
    if method not in {"fft", "direct"}:
        raise ValueError("method must be 'fft' or 'direct'")
    return convolve_spatially_invariant(
        array,
        gaussian_kernel(config),
        boundary=config.boundary,
        constant_value=constant_value,
        method=method,  # type: ignore[arg-type]
    )


def gaussian_mtf(sigma_pixels: float, frequency_cycles_per_pixel: ArrayLike) -> MTFCurve:
    """Evaluate the continuous Gaussian MTF on a declared frequency grid."""

    sigma = float(sigma_pixels)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma_pixels must be finite and nonnegative")
    frequency = np.asarray(frequency_cycles_per_pixel, dtype=np.float64)
    if frequency.ndim != 1 or frequency.size < 2:
        raise ValueError("frequency must be one-dimensional with at least two samples")
    values = np.exp(-2.0 * np.pi**2 * sigma**2 * frequency**2)
    return MTFCurve(frequency, values, "cycles/pixel", "analytic-continuous-gaussian")


def sigma_for_mtf50(mtf50_cycles_per_pixel: float) -> float:
    """Invert ``exp(-2*pi^2*sigma^2*f^2)=0.5``."""

    frequency = float(mtf50_cycles_per_pixel)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("mtf50_cycles_per_pixel must be finite and positive")
    return math.sqrt(math.log(2.0)) / (math.sqrt(2.0) * math.pi * frequency)


def sigma_for_encircled_energy(radius_pixels: float, fraction: float) -> float:
    """Invert the continuous circular-Gaussian encircled-energy equation."""

    radius = float(radius_pixels)
    probability = float(fraction)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_pixels must be finite and positive")
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("fraction must be finite and in (0, 1)")
    return radius / math.sqrt(-2.0 * math.log1p(-probability))


def match_gaussian_to_kernel(
    kernel: PixelIntegratedKernel | ArrayLike,
    *,
    criterion: GaussianMatchCriterion,
    boundary: BoundaryPolicy,
) -> MatchedGaussian:
    """Fit an analytic width and report the response of the finite executed kernel."""

    criterion = GaussianMatchCriterion(criterion)
    values = kernel.values if isinstance(kernel, PixelIntegratedKernel) else np.asarray(kernel)
    if criterion is GaussianMatchCriterion.MTF50:
        target = mtf50(*(lambda curve: (curve.frequency, curve.mtf))(kernel_axis_mtf(values)))
        if not math.isfinite(target):
            raise ValueError("kernel MTF50 is right-censored at image Nyquist")
        sigma = sigma_for_mtf50(target)
        unit = "cycles/pixel"
    else:
        fraction = 0.5 if criterion is GaussianMatchCriterion.EE50 else 0.8
        target = encircled_energy_curve(values).radius_at(fraction, interpolate=True)
        if target <= 0.0:
            raise ValueError("kernel EE radius is unresolved at the central sample")
        sigma = sigma_for_encircled_energy(target, fraction)
        unit = "pixel"
    config = GaussianComparatorConfig(sigma, boundary)
    executed = gaussian_kernel(config)
    if criterion is GaussianMatchCriterion.MTF50:
        curve = kernel_axis_mtf(executed)
        achieved = mtf50(curve.frequency, curve.mtf)
        if not math.isfinite(achieved):
            raise ValueError("matched finite Gaussian kernel MTF50 is right-censored at Nyquist")
    else:
        achieved = encircled_energy_curve(executed).radius_at(fraction, interpolate=True)
    return MatchedGaussian(config, criterion, target, unit, achieved)


__all__ = [
    "GaussianComparatorConfig",
    "GaussianMatchCriterion",
    "MatchedGaussian",
    "apply_gaussian_comparator",
    "gaussian_kernel",
    "gaussian_mtf",
    "match_gaussian_to_kernel",
    "sigma_for_encircled_energy",
    "sigma_for_mtf50",
]
