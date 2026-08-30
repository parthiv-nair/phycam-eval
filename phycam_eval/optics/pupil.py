"""Centered pupil sampling and physical defocus phase.

This module intentionally contains no image-domain Fourier filtering.  The
complex pupil is first transformed to a coherent amplitude response; image
intensity is filtered only by the resulting incoherent PSF/OTF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def _odd_size(value: int, name: str, minimum: int = 3) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or value % 2 != 1:
        raise ValueError(f"{name} must be odd and at least {minimum}, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class PupilSampling:
    """Numerical sampling identity for an odd, centered normalized pupil.

    ``base_size`` samples include both endpoints ``[-q_max, q_max]``.
    ``fft_size`` is the odd centered zero-padded transform length.
    """

    base_size: int
    q_max: float
    fft_size: int

    def __post_init__(self) -> None:
        base_size = _odd_size(self.base_size, "base_size")
        fft_size = _odd_size(self.fft_size, "fft_size")
        q_max = _positive_finite(self.q_max, "q_max")
        if q_max <= 1.0:
            raise ValueError("q_max must be greater than 1 so the aperture has exterior support")
        if fft_size < base_size:
            raise ValueError("fft_size must be greater than or equal to base_size")
        object.__setattr__(self, "base_size", base_size)
        object.__setattr__(self, "q_max", q_max)
        object.__setattr__(self, "fft_size", fft_size)

    @property
    def delta_q(self) -> float:
        """Normalized-pupil sample spacing, including endpoint convention."""

        return 2.0 * self.q_max / (self.base_size - 1)

    def axis(self) -> FloatArray:
        """Return the exact odd centered pupil coordinate axis."""

        center = (self.base_size - 1) / 2.0
        return (np.arange(self.base_size, dtype=np.float64) - center) * self.delta_q

    def coordinates(self) -> tuple[FloatArray, FloatArray]:
        """Return ``(q_x, q_y)`` arrays with matrix/image indexing."""

        q = self.axis()
        q_x, q_y = np.meshgrid(q, q, indexing="xy")
        return q_x, q_y


def circular_aperture(q_x: NDArray[np.floating], q_y: NDArray[np.floating]) -> FloatArray:
    """Return an ideal unit-amplitude circular aperture, including ``q == 1``."""

    q_x = np.asarray(q_x, dtype=np.float64)
    q_y = np.asarray(q_y, dtype=np.float64)
    if q_x.shape != q_y.shape:
        raise ValueError("q_x and q_y must have identical shapes")
    if q_x.ndim != 2:
        raise ValueError("pupil coordinates must be two-dimensional")
    return (q_x * q_x + q_y * q_y <= 1.0).astype(np.float64)


def wavelength_scaled_defocus(
    edge_waves_ref: float,
    reference_wavelength_m: float,
    wavelength_m: float,
) -> float:
    """Convert reference edge-to-center waves to a channel wavelength."""

    edge_waves_ref = float(edge_waves_ref)
    if not math.isfinite(edge_waves_ref):
        raise ValueError("edge_waves_ref must be finite")
    reference_wavelength_m = _positive_finite(reference_wavelength_m, "reference_wavelength_m")
    wavelength_m = _positive_finite(wavelength_m, "wavelength_m")
    return edge_waves_ref * reference_wavelength_m / wavelength_m


def defocus_phase(radius_squared: NDArray[np.floating], edge_waves: float) -> FloatArray:
    """Return ``2*pi*W*q^2`` radians for edge-to-center defocus ``W``."""

    edge_waves = float(edge_waves)
    if not math.isfinite(edge_waves):
        raise ValueError("edge_waves must be finite")
    radius_squared = np.asarray(radius_squared, dtype=np.float64)
    if np.any(radius_squared < 0.0) or not np.all(np.isfinite(radius_squared)):
        raise ValueError("radius_squared must contain finite nonnegative values")
    return 2.0 * np.pi * edge_waves * radius_squared


def complex_pupil(sampling: PupilSampling, edge_waves: float) -> ComplexArray:
    """Construct the circular complex pupil for one wavelength.

    A piston term is not subtracted: it has no effect on the resulting
    incoherent intensity PSF and retaining the literal edge-to-center phase
    keeps the convention directly inspectable.
    """

    q_x, q_y = sampling.coordinates()
    radius_squared = q_x * q_x + q_y * q_y
    aperture = circular_aperture(q_x, q_y)
    pupil = aperture * np.exp(1j * defocus_phase(radius_squared, edge_waves))
    return np.asarray(pupil, dtype=np.complex128)


def centered_zero_pad(array: NDArray[np.generic], target_size: int) -> NDArray[np.generic]:
    """Center an odd square array in an odd square zero-padded array."""

    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("array must be square and two-dimensional")
    source_size = _odd_size(array.shape[0], "source size", minimum=1)
    target_size = _odd_size(target_size, "target_size", minimum=1)
    if target_size < source_size:
        raise ValueError("target_size cannot be smaller than the source array")
    before = (target_size - source_size) // 2
    after = target_size - source_size - before
    return np.pad(array, ((before, after), (before, after)), mode="constant")
