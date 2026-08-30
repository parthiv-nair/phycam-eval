"""Physical pupil-to-PSF/OTF conversion and exactly-once pixel integration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import j1

from .pupil import PupilSampling, centered_zero_pad

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _readonly_float_array(value: Any, *, ndim: int, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    immutable_storage = array.tobytes(order="C")
    return np.frombuffer(immutable_storage, dtype=array.dtype).reshape(array.shape)


def _readonly_complex_array(value: Any, *, ndim: int, name: str) -> ComplexArray:
    array = np.array(value, dtype=np.complex128, copy=True, order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    immutable_storage = array.tobytes(order="C")
    return np.frombuffer(immutable_storage, dtype=array.dtype).reshape(array.shape)


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class ContinuousPSF:
    """Oversampled continuous-density incoherent PSF.

    ``density`` has units m^-2, and its Riemann sum with ``sample_spacing_m``
    is one.  It has not been integrated over a photosite.
    """

    density: FloatArray
    axis_m: FloatArray
    sample_spacing_m: float
    wavelength_m: float
    f_number: float
    delta_q: float
    edge_waves: float
    model_identity: str = ""
    pixel_integrated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        density = _readonly_float_array(self.density, ndim=2, name="density")
        axis = _readonly_float_array(self.axis_m, ndim=1, name="axis_m")
        if density.shape[0] != density.shape[1] or density.shape[0] != axis.size:
            raise ValueError("density must be square and match axis_m")
        if density.shape[0] % 2 != 1:
            raise ValueError("density must use an odd centered grid")
        if np.any(density < 0.0):
            raise ValueError("PSF density cannot be negative")
        sample_spacing_m = _positive_finite(self.sample_spacing_m, "sample_spacing_m")
        wavelength_m = _positive_finite(self.wavelength_m, "wavelength_m")
        f_number = _positive_finite(self.f_number, "f_number")
        delta_q = _positive_finite(self.delta_q, "delta_q")
        edge_waves = float(self.edge_waves)
        if not math.isfinite(edge_waves):
            raise ValueError("edge_waves must be finite")
        if axis.size > 1 and not np.allclose(np.diff(axis), sample_spacing_m, rtol=2e-13, atol=0.0):
            raise ValueError("axis_m spacing is inconsistent with sample_spacing_m")
        center = axis.size // 2
        if axis[center] != 0.0 or not np.allclose(
            axis,
            -axis[::-1],
            rtol=2e-13,
            atol=0.0,
        ):
            raise ValueError("axis_m must be centered at zero and antisymmetric")
        energy = float(density.sum(dtype=np.float64) * sample_spacing_m**2)
        if not math.isclose(energy, 1.0, rel_tol=5e-13, abs_tol=5e-13):
            raise ValueError(f"PSF density integral must equal one, got {energy:.17g}")
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "axis_m", axis)
        object.__setattr__(self, "sample_spacing_m", sample_spacing_m)
        object.__setattr__(self, "wavelength_m", wavelength_m)
        object.__setattr__(self, "f_number", f_number)
        object.__setattr__(self, "delta_q", delta_q)
        object.__setattr__(self, "edge_waves", edge_waves)

    @property
    def energy(self) -> float:
        return float(self.density.sum(dtype=np.float64) * self.sample_spacing_m**2)


@dataclass(frozen=True, slots=True)
class OpticalTransferFunction:
    """Centered optical transfer function on physical cycles-per-meter bins."""

    values: ComplexArray
    frequency_axis_cpm: FloatArray
    model_identity: str = ""

    def __post_init__(self) -> None:
        values = _readonly_complex_array(self.values, ndim=2, name="values")
        frequency_axis = _readonly_float_array(
            self.frequency_axis_cpm, ndim=1, name="frequency_axis_cpm"
        )
        if values.shape[0] != values.shape[1] or values.shape[0] != frequency_axis.size:
            raise ValueError("OTF must be square and match frequency_axis_cpm")
        if values.shape[0] % 2 != 1:
            raise ValueError("OTF must use an odd centered grid")
        center = values.shape[0] // 2
        if not np.isclose(values[center, center], 1.0 + 0.0j, rtol=1e-13, atol=1e-13):
            raise ValueError("OTF centered DC value must equal one")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "frequency_axis_cpm", frequency_axis)


@dataclass(frozen=True, slots=True)
class PixelIntegratedKernel:
    """Dimensionless sampled kernel with pixel aperture applied exactly once."""

    values: FloatArray
    pixel_pitch_m: float
    retained_energy: float
    requested_encircled_energy: float
    source_identity: str = ""
    pixel_integrated: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        values = _readonly_float_array(self.values, ndim=2, name="values")
        if values.shape[0] != values.shape[1] or values.shape[0] % 2 != 1:
            raise ValueError("pixel-integrated kernel must be odd and square")
        if np.any(values < 0.0):
            raise ValueError("pixel-integrated kernel cannot be negative")
        if not np.isclose(values.sum(dtype=np.float64), 1.0, rtol=5e-13, atol=5e-13):
            raise ValueError("pixel-integrated kernel samples must sum to one")
        pixel_pitch_m = _positive_finite(self.pixel_pitch_m, "pixel_pitch_m")
        retained_energy = float(self.retained_energy)
        requested = float(self.requested_encircled_energy)
        if not math.isfinite(retained_energy) or not (0.0 < retained_energy <= 1.0 + 5e-13):
            raise ValueError("retained_energy must be finite and in (0, 1]")
        if not math.isfinite(requested) or not (0.0 < requested <= 1.0):
            raise ValueError("requested_encircled_energy must be finite and in (0, 1]")
        if retained_energy + 5e-13 < requested:
            raise ValueError("retained_energy cannot be below requested_encircled_energy")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "pixel_pitch_m", pixel_pitch_m)
        object.__setattr__(self, "retained_energy", min(retained_energy, 1.0))
        object.__setattr__(self, "requested_encircled_energy", requested)


class PixelIntegrationError(ValueError):
    """Raised when a photosite aperture would be integrated more than once."""


def pupil_to_psf(
    pupil: NDArray[np.complexfloating],
    sampling: PupilSampling,
    wavelength_m: float,
    f_number: float,
    *,
    edge_waves: float = 0.0,
    model_identity: str = "",
) -> ContinuousPSF:
    """Transform a complex pupil into a normalized continuous-density PSF."""

    wavelength_m = _positive_finite(wavelength_m, "wavelength_m")
    f_number = _positive_finite(f_number, "f_number")
    edge_waves = float(edge_waves)
    if not math.isfinite(edge_waves):
        raise ValueError("edge_waves must be finite")
    pupil = np.asarray(pupil, dtype=np.complex128)
    expected_shape = (sampling.base_size, sampling.base_size)
    if pupil.shape != expected_shape:
        raise ValueError(f"pupil shape must be {expected_shape}, got {pupil.shape}")
    if not np.all(np.isfinite(pupil)):
        raise ValueError("pupil must contain finite values")
    if not np.any(np.abs(pupil) > 0.0):
        raise ValueError("pupil cannot be identically zero")

    padded = centered_zero_pad(pupil, sampling.fft_size)
    amplitude = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded))) * sampling.delta_q**2
    intensity = np.abs(amplitude) ** 2
    sample_spacing_m = 2.0 * wavelength_m * f_number / (sampling.fft_size * sampling.delta_q)
    normalizer = intensity.sum(dtype=np.float64) * sample_spacing_m**2
    if not math.isfinite(float(normalizer)) or normalizer <= 0.0:
        raise RuntimeError("invalid pupil transform energy")
    density = np.asarray(intensity / normalizer, dtype=np.float64)

    nu = np.fft.fftshift(np.fft.fftfreq(sampling.fft_size, d=sampling.delta_q))
    axis_m = np.asarray(2.0 * wavelength_m * f_number * nu, dtype=np.float64)
    return ContinuousPSF(
        density=density,
        axis_m=axis_m,
        sample_spacing_m=sample_spacing_m,
        wavelength_m=wavelength_m,
        f_number=f_number,
        delta_q=sampling.delta_q,
        edge_waves=edge_waves,
        model_identity=model_identity,
    )


def psf_to_otf(psf: ContinuousPSF) -> OpticalTransferFunction:
    """Return the centered, DC-normalized optical OTF of an uncropped PSF."""

    transform = (
        np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf.density))) * psf.sample_spacing_m**2
    )
    center = transform.shape[0] // 2
    dc = transform[center, center]
    if abs(dc) == 0.0 or not np.isfinite(dc):
        raise RuntimeError("PSF transform has invalid DC value")
    values = np.asarray(transform / dc, dtype=np.complex128)
    values[center, center] = 1.0 + 0.0j
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(psf.density.shape[0], d=psf.sample_spacing_m))
    return OpticalTransferFunction(
        values=values,
        frequency_axis_cpm=np.asarray(frequency_axis, dtype=np.float64),
        model_identity=psf.model_identity,
    )


def airy_psf_density(
    radius_m: NDArray[np.floating] | float,
    wavelength_m: float,
    f_number: float,
) -> FloatArray:
    """Analytic unit-energy Airy intensity density for a clear circular pupil."""

    wavelength_m = _positive_finite(wavelength_m, "wavelength_m")
    f_number = _positive_finite(f_number, "f_number")
    radius = np.asarray(radius_m, dtype=np.float64)
    if np.any(radius < 0.0) or not np.all(np.isfinite(radius)):
        raise ValueError("radius_m must contain finite nonnegative values")
    scale = wavelength_m * f_number
    z = np.pi * radius / scale
    amplitude = np.ones_like(z)
    nonzero = z != 0.0
    amplitude[nonzero] = 2.0 * j1(z[nonzero]) / z[nonzero]
    return np.asarray((np.pi / (4.0 * scale**2)) * amplitude**2, dtype=np.float64)


def _pixel_overlap_matrix(axis_m: FloatArray, spacing_m: float, pixel_pitch_m: float) -> FloatArray:
    """Lengths shared by piecewise-constant PSF cells and sensor pixels."""

    source_half_extent = axis_m.size * spacing_m / 2.0
    max_pixel_index = max(0, math.ceil(source_half_extent / pixel_pitch_m + 0.5) - 1)
    pixel_centers = (
        np.arange(-max_pixel_index, max_pixel_index + 1, dtype=np.float64) * pixel_pitch_m
    )
    source_left = axis_m - spacing_m / 2.0
    source_right = axis_m + spacing_m / 2.0
    pixel_left = pixel_centers - pixel_pitch_m / 2.0
    pixel_right = pixel_centers + pixel_pitch_m / 2.0
    overlap = np.minimum(pixel_right[:, None], source_right[None, :]) - np.maximum(
        pixel_left[:, None], source_left[None, :]
    )
    return np.maximum(overlap, 0.0)


def _crop_encircled_energy(
    values: FloatArray,
    pixel_pitch_m: float,
    threshold: float,
) -> tuple[FloatArray, float]:
    center = values.shape[0] // 2
    coordinates = (np.arange(values.shape[0], dtype=np.float64) - center) * pixel_pitch_m
    x_m, y_m = np.meshgrid(coordinates, coordinates, indexing="xy")
    radius = np.hypot(x_m, y_m).ravel()
    mass = values.ravel()
    order = np.argsort(radius, kind="stable")
    cumulative = np.cumsum(mass[order], dtype=np.float64)
    target = threshold * float(mass.sum(dtype=np.float64))
    selected = min(int(np.searchsorted(cumulative, target, side="left")), order.size - 1)
    support_radius_m = radius[order[selected]]
    half_width = min(center, int(math.ceil(support_radius_m / pixel_pitch_m)))
    cropped = np.array(
        values[
            center - half_width : center + half_width + 1,
            center - half_width : center + half_width + 1,
        ],
        dtype=np.float64,
        copy=True,
    )
    retained = float(cropped.sum(dtype=np.float64))
    if retained <= 0.0:
        raise RuntimeError("energy crop produced an empty kernel")
    cropped /= retained
    return cropped, retained


def pixel_integrate_psf(
    psf: ContinuousPSF | PixelIntegratedKernel,
    pixel_pitch_m: float,
    *,
    encircled_energy: float = 1.0,
) -> PixelIntegratedKernel:
    """Integrate an oversampled PSF into sensor pixels exactly once.

    The continuous samples are treated as density values on centered square
    cells.  Exact overlap lengths integrate those cells into the sensor-pixel
    lattice, including non-integer ratios of pixel pitch to PSF spacing.
    """

    if getattr(psf, "pixel_integrated", False):
        raise PixelIntegrationError("pixel aperture has already been integrated")
    if not isinstance(psf, ContinuousPSF):
        raise TypeError("psf must be a ContinuousPSF")
    pixel_pitch_m = _positive_finite(pixel_pitch_m, "pixel_pitch_m")
    if psf.sample_spacing_m > pixel_pitch_m * (1.0 + 1e-12):
        raise ValueError(
            "continuous PSF must be sampled at least as finely as the sensor pixel pitch"
        )
    encircled_energy = float(encircled_energy)
    if not (0.0 < encircled_energy <= 1.0) or not math.isfinite(encircled_energy):
        raise ValueError("encircled_energy must be finite and in (0, 1]")

    overlap = _pixel_overlap_matrix(psf.axis_m, psf.sample_spacing_m, pixel_pitch_m)
    integrated = np.asarray(overlap @ psf.density @ overlap.T, dtype=np.float64)
    integrated[integrated < 0.0] = 0.0
    total = float(integrated.sum(dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("pixel integration produced invalid energy")
    if not math.isclose(total, psf.energy, rel_tol=5e-12, abs_tol=5e-13):
        raise RuntimeError("pixel integration did not cover the continuous PSF support")
    integrated /= total
    cropped, retained = _crop_encircled_energy(integrated, pixel_pitch_m, encircled_energy)
    return PixelIntegratedKernel(
        values=cropped,
        pixel_pitch_m=pixel_pitch_m,
        retained_energy=retained,
        requested_encircled_energy=encircled_energy,
        source_identity=psf.model_identity,
    )
