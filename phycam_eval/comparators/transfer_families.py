"""Boundary-harmonized transfer-family comparators for the main LDR study.

These operators adapt two legacy stress families to the v2 experimental
contract. Both consume linear-light arrays. The sampled-incoherent family uses
the same explicit padded spatial convolution boundary as the physical model;
the signed quadratic-cosine family uses a DCT-I whole-sample even extension,
matching the physical branch's reflect continuation rather than a circular
FFT. Exact schema-v1 gamma/circular behavior
belongs only in a separately labelled historical supplement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import dctn, idctn

from .._canonical import canonical_sha256
from ..optics.convolution import BoundaryPolicy, convolve_spatially_invariant

FloatArray = NDArray[np.float64]


def _alpha(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("alpha must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("alpha must be a real number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    return 0.0 if result == 0.0 else result


def _image(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim not in {2, 3} or any(size <= 0 for size in array.shape):
        raise ValueError("image must have shape (H,W) or (H,W,C)")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class QuadraticCosineComparatorConfig:
    """Signed ``cos(alpha*rho^2)`` response under whole-sample reflection."""

    alpha: float
    implementation_id: str = "adapted-quadratic-cosine-dct-i-reflect-linear-v2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _alpha(self.alpha))
        if not isinstance(self.implementation_id, str) or not self.implementation_id:
            raise ValueError("implementation_id must be a nonempty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": "adapted_quadratic_cosine",
            "physical_model": "none",
            "legacy_family": "quadratic_phase_real_projection",
            "input_light": "linear",
            "alpha": self.alpha,
            "radial_normalization": "theoretical_2d_nyquist_corner",
            "boundary": "dct_i_whole_sample_reflect_extension",
            "output_clipping": None,
            "implementation_id": self.implementation_id,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def quadratic_cosine_response(
    shape: tuple[int, int],
    config: QuadraticCosineComparatorConfig,
) -> FloatArray:
    """Return the signed DCT-domain response for one spatial shape."""

    if not isinstance(config, QuadraticCosineComparatorConfig):
        raise TypeError("config must be a QuadraticCosineComparatorConfig")
    if (
        isinstance(shape, (str, bytes))
        or len(shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in shape
        )
    ):
        raise TypeError("shape must contain integer height and width")
    height, width = (int(value) for value in shape)
    if height <= 0 or width <= 0:
        raise ValueError("shape dimensions must be positive")
    frequency_y = (
        np.zeros(1, dtype=np.float64)
        if height == 1
        else np.arange(height, dtype=np.float64) / (2.0 * (height - 1))
    )
    frequency_x = (
        np.zeros(1, dtype=np.float64)
        if width == 1
        else np.arange(width, dtype=np.float64) / (2.0 * (width - 1))
    )
    radial_squared = (np.square(frequency_y[:, None]) + np.square(frequency_x[None, :])) / 0.5
    response = np.cos(config.alpha * radial_squared)
    response.setflags(write=False)
    return response


def apply_quadratic_cosine_comparator(
    image: ArrayLike,
    config: QuadraticCosineComparatorConfig,
) -> FloatArray:
    """Apply the signed comparator without nonlinear clipping."""

    values = _image(image)
    if not isinstance(config, QuadraticCosineComparatorConfig):
        raise TypeError("config must be a QuadraticCosineComparatorConfig")
    if config.alpha == 0.0:
        return np.array(values, dtype=np.float64, copy=True)
    response = quadratic_cosine_response(values.shape[:2], config)
    # DCT-I is the real Fourier basis of np.pad(mode="reflect")'s
    # whole-sample even continuation.  The unnormalized forward/inverse pair
    # preserves a constant exactly; orthonormal DCT-I endpoint weights would
    # not represent a constant with a single DC coefficient.
    axes = tuple(axis for axis in (0, 1) if values.shape[axis] > 1)
    transformed = (
        dctn(values, axes=axes, type=1, norm=None)
        if axes
        else np.array(values, dtype=np.float64, copy=True)
    )
    if values.ndim == 3:
        response = response[:, :, None]
    return np.asarray(
        idctn(transformed * response, axes=axes, type=1, norm=None)
        if axes
        else transformed * response,
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class SampledIncoherentComparatorConfig:
    """Finite nonnegative PSF derived from the legacy full-grid pupil family."""

    alpha: float
    boundary: BoundaryPolicy = "reflect"
    pupil_grid_size: int = 257
    encircled_energy: float = 0.999
    implementation_id: str = "adapted-sampled-incoherent-spatial-linear-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _alpha(self.alpha))
        if self.boundary not in {"reflect", "constant", "zero"}:
            raise ValueError("boundary must be reflect, constant, or zero")
        if (
            isinstance(self.pupil_grid_size, bool)
            or not isinstance(self.pupil_grid_size, (int, np.integer))
            or int(self.pupil_grid_size) < 33
            or int(self.pupil_grid_size) % 2 != 1
        ):
            raise ValueError("pupil_grid_size must be an odd integer of at least 33")
        energy = float(self.encircled_energy)
        if not math.isfinite(energy) or not 0.0 < energy <= 1.0:
            raise ValueError("encircled_energy must lie in (0, 1]")
        if not isinstance(self.implementation_id, str) or not self.implementation_id:
            raise ValueError("implementation_id must be a nonempty string")
        object.__setattr__(self, "pupil_grid_size", int(self.pupil_grid_size))
        object.__setattr__(self, "encircled_energy", energy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": "adapted_sampled_incoherent_quadratic_pupil",
            "physical_model": "none",
            "legacy_family": "full_grid_sampled_incoherent_quadratic_pupil",
            "input_light": "linear",
            "alpha": self.alpha,
            "pupil_grid_size": self.pupil_grid_size,
            "radial_normalization": "sampled_grid_max_radius",
            "encircled_energy": self.encircled_energy,
            "boundary": self.boundary,
            "output_clipping": None,
            "implementation_id": self.implementation_id,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _energy_crop(values: FloatArray, fraction: float) -> tuple[FloatArray, float]:
    center = values.shape[0] // 2
    y, x = np.indices(values.shape, dtype=np.float64)
    radius = np.hypot(y - center, x - center).ravel()
    mass = values.ravel()
    order = np.argsort(radius, kind="stable")
    cumulative = np.cumsum(mass[order], dtype=np.float64)
    selected = min(
        int(np.searchsorted(cumulative, fraction * float(mass.sum()), side="left")),
        len(order) - 1,
    )
    half_width = min(center, int(math.ceil(radius[order[selected]])))
    cropped = np.array(
        values[
            center - half_width : center + half_width + 1,
            center - half_width : center + half_width + 1,
        ],
        dtype=np.float64,
        copy=True,
    )
    retained = float(cropped.sum(dtype=np.float64))
    cropped /= retained
    immutable_storage = cropped.tobytes(order="C")
    immutable = np.frombuffer(immutable_storage, dtype=cropped.dtype).reshape(cropped.shape)
    return immutable, retained


@lru_cache(maxsize=128)
def _sampled_incoherent_kernel_cached(
    config: SampledIncoherentComparatorConfig,
) -> tuple[FloatArray, float]:
    size = config.pupil_grid_size
    frequency = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    radial = np.hypot(fy, fx)
    radial /= float(np.max(radial))
    pupil = np.exp(1j * config.alpha * np.square(radial))
    coherent = np.fft.ifft2(pupil)
    intensity = np.fft.fftshift(np.square(np.abs(coherent)))
    intensity /= intensity.sum(dtype=np.float64)
    return _energy_crop(np.asarray(intensity, dtype=np.float64), config.encircled_energy)


def sampled_incoherent_kernel(
    config: SampledIncoherentComparatorConfig,
) -> FloatArray:
    """Return the finite normalized PSF used by the adapted comparator."""

    if not isinstance(config, SampledIncoherentComparatorConfig):
        raise TypeError("config must be a SampledIncoherentComparatorConfig")
    return _sampled_incoherent_kernel_cached(config)[0]


def sampled_incoherent_retained_energy(
    config: SampledIncoherentComparatorConfig,
) -> float:
    """Return the pre-renormalization mass retained by the finite support."""

    if not isinstance(config, SampledIncoherentComparatorConfig):
        raise TypeError("config must be a SampledIncoherentComparatorConfig")
    return _sampled_incoherent_kernel_cached(config)[1]


def apply_sampled_incoherent_comparator(
    image: ArrayLike,
    config: SampledIncoherentComparatorConfig,
    *,
    constant_value: float = 0.0,
) -> FloatArray:
    """Apply the finite PSF under the declared non-circular boundary."""

    values = _image(image)
    if not isinstance(config, SampledIncoherentComparatorConfig):
        raise TypeError("config must be a SampledIncoherentComparatorConfig")
    if config.alpha == 0.0:
        return np.array(values, dtype=np.float64, copy=True)
    return convolve_spatially_invariant(
        values,
        sampled_incoherent_kernel(config),
        boundary=config.boundary,
        constant_value=constant_value,
        method="fft",
    )


__all__ = [
    "QuadraticCosineComparatorConfig",
    "SampledIncoherentComparatorConfig",
    "apply_quadratic_cosine_comparator",
    "apply_sampled_incoherent_comparator",
    "quadratic_cosine_response",
    "sampled_incoherent_kernel",
    "sampled_incoherent_retained_energy",
]
