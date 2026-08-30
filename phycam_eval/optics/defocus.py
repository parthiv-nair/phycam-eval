"""Physical defocus construction, caching, and point-sample diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from threading import RLock
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .convolution import BoundaryPolicy, ConvolutionMethod, convolve_spatially_invariant
from .psf import (
    ContinuousPSF,
    OpticalTransferFunction,
    PixelIntegratedKernel,
    pixel_integrate_psf,
    psf_to_otf,
    pupil_to_psf,
)
from .pupil import PupilSampling, complex_pupil, wavelength_scaled_defocus

_MODEL_VERSION = "physical-defocus-v1"


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def _float_identity(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("cache identity cannot contain non-finite floats")
    if value == 0.0:
        value = 0.0
    return value.hex()


@dataclass(frozen=True, slots=True)
class DefocusConfig:
    """Complete identity of a circular-pupil PSF and diagnostic model."""

    f_number: float
    pixel_pitch_m: float
    wavelengths_m: tuple[float, ...]
    reference_wavelength_m: float
    edge_waves_ref: float
    pupil_sampling: PupilSampling
    encircled_energy: float

    def __post_init__(self) -> None:
        f_number = _positive_finite(self.f_number, "f_number")
        pixel_pitch_m = _positive_finite(self.pixel_pitch_m, "pixel_pitch_m")
        reference_wavelength_m = _positive_finite(
            self.reference_wavelength_m, "reference_wavelength_m"
        )
        wavelengths = tuple(
            _positive_finite(value, f"wavelengths_m[{index}]")
            for index, value in enumerate(self.wavelengths_m)
        )
        if not wavelengths:
            raise ValueError("wavelengths_m cannot be empty")
        edge_waves_ref = float(self.edge_waves_ref)
        if not math.isfinite(edge_waves_ref):
            raise ValueError("edge_waves_ref must be finite")
        if not isinstance(self.pupil_sampling, PupilSampling):
            raise TypeError("pupil_sampling must be a PupilSampling")
        encircled_energy = float(self.encircled_energy)
        if not math.isfinite(encircled_energy) or not (0.0 < encircled_energy <= 1.0):
            raise ValueError("encircled_energy must be finite and in (0, 1]")

        finest_dx = max(
            2.0
            * wavelength
            * f_number
            / (self.pupil_sampling.fft_size * self.pupil_sampling.delta_q)
            for wavelength in wavelengths
        )
        if finest_dx > pixel_pitch_m * (1.0 + 1e-12):
            raise ValueError(
                "pupil FFT must oversample the sensor grid for every configured wavelength"
            )
        object.__setattr__(self, "f_number", f_number)
        object.__setattr__(self, "pixel_pitch_m", pixel_pitch_m)
        object.__setattr__(self, "wavelengths_m", wavelengths)
        object.__setattr__(self, "reference_wavelength_m", reference_wavelength_m)
        object.__setattr__(self, "edge_waves_ref", edge_waves_ref)
        object.__setattr__(self, "encircled_energy", encircled_energy)

    def identity_payload(self) -> dict[str, Any]:
        """Return an exact, language-independent cache identity payload."""

        sampling = self.pupil_sampling
        return {
            "model": _MODEL_VERSION,
            "aperture": "ideal-circular-inclusive-edge",
            "phase": "edge-to-center-quadratic-waves",
            "fft": "numpy-forward-centered-odd-v1",
            "pixel_integration": "piecewise-constant-cell-overlap-v1",
            "energy_crop": "center-radius-square-support-v1",
            "f_number": _float_identity(self.f_number),
            "pixel_pitch_m": _float_identity(self.pixel_pitch_m),
            "wavelengths_m": [_float_identity(value) for value in self.wavelengths_m],
            "reference_wavelength_m": _float_identity(self.reference_wavelength_m),
            "edge_waves_ref": _float_identity(self.edge_waves_ref),
            "encircled_energy": _float_identity(self.encircled_energy),
            "pupil": {
                "base_size": sampling.base_size,
                "q_max": _float_identity(sampling.q_max),
                "fft_size": sampling.fft_size,
            },
        }

    @property
    def cache_key(self) -> str:
        canonical = json.dumps(
            self.identity_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class ChannelDefocus:
    wavelength_m: float
    edge_waves: float
    psf: ContinuousPSF
    otf: OpticalTransferFunction
    kernel: PixelIntegratedKernel


@dataclass(frozen=True, slots=True)
class DefocusModel:
    config: DefocusConfig
    channels: tuple[ChannelDefocus, ...]
    cache_key: str


_CACHE: dict[str, DefocusModel] = {}
_CACHE_LOCK = RLock()


def clear_defocus_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def defocus_cache_size() -> int:
    with _CACHE_LOCK:
        return len(_CACHE)


def _construct_model(config: DefocusConfig) -> DefocusModel:
    identity = config.cache_key
    channels = []
    for wavelength_m in config.wavelengths_m:
        edge_waves = wavelength_scaled_defocus(
            config.edge_waves_ref,
            config.reference_wavelength_m,
            wavelength_m,
        )
        pupil = complex_pupil(config.pupil_sampling, edge_waves)
        psf = pupil_to_psf(
            pupil,
            config.pupil_sampling,
            wavelength_m,
            config.f_number,
            edge_waves=edge_waves,
            model_identity=identity,
        )
        otf = psf_to_otf(psf)
        kernel = pixel_integrate_psf(
            psf,
            config.pixel_pitch_m,
            encircled_energy=config.encircled_energy,
        )
        channels.append(
            ChannelDefocus(
                wavelength_m=wavelength_m,
                edge_waves=edge_waves,
                psf=psf,
                otf=otf,
                kernel=kernel,
            )
        )
    return DefocusModel(config=config, channels=tuple(channels), cache_key=identity)


def build_defocus_model(config: DefocusConfig, *, use_cache: bool = True) -> DefocusModel:
    """Build or retrieve a physical defocus model by complete identity."""

    if not isinstance(config, DefocusConfig):
        raise TypeError("config must be a DefocusConfig")
    if not use_cache:
        return _construct_model(config)
    key = config.cache_key
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached
    constructed = _construct_model(config)
    with _CACHE_LOCK:
        return _CACHE.setdefault(key, constructed)


def apply_defocus(
    image: ArrayLike,
    config: DefocusConfig,
    *,
    boundary: BoundaryPolicy,
    constant_value: float = 0.0,
    method: ConvolutionMethod = "fft",
    use_cache: bool = True,
) -> NDArray[np.float64]:
    """Apply the single-aperture point-sample diagnostic to linear light.

    Native LDR formation instead uses ``collapse_cell_average_transfer`` so
    its declared finite source cells and target photosites are both present.
    """

    image_array = np.asarray(image, dtype=np.float64)
    model = build_defocus_model(config, use_cache=use_cache)
    if image_array.ndim == 2:
        if len(model.channels) != 1:
            raise ValueError("a 2-D image requires exactly one configured wavelength")
        kernel: NDArray[np.float64] | PixelIntegratedKernel = model.channels[0].kernel
    elif image_array.ndim == 3:
        if image_array.shape[2] != len(model.channels):
            raise ValueError("image channels must match configured wavelengths")
        max_size = max(channel.kernel.values.shape[0] for channel in model.channels)
        channel_kernels = []
        for channel in model.channels:
            values = channel.kernel.values
            before = (max_size - values.shape[0]) // 2
            after = max_size - values.shape[0] - before
            channel_kernels.append(
                np.pad(values, ((before, after), (before, after)), mode="constant")
            )
        kernel = np.stack(channel_kernels, axis=2)
    else:
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    return convolve_spatially_invariant(
        image_array,
        kernel,
        boundary=boundary,
        constant_value=constant_value,
        method=method,
    )
