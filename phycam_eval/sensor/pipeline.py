"""Sensor-tier orchestration for linear-RGB approximation and Bayer RAW capture.

The two public capture functions share the same electron, gain, and ADC model,
but their sampling contracts remain deliberately distinct:

* ``capture_linear_rgb`` retains three camera channels and does not mosaic.
* ``capture_bayer_raw`` selects one globally positioned CFA photosite channel
  and returns a signed, black-subtracted RAW mosaic without demosaicing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..profiles import SensorProfile
from .adc import (
    ADCProfile,
    GainSetting,
    black_subtract_and_normalize,
    quantize_adc,
    validate_headroom,
)
from .cfa import BayerPattern, cfa_channel_indices
from .exposure import ExposureSetting, expected_dark_electrons, expected_photoelectrons
from .noise import ElectronCapture, RNGKey, StatelessRNG, capture_electrons

FloatArray = NDArray[np.float64]
UInt32Array = NDArray[np.uint32]
UInt64Array = NDArray[np.uint64]
BoolArray = NDArray[np.bool_]
Int8Array = NDArray[np.int8]

SENSOR_CAPTURE_NOISE_SOURCE = "sensor_capture"


class GainPolicy(str, Enum):
    """Explicit gain behavior for a photon-budget sweep."""

    FIXED_PROFILE = "fixed_profile"
    EXPOSURE_COMPENSATED = "exposure_compensated"


def _readonly_array(value: Any, dtype: np.dtype[Any], *, name: str) -> NDArray[Any]:
    array = np.array(value, dtype=dtype, copy=True)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class HeadroomDiagnostic:
    """Fixed-profile and active-exposure analog/ADC headroom."""

    profile_adc_limit_electrons: float
    profile_limiting_stage: str
    profile_neutral_headroom_electrons: float
    active_reference_electrons: float
    active_adc_limit_electrons: float
    active_limiting_stage: str
    active_neutral_headroom_electrons: float


@dataclass(frozen=True, slots=True)
class SensorDiagnostics:
    """Read-only intermediate values and complete stochastic sample identity."""

    expected_photoelectrons: FloatArray
    expected_dark_electrons: FloatArray
    photoelectrons: FloatArray
    dark_electrons: FloatArray
    well_electrons: FloatArray
    read_noise_electrons: FloatArray
    electrons: FloatArray
    adc_dn: UInt32Array
    full_well_saturation: BoolArray
    adc_saturation: BoolArray
    counters: UInt64Array
    rng_key: RNGKey
    rng_algorithm: str
    gain: GainSetting
    gain_policy: GainPolicy
    adc_profile: ADCProfile
    exposure: ExposureSetting
    headroom: HeadroomDiagnostic
    stochastic: bool

    def __post_init__(self) -> None:
        float_fields = (
            "expected_photoelectrons",
            "expected_dark_electrons",
            "photoelectrons",
            "dark_electrons",
            "well_electrons",
            "read_noise_electrons",
            "electrons",
        )
        shape: tuple[int, ...] | None = None
        for field_name in float_fields:
            array = _readonly_array(
                getattr(self, field_name), np.dtype(np.float64), name=field_name
            )
            if shape is None:
                shape = array.shape
            elif array.shape != shape:
                raise ValueError("all sensor diagnostic arrays must have the same shape")
            object.__setattr__(self, field_name, array)
        for field_name, dtype in (
            ("adc_dn", np.dtype(np.uint32)),
            ("full_well_saturation", np.dtype(np.bool_)),
            ("adc_saturation", np.dtype(np.bool_)),
            ("counters", np.dtype(np.uint64)),
        ):
            array = _readonly_array(getattr(self, field_name), dtype, name=field_name)
            if array.shape != shape:
                raise ValueError("all sensor diagnostic arrays must have the same shape")
            object.__setattr__(self, field_name, array)
        if not isinstance(self.rng_key, RNGKey):
            raise TypeError("rng_key must be an RNGKey")
        if not isinstance(self.gain, GainSetting):
            raise TypeError("gain must be a GainSetting")
        if not isinstance(self.adc_profile, ADCProfile):
            raise TypeError("adc_profile must be an ADCProfile")
        if not isinstance(self.exposure, ExposureSetting):
            raise TypeError("exposure must be an ExposureSetting")
        if not isinstance(self.headroom, HeadroomDiagnostic):
            raise TypeError("headroom must be a HeadroomDiagnostic")
        object.__setattr__(self, "gain_policy", GainPolicy(self.gain_policy))
        object.__setattr__(self, "rng_algorithm", str(self.rng_algorithm))
        object.__setattr__(self, "stochastic", bool(self.stochastic))


@dataclass(frozen=True, slots=True)
class LinearRGBCapture:
    """Three-channel signed camera-linear output; CFA and demosaic are bypassed."""

    signed_camera_linear_rgb: FloatArray
    diagnostics: SensorDiagnostics

    def __post_init__(self) -> None:
        output = _readonly_array(
            self.signed_camera_linear_rgb,
            np.dtype(np.float64),
            name="signed_camera_linear_rgb",
        )
        if output.ndim != 3 or output.shape[-1] != 3:
            raise ValueError("signed_camera_linear_rgb must have shape (H, W, 3)")
        if output.shape != self.diagnostics.adc_dn.shape:
            raise ValueError("output and diagnostics must have the same shape")
        object.__setattr__(self, "signed_camera_linear_rgb", output)


@dataclass(frozen=True, slots=True)
class BayerRawCapture:
    """One-channel signed RAW mosaic before demosaicing or color processing."""

    raw_signed_mosaic: FloatArray
    cfa_channel_indices: Int8Array
    cfa_pattern: BayerPattern
    diagnostics: SensorDiagnostics

    def __post_init__(self) -> None:
        raw = _readonly_array(
            self.raw_signed_mosaic, np.dtype(np.float64), name="raw_signed_mosaic"
        )
        channels = _readonly_array(
            self.cfa_channel_indices,
            np.dtype(np.int8),
            name="cfa_channel_indices",
        )
        if raw.ndim != 2 or channels.shape != raw.shape:
            raise ValueError("RAW mosaic and CFA channel map must have the same 2-D shape")
        if self.diagnostics.adc_dn.shape != raw.shape:
            raise ValueError("RAW output and diagnostics must have the same shape")
        if np.any((channels < 0) | (channels > 2)):
            raise ValueError("CFA channel indices must lie in [0, 2]")
        object.__setattr__(self, "raw_signed_mosaic", raw)
        object.__setattr__(self, "cfa_channel_indices", channels)
        object.__setattr__(self, "cfa_pattern", BayerPattern(self.cfa_pattern))


def make_sensor_rng_key(
    profile: SensorProfile,
    *,
    seed: int,
    image_id: str,
    coupling_id: str,
    realization: int,
) -> RNGKey:
    """Construct the canonical base key used by both sensor tiers."""

    if not isinstance(profile, SensorProfile):
        raise TypeError("profile must be a SensorProfile")
    return RNGKey(
        profile_hash=profile.sha256,
        seed=seed,
        image_id=image_id,
        coupling_id=coupling_id,
        realization=realization,
        noise_source=SENSOR_CAPTURE_NOISE_SOURCE,
    )


def _validate_rng_key(key: RNGKey, profile: SensorProfile) -> RNGKey:
    if not isinstance(key, RNGKey):
        raise TypeError("rng_key must be an RNGKey")
    if key.profile_hash != profile.sha256:
        raise ValueError("rng_key profile_hash does not match the SensorProfile")
    if key.noise_source != SENSOR_CAPTURE_NOISE_SOURCE:
        raise ValueError(f"rng_key noise_source must be {SENSOR_CAPTURE_NOISE_SOURCE!r}")
    return key


def _camera_rgb_and_origin(
    camera_linear_rgb: ArrayLike,
    profile: SensorProfile,
    sensor_origin: tuple[int, int] | None,
) -> tuple[FloatArray, tuple[int, int]]:
    values = np.asarray(camera_linear_rgb, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("camera_linear_rgb must have shape (H, W, 3)")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("camera_linear_rgb spatial dimensions cannot be empty")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("camera_linear_rgb must be finite and nonnegative")
    sensor_shape = profile.sensor_shape_pixels
    if sensor_origin is None:
        if values.shape[:2] != sensor_shape:
            raise ValueError(
                "full-frame input shape must equal SensorProfile.sensor_shape_pixels; "
                "pass sensor_origin explicitly for a bounded sensor tile"
            )
        return values, (0, 0)
    if not isinstance(sensor_origin, tuple) or len(sensor_origin) != 2:
        raise TypeError("sensor_origin must be a (row, column) integer tuple")
    origin_values = []
    for coordinate in sensor_origin:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, np.integer)):
            raise TypeError("sensor_origin coordinates must be integers")
        coordinate = int(coordinate)
        if coordinate < 0:
            raise ValueError("sensor_origin coordinates must be nonnegative")
        origin_values.append(coordinate)
    origin = (origin_values[0], origin_values[1])
    if (
        origin[0] + values.shape[0] > sensor_shape[0]
        or origin[1] + values.shape[1] > sensor_shape[1]
    ):
        raise ValueError("sensor tile lies outside SensorProfile.sensor_shape_pixels")
    return values, origin


def _quantum_efficiency(profile: SensorProfile) -> FloatArray:
    missing = [channel for channel in "RGB" if channel not in profile.quantum_efficiency]
    if missing:
        raise ValueError(
            "SensorProfile.quantum_efficiency must declare R, G, and B; "
            f"missing {', '.join(missing)}"
        )
    return np.asarray([profile.quantum_efficiency[channel] for channel in "RGB"], dtype=np.float64)


def _adc_profile(profile: SensorProfile) -> ADCProfile:
    return ADCProfile(
        conversion_dn_per_electron=profile.base_conversion_gain_dn_per_electron,
        black_level_dn=profile.black_level_dn,
        bit_depth=profile.adc_bit_depth,
    )


def _gain_and_headroom(
    profile: SensorProfile,
    exposure: ExposureSetting,
    gain_policy: GainPolicy | str,
) -> tuple[GainSetting, GainPolicy, ADCProfile, HeadroomDiagnostic]:
    if not isinstance(exposure, ExposureSetting):
        raise TypeError("exposure must be an ExposureSetting")
    policy = GainPolicy(gain_policy)
    fixed_gain = GainSetting(profile.analog_gain, profile.digital_gain)
    adc_profile = _adc_profile(profile)
    profile_headroom = validate_headroom(
        reference_electrons=profile.reference_electron_budget_electrons,
        full_well_electrons=profile.full_well_capacity_electrons,
        gain=fixed_gain,
        profile=adc_profile,
    )
    profile_neutral = float(profile_headroom["neutral_headroom_electrons"])
    tolerance = 1e-12 * profile.reference_electron_budget_electrons
    if profile_neutral < -tolerance:
        raise ValueError("SensorProfile has negative neutral headroom at its fixed analog gain")

    if policy is GainPolicy.FIXED_PROFILE:
        gain = fixed_gain
    else:
        gain = GainSetting.compensate(
            exposure.photon_budget_factor, analog_gain=profile.analog_gain
        )
    active_reference = exposure.photon_budget_factor * profile.reference_electron_budget_electrons
    active_headroom = validate_headroom(
        reference_electrons=active_reference,
        full_well_electrons=profile.full_well_capacity_electrons,
        gain=gain,
        profile=adc_profile,
    )
    active_neutral = float(active_headroom["neutral_headroom_electrons"])
    if active_neutral < -tolerance:
        raise ValueError("selected gain policy has negative active neutral headroom")
    diagnostic = HeadroomDiagnostic(
        profile_adc_limit_electrons=float(profile_headroom["adc_limit_electrons"]),
        profile_limiting_stage=str(profile_headroom["limiting_stage"]),
        profile_neutral_headroom_electrons=profile_neutral,
        active_reference_electrons=active_reference,
        active_adc_limit_electrons=float(active_headroom["adc_limit_electrons"]),
        active_limiting_stage=str(active_headroom["limiting_stage"]),
        active_neutral_headroom_electrons=active_neutral,
    )
    return gain, policy, adc_profile, diagnostic


def _counter_array(
    shape: tuple[int, ...],
    *,
    sensor_shape: tuple[int, int],
    sensor_origin: tuple[int, int],
    rgb_channels: bool,
    counters: ArrayLike | None,
) -> UInt64Array:
    count = int(np.prod(shape, dtype=np.int64))
    maximum_counter = sensor_shape[0] * sensor_shape[1] * (3 if rgb_channels else 1)
    if counters is not None:
        counter_array = np.asarray(counters)
        if counter_array.shape != shape:
            if counter_array.size != count:
                raise ValueError("counters must have the capture output shape or size")
            counter_array = counter_array.reshape(shape)
        if not np.issubdtype(counter_array.dtype, np.integer) or np.any(counter_array < 0):
            raise ValueError("counters must be nonnegative integers")
        counter_array = counter_array.astype(np.uint64, copy=False)
        if np.any(counter_array >= np.uint64(maximum_counter)):
            raise ValueError("counter lies outside the configured sensor identity range")
        return np.array(counter_array, dtype=np.uint64, copy=True)

    height, width = shape[:2]
    rows = np.arange(sensor_origin[0], sensor_origin[0] + height, dtype=np.uint64)[:, None]
    columns = np.arange(sensor_origin[1], sensor_origin[1] + width, dtype=np.uint64)[None, :]
    photosites = rows * np.uint64(sensor_shape[1]) + columns
    if rgb_channels:
        channels = np.arange(3, dtype=np.uint64)[None, None, :]
        return photosites[..., None] * np.uint64(3) + channels
    return photosites


def _deterministic_electron_capture(
    photo_expectation: FloatArray,
    dark_expectation: FloatArray,
    full_well_electrons: float,
) -> ElectronCapture:
    photo = np.array(photo_expectation, dtype=np.float64, copy=True)
    dark = np.array(dark_expectation, dtype=np.float64, copy=True)
    well = np.minimum(photo + dark, full_well_electrons)
    read = np.zeros_like(well)
    return ElectronCapture(photo, dark, well, read, well.copy())


def _capture_samples(
    normalized_signal: FloatArray,
    *,
    profile: SensorProfile,
    exposure: ExposureSetting,
    rng_key: RNGKey,
    rng: StatelessRNG,
    gain_policy: GainPolicy | str,
    stochastic: bool,
    counters: UInt64Array,
) -> tuple[FloatArray, SensorDiagnostics]:
    gain, policy, adc_profile, headroom = _gain_and_headroom(profile, exposure, gain_policy)
    expected_photo = expected_photoelectrons(
        normalized_signal,
        reference_electrons=profile.reference_electron_budget_electrons,
        exposure=exposure,
    )
    expected_dark_scalar = expected_dark_electrons(
        profile.dark_current_electrons_per_second,
        exposure=exposure,
    )
    expected_dark = np.full_like(expected_photo, float(expected_dark_scalar))
    if stochastic:
        electron_capture = capture_electrons(
            expected_photo,
            full_well_electrons=profile.full_well_capacity_electrons,
            read_noise_electrons=profile.read_noise_rms_electrons,
            expected_dark_electrons=expected_dark,
            rng=rng,
            key=rng_key,
            counters=counters,
        )
    else:
        electron_capture = _deterministic_electron_capture(
            expected_photo,
            expected_dark,
            profile.full_well_capacity_electrons,
        )
    adc_dn = quantize_adc(electron_capture.electrons, gain=gain, profile=adc_profile)
    signed = black_subtract_and_normalize(
        adc_dn,
        gain=gain,
        profile=adc_profile,
        reference_electrons=profile.reference_electron_budget_electrons,
    )
    diagnostics = SensorDiagnostics(
        expected_photoelectrons=expected_photo,
        expected_dark_electrons=expected_dark,
        photoelectrons=electron_capture.photoelectrons,
        dark_electrons=electron_capture.dark_electrons,
        well_electrons=electron_capture.well_electrons,
        read_noise_electrons=electron_capture.read_noise_electrons,
        electrons=electron_capture.electrons,
        adc_dn=adc_dn,
        full_well_saturation=(
            electron_capture.well_electrons >= profile.full_well_capacity_electrons
        ),
        adc_saturation=(adc_dn >= adc_profile.maximum_dn),
        counters=counters,
        rng_key=rng_key,
        rng_algorithm=rng.algorithm,
        gain=gain,
        gain_policy=policy,
        adc_profile=adc_profile,
        exposure=exposure,
        headroom=headroom,
        stochastic=stochastic,
    )
    return np.asarray(signed, dtype=np.float64), diagnostics


def capture_linear_rgb(
    camera_linear_rgb: ArrayLike,
    *,
    profile: SensorProfile,
    exposure: ExposureSetting,
    rng_key: RNGKey,
    gain_policy: GainPolicy | str = GainPolicy.FIXED_PROFILE,
    stochastic: bool = True,
    rng: StatelessRNG | None = None,
    sensor_origin: tuple[int, int] | None = None,
    counters: ArrayLike | None = None,
) -> LinearRGBCapture:
    """Capture the declared three-channel LDR sensor approximation.

    The input must already be camera-linear RGB.  Per-channel QE is applied,
    but CFA mosaicing and demosaicing are both bypassed.
    """

    if not isinstance(profile, SensorProfile):
        raise TypeError("profile must be a SensorProfile")
    values, origin = _camera_rgb_and_origin(camera_linear_rgb, profile, sensor_origin)
    key = _validate_rng_key(rng_key, profile)
    rng = rng or StatelessRNG()
    if not isinstance(rng, StatelessRNG):
        raise TypeError("rng must be a StatelessRNG")
    qe_weighted = values * _quantum_efficiency(profile)[None, None, :]
    counter_array = _counter_array(
        qe_weighted.shape,
        sensor_shape=profile.sensor_shape_pixels,
        sensor_origin=origin,
        rgb_channels=True,
        counters=counters,
    )
    signed, diagnostics = _capture_samples(
        qe_weighted,
        profile=profile,
        exposure=exposure,
        rng_key=key,
        rng=rng,
        gain_policy=gain_policy,
        stochastic=stochastic,
        counters=counter_array,
    )
    return LinearRGBCapture(signed_camera_linear_rgb=signed, diagnostics=diagnostics)


def capture_bayer_raw(
    camera_linear_rgb: ArrayLike,
    *,
    profile: SensorProfile,
    exposure: ExposureSetting,
    rng_key: RNGKey,
    gain_policy: GainPolicy | str = GainPolicy.FIXED_PROFILE,
    stochastic: bool = True,
    rng: StatelessRNG | None = None,
    sensor_origin: tuple[int, int] | None = None,
    counters: ArrayLike | None = None,
) -> BayerRawCapture:
    """Capture one QE-weighted channel per Bayer photosite as signed RAW."""

    if not isinstance(profile, SensorProfile):
        raise TypeError("profile must be a SensorProfile")
    if profile.cfa_pattern is None:
        raise ValueError("Bayer RAW capture requires SensorProfile.cfa_pattern")
    try:
        pattern = BayerPattern(profile.cfa_pattern)
    except ValueError as error:
        raise ValueError("SensorProfile.cfa_pattern must be a supported Bayer pattern") from error
    values, origin = _camera_rgb_and_origin(camera_linear_rgb, profile, sensor_origin)
    key = _validate_rng_key(rng_key, profile)
    rng = rng or StatelessRNG()
    if not isinstance(rng, StatelessRNG):
        raise TypeError("rng must be a StatelessRNG")
    qe_weighted = values * _quantum_efficiency(profile)[None, None, :]
    full_channel_map = cfa_channel_indices(profile.sensor_shape_pixels, pattern)
    channel_map = full_channel_map[
        origin[0] : origin[0] + values.shape[0],
        origin[1] : origin[1] + values.shape[1],
    ]
    mosaiced = np.take_along_axis(qe_weighted, channel_map[..., None], axis=-1)[..., 0]
    counter_array = _counter_array(
        mosaiced.shape,
        sensor_shape=profile.sensor_shape_pixels,
        sensor_origin=origin,
        rgb_channels=False,
        counters=counters,
    )
    signed, diagnostics = _capture_samples(
        mosaiced,
        profile=profile,
        exposure=exposure,
        rng_key=key,
        rng=rng,
        gain_policy=gain_policy,
        stochastic=stochastic,
        counters=counter_array,
    )
    return BayerRawCapture(
        raw_signed_mosaic=signed,
        cfa_channel_indices=channel_map,
        cfa_pattern=pattern,
        diagnostics=diagnostics,
    )
