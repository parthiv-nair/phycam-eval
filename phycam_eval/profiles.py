"""Immutable, serializable, content-addressed camera profiles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Type, TypeVar

from ._canonical import (
    canonical_bytes,
    canonical_sha256,
    finite_float,
    freeze_json_value,
    json_value,
    nfc_string,
    positive_float,
    positive_int,
)
from .boundary import BoundaryContract
from .domains import ColorSpace, DataMode
from .isp.gamut import PostToneGamutPolicy, PreToneGamutPolicy

T = TypeVar("T", bound="SerializableProfile")


def _nonempty_string(value: str, *, field_name: str) -> str:
    result = nfc_string(value, field_name=field_name)
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _reject_unknown_fields(
    value: Mapping[str, Any],
    allowed_fields: Sequence[str],
    *,
    profile_name: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{profile_name} record must be a mapping")
    unknown = set(value).difference(allowed_fields)
    if unknown:
        rendered = ", ".join(sorted(repr(field) for field in unknown))
        raise ValueError(f"{profile_name} record contains unknown fields: {rendered}")


def _float_tuple(
    value: Sequence[float],
    *,
    field_name: str,
    length: int,
    positive: bool = False,
) -> tuple[float, ...]:
    if len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values")
    validator = positive_float if positive else finite_float
    return tuple(
        validator(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)
    )


def _matrix3(value: Sequence[Sequence[float]], *, field_name: str) -> tuple[tuple[float, ...], ...]:
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three rows")
    return tuple(
        _float_tuple(row, field_name=f"{field_name}[{index}]", length=3)
        for index, row in enumerate(value)
    )


def _float_mapping(
    value: Mapping[str, float],
    *,
    field_name: str,
    positive: bool,
    bounded_one: bool = False,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    normalized: dict[str, float] = {}
    for original_key, original_value in value.items():
        key = _nonempty_string(original_key, field_name=f"{field_name} key")
        if key in normalized:
            raise ValueError(f"{field_name} contains duplicate NFC-normalized keys")
        number = (
            positive_float(original_value, field_name=f"{field_name}[{key!r}]")
            if positive
            else finite_float(original_value, field_name=f"{field_name}[{key!r}]")
        )
        if bounded_one and not 0.0 <= number <= 1.0:
            raise ValueError(f"{field_name}[{key!r}] must lie in [0, 1]")
        normalized[key] = number
    return MappingProxyType(dict(sorted(normalized.items())))


class SerializableProfile:
    """Shared serialization and identity behavior for frozen profile records."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Return human-readable stable JSON (not the identity encoding)."""

        kwargs: dict[str, Any] = {
            "allow_nan": False,
            "ensure_ascii": False,
            "sort_keys": True,
        }
        if indent is None:
            kwargs["separators"] = (",", ":")
        else:
            kwargs["indent"] = indent
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls: Type[T], payload: str) -> T:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise TypeError("profile JSON must contain an object")
        return cls.from_dict(value)  # type: ignore[attr-defined, no-any-return]

    @property
    def identity_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class OpticsProfile(SerializableProfile):
    """Fixed optics and numerical pupil-sampling configuration."""

    f_number: float
    reference_wavelength_m: float
    channel_wavelengths_m: Mapping[str, float]
    pupil_grid_size: int
    pupil_q_max: float
    pupil_fft_size: int
    psf_energy_fraction: float
    boundary_policy: str
    boundary_constant_value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "f_number", positive_float(self.f_number, field_name="f_number"))
        object.__setattr__(
            self,
            "reference_wavelength_m",
            positive_float(
                self.reference_wavelength_m,
                field_name="reference_wavelength_m",
            ),
        )
        object.__setattr__(
            self,
            "channel_wavelengths_m",
            _float_mapping(
                self.channel_wavelengths_m,
                field_name="channel_wavelengths_m",
                positive=True,
            ),
        )
        grid_size = positive_int(self.pupil_grid_size, field_name="pupil_grid_size")
        if grid_size < 3 or grid_size % 2 == 0:
            raise ValueError("pupil_grid_size must be an odd integer >= 3")
        object.__setattr__(self, "pupil_grid_size", grid_size)
        q_max = positive_float(self.pupil_q_max, field_name="pupil_q_max")
        if q_max <= 1.0:
            raise ValueError("pupil_q_max must be greater than 1")
        object.__setattr__(self, "pupil_q_max", q_max)
        fft_size = positive_int(self.pupil_fft_size, field_name="pupil_fft_size")
        if fft_size < grid_size or fft_size % 2 == 0:
            raise ValueError("pupil_fft_size must be odd and no smaller than pupil_grid_size")
        object.__setattr__(self, "pupil_fft_size", fft_size)
        fraction = positive_float(self.psf_energy_fraction, field_name="psf_energy_fraction")
        if fraction > 1.0:
            raise ValueError("psf_energy_fraction must not exceed 1")
        object.__setattr__(self, "psf_energy_fraction", fraction)
        boundary = BoundaryContract(
            _nonempty_string(self.boundary_policy, field_name="boundary_policy"),
            self.boundary_constant_value,
        )
        object.__setattr__(self, "boundary_policy", boundary.mode.value)
        object.__setattr__(self, "boundary_constant_value", boundary.constant_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "f_number": self.f_number,
            "reference_wavelength_m": self.reference_wavelength_m,
            "channel_wavelengths_m": json_value(self.channel_wavelengths_m),
            "pupil_grid_size": self.pupil_grid_size,
            "pupil_q_max": self.pupil_q_max,
            "pupil_fft_size": self.pupil_fft_size,
            "psf_energy_fraction": self.psf_energy_fraction,
            "boundary_policy": self.boundary_policy,
            "boundary_constant_value": self.boundary_constant_value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OpticsProfile":
        allowed_fields = (
            "f_number",
            "reference_wavelength_m",
            "channel_wavelengths_m",
            "pupil_grid_size",
            "pupil_q_max",
            "pupil_fft_size",
            "psf_energy_fraction",
            "boundary_policy",
            "boundary_constant_value",
        )
        _reject_unknown_fields(value, allowed_fields, profile_name="OpticsProfile")
        payload = {key: value[key] for key in allowed_fields if key != "boundary_constant_value"}
        payload["boundary_constant_value"] = value.get("boundary_constant_value", 0.0)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class SensorProfile(SerializableProfile):
    """Fixed photosite, electron, gain, and ADC configuration."""

    sensor_shape_pixels: tuple[int, int]
    pixel_pitch_m: float
    reference_electron_budget_electrons: float
    full_well_capacity_electrons: float
    read_noise_rms_electrons: float
    dark_current_electrons_per_second: float
    quantum_efficiency: Mapping[str, float]
    base_conversion_gain_dn_per_electron: float
    analog_gain: float
    digital_gain: float
    black_level_dn: float
    adc_bit_depth: int
    cfa_pattern: Optional[str]

    def __post_init__(self) -> None:
        if len(self.sensor_shape_pixels) != 2:
            raise ValueError("sensor_shape_pixels must contain height and width")
        shape = tuple(
            positive_int(item, field_name=f"sensor_shape_pixels[{index}]")
            for index, item in enumerate(self.sensor_shape_pixels)
        )
        object.__setattr__(self, "sensor_shape_pixels", shape)
        object.__setattr__(
            self,
            "pixel_pitch_m",
            positive_float(self.pixel_pitch_m, field_name="pixel_pitch_m"),
        )
        q0 = positive_float(
            self.reference_electron_budget_electrons,
            field_name="reference_electron_budget_electrons",
        )
        full_well = positive_float(
            self.full_well_capacity_electrons,
            field_name="full_well_capacity_electrons",
        )
        if q0 > full_well:
            raise ValueError("reference_electron_budget_electrons must not exceed full well")
        object.__setattr__(self, "reference_electron_budget_electrons", q0)
        object.__setattr__(self, "full_well_capacity_electrons", full_well)
        object.__setattr__(
            self,
            "read_noise_rms_electrons",
            positive_float(
                self.read_noise_rms_electrons,
                field_name="read_noise_rms_electrons",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "dark_current_electrons_per_second",
            positive_float(
                self.dark_current_electrons_per_second,
                field_name="dark_current_electrons_per_second",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "quantum_efficiency",
            _float_mapping(
                self.quantum_efficiency,
                field_name="quantum_efficiency",
                positive=False,
                bounded_one=True,
            ),
        )
        for field_name in (
            "base_conversion_gain_dn_per_electron",
            "analog_gain",
            "digital_gain",
        ):
            object.__setattr__(
                self,
                field_name,
                positive_float(getattr(self, field_name), field_name=field_name),
            )
        black = positive_float(self.black_level_dn, field_name="black_level_dn", allow_zero=True)
        object.__setattr__(self, "black_level_dn", black)
        bit_depth = positive_int(self.adc_bit_depth, field_name="adc_bit_depth")
        if bit_depth > 32:
            raise ValueError("adc_bit_depth must not exceed 32")
        if black > (2**bit_depth - 1):
            raise ValueError("black_level_dn must lie within the ADC rails")
        object.__setattr__(self, "adc_bit_depth", bit_depth)
        if self.cfa_pattern is not None:
            object.__setattr__(
                self,
                "cfa_pattern",
                _nonempty_string(self.cfa_pattern, field_name="cfa_pattern"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_shape_pixels": list(self.sensor_shape_pixels),
            "pixel_pitch_m": self.pixel_pitch_m,
            "reference_electron_budget_electrons": self.reference_electron_budget_electrons,
            "full_well_capacity_electrons": self.full_well_capacity_electrons,
            "read_noise_rms_electrons": self.read_noise_rms_electrons,
            "dark_current_electrons_per_second": self.dark_current_electrons_per_second,
            "quantum_efficiency": json_value(self.quantum_efficiency),
            "base_conversion_gain_dn_per_electron": self.base_conversion_gain_dn_per_electron,
            "analog_gain": self.analog_gain,
            "digital_gain": self.digital_gain,
            "black_level_dn": self.black_level_dn,
            "adc_bit_depth": self.adc_bit_depth,
            "cfa_pattern": self.cfa_pattern,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SensorProfile":
        keys = (
            "sensor_shape_pixels",
            "pixel_pitch_m",
            "reference_electron_budget_electrons",
            "full_well_capacity_electrons",
            "read_noise_rms_electrons",
            "dark_current_electrons_per_second",
            "quantum_efficiency",
            "base_conversion_gain_dn_per_electron",
            "analog_gain",
            "digital_gain",
            "black_level_dn",
            "adc_bit_depth",
            "cfa_pattern",
        )
        _reject_unknown_fields(value, keys, profile_name="SensorProfile")
        return cls(**{key: value[key] for key in keys})


@dataclass(frozen=True, slots=True)
class ReadoutProfile(SerializableProfile):
    """Fixed exposure timing and numerical quadrature configuration."""

    frame_start_time_s: float
    line_time_s: float
    exposure_time_s: float
    reference_time_s: float
    annotation_time_s: float
    quadrature_order: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_start_time_s",
            finite_float(self.frame_start_time_s, field_name="frame_start_time_s"),
        )
        object.__setattr__(
            self,
            "line_time_s",
            positive_float(self.line_time_s, field_name="line_time_s", allow_zero=True),
        )
        object.__setattr__(
            self,
            "exposure_time_s",
            positive_float(self.exposure_time_s, field_name="exposure_time_s"),
        )
        object.__setattr__(
            self,
            "reference_time_s",
            finite_float(self.reference_time_s, field_name="reference_time_s"),
        )
        object.__setattr__(
            self,
            "annotation_time_s",
            finite_float(self.annotation_time_s, field_name="annotation_time_s"),
        )
        object.__setattr__(
            self,
            "quadrature_order",
            positive_int(self.quadrature_order, field_name="quadrature_order"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_start_time_s": self.frame_start_time_s,
            "line_time_s": self.line_time_s,
            "exposure_time_s": self.exposure_time_s,
            "reference_time_s": self.reference_time_s,
            "annotation_time_s": self.annotation_time_s,
            "quadrature_order": self.quadrature_order,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReadoutProfile":
        keys = (
            "frame_start_time_s",
            "line_time_s",
            "exposure_time_s",
            "reference_time_s",
            "annotation_time_s",
            "quadrature_order",
        )
        _reject_unknown_fields(value, keys, profile_name="ReadoutProfile")
        return cls(**{key: value[key] for key in keys})


@dataclass(frozen=True, slots=True)
class ISPProfile(SerializableProfile):
    """Fixed color, gamut, and global tone configuration."""

    white_balance_gains: tuple[float, float, float]
    camera_to_output_matrix: tuple[tuple[float, float, float], ...]
    output_luminance_coefficients: tuple[float, float, float]
    pre_tone_gamut_policy: str
    tone_pivot: float
    post_tone_gamut_policy: str
    output_color_space: ColorSpace

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "white_balance_gains",
            _float_tuple(
                self.white_balance_gains,
                field_name="white_balance_gains",
                length=3,
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "camera_to_output_matrix",
            _matrix3(self.camera_to_output_matrix, field_name="camera_to_output_matrix"),
        )
        luminance = _float_tuple(
            self.output_luminance_coefficients,
            field_name="output_luminance_coefficients",
            length=3,
        )
        if any(item <= 0.0 for item in luminance) or not math.isclose(
            sum(luminance), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("output_luminance_coefficients must be positive and sum to one")
        object.__setattr__(self, "output_luminance_coefficients", luminance)
        pre_tone_name = _nonempty_string(
            self.pre_tone_gamut_policy, field_name="pre_tone_gamut_policy"
        )
        try:
            pre_tone_policy = PreToneGamutPolicy(pre_tone_name)
        except ValueError as exc:
            raise ValueError("pre_tone_gamut_policy is not implemented") from exc
        object.__setattr__(self, "pre_tone_gamut_policy", pre_tone_policy.value)
        object.__setattr__(
            self,
            "tone_pivot",
            positive_float(self.tone_pivot, field_name="tone_pivot"),
        )
        post_tone_name = _nonempty_string(
            self.post_tone_gamut_policy, field_name="post_tone_gamut_policy"
        )
        try:
            post_tone_policy = PostToneGamutPolicy(post_tone_name)
        except ValueError as exc:
            raise ValueError("post_tone_gamut_policy is not implemented") from exc
        object.__setattr__(self, "post_tone_gamut_policy", post_tone_policy.value)
        output_space = ColorSpace(self.output_color_space)
        if output_space in {
            ColorSpace.NONE,
            ColorSpace.RAW_MOSAIC,
            ColorSpace.SCENE_SPECTRAL,
        }:
            raise ValueError("output_color_space must be an RGB color space")
        object.__setattr__(self, "output_color_space", output_space)

    def to_dict(self) -> dict[str, Any]:
        return {
            "white_balance_gains": list(self.white_balance_gains),
            "camera_to_output_matrix": [list(row) for row in self.camera_to_output_matrix],
            "output_luminance_coefficients": list(self.output_luminance_coefficients),
            "pre_tone_gamut_policy": self.pre_tone_gamut_policy,
            "tone_pivot": self.tone_pivot,
            "post_tone_gamut_policy": self.post_tone_gamut_policy,
            "output_color_space": self.output_color_space.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ISPProfile":
        keys = (
            "white_balance_gains",
            "camera_to_output_matrix",
            "output_luminance_coefficients",
            "pre_tone_gamut_policy",
            "tone_pivot",
            "post_tone_gamut_policy",
            "output_color_space",
        )
        _reject_unknown_fields(value, keys, profile_name="ISPProfile")
        return cls(**{key: value[key] for key in keys})


@dataclass(frozen=True, slots=True)
class CameraProfile(SerializableProfile):
    """Complete fixed camera profile; its content hash is its identity."""

    name: str
    data_mode: DataMode
    optics: OpticsProfile
    sensor: SensorProfile
    readout: ReadoutProfile
    isp: ISPProfile
    calibration_reference: Optional[str]
    fixed_parameters: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = field(default=2, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, field_name="name"))
        object.__setattr__(self, "data_mode", DataMode(self.data_mode))
        for field_name, field_type in (
            ("optics", OpticsProfile),
            ("sensor", SensorProfile),
            ("readout", ReadoutProfile),
            ("isp", ISPProfile),
        ):
            if not isinstance(getattr(self, field_name), field_type):
                raise TypeError(f"{field_name} must be a {field_type.__name__}")
        if self.calibration_reference is not None:
            object.__setattr__(
                self,
                "calibration_reference",
                _nonempty_string(self.calibration_reference, field_name="calibration_reference"),
            )
        frozen = freeze_json_value(self.fixed_parameters)
        if not isinstance(frozen, MappingProxyType):
            raise TypeError("fixed_parameters must be a mapping")
        object.__setattr__(self, "fixed_parameters", frozen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "data_mode": self.data_mode.value,
            "optics": self.optics.to_dict(),
            "sensor": self.sensor.to_dict(),
            "readout": self.readout.to_dict(),
            "isp": self.isp.to_dict(),
            "calibration_reference": self.calibration_reference,
            "fixed_parameters": json_value(self.fixed_parameters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CameraProfile":
        _reject_unknown_fields(
            value,
            (
                "schema_version",
                "name",
                "data_mode",
                "optics",
                "sensor",
                "readout",
                "isp",
                "calibration_reference",
                "fixed_parameters",
            ),
            profile_name="CameraProfile",
        )
        if value.get("schema_version") != 2:
            raise ValueError("CameraProfile requires schema_version 2")
        return cls(
            name=value["name"],
            data_mode=DataMode(value["data_mode"]),
            optics=OpticsProfile.from_dict(value["optics"]),
            sensor=SensorProfile.from_dict(value["sensor"]),
            readout=ReadoutProfile.from_dict(value["readout"]),
            isp=ISPProfile.from_dict(value["isp"]),
            calibration_reference=value["calibration_reference"],
            fixed_parameters=value.get("fixed_parameters", {}),
        )

    @property
    def profile_hash(self) -> str:
        """Compatibility-readable name for the content-addressed identity."""

        return self.sha256


__all__ = [
    "CameraProfile",
    "ISPProfile",
    "OpticsProfile",
    "ReadoutProfile",
    "SensorProfile",
    "SerializableProfile",
    "canonical_bytes",
]
