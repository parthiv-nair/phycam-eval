"""Analog gain, black pedestal, ADC, and post-ADC digital normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class GainSetting:
    analog_gain: float
    digital_gain: float

    def __post_init__(self) -> None:
        for name in ("analog_gain", "digital_gain"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    @classmethod
    def compensate(
        cls,
        photon_budget_factor: float,
        *,
        analog_gain: float,
    ) -> GainSetting:
        factor = float(photon_budget_factor)
        analog = float(analog_gain)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("photon_budget_factor must be finite and positive")
        if not np.isfinite(analog) or analog <= 0.0:
            raise ValueError("analog_gain must be finite and positive")
        return cls(analog, 1.0 / (factor * analog))


@dataclass(frozen=True, slots=True)
class ADCProfile:
    conversion_dn_per_electron: float
    black_level_dn: float
    bit_depth: int

    def __post_init__(self) -> None:
        conversion = float(self.conversion_dn_per_electron)
        black = float(self.black_level_dn)
        if not np.isfinite(conversion) or conversion <= 0.0:
            raise ValueError("conversion_dn_per_electron must be finite and positive")
        if not np.isfinite(black) or black < 0.0:
            raise ValueError("black_level_dn must be finite and nonnegative")
        if isinstance(self.bit_depth, bool) or not isinstance(self.bit_depth, (int, np.integer)):
            raise TypeError("bit_depth must be an integer")
        bits = int(self.bit_depth)
        if bits <= 0 or bits > 32:
            raise ValueError("bit_depth must be between 1 and 32")
        if black > 2**bits - 1:
            raise ValueError("black level lies above the ADC rail")
        object.__setattr__(self, "conversion_dn_per_electron", conversion)
        object.__setattr__(self, "black_level_dn", black)
        object.__setattr__(self, "bit_depth", bits)

    @property
    def maximum_dn(self) -> int:
        return 2**self.bit_depth - 1


def quantize_adc(
    electrons: ArrayLike,
    *,
    gain: GainSetting,
    profile: ADCProfile,
) -> NDArray[np.uint32]:
    """Apply analog gain, black level, ties-to-even rounding, and ADC rails."""

    values = np.asarray(electrons, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("electrons must be nonempty and finite")
    scale = profile.conversion_dn_per_electron * gain.analog_gain
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("combined analog/conversion gain must be finite and positive")
    # Keep multiplication and addition as separate NumPy operations.  The C++
    # exact-parity implementation deliberately follows this binary64 rounding
    # boundary and therefore disables contraction into a fused multiply-add.
    scaled_electrons = scale * values
    unquantized = profile.black_level_dn + scaled_electrons
    quantized = np.rint(unquantized)
    quantized = np.clip(quantized, 0, profile.maximum_dn)
    return quantized.astype(np.uint32)


def black_subtract_and_normalize(
    adc_dn: ArrayLike,
    *,
    gain: GainSetting,
    profile: ADCProfile,
    reference_electrons: float,
) -> NDArray[np.float64]:
    """Apply signed black subtraction, digital gain, and Q0 normalization."""

    values = np.asarray(adc_dn)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("adc_dn must be nonempty and finite")
    reference = float(reference_electrons)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_electrons must be finite and positive")
    return (
        gain.digital_gain
        * (values.astype(np.float64) - profile.black_level_dn)
        / (profile.conversion_dn_per_electron * reference)
    )


def validate_headroom(
    *,
    reference_electrons: float,
    full_well_electrons: float,
    gain: GainSetting,
    profile: ADCProfile,
) -> dict[str, float | str]:
    """Describe which charge/ADC rail limits the current gain mode."""

    reference = float(reference_electrons)
    full_well = float(full_well_electrons)
    if not np.isfinite(reference) or not np.isfinite(full_well):
        raise ValueError("electron budgets must be finite")
    if reference <= 0.0 or full_well <= 0.0 or reference > full_well:
        raise ValueError("require 0 < reference_electrons <= full_well_electrons")
    adc_electrons = (profile.maximum_dn - profile.black_level_dn) / (
        profile.conversion_dn_per_electron * gain.analog_gain
    )
    limiting = "adc" if adc_electrons < full_well else "full_well"
    return {
        "reference_electrons": reference,
        "full_well_electrons": full_well,
        "adc_limit_electrons": adc_electrons,
        "limiting_stage": limiting,
        "neutral_headroom_electrons": min(full_well, adc_electrons) - reference,
    }
