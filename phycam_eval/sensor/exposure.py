"""Photon-budget policies and deterministic electron expectations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ExposurePolicy(str, Enum):
    FIXED_DURATION_ATTENUATION = "fixed_duration_attenuation"
    SHUTTER_TIME = "shutter_time"


@dataclass(frozen=True, slots=True)
class ExposureSetting:
    """One-knob photon-budget loss with an explicit physical mechanism."""

    photon_loss_stops: float
    baseline_exposure_s: float
    policy: ExposurePolicy = ExposurePolicy.FIXED_DURATION_ATTENUATION

    def __post_init__(self) -> None:
        stops = float(self.photon_loss_stops)
        baseline = float(self.baseline_exposure_s)
        if not np.isfinite(stops) or stops < 0.0:
            raise ValueError("photon_loss_stops must be finite and nonnegative")
        if not np.isfinite(baseline) or baseline <= 0.0:
            raise ValueError("baseline_exposure_s must be finite and positive")
        object.__setattr__(self, "photon_loss_stops", stops)
        object.__setattr__(self, "baseline_exposure_s", baseline)
        object.__setattr__(self, "policy", ExposurePolicy(self.policy))

    @property
    def photon_budget_factor(self) -> float:
        return 2.0 ** (-self.photon_loss_stops)

    @property
    def exposure_s(self) -> float:
        if self.policy is ExposurePolicy.SHUTTER_TIME:
            return self.baseline_exposure_s * self.photon_budget_factor
        return self.baseline_exposure_s

    @property
    def illumination_factor(self) -> float:
        if self.policy is ExposurePolicy.FIXED_DURATION_ATTENUATION:
            return self.photon_budget_factor
        return 1.0


def expected_photoelectrons(
    normalized_signal: ArrayLike,
    *,
    reference_electrons: float,
    exposure: ExposureSetting,
) -> NDArray[np.float64]:
    """Return ``mu = 2**(-s) * Q0 * x`` for normalized nonnegative signal."""

    signal = np.asarray(normalized_signal, dtype=np.float64)
    reference = float(reference_electrons)
    if signal.size == 0 or not np.all(np.isfinite(signal)) or np.any(signal < 0.0):
        raise ValueError("normalized_signal must be nonempty, finite, and nonnegative")
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_electrons must be finite and positive")
    return exposure.photon_budget_factor * reference * signal


def expected_dark_electrons(
    dark_current_e_per_s: ArrayLike,
    *,
    exposure: ExposureSetting,
) -> NDArray[np.float64]:
    rates = np.asarray(dark_current_e_per_s, dtype=np.float64)
    if rates.size == 0 or not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("dark current must be nonempty, finite, and nonnegative")
    return rates * exposure.exposure_s
