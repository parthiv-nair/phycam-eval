"""Global stop-ratio tone compression in output-linear RGB."""

from __future__ import annotations

import math

import numpy as np

from ..color import (
    DEFAULT_OUTPUT_COLORIMETRY,
    OutputColorimetry,
    luminance,
    validate_rgb_array,
)

__all__ = [
    "apply_global_tone",
    "global_stop_ratio_tone",
    "stop_ratio_curve",
]


def _positive_scalar(value: float, *, name: str, maximum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if scalar <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    if maximum is not None and scalar > maximum:
        raise ValueError(f"{name} must be less than or equal to {maximum}")
    return scalar


def _validate_luminance_array(values: np.ndarray) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError("luminance_values must be a NumPy array")
    if values.size == 0:
        raise ValueError("luminance_values must not be empty")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("luminance_values must have a real floating-point dtype")
    if not np.all(np.isfinite(values)):
        raise ValueError("luminance_values must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("luminance_values must be nonnegative")
    return values


def stop_ratio_curve(
    luminance_values: np.ndarray,
    rho: float,
    *,
    pivot: float = 0.18,
) -> np.ndarray:
    """Apply ``Y_p * (Y / Y_p) ** rho`` to every positive luminance.

    Zero maps exactly to zero.  There is no epsilon, low-value threshold, or
    additive offset.  The ``rho == 1`` branch returns an exact elementwise copy
    so identity also holds for the smallest representable positive value.
    """

    values = _validate_luminance_array(luminance_values)
    ratio = _positive_scalar(rho, name="rho", maximum=1.0)
    pivot_value = _positive_scalar(pivot, name="pivot")
    if ratio == 1.0:
        return values.copy()

    result = np.zeros_like(values)
    positive = values > 0.0
    typed_pivot = np.asarray(pivot_value, dtype=values.dtype)
    typed_ratio = np.asarray(ratio, dtype=values.dtype)
    result[positive] = typed_pivot * np.power(
        values[positive] / typed_pivot,
        typed_ratio,
    )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("stop-ratio tone curve produced a nonfinite value")
    return result


def apply_global_tone(
    nonnegative_output_rgb: np.ndarray,
    rho: float,
    *,
    pivot: float = 0.18,
    colorimetry: OutputColorimetry = DEFAULT_OUTPUT_COLORIMETRY,
) -> np.ndarray:
    """Apply the global luminance curve while preserving RGB chromaticity.

    The input must already have passed a named pre-tone gamut policy.  Output
    remains unbounded above; a separate post-tone policy owns output clipping.
    """

    values = validate_rgb_array(
        nonnegative_output_rgb,
        name="nonnegative_output_rgb",
        lower_bound=0.0,
    )
    if not isinstance(colorimetry, OutputColorimetry):
        raise TypeError("colorimetry must be an OutputColorimetry")
    ratio = _positive_scalar(rho, name="rho", maximum=1.0)
    pivot_value = _positive_scalar(pivot, name="pivot")

    # Bypass all arithmetic at the identity endpoint.  This is stronger than
    # an allclose guarantee and preserves subnormal positive channel values.
    if ratio == 1.0:
        return values.copy()

    input_luminance = luminance(values, colorimetry=colorimetry)
    mapped_luminance = stop_ratio_curve(input_luminance, ratio, pivot=pivot_value)

    chromaticity = np.zeros_like(values)
    denominator = input_luminance[..., np.newaxis]
    np.divide(
        values,
        denominator,
        out=chromaticity,
        where=denominator > 0.0,
    )
    result = chromaticity * mapped_luminance[..., np.newaxis]
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("global tone mapping produced a nonfinite value")
    return result


global_stop_ratio_tone = apply_global_tone
