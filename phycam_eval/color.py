"""Colorimetry and exact sRGB transfer-function primitives.

The functions in this module deliberately accept only floating-point NumPy
arrays whose final axis is RGB.  Integer conversion, clipping, and gamut
mapping are separate camera stages and must never happen implicitly here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "DEFAULT_OUTPUT_COLORIMETRY",
    "OutputColorimetry",
    "SRGB_REC709_D65",
    "decode_srgb",
    "encode_srgb",
    "luminance",
    "srgb_decode",
    "srgb_encode",
    "validate_rgb_array",
]


def _canonical_xy(value: Sequence[float], *, name: str) -> tuple[float, float]:
    """Validate and canonicalize one CIE xy chromaticity coordinate."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    try:
        x, y = (float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} coordinates must be real numbers") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{name} coordinates must be finite")
    if x < 0.0 or y <= 0.0 or x + y > 1.0:
        raise ValueError(f"{name} must be a valid CIE xy coordinate")
    return (x, y)


@dataclass(frozen=True)
class OutputColorimetry:
    """Fixed output primaries, white point, and corresponding luminance row.

    ``primaries_xy`` is ordered red, green, blue.  The luminance weights must
    be the normalized Y row derived for those primaries and the white point.
    Their derivation is intentionally not guessed at runtime: the profile is
    responsible for declaring the row used by tone mapping.
    """

    name: str
    primaries_xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    white_point_xy: tuple[float, float]
    luminance_weights: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("colorimetry name must be a nonempty string")
        if not isinstance(self.primaries_xy, Sequence) or len(self.primaries_xy) != 3:
            raise ValueError("primaries_xy must contain red, green, and blue coordinates")

        primaries = tuple(
            _canonical_xy(primary, name=f"primaries_xy[{index}]")
            for index, primary in enumerate(self.primaries_xy)
        )
        white_point = _canonical_xy(self.white_point_xy, name="white_point_xy")

        if not isinstance(self.luminance_weights, Sequence) or len(self.luminance_weights) != 3:
            raise ValueError("luminance_weights must contain three values")
        try:
            weights = tuple(float(weight) for weight in self.luminance_weights)
        except (TypeError, ValueError) as exc:
            raise TypeError("luminance_weights must be real numbers") from exc
        if not all(math.isfinite(weight) for weight in weights):
            raise ValueError("luminance_weights must be finite")
        if not all(weight > 0.0 for weight in weights):
            raise ValueError("luminance_weights must be strictly positive")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("luminance_weights must sum to one")

        # Canonical tuples make the frozen object deeply immutable even when a
        # caller supplied lists at construction time.
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "primaries_xy", primaries)
        object.__setattr__(self, "white_point_xy", white_point)
        object.__setattr__(self, "luminance_weights", weights)


SRGB_REC709_D65 = OutputColorimetry(
    name="linear-sRGB/Rec.709 D65",
    primaries_xy=((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
    white_point_xy=(0.3127, 0.3290),
    luminance_weights=(0.2126, 0.7152, 0.0722),
)
DEFAULT_OUTPUT_COLORIMETRY = SRGB_REC709_D65


def validate_rgb_array(
    rgb: np.ndarray,
    *,
    name: str = "rgb",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> np.ndarray:
    """Return ``rgb`` after validating the common RGB-array contract.

    A color vector ``(3,)``, a swatch table ``(N, 3)``, and arbitrary image or
    batch dimensions ``(..., 3)`` are supported.  Empty arrays are rejected
    because they cannot represent a camera frame or a color sample.
    """

    if not isinstance(rgb, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if rgb.ndim < 1 or rgb.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3)")
    if rgb.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(rgb.dtype, np.floating):
        raise TypeError(f"{name} must have a real floating-point dtype")
    if not np.all(np.isfinite(rgb)):
        raise ValueError(f"{name} must contain only finite values")
    if lower_bound is not None and np.any(rgb < lower_bound):
        raise ValueError(f"{name} must be greater than or equal to {lower_bound}")
    if upper_bound is not None and np.any(rgb > upper_bound):
        raise ValueError(f"{name} must be less than or equal to {upper_bound}")
    return rgb


def srgb_decode(code_rgb: np.ndarray) -> np.ndarray:
    """Decode display-sRGB code values in ``[0, 1]`` to linear light.

    The result has the same shape and floating-point dtype as ``code_rgb``.
    No camera-response inversion or color-space conversion is implied.
    """

    code = validate_rgb_array(
        code_rgb,
        name="code_rgb",
        lower_bound=0.0,
        upper_bound=1.0,
    )
    result = np.empty_like(code)
    threshold = np.asarray(0.04045, dtype=code.dtype)
    linear_branch = code <= threshold
    result[linear_branch] = code[linear_branch] / np.asarray(12.92, dtype=code.dtype)
    nonlinear = code[~linear_branch]
    result[~linear_branch] = np.power(
        (nonlinear + np.asarray(0.055, dtype=code.dtype)) / np.asarray(1.055, dtype=code.dtype),
        np.asarray(2.4, dtype=code.dtype),
    )
    return result


def srgb_encode(linear_rgb: np.ndarray) -> np.ndarray:
    """Encode linear output-sRGB values in ``[0, 1]`` as display sRGB."""

    linear = validate_rgb_array(
        linear_rgb,
        name="linear_rgb",
        lower_bound=0.0,
        upper_bound=1.0,
    )
    result = np.empty_like(linear)
    threshold = np.asarray(0.0031308, dtype=linear.dtype)
    linear_branch = linear <= threshold
    result[linear_branch] = linear[linear_branch] * np.asarray(12.92, dtype=linear.dtype)
    nonlinear = linear[~linear_branch]
    result[~linear_branch] = np.asarray(1.055, dtype=linear.dtype) * np.power(
        nonlinear, np.asarray(1.0 / 2.4, dtype=linear.dtype)
    ) - np.asarray(0.055, dtype=linear.dtype)
    return result


# Both word orders are exposed because ``decode_srgb`` reads naturally at an
# ingestion call site while ``srgb_decode`` mirrors the transfer-function name.
decode_srgb = srgb_decode
encode_srgb = srgb_encode


def luminance(
    rgb: np.ndarray,
    *,
    colorimetry: OutputColorimetry = DEFAULT_OUTPUT_COLORIMETRY,
) -> np.ndarray:
    """Compute output-space luminance using the profile's declared Y row.

    Scaling by the largest channel before the dot product avoids intermediate
    overflow and preserves representable subnormal neutral-gray luminances.
    The returned array has shape ``rgb.shape[:-1]`` and the input dtype.
    """

    values = validate_rgb_array(rgb)
    if not isinstance(colorimetry, OutputColorimetry):
        raise TypeError("colorimetry must be an OutputColorimetry")

    scale = np.max(np.abs(values), axis=-1)
    normalized = np.zeros_like(values)
    np.divide(
        values,
        scale[..., np.newaxis],
        out=normalized,
        where=scale[..., np.newaxis] != 0.0,
    )
    weights = np.asarray(colorimetry.luminance_weights, dtype=values.dtype)
    weighted = np.sum(normalized * weights, axis=-1, dtype=values.dtype)
    return scale * weighted
