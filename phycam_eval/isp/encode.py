"""Output encoding stage.

Range and gamut mapping are intentionally not performed here; callers must
apply a named post-tone policy before requesting sRGB encoding.
"""

from __future__ import annotations

import numpy as np

from ..color import srgb_encode as _srgb_encode

__all__ = ["encode_srgb", "srgb_encode"]


def srgb_encode(linear_output_rgb: np.ndarray) -> np.ndarray:
    """Encode bounded linear output-sRGB using the exact sRGB transfer."""

    return _srgb_encode(linear_output_rgb)


encode_srgb = srgb_encode
