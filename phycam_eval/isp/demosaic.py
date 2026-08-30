"""Fixed, deterministic Bayer demosaicing for the RAW reference tier."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import convolve

from phycam_eval.sensor.cfa import BayerPattern, cfa_channel_indices

_BILINEAR_KERNEL = np.array(
    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
    dtype=np.float64,
)


def demosaic_bilinear(
    raw: ArrayLike,
    pattern: BayerPattern | str,
    *,
    boundary: str = "reflect",
) -> NDArray[np.float64]:
    """Interpolate each CFA plane with normalized fixed bilinear weights.

    Signed RAW values are retained.  Observed photosites are restored exactly
    after interpolation, so this stage does not alter measured samples.
    """

    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("raw must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("raw must be finite")
    if boundary not in {"reflect", "mirror", "nearest", "constant", "wrap"}:
        raise ValueError("unsupported boundary mode")

    channel_map = cfa_channel_indices(values.shape, pattern)
    output = np.empty((*values.shape, 3), dtype=np.float64)
    for channel in range(3):
        mask = channel_map == channel
        numerator = convolve(np.where(mask, values, 0.0), _BILINEAR_KERNEL, mode=boundary, cval=0.0)
        denominator = convolve(mask.astype(np.float64), _BILINEAR_KERNEL, mode=boundary, cval=0.0)
        if np.any(denominator <= 0.0):
            raise RuntimeError("demosaic interpolation has uncovered pixels")
        plane = numerator / denominator
        plane[mask] = values[mask]
        output[..., channel] = plane
    return output
