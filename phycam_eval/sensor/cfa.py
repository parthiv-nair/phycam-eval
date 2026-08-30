"""Deterministic Bayer color-filter-array selection."""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BayerPattern(str, Enum):
    RGGB = "RGGB"
    BGGR = "BGGR"
    GRBG = "GRBG"
    GBRG = "GBRG"


_CHANNELS = {"R": 0, "G": 1, "B": 2}


def cfa_channel_indices(shape: tuple[int, int], pattern: BayerPattern | str) -> NDArray[np.int8]:
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape must contain two positive dimensions")
    pattern_value = BayerPattern(pattern).value
    tile = np.array(
        [
            [_CHANNELS[pattern_value[0]], _CHANNELS[pattern_value[1]]],
            [_CHANNELS[pattern_value[2]], _CHANNELS[pattern_value[3]]],
        ],
        dtype=np.int8,
    )
    rows = np.arange(shape[0])[:, None] % 2
    columns = np.arange(shape[1])[None, :] % 2
    return tile[rows, columns]


def mosaic_rgb(camera_linear_rgb: ArrayLike, pattern: BayerPattern | str) -> NDArray[np.float64]:
    values = np.asarray(camera_linear_rgb, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("camera_linear_rgb must have shape (H, W, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("camera_linear_rgb must be finite")
    channels = cfa_channel_indices(values.shape[:2], pattern)
    return np.take_along_axis(values, channels[..., None], axis=-1)[..., 0]
