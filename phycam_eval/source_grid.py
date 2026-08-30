"""Explicit mappings between source samples and a modeled sensor grid.

Images are treated as arrays of cell-average linear-light values.  A native
LDR approximation may assign one source cell to one synthetic sensor pixel.
Forward-camera inputs instead carry physical source geometry and are area
resampled onto a declared sensor window before capture.  Array shape alone is
never interpreted as a physical scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SourceGridError(ValueError):
    """Raised when a source-to-sensor mapping is physically inconsistent."""


class SourceGridMode(str, Enum):
    """Supported interpretations of an input image grid."""

    NATIVE_AS_SYNTHETIC_SENSOR = "native_as_synthetic_sensor"
    MATCHED_WINDOW_AREA_RESAMPLE = "matched_window_area_resample"


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise SourceGridError(f"{name} must be positive")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise SourceGridError(f"{name} must be finite and positive")
    return result


def _finite_pair(values: Iterable[float], name: str) -> tuple[float, float]:
    pair = tuple(float(value) for value in values)
    if len(pair) != 2:
        raise SourceGridError(f"{name} must contain exactly two values")
    if not all(np.isfinite(value) for value in pair):
        raise SourceGridError(f"{name} must be finite")
    return pair[0], pair[1]


@dataclass(frozen=True, slots=True)
class GridGeometry:
    """A rectangular cell grid in a common sensor-plane coordinate system.

    ``origin_m`` is the upper-left window boundary as ``(y, x)``.  Pixel pitch
    is also ordered ``(y, x)``.  Values are in meters.
    """

    height: int
    width: int
    pixel_pitch_m: tuple[float, float]
    origin_m: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "height", _positive_int(self.height, "height"))
        object.__setattr__(self, "width", _positive_int(self.width, "width"))
        pitch = _finite_pair(self.pixel_pitch_m, "pixel_pitch_m")
        pitch = (
            _finite_positive(pitch[0], "pixel_pitch_m[0]"),
            _finite_positive(pitch[1], "pixel_pitch_m[1]"),
        )
        object.__setattr__(self, "pixel_pitch_m", pitch)
        object.__setattr__(self, "origin_m", _finite_pair(self.origin_m, "origin_m"))

    @classmethod
    def square_pixels(
        cls,
        height: int,
        width: int,
        pixel_pitch_m: float,
        *,
        origin_m: tuple[float, float] = (0.0, 0.0),
    ) -> GridGeometry:
        pitch = _finite_positive(pixel_pitch_m, "pixel_pitch_m")
        return cls(height, width, (pitch, pitch), origin_m)

    @property
    def extent_m(self) -> tuple[float, float]:
        return (
            self.height * self.pixel_pitch_m[0],
            self.width * self.pixel_pitch_m[1],
        )

    @property
    def bounds_m(self) -> tuple[float, float, float, float]:
        extent_y, extent_x = self.extent_m
        return (
            self.origin_m[0],
            self.origin_m[0] + extent_y,
            self.origin_m[1],
            self.origin_m[1] + extent_x,
        )

    @property
    def pixel_area_m2(self) -> float:
        return self.pixel_pitch_m[0] * self.pixel_pitch_m[1]


@dataclass(frozen=True, slots=True)
class MappedGrid:
    """A mapped array paired with its explicit modeled geometry."""

    values: NDArray[np.float64]
    geometry: GridGeometry
    mode: SourceGridMode

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=np.float64)
        if array.ndim < 2:
            raise SourceGridError("mapped values must have at least two dimensions")
        if array.shape[:2] != (self.geometry.height, self.geometry.width):
            raise SourceGridError("mapped values do not match the declared geometry")
        if not np.all(np.isfinite(array)):
            raise SourceGridError("mapped values must be finite")
        array = np.array(array, dtype=np.float64, copy=True, order="C")
        array.setflags(write=False)
        object.__setattr__(self, "values", array)


def native_as_synthetic_sensor(
    image: ArrayLike,
    *,
    pixel_pitch_m: float | tuple[float, float],
    origin_m: tuple[float, float] = (0.0, 0.0),
) -> MappedGrid:
    """Assign each native sample to one explicitly sized synthetic pixel."""

    values = _validated_image(image)
    if np.isscalar(pixel_pitch_m):
        geometry = GridGeometry.square_pixels(
            values.shape[0], values.shape[1], float(pixel_pitch_m), origin_m=origin_m
        )
    else:
        geometry = GridGeometry(values.shape[0], values.shape[1], tuple(pixel_pitch_m), origin_m)
    return MappedGrid(values, geometry, SourceGridMode.NATIVE_AS_SYNTHETIC_SENSOR)


def area_resample_to_sensor(
    image: ArrayLike,
    *,
    source: GridGeometry,
    sensor: GridGeometry,
    window_tolerance_m: float = 1e-12,
) -> MappedGrid:
    """Area-resample cell averages between identical physical windows.

    The separable overlap operator exactly preserves constants and the spatial
    integral (up to floating-point error).  It is an explicit box/cell-average
    reconstruction, not bilinear interpolation and not detector preprocessing.
    """

    values = _validated_image(image)
    if values.shape[:2] != (source.height, source.width):
        raise SourceGridError("source array shape does not match source geometry")
    tolerance = float(window_tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise SourceGridError("window_tolerance_m must be finite and nonnegative")
    if not np.allclose(
        source.bounds_m,
        sensor.bounds_m,
        rtol=0.0,
        atol=tolerance,
    ):
        raise SourceGridError(
            "source and sensor windows must match; cropping/FOV changes require "
            "a separate declared adapter"
        )

    if source == sensor:
        # The native active-ROI contract uses the stored image cells as the
        # sensor photosites. Preserve this exact identity without constructing
        # dense square overlap matrices.
        return MappedGrid(values, sensor, SourceGridMode.MATCHED_WINDOW_AREA_RESAMPLE)

    weights_y = _overlap_average_weights(
        source.height,
        source.pixel_pitch_m[0],
        source.origin_m[0],
        sensor.height,
        sensor.pixel_pitch_m[0],
        sensor.origin_m[0],
    )
    weights_x = _overlap_average_weights(
        source.width,
        source.pixel_pitch_m[1],
        source.origin_m[1],
        sensor.width,
        sensor.pixel_pitch_m[1],
        sensor.origin_m[1],
    )
    # The rectangular overlap operator is exactly separable. A single
    # three-operand contraction can select a near-naive path whose work scales
    # with source_area * target_area. Contracting one spatial axis at a time
    # evaluates the same finite-volume sum while scaling independently in y
    # and x. The second tensordot appends target-x after any trailing channels,
    # so restore it to the second spatial axis.
    vertical = np.tensordot(weights_y, values, axes=((1,), (0,)))
    mapped = np.tensordot(vertical, weights_x, axes=((1,), (1,)))
    if mapped.ndim > 2:
        mapped = np.moveaxis(mapped, -1, 1)
    return MappedGrid(
        mapped,
        sensor,
        SourceGridMode.MATCHED_WINDOW_AREA_RESAMPLE,
    )


def spatial_integral(values: ArrayLike, geometry: GridGeometry) -> NDArray[np.float64]:
    """Return the per-trailing-component integral over the sensor window."""

    array = _validated_image(values)
    if array.shape[:2] != (geometry.height, geometry.width):
        raise SourceGridError("array shape does not match geometry")
    return np.sum(array, axis=(0, 1)) * geometry.pixel_area_m2


def _validated_image(image: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim < 2:
        raise SourceGridError("image must have at least two dimensions")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise SourceGridError("image spatial dimensions must be nonempty")
    if not np.all(np.isfinite(array)):
        raise SourceGridError("image values must be finite")
    return array


def _overlap_average_weights(
    source_count: int,
    source_pitch: float,
    source_origin: float,
    target_count: int,
    target_pitch: float,
    target_origin: float,
) -> NDArray[np.float64]:
    source_edges = source_origin + np.arange(source_count + 1) * source_pitch
    target_edges = target_origin + np.arange(target_count + 1) * target_pitch
    left = np.maximum(target_edges[:-1, None], source_edges[None, :-1])
    right = np.minimum(target_edges[1:, None], source_edges[None, 1:])
    overlap = np.maximum(0.0, right - left)
    weights = overlap / target_pitch
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise SourceGridError("target cells are not fully covered by the source window")
    return weights
