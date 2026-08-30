"""Joint scene-warp, optics, and row-exposure reference rendering."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import map_coordinates

from .motion import translation_homography
from .timing import ReadoutTiming

HomographyAtTime = Callable[[float], ArrayLike]
OpticalOperator = Callable[[NDArray[np.float64]], ArrayLike]

_BOUNDARY_MODES = {"constant", "nearest", "reflect", "mirror", "wrap"}


def warp_homography(
    image: ArrayLike,
    reference_to_output: ArrayLike,
    *,
    output_shape: tuple[int, int] | None = None,
    interpolation_order: int = 1,
    boundary: str = "reflect",
    cval: float = 0.0,
) -> NDArray[np.float64]:
    """Warp a reference image with a homography mapping reference to output."""

    values = _validated_image(image)
    homography = np.asarray(reference_to_output, dtype=np.float64)
    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        raise ValueError("reference_to_output must be a finite 3 by 3 matrix")
    determinant = float(np.linalg.det(homography))
    if not np.isfinite(determinant) or abs(determinant) < 1e-15:
        raise ValueError("homography must be invertible")
    if interpolation_order < 0 or interpolation_order > 5:
        raise ValueError("interpolation_order must be between 0 and 5")
    if boundary not in _BOUNDARY_MODES:
        raise ValueError(f"unsupported boundary mode: {boundary}")
    if output_shape is None:
        output_shape = values.shape[:2]
    if len(output_shape) != 2 or min(output_shape) <= 0:
        raise ValueError("output_shape must contain two positive dimensions")

    output_y, output_x = np.indices(output_shape, dtype=np.float64)
    homogeneous = np.stack([output_x.ravel(), output_y.ravel(), np.ones(output_x.size)], axis=0)
    source = np.linalg.inv(homography) @ homogeneous
    denominator = source[2]
    valid = np.abs(denominator) > 1e-15
    source_x = np.full_like(denominator, -1e12)
    source_y = np.full_like(denominator, -1e12)
    source_x[valid] = source[0, valid] / denominator[valid]
    source_y[valid] = source[1, valid] / denominator[valid]
    coordinates = np.stack([source_y, source_x], axis=0)

    trailing_shape = values.shape[2:]
    flat_channels = values.reshape(values.shape[0], values.shape[1], -1)
    warped_channels = [
        map_coordinates(
            flat_channels[..., channel],
            coordinates,
            order=interpolation_order,
            mode=boundary,
            cval=float(cval),
            prefilter=interpolation_order > 1,
        ).reshape(output_shape)
        for channel in range(flat_channels.shape[-1])
    ]
    warped = np.stack(warped_channels, axis=-1)
    if not trailing_shape:
        return warped[..., 0]
    return warped.reshape((*output_shape, *trailing_shape))


def render_rolling_exposure(
    image: ArrayLike,
    timing: ReadoutTiming,
    homography_at_time: HomographyAtTime,
    *,
    optical_operator: OpticalOperator | None = None,
    quadrature_order: int = 3,
    interpolation_order: int = 1,
    boundary: str = "reflect",
    cval: float = 0.0,
) -> NDArray[np.float64]:
    """Render positive-duration rolling exposure with Gauss-Legendre quadrature.

    At every quadrature time this function first projects the scene, then calls
    the optical operator on that instantaneous full image, and only then
    accumulates the exposed row.  It never warps a single preblurred reference.
    """

    values = _validated_image(image)
    if values.shape[0] != timing.height:
        raise ValueError("image height must match readout timing")
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, (int, np.integer)):
        raise TypeError("quadrature_order must be an integer")
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive")

    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    output = np.empty_like(values, dtype=np.float64)
    cache: dict[float, NDArray[np.float64]] = {}

    for row in range(timing.height):
        start = timing.row_start_s(row)
        times = start + 0.5 * timing.exposure_s * (nodes + 1.0)
        accumulated = np.zeros_like(values[row], dtype=np.float64)
        for time_s, weight in zip(times, weights, strict=True):
            key = float(time_s)
            formed = cache.get(key)
            if formed is None:
                formed = _formed_at_time(
                    values,
                    key,
                    homography_at_time,
                    optical_operator,
                    interpolation_order,
                    boundary,
                    cval,
                )
                if timing.is_global:
                    cache[key] = formed
            accumulated += 0.5 * float(weight) * formed[row]
        output[row] = accumulated
    return output


def render_instantaneous_row_geometry(
    image: ArrayLike,
    timing: ReadoutTiming,
    homography_at_time: HomographyAtTime,
    *,
    optical_operator: OpticalOperator | None = None,
    interpolation_order: int = 1,
    boundary: str = "reflect",
    cval: float = 0.0,
) -> NDArray[np.float64]:
    """Render the nonphysical ``exposure_s -> 0+`` row-geometry limit."""

    values = _validated_image(image)
    if values.shape[0] != timing.height:
        raise ValueError("image height must match readout timing")
    output = np.empty_like(values, dtype=np.float64)
    cache: dict[float, NDArray[np.float64]] = {}
    for row in range(timing.height):
        time_s = timing.row_start_s(row)
        formed = cache.get(time_s)
        if formed is None:
            formed = _formed_at_time(
                values,
                time_s,
                homography_at_time,
                optical_operator,
                interpolation_order,
                boundary,
                cval,
            )
            if timing.is_global:
                cache[time_s] = formed
        output[row] = formed[row]
    return output


def analytic_horizontal_row_shear(
    image: ArrayLike,
    timing: ReadoutTiming,
    *,
    velocity_px_s: float,
    reference_time_s: float | None = None,
    interpolation_order: int = 1,
    boundary: str = "reflect",
    cval: float = 0.0,
) -> NDArray[np.float64]:
    """Closed-form constant-horizontal-velocity instantaneous row shear."""

    reference = (
        float(timing.reference_time_s) if reference_time_s is None else float(reference_time_s)
    )
    velocity = float(velocity_px_s)
    if not np.isfinite(reference) or not np.isfinite(velocity):
        raise ValueError("velocity and reference time must be finite")
    return render_instantaneous_row_geometry(
        image,
        timing,
        lambda time_s: translation_homography(velocity * (time_s - reference), 0.0),
        interpolation_order=interpolation_order,
        boundary=boundary,
        cval=cval,
    )


def _formed_at_time(
    image: NDArray[np.float64],
    time_s: float,
    homography_at_time: HomographyAtTime,
    optical_operator: OpticalOperator | None,
    interpolation_order: int,
    boundary: str,
    cval: float,
) -> NDArray[np.float64]:
    warped = warp_homography(
        image,
        homography_at_time(time_s),
        interpolation_order=interpolation_order,
        boundary=boundary,
        cval=cval,
    )
    if optical_operator is None:
        return warped
    formed = np.asarray(optical_operator(warped), dtype=np.float64)
    if formed.shape != image.shape:
        raise ValueError("optical_operator must preserve image shape")
    if not np.all(np.isfinite(formed)):
        raise ValueError("optical_operator returned nonfinite values")
    return formed


def _validated_image(image: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(image, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("image must have two nonempty spatial dimensions")
    if not np.all(np.isfinite(values)):
        raise ValueError("image values must be finite")
    return values
