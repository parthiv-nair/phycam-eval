"""Joint motion, continuous optics, and photosite-area image formation.

The forward path represented here deliberately keeps two spatial lattices
distinct. ``scene_rgb`` is a cell-average reconstruction on a declared source
grid. On an oversampled source, optical PSFs are projected as
``PSFQuadratureKernel`` objects and the irradiance is subsequently averaged
onto photosites. On an equal source/sensor grid, an analytically integrated
``CellAverageTransferKernel`` collapses the source-cell reconstruction,
continuous PSF, and one target-area average into the same linear operator.
Both paths apply the detector aperture exactly once.

Homographies supplied to this module use the modeled sensor's pixel-center
coordinates.  They are conjugated into source-grid pixel-center coordinates
before the oversampled scene is warped.  This matters whenever source and
sensor pitches differ: a one-sensor-pixel translation is several source-grid
samples on an oversampled reconstruction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .optics.convolution import convolve_spatially_invariant
from .optics.sampling import CellAverageTransferKernel, PSFQuadratureKernel
from .readout.rolling_shutter import warp_homography
from .readout.timing import ReadoutTiming
from .source_grid import GridGeometry, area_resample_to_sensor

HomographyAtTime = Callable[[float], ArrayLike]
OpticsBoundary = Literal["reflect", "constant", "zero"]
WarpBoundary = Literal["constant", "nearest", "reflect", "mirror", "wrap"]
ConvolutionMethod = Literal["fft", "direct"]
OpticalFormationKernel = PSFQuadratureKernel | CellAverageTransferKernel


def source_to_sensor_coordinate_transform(
    source: GridGeometry,
    sensor: GridGeometry,
) -> NDArray[np.float64]:
    """Map source-grid pixel-center coordinates to sensor-pixel coordinates.

    Array coordinates are center based: coordinate zero denotes the center of
    the first cell.  ``GridGeometry.origin_m`` instead denotes the upper-left
    *boundary*.  For the x coordinate, for example, the mapping is

    ``x_sensor = (origin_source + (x_source + 1/2) pitch_source
                  - origin_sensor) / pitch_sensor - 1/2``.

    The returned matrix applies that mapping in homogeneous ``(x, y, 1)``
    order.  The geometries need not have equal extents for this coordinate
    helper, although joint formation itself requires a matched physical
    window.
    """

    scale_x = source.pixel_pitch_m[1] / sensor.pixel_pitch_m[1]
    scale_y = source.pixel_pitch_m[0] / sensor.pixel_pitch_m[0]
    offset_x = (
        (source.origin_m[1] - sensor.origin_m[1]) / sensor.pixel_pitch_m[1] + 0.5 * scale_x - 0.5
    )
    offset_y = (
        (source.origin_m[0] - sensor.origin_m[0]) / sensor.pixel_pitch_m[0] + 0.5 * scale_y - 0.5
    )
    return np.array(
        [
            [scale_x, 0.0, offset_x],
            [0.0, scale_y, offset_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def transform_sensor_homography_to_source_grid(
    sensor_homography: ArrayLike,
    *,
    source: GridGeometry,
    sensor: GridGeometry,
) -> NDArray[np.float64]:
    """Conjugate a sensor-pixel homography onto the source-grid lattice.

    ``sensor_homography`` maps reference sensor coordinates to coordinates at
    the requested time.  The returned homography has the same direction but
    acts on the oversampled source array consumed by :func:`warp_homography`.
    """

    homography = np.asarray(sensor_homography, dtype=np.float64)
    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        raise ValueError("sensor_homography must be a finite 3 by 3 matrix")
    determinant = float(np.linalg.det(homography))
    if not np.isfinite(determinant) or abs(determinant) < 1e-15:
        raise ValueError("sensor_homography must be invertible")

    sensor_from_source = source_to_sensor_coordinate_transform(source, sensor)
    source_from_sensor = np.linalg.inv(sensor_from_source)
    transformed = source_from_sensor @ homography @ sensor_from_source
    scale = transformed[2, 2]
    if abs(scale) > 1e-15:
        transformed = transformed / scale
    return np.asarray(transformed, dtype=np.float64)


def render_joint_photosite_exposure(
    scene_rgb: ArrayLike,
    *,
    source: GridGeometry,
    sensor: GridGeometry,
    timing: ReadoutTiming,
    optical_kernels: Sequence[OpticalFormationKernel],
    homography_at_time: HomographyAtTime | None = None,
    quadrature_order: int = 3,
    interpolation_order: int = 1,
    optics_boundary: OpticsBoundary,
    warp_boundary: WarpBoundary,
    constant_value: float = 0.0,
    convolution_method: ConvolutionMethod = "fft",
    window_tolerance_m: float = 1e-12,
) -> NDArray[np.float64]:
    """Form time-averaged camera RGB values on the target photosites.

    For every row-exposure quadrature time the operation order is fixed:

    1. transform the sensor-coordinate homography to the source lattice;
    2. warp the oversampled instantaneous scene;
    3. convolve each camera channel with its declared optical representation;
    4. area-average onto target photosites exactly once, explicitly or inside
       an equal-grid cell-average transfer kernel; and
    5. accumulate only the currently exposed target row.

    ``homography_at_time=None`` declares a static scene and enables a direct
    static path.  A callable that returns identity matrices produces the same
    numerical result through the general quadrature path.  ``line_time_s=0``
    enables a global-shutter path that forms each quadrature image only once.

    The returned values are exposure-time averages, not a second time
    integral.  A later electron-budget stage may therefore multiply them by
    the declared exposure duration exactly once.
    """

    values, kernels = _validate_inputs(
        scene_rgb,
        source=source,
        sensor=sensor,
        timing=timing,
        optical_kernels=optical_kernels,
        quadrature_order=quadrature_order,
        interpolation_order=interpolation_order,
        optics_boundary=optics_boundary,
        warp_boundary=warp_boundary,
        constant_value=constant_value,
        convolution_method=convolution_method,
        window_tolerance_m=window_tolerance_m,
    )

    def form_at_time(time_s: float) -> NDArray[np.float64]:
        if homography_at_time is None:
            warped = values
        else:
            source_homography = transform_sensor_homography_to_source_grid(
                homography_at_time(float(time_s)),
                source=source,
                sensor=sensor,
            )
            warped = warp_homography(
                values,
                source_homography,
                interpolation_order=interpolation_order,
                boundary=warp_boundary,
                cval=constant_value,
            )

        formed_channels = [
            convolve_spatially_invariant(
                warped[:, :, channel],
                kernel,
                boundary=optics_boundary,
                constant_value=constant_value,
                method=convolution_method,
            )
            for channel, kernel in enumerate(kernels)
        ]
        source_irradiance = np.stack(formed_channels, axis=2)
        if isinstance(kernels[0], CellAverageTransferKernel):
            # Equal-grid collapsed kernels already include the one target
            # photosite average as well as finite source-cell reconstruction.
            return source_irradiance
        return area_resample_to_sensor(
            source_irradiance,
            source=source,
            sensor=sensor,
            window_tolerance_m=window_tolerance_m,
        ).values

    # With no motion, neither optics nor photosite integration changes during
    # the exposure; evaluating them once is both exact and a useful statement
    # of the static model.
    if homography_at_time is None:
        return np.array(form_at_time(float(timing.reference_time_s)), copy=True)

    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))

    # All rows share the same exposure interval under a global shutter.  Form
    # complete target images at the quadrature nodes and integrate them once.
    if timing.is_global:
        start = timing.frame_start_s
        output = np.zeros((sensor.height, sensor.width, 3), dtype=np.float64)
        for node, weight in zip(nodes, weights, strict=True):
            time_s = start + 0.5 * timing.exposure_s * (float(node) + 1.0)
            output += 0.5 * float(weight) * form_at_time(time_s)
        return output

    # Rolling rows generally sample different times.  Avoid retaining every
    # full oversampled/target image merely to seek rare floating-point time
    # coincidences: that would turn this reference into O(H^2 W) storage.
    output = np.empty((sensor.height, sensor.width, 3), dtype=np.float64)
    for row in range(sensor.height):
        start = timing.row_start_s(row)
        accumulated = np.zeros((sensor.width, 3), dtype=np.float64)
        for node, weight in zip(nodes, weights, strict=True):
            time_s = start + 0.5 * timing.exposure_s * (float(node) + 1.0)
            formed = form_at_time(float(time_s))
            accumulated += 0.5 * float(weight) * formed[row]
        output[row] = accumulated
    return output


def _validate_inputs(
    scene_rgb: ArrayLike,
    *,
    source: GridGeometry,
    sensor: GridGeometry,
    timing: ReadoutTiming,
    optical_kernels: Sequence[OpticalFormationKernel],
    quadrature_order: int,
    interpolation_order: int,
    optics_boundary: str,
    warp_boundary: str,
    constant_value: float,
    convolution_method: str,
    window_tolerance_m: float,
) -> tuple[NDArray[np.float64], tuple[OpticalFormationKernel, ...]]:
    values = np.asarray(scene_rgb, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("scene_rgb must have shape (H, W, 3)")
    if values.shape[:2] != (source.height, source.width):
        raise ValueError("scene_rgb spatial shape must match source geometry")
    if not np.all(np.isfinite(values)):
        raise ValueError("scene_rgb must contain only finite values")
    if timing.height != sensor.height:
        raise ValueError("readout timing height must match sensor geometry")

    tolerance = float(window_tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("window_tolerance_m must be finite and nonnegative")
    if not np.allclose(source.bounds_m, sensor.bounds_m, rtol=0.0, atol=tolerance):
        raise ValueError("source and sensor geometries must cover the same physical window")

    try:
        kernels = tuple(optical_kernels)
    except TypeError as error:
        raise TypeError("optical_kernels must be a sequence of three kernels") from error
    if len(kernels) != 3:
        raise ValueError("optical_kernels must contain one kernel per camera RGB channel")
    cell_average = tuple(isinstance(kernel, CellAverageTransferKernel) for kernel in kernels)
    if any(cell_average) and not all(cell_average):
        raise TypeError("optical_kernels cannot mix quadrature and cell-average representations")
    if all(cell_average) and source != sensor:
        raise ValueError("cell-average transfer kernels require identical source and sensor grids")
    if not any(cell_average) and (source.height <= sensor.height or source.width <= sensor.width):
        raise ValueError(
            "PSF quadrature formation requires a source grid strictly finer than "
            "the sensor in both axes; use a cell-average transfer kernel on equal grids"
        )
    for channel, kernel in enumerate(kernels):
        if not isinstance(kernel, (PSFQuadratureKernel, CellAverageTransferKernel)):
            raise TypeError(
                "optical_kernels must contain PSFQuadratureKernel or "
                f"CellAverageTransferKernel objects (channel {channel})"
            )
        if isinstance(kernel, PSFQuadratureKernel) and getattr(kernel, "pixel_integrated", True):
            raise ValueError("a photosite-integrated quadrature kernel cannot enter formation")
        if not np.allclose(
            kernel.sample_spacing_m,
            source.pixel_pitch_m,
            rtol=5e-13,
            atol=0.0,
        ):
            raise ValueError(f"optical kernel channel {channel} spacing must match source geometry")

    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, (int, np.integer)):
        raise TypeError("quadrature_order must be an integer")
    if int(quadrature_order) <= 0:
        raise ValueError("quadrature_order must be positive")
    if isinstance(interpolation_order, bool) or not isinstance(
        interpolation_order, (int, np.integer)
    ):
        raise TypeError("interpolation_order must be an integer")
    if int(interpolation_order) < 0 or int(interpolation_order) > 5:
        raise ValueError("interpolation_order must be between 0 and 5")
    if optics_boundary not in {"reflect", "constant", "zero"}:
        raise ValueError("optics_boundary must be 'reflect', 'constant', or 'zero'")
    if warp_boundary not in {"constant", "nearest", "reflect", "mirror", "wrap"}:
        raise ValueError("unsupported warp_boundary")
    if not np.isfinite(float(constant_value)):
        raise ValueError("constant_value must be finite")
    if convolution_method not in {"fft", "direct"}:
        raise ValueError("convolution_method must be 'fft' or 'direct'")
    return values, kernels


__all__ = [
    "HomographyAtTime",
    "render_joint_photosite_exposure",
    "source_to_sensor_coordinate_transform",
    "transform_sensor_homography_to_source_grid",
]
