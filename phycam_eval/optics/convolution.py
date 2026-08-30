"""Padded spatially invariant convolution for collapsed optical kernels."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import convolve, fftconvolve

from .psf import PixelIntegratedKernel
from .sampling import CellAverageTransferKernel, PSFQuadratureKernel

BoundaryPolicy = Literal["reflect", "constant", "zero", "valid"]
ConvolutionMethod = Literal["fft", "direct"]


def _kernel_array(
    kernel: ArrayLike | PixelIntegratedKernel | PSFQuadratureKernel | CellAverageTransferKernel,
) -> NDArray[np.float64]:
    if isinstance(kernel, (PixelIntegratedKernel, PSFQuadratureKernel, CellAverageTransferKernel)):
        values = kernel.values
    else:
        values = np.asarray(kernel, dtype=np.float64)
    if values.ndim not in (2, 3):
        raise ValueError("kernel must be a 2-D array or a channel-last 3-D array")
    if values.shape[0] % 2 != 1 or values.shape[1] % 2 != 1:
        raise ValueError("kernel spatial dimensions must be odd")
    if not np.all(np.isfinite(values)):
        raise ValueError("kernel must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("optical kernel cannot contain negative values")
    if values.ndim == 2:
        sums = np.array([values.sum(dtype=np.float64)])
    else:
        sums = values.sum(axis=(0, 1), dtype=np.float64)
    if not np.allclose(sums, 1.0, rtol=5e-11, atol=5e-13):
        raise ValueError("every optical kernel channel must sum to one")
    return np.asarray(values, dtype=np.float64)


def _convolve_plane(
    plane: NDArray[np.float64],
    kernel: NDArray[np.float64],
    boundary: BoundaryPolicy,
    constant_value: float,
    method: ConvolutionMethod,
) -> NDArray[np.float64]:
    if method == "fft":
        operation = fftconvolve
    elif method == "direct":

        def operation(a, b, mode):
            return convolve(a, b, mode=mode, method="direct")
    else:
        raise ValueError("method must be 'fft' or 'direct'")

    if boundary == "valid":
        if plane.shape[0] < kernel.shape[0] or plane.shape[1] < kernel.shape[1]:
            raise ValueError("valid convolution requires image dimensions >= kernel dimensions")
        return np.asarray(operation(plane, kernel, mode="valid"), dtype=np.float64)

    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padding = ((pad_y, pad_y), (pad_x, pad_x))
    if boundary == "reflect":
        padded = np.pad(plane, padding, mode="reflect")
    elif boundary == "constant":
        if not np.isfinite(constant_value):
            raise ValueError("constant_value must be finite")
        padded = np.pad(plane, padding, mode="constant", constant_values=constant_value)
    elif boundary == "zero":
        padded = np.pad(plane, padding, mode="constant", constant_values=0.0)
    else:
        raise ValueError("boundary must be 'reflect', 'constant', 'zero', or 'valid'")
    return np.asarray(operation(padded, kernel, mode="valid"), dtype=np.float64)


def convolve_spatially_invariant(
    image: ArrayLike,
    kernel: ArrayLike | PixelIntegratedKernel | PSFQuadratureKernel | CellAverageTransferKernel,
    *,
    boundary: BoundaryPolicy,
    constant_value: float = 0.0,
    method: ConvolutionMethod = "fft",
) -> NDArray[np.float64]:
    """Apply a declared-boundary linear convolution to one or more channels.

    Images are either ``(H, W)`` or channel-last ``(H, W, C)``.  A 2-D
    kernel is shared across channels; a 3-D kernel supplies one kernel per
    image channel.  ``PSFQuadratureKernel`` is the oversampled full-forward
    representation; ``CellAverageTransferKernel`` is the exact equal-grid
    source-cell-to-photosite collapse; and ``PixelIntegratedKernel`` is the
    single-aperture point-sample diagnostic representation. FFT mode is
    explicitly padded and never circular.
    """

    image_array = np.asarray(image, dtype=np.float64)
    if image_array.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    if image_array.shape[0] == 0 or image_array.shape[1] == 0:
        raise ValueError("image spatial dimensions cannot be empty")
    if not np.all(np.isfinite(image_array)):
        raise ValueError("image must contain only finite values")
    kernel_array = _kernel_array(kernel)

    if image_array.ndim == 2:
        if kernel_array.ndim != 2:
            raise ValueError("a single-channel image requires a 2-D kernel")
        return _convolve_plane(image_array, kernel_array, boundary, constant_value, method)

    channels = image_array.shape[2]
    if channels == 0:
        raise ValueError("image channel dimension cannot be empty")
    if kernel_array.ndim == 3 and kernel_array.shape[2] != channels:
        raise ValueError("kernel and image channel counts must match")
    outputs = []
    for channel in range(channels):
        channel_kernel = kernel_array if kernel_array.ndim == 2 else kernel_array[:, :, channel]
        outputs.append(
            _convolve_plane(
                image_array[:, :, channel],
                channel_kernel,
                boundary,
                constant_value,
                method,
            )
        )
    return np.stack(outputs, axis=2)
