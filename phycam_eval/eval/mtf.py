"""Exact transfer-function and sampled slanted-edge MTF diagnostics.

The exact OTF/kernel paths should be preferred for a known linear,
shift-invariant model.  The slanted-edge path is a reproducible research
estimator inspired by ISO 12233; it is not an ISO-conformance implementation.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..optics.psf import OpticalTransferFunction, PixelIntegratedKernel

FloatArray = NDArray[np.float64]


def _readonly_vector(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MTFCurve:
    """A DC-normalized one-dimensional MTF curve with explicit frequency unit."""

    frequency: FloatArray
    mtf: FloatArray
    frequency_unit: str
    method: str

    def __post_init__(self) -> None:
        frequency = _readonly_vector(self.frequency, name="frequency")
        mtf = _readonly_vector(self.mtf, name="mtf")
        if frequency.shape != mtf.shape:
            raise ValueError("frequency and mtf must have equal shape")
        if frequency[0] < 0.0 or np.any(np.diff(frequency) <= 0.0):
            raise ValueError("frequency must be nonnegative and strictly increasing")
        if not np.isclose(frequency[0], 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("frequency must begin at DC")
        if np.any(mtf < 0.0):
            raise ValueError("mtf must be nonnegative")
        if not np.isclose(mtf[0], 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError("mtf must be DC-normalized")
        if not isinstance(self.frequency_unit, str) or not self.frequency_unit:
            raise ValueError("frequency_unit must be a non-empty string")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a non-empty string")
        mtf_copy = np.array(mtf, copy=True)
        mtf_copy[0] = 1.0
        mtf_copy.setflags(write=False)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "mtf", mtf_copy)

    def crossing(self, level: float = 0.5) -> float:
        return first_downward_crossing(self.frequency, self.mtf, level=level)


def first_downward_crossing(frequency: ArrayLike, mtf: ArrayLike, *, level: float = 0.5) -> float:
    """Interpolate the first downward crossing, returning NaN if right-censored."""

    curve = MTFCurve(frequency, mtf, "unspecified", "crossing-input")
    if isinstance(level, bool) or not isinstance(level, numbers.Real):
        raise TypeError("level must be a real number")
    target = float(level)
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("level must be finite and in (0, 1)")
    values = curve.mtf
    if values[0] <= target:
        raise ValueError("crossing is left-censored at DC")
    pairs = np.flatnonzero((values[:-1] > target) & (values[1:] <= target))
    if pairs.size == 0:
        if np.any(values <= target):
            raise ValueError("curve contains no valid downward crossing")
        return float("nan")
    index = int(pairs[0] + 1)
    f0, f1 = curve.frequency[index - 1], curve.frequency[index]
    m0, m1 = values[index - 1], values[index]
    fraction = (target - m0) / (m1 - m0)
    return float(f0 + fraction * (f1 - f0))


def mtf50(frequency: ArrayLike, mtf: ArrayLike) -> float:
    """Return the first observed 50% crossing or NaN when right-censored."""

    return first_downward_crossing(frequency, mtf, level=0.5)


def otf_axis_mtf(otf: OpticalTransferFunction, *, axis: Literal["x", "y"] = "x") -> MTFCurve:
    """Extract the positive-frequency central axis of an exact physical OTF."""

    if not isinstance(otf, OpticalTransferFunction):
        raise TypeError("otf must be an OpticalTransferFunction")
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    center = otf.values.shape[0] // 2
    values = otf.values[center, center:] if axis == "x" else otf.values[center:, center]
    frequency = otf.frequency_axis_cpm[center:]
    magnitude = np.abs(values)
    magnitude /= magnitude[0]
    return MTFCurve(frequency, magnitude, "cycles/m", f"exact-otf-{axis}-axis")


def kernel_axis_mtf(
    kernel: PixelIntegratedKernel | ArrayLike,
    *,
    axis: Literal["x", "y"] = "x",
    sample_count: int = 2049,
    pixel_pitch_m: float | None = None,
) -> MTFCurve:
    """Evaluate a discrete kernel's axis DTFT from DC through Nyquist."""

    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if not isinstance(sample_count, numbers.Integral) or isinstance(sample_count, bool):
        raise TypeError("sample_count must be an integer")
    sample_count = int(sample_count)
    if sample_count < 3:
        raise ValueError("sample_count must be at least 3")
    if isinstance(kernel, PixelIntegratedKernel):
        values = np.asarray(kernel.values, dtype=np.float64)
        pitch = kernel.pixel_pitch_m
        if pixel_pitch_m is not None and not math.isclose(
            float(pixel_pitch_m), pitch, rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError("pixel_pitch_m disagrees with PixelIntegratedKernel")
    else:
        values = np.asarray(kernel, dtype=np.float64)
        pitch = None if pixel_pitch_m is None else float(pixel_pitch_m)
    if values.ndim != 2 or any(size == 0 for size in values.shape):
        raise ValueError("kernel must be a non-empty 2-D array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("kernel must contain finite nonnegative values")
    total = float(values.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("kernel must have positive DC gain")
    values = values / total
    line_spread = values.sum(axis=0 if axis == "x" else 1, dtype=np.float64)
    positions = np.arange(line_spread.size, dtype=np.float64) - (line_spread.size - 1) / 2.0
    cycles_per_pixel = np.linspace(0.0, 0.5, sample_count, dtype=np.float64)
    transform = np.exp(-2j * np.pi * cycles_per_pixel[:, None] * positions[None, :]) @ line_spread
    magnitude = np.abs(transform)
    magnitude /= magnitude[0]
    if pitch is None:
        frequency = cycles_per_pixel
        unit = "cycles/pixel"
    else:
        if not math.isfinite(pitch) or pitch <= 0.0:
            raise ValueError("pixel_pitch_m must be finite and positive")
        frequency = cycles_per_pixel / pitch
        unit = "cycles/m"
    return MTFCurve(frequency, magnitude, unit, f"kernel-dtft-{axis}-axis")


def esf_to_mtf(esf: ArrayLike, *, sample_spacing: float) -> MTFCurve:
    """Differentiate an ESF, window its LSF, and transform to a sampled MTF."""

    samples = np.asarray(esf, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 8:
        raise ValueError("esf must be one-dimensional with at least 8 samples")
    if not np.all(np.isfinite(samples)):
        raise ValueError("esf must contain only finite values")
    spacing = float(sample_spacing)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("sample_spacing must be finite and positive")
    lsf = np.gradient(samples, spacing)
    transform = np.fft.rfft(lsf * np.hanning(samples.size))
    magnitude = np.abs(transform)
    if magnitude[0] <= 1e-12 * max(1.0, float(magnitude.max())):
        raise ValueError("ESF derivative has negligible DC energy")
    magnitude /= magnitude[0]
    frequency = np.fft.rfftfreq(samples.size, d=spacing)
    return MTFCurve(frequency, magnitude, "inverse-sample-unit", "slanted-edge-esf")


def _to_gray(image: ArrayLike) -> FloatArray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim not in {2, 3} or any(size == 0 for size in array.shape):
        raise ValueError("image must have shape (H,W), (H,W,C), or (C,H,W)")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must contain only finite values")
    if array.ndim == 2:
        return array
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = array.transpose(1, 2, 0)
    if array.shape[-1] not in {1, 3, 4}:
        raise ValueError("color images must have 1, 3, or 4 channels")
    if array.shape[-1] == 1:
        return array[..., 0]
    return np.asarray(
        0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2],
        dtype=np.float64,
    )


def _measure_esf(roi: FloatArray, *, n_bins: int) -> tuple[FloatArray, FloatArray]:
    if roi.ndim != 2 or min(roi.shape) < 16:
        raise ValueError("slanted-edge ROI must be a 2-D array at least 16x16")
    if not isinstance(n_bins, numbers.Integral) or isinstance(n_bins, bool) or n_bins < 32:
        raise ValueError("n_bins must be an integer >= 32")
    n_bins = int(n_bins)
    contrast = float(np.ptp(roi))
    if contrast <= 1e-10:
        raise ValueError("ROI must contain a non-zero-contrast edge")
    threshold = 0.5 * (float(roi.min()) + float(roi.max()))
    height, width = roi.shape
    edge_x = np.full(height, np.nan, dtype=np.float64)
    strength = np.zeros(height, dtype=np.float64)
    for row_index, row in enumerate(roi):
        difference = np.diff(row)
        column = int(np.argmax(np.abs(difference)))
        strength[row_index] = abs(float(difference[column]))
        if strength[row_index] <= 1e-12:
            continue
        v0, v1 = float(row[column]), float(row[column + 1])
        fraction = (threshold - v0) / (v1 - v0)
        edge_x[row_index] = column + (fraction if 0.0 <= fraction <= 1.0 else 0.5)
    valid = np.isfinite(edge_x) & (strength >= max(1e-12, 0.05 * float(strength.max())))
    if np.count_nonzero(valid) < max(16, height // 4):
        raise ValueError("ROI does not contain a measurable edge across rows")
    rows = np.arange(height, dtype=np.float64)
    slope, intercept = np.polyfit(rows[valid], edge_x[valid], 1)
    angle = math.degrees(math.atan(abs(float(slope))))
    if not 1.0 <= angle <= 15.0:
        raise ValueError("ROI edge must be slanted between 1 and 15 degrees from vertical")
    if abs(float(slope)) * (np.flatnonzero(valid)[-1] - np.flatnonzero(valid)[0]) < 1.0:
        raise ValueError("ROI has insufficient sub-pixel phase coverage")
    center_by_row = slope * rows[:, None] + intercept
    columns = np.arange(width, dtype=np.float64)[None, :]
    position = ((columns - center_by_row) / math.sqrt(1.0 + slope**2)).ravel()
    values = roi.ravel()
    edges = np.linspace(float(position.min()), float(position.max()), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices = np.clip(np.digitize(position, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(indices, minlength=n_bins).astype(np.float64)
    sums = np.bincount(indices, weights=values, minlength=n_bins).astype(np.float64)
    occupied = counts > 0.0
    if np.count_nonzero(occupied) < max(8, n_bins // 4):
        raise ValueError("ROI provides too few occupied ESF bins")
    esf = np.empty(n_bins, dtype=np.float64)
    esf[occupied] = sums[occupied] / counts[occupied]
    esf[~occupied] = np.interp(centers[~occupied], centers[occupied], esf[occupied])
    return centers, esf


def measure_slanted_edge_mtf(
    image: ArrayLike,
    *,
    roi: tuple[int, int, int, int],
    linear_light: bool,
    pixel_pitch_m: float | None = None,
    n_bins: int = 512,
) -> MTFCurve:
    """Measure a declared linear-light, near-vertical slanted edge.

    ``linear_light`` is intentionally required so encoded display values are
    not silently treated as irradiance.  Frequencies above the original image
    Nyquist limit are discarded even though the ESF is numerically oversampled.
    """

    if linear_light is not True:
        raise ValueError("slanted-edge MTF requires declared linear-light input")
    gray = _to_gray(image)
    dynamic_range = float(np.ptp(gray))
    if dynamic_range <= 1e-10:
        raise ValueError("image must contain non-zero contrast")
    gray = (gray - float(gray.min())) / dynamic_range
    if len(roi) != 4 or not all(isinstance(value, numbers.Integral) for value in roi):
        raise ValueError("roi must contain four integer bounds")
    row0, row1, col0, col1 = (int(value) for value in roi)
    if not (0 <= row0 < row1 <= gray.shape[0] and 0 <= col0 < col1 <= gray.shape[1]):
        raise ValueError("roi is outside the image")
    positions, esf = _measure_esf(gray[row0:row1, col0:col1], n_bins=n_bins)
    spacing_pixels = float(np.median(np.diff(positions)))
    base_curve = esf_to_mtf(esf, sample_spacing=spacing_pixels)
    supported = base_curve.frequency <= 0.5 + 1e-12
    frequency_cpp = base_curve.frequency[supported]
    mtf = base_curve.mtf[supported]
    if frequency_cpp.size < 2:
        raise ValueError("ROI provides insufficient image-supported frequencies")
    if pixel_pitch_m is None:
        frequency = frequency_cpp
        unit = "cycles/pixel"
    else:
        pitch = float(pixel_pitch_m)
        if not math.isfinite(pitch) or pitch <= 0.0:
            raise ValueError("pixel_pitch_m must be finite and positive")
        frequency = frequency_cpp / pitch
        unit = "cycles/m"
    return MTFCurve(frequency, mtf, unit, "slanted-edge-linear-light")


def make_slanted_edge_chart(
    height: int = 256, width: int = 256, *, angle_deg: float = 5.0
) -> tuple[FloatArray, tuple[int, int, int, int]]:
    """Generate a deterministic linear-light synthetic chart and centered ROI."""

    if not isinstance(height, numbers.Integral) or isinstance(height, bool):
        raise TypeError("height must be an integer")
    if not isinstance(width, numbers.Integral) or isinstance(width, bool):
        raise TypeError("width must be an integer")
    height, width = int(height), int(width)
    if height < 16 or width < 16:
        raise ValueError("height and width must each be at least 16")
    angle = float(angle_deg)
    if not math.isfinite(angle) or not 1.0 <= abs(angle) <= 15.0:
        raise ValueError("angle_deg magnitude must lie in [1, 15]")
    slope = math.tan(math.radians(angle))
    rows = np.arange(height, dtype=np.float64)
    columns = np.arange(width, dtype=np.float64)
    edge = (width - 1) / 2.0 + (rows - (height - 1) / 2.0) * slope
    chart = (columns[None, :] >= edge[:, None]).astype(np.float64)
    half_width = min(64, width // 4)
    roi = (0, height, width // 2 - half_width, width // 2 + half_width)
    return chart, roi


__all__ = [
    "MTFCurve",
    "esf_to_mtf",
    "first_downward_crossing",
    "kernel_axis_mtf",
    "make_slanted_edge_chart",
    "measure_slanted_edge_mtf",
    "mtf50",
    "otf_axis_mtf",
]
