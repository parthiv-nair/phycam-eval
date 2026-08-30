"""Finite-volume transfers from a continuous optical PSF to scene grids.

``sample_continuous_psf`` projects the continuous-density PSF onto an
oversampled numerical scene grid and deliberately does *not* model a detector
aperture; the detector/photosite stage must subsequently integrate irradiance
over each physical pixel exactly once.  For equal source and target grids,
``collapse_cell_average_transfer`` instead includes that one target-photosite
average together with finite source-cell reconstruction in a single exact
discrete transfer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .psf import ContinuousPSF, _readonly_float_array

FloatArray = NDArray[np.float64]


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {result!r}")
    return result


def _sample_spacing_pair(
    value: float | Iterable[float],
) -> tuple[float, float]:
    if np.isscalar(value):
        spacing = _positive_finite(float(value), "sample_spacing_m")
        return spacing, spacing
    pair = tuple(float(component) for component in value)
    if len(pair) != 2:
        raise ValueError("sample_spacing_m must be a scalar or a (y, x) pair")
    return (
        _positive_finite(pair[0], "sample_spacing_m[0]"),
        _positive_finite(pair[1], "sample_spacing_m[1]"),
    )


@dataclass(frozen=True, slots=True)
class PSFQuadratureKernel:
    """Dimensionless optical-convolution weights on a numerical scene grid.

    ``values[j, i]`` is the integral of the continuous PSF density over one
    numerical reconstruction cell centered at offset ``(j * dy, i * dx)``.
    The finite-volume projection is a quadrature choice, not a physical pixel
    aperture.  Consequently ``pixel_integrated`` is always false.

    The retained finite support is renormalized to sum to one.  Its pre-
    normalization energy is recorded in ``retained_energy`` so truncation is
    explicit and reproducible.
    """

    values: FloatArray
    sample_spacing_m: tuple[float, float]
    retained_energy: float
    requested_encircled_energy: float
    source_identity: str = ""
    quadrature_rule: str = "centered-piecewise-constant-cell-overlap-v1"
    pixel_integrated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        values = _readonly_float_array(self.values, ndim=2, name="values")
        if values.shape[0] % 2 != 1 or values.shape[1] % 2 != 1:
            raise ValueError("quadrature kernel dimensions must be odd and centered")
        if not np.all(np.isfinite(values)):
            raise ValueError("quadrature kernel must contain only finite values")
        if np.any(values < 0.0):
            raise ValueError("quadrature kernel cannot contain negative values")
        if not np.isclose(values.sum(dtype=np.float64), 1.0, rtol=5e-13, atol=5e-13):
            raise ValueError("quadrature kernel weights must sum to one")

        spacing = _sample_spacing_pair(self.sample_spacing_m)
        retained = float(self.retained_energy)
        requested = float(self.requested_encircled_energy)
        if not math.isfinite(retained) or not (0.0 < retained <= 1.0 + 5e-13):
            raise ValueError("retained_energy must be finite and in (0, 1]")
        if not math.isfinite(requested) or not (0.0 < requested <= 1.0):
            raise ValueError("requested_encircled_energy must be finite and in (0, 1]")
        if retained + 5e-13 < requested:
            raise ValueError("retained_energy cannot be below requested_encircled_energy")
        if not isinstance(self.source_identity, str):
            raise TypeError("source_identity must be a string")
        if not isinstance(self.quadrature_rule, str) or not self.quadrature_rule:
            raise ValueError("quadrature_rule must be a nonempty string")

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sample_spacing_m", spacing)
        object.__setattr__(self, "retained_energy", min(retained, 1.0))
        object.__setattr__(self, "requested_encircled_energy", requested)

    @property
    def cell_area_m2(self) -> float:
        """Area represented by one numerical source-grid cell."""

        return self.sample_spacing_m[0] * self.sample_spacing_m[1]

    @property
    def energy(self) -> float:
        """Dimensionless energy of the renormalized discrete measure."""

        return float(self.values.sum(dtype=np.float64))

    @property
    def axis_y_m(self) -> FloatArray:
        """Centered physical y offsets of the kernel cells."""

        axis = (
            np.arange(self.values.shape[0], dtype=np.float64) - self.values.shape[0] // 2
        ) * self.sample_spacing_m[0]
        return _readonly_float_array(axis, ndim=1, name="axis_y_m")

    @property
    def axis_x_m(self) -> FloatArray:
        """Centered physical x offsets of the kernel cells."""

        axis = (
            np.arange(self.values.shape[1], dtype=np.float64) - self.values.shape[1] // 2
        ) * self.sample_spacing_m[1]
        return _readonly_float_array(axis, ndim=1, name="axis_x_m")


@dataclass(frozen=True, slots=True)
class CellAverageTransferKernel:
    """Exact collapsed transfer from source-cell to photosite cell averages.

    The kernel integrates the continuous PSF against the autocorrelation of
    the declared piecewise-constant source cell and the equal target
    photosite.  It therefore includes the target aperture exactly once while
    also accounting for the finite support of each source cell.  This is not
    the single-aperture ``PixelIntegratedKernel`` used for point-sample
    diagnostics.
    """

    values: FloatArray
    sample_spacing_m: tuple[float, float]
    retained_energy: float
    requested_encircled_energy: float
    source_identity: str = ""
    integration_rule: str = "piecewise_constant_source_x_photosite_tent_exact_v1"
    pixel_integrated: bool = field(default=True, init=False)
    source_cell_integrated: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        values = _readonly_float_array(self.values, ndim=2, name="values")
        if any(size % 2 != 1 for size in values.shape):
            raise ValueError("cell-average transfer kernel must be a centered odd 2-D array")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("cell-average transfer kernel must be finite and nonnegative")
        if not np.isclose(values.sum(dtype=np.float64), 1.0, rtol=5e-13, atol=5e-13):
            raise ValueError("cell-average transfer kernel weights must sum to one")
        spacing = _sample_spacing_pair(self.sample_spacing_m)
        retained = float(self.retained_energy)
        requested = float(self.requested_encircled_energy)
        if not math.isfinite(retained) or not (0.0 < retained <= 1.0 + 5e-13):
            raise ValueError("retained_energy must be finite and in (0, 1]")
        if not math.isfinite(requested) or not (0.0 < requested <= 1.0):
            raise ValueError("requested_encircled_energy must be finite and in (0, 1]")
        if retained + 5e-13 < requested:
            raise ValueError("retained_energy cannot be below requested_encircled_energy")
        if not isinstance(self.source_identity, str):
            raise TypeError("source_identity must be a string")
        if not isinstance(self.integration_rule, str) or not self.integration_rule:
            raise ValueError("integration_rule must be a nonempty string")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sample_spacing_m", spacing)
        object.__setattr__(self, "retained_energy", min(retained, 1.0))
        object.__setattr__(self, "requested_encircled_energy", requested)

    @property
    def cell_area_m2(self) -> float:
        """Area of the equal source and target cells represented here."""

        return self.sample_spacing_m[0] * self.sample_spacing_m[1]

    @property
    def energy(self) -> float:
        """Dimensionless energy of the renormalized transfer."""

        return float(self.values.sum(dtype=np.float64))

    @property
    def axis_y_m(self) -> FloatArray:
        """Centered physical y offsets of the transfer coefficients."""

        axis = (
            np.arange(self.values.shape[0], dtype=np.float64) - self.values.shape[0] // 2
        ) * self.sample_spacing_m[0]
        return _readonly_float_array(axis, ndim=1, name="axis_y_m")

    @property
    def axis_x_m(self) -> FloatArray:
        """Centered physical x offsets of the transfer coefficients."""

        axis = (
            np.arange(self.values.shape[1], dtype=np.float64) - self.values.shape[1] // 2
        ) * self.sample_spacing_m[1]
        return _readonly_float_array(axis, ndim=1, name="axis_x_m")


def _cell_overlap_matrix(
    source_axis_m: FloatArray,
    source_spacing_m: float,
    target_spacing_m: float,
) -> FloatArray:
    """Exact overlaps between finite PSF cells and centered numerical cells."""

    source_half_extent = source_axis_m.size * source_spacing_m / 2.0
    max_target_index = max(
        0,
        math.ceil(source_half_extent / target_spacing_m + 0.5) - 1,
    )
    target_centers = (
        np.arange(-max_target_index, max_target_index + 1, dtype=np.float64) * target_spacing_m
    )
    source_left = source_axis_m - source_spacing_m / 2.0
    source_right = source_axis_m + source_spacing_m / 2.0
    target_left = target_centers - target_spacing_m / 2.0
    target_right = target_centers + target_spacing_m / 2.0
    overlap = np.minimum(target_right[:, None], source_right[None, :]) - np.maximum(
        target_left[:, None], source_left[None, :]
    )
    return np.asarray(np.maximum(overlap, 0.0), dtype=np.float64)


def _crop_encircled_energy(
    mass: FloatArray,
    sample_spacing_m: tuple[float, float],
    threshold: float,
) -> tuple[FloatArray, float]:
    center_y = mass.shape[0] // 2
    center_x = mass.shape[1] // 2
    y_m = (np.arange(mass.shape[0], dtype=np.float64) - center_y) * sample_spacing_m[0]
    x_m = (np.arange(mass.shape[1], dtype=np.float64) - center_x) * sample_spacing_m[1]
    radius_m = np.hypot(y_m[:, None], x_m[None, :]).ravel()
    flattened_mass = mass.ravel()
    order = np.argsort(radius_m, kind="stable")
    cumulative = np.cumsum(flattened_mass[order], dtype=np.float64)
    target = threshold * float(flattened_mass.sum(dtype=np.float64))
    selected = min(
        int(np.searchsorted(cumulative, target, side="left")),
        order.size - 1,
    )
    support_radius_m = float(radius_m[order[selected]])
    half_height = min(center_y, int(math.ceil(support_radius_m / sample_spacing_m[0])))
    half_width = min(center_x, int(math.ceil(support_radius_m / sample_spacing_m[1])))
    cropped = np.array(
        mass[
            center_y - half_height : center_y + half_height + 1,
            center_x - half_width : center_x + half_width + 1,
        ],
        dtype=np.float64,
        copy=True,
    )
    retained = float(cropped.sum(dtype=np.float64))
    if not math.isfinite(retained) or retained <= 0.0:
        raise RuntimeError("encircled-energy crop produced invalid energy")
    cropped /= retained
    return cropped, retained


def sample_continuous_psf(
    psf: ContinuousPSF,
    sample_spacing_m: float | Iterable[float],
    *,
    encircled_energy: float = 1.0,
) -> PSFQuadratureKernel:
    """Project ``psf`` onto an oversampled numerical scene lattice.

    The PSF is represented as a piecewise-constant density on its native FFT
    cells.  Exact cell overlaps produce dimensionless finite-volume weights
    on the requested ``(dy, dx)`` lattice.  This operation is numerical
    quadrature only: it does not apply the sensor pixel response and the
    returned kernel must be followed by the forward path's one photosite-area
    integration.

    The pupil FFT must sample the PSF at least as finely as the requested
    scene lattice.  Refining the scene lattice therefore requires a matching
    refinement of the pupil/PSF sampling rather than interpolation that hides
    an under-resolved optical model.
    """

    if not isinstance(psf, ContinuousPSF):
        raise TypeError("psf must be a ContinuousPSF")
    spacing = _sample_spacing_pair(sample_spacing_m)
    if psf.sample_spacing_m > min(spacing) * (1.0 + 1e-12):
        raise ValueError(
            "continuous PSF must be sampled at least as finely as the numerical scene grid"
        )
    threshold = float(encircled_energy)
    if not math.isfinite(threshold) or not (0.0 < threshold <= 1.0):
        raise ValueError("encircled_energy must be finite and in (0, 1]")

    overlap_y = _cell_overlap_matrix(
        psf.axis_m,
        psf.sample_spacing_m,
        spacing[0],
    )
    overlap_x = _cell_overlap_matrix(
        psf.axis_m,
        psf.sample_spacing_m,
        spacing[1],
    )
    mass = np.asarray(overlap_y @ psf.density @ overlap_x.T, dtype=np.float64)
    mass[mass < 0.0] = 0.0
    represented_energy = float(mass.sum(dtype=np.float64))
    if not math.isfinite(represented_energy) or represented_energy <= 0.0:
        raise RuntimeError("PSF grid projection produced invalid energy")
    if not math.isclose(represented_energy, psf.energy, rel_tol=5e-12, abs_tol=5e-13):
        raise RuntimeError("PSF grid projection did not cover the continuous PSF support")
    mass /= represented_energy

    cropped, retained = _crop_encircled_energy(mass, spacing, threshold)
    return PSFQuadratureKernel(
        values=cropped,
        sample_spacing_m=spacing,
        retained_energy=retained,
        requested_encircled_energy=threshold,
        source_identity=psf.model_identity,
    )


def _tent_antiderivative(values: FloatArray, pitch_m: float) -> FloatArray:
    """Antiderivative of ``max(0, pitch - abs(x))`` with zero at -infinity."""

    x = np.asarray(values, dtype=np.float64)
    result = np.empty_like(x)
    below = x <= -pitch_m
    rising = (x > -pitch_m) & (x <= 0.0)
    falling = (x > 0.0) & (x < pitch_m)
    above = x >= pitch_m
    result[below] = 0.0
    result[rising] = 0.5 * np.square(x[rising] + pitch_m)
    result[falling] = pitch_m**2 - 0.5 * np.square(pitch_m - x[falling])
    result[above] = pitch_m**2
    return result


def _cell_average_tent_matrix(
    psf_axis_m: FloatArray,
    psf_spacing_m: float,
    cell_pitch_m: float,
) -> FloatArray:
    """Integrate equal-cell source/target overlap over each finite PSF cell."""

    support_half_extent = float(np.max(np.abs(psf_axis_m))) + 0.5 * psf_spacing_m
    maximum_offset = max(
        0,
        int(math.ceil((support_half_extent + cell_pitch_m) / cell_pitch_m) - 1),
    )
    centers = np.arange(-maximum_offset, maximum_offset + 1, dtype=np.float64) * cell_pitch_m
    left = psf_axis_m - 0.5 * psf_spacing_m
    right = psf_axis_m + 0.5 * psf_spacing_m
    integrated = _tent_antiderivative(
        right[None, :] - centers[:, None],
        cell_pitch_m,
    ) - _tent_antiderivative(
        left[None, :] - centers[:, None],
        cell_pitch_m,
    )
    integrated[integrated < 0.0] = 0.0
    return np.asarray(integrated, dtype=np.float64)


def collapse_cell_average_transfer(
    psf: ContinuousPSF,
    sample_spacing_m: float | Iterable[float],
    *,
    encircled_energy: float = 1.0,
) -> CellAverageTransferKernel:
    """Collapse exact equal-grid source reconstruction and target integration.

    For source-cell value :math:`s_j`, continuous PSF density :math:`h`, and
    equal source/target cell :math:`C`, the returned offset weight is

    ``1/area(C) * integral h(q) * area(C intersect (C + q - offset)) dq``.

    The overlap is a separable tent.  Integrating its analytic antiderivative
    over every finite PSF-density cell makes this equivalent to explicit
    piecewise-constant reconstruction, continuous convolution, and one target
    photosite average, without constructing an oversampled image per frame.
    """

    if not isinstance(psf, ContinuousPSF):
        raise TypeError("psf must be a ContinuousPSF")
    spacing = _sample_spacing_pair(sample_spacing_m)
    if psf.sample_spacing_m > min(spacing) * (1.0 + 1e-12):
        raise ValueError("continuous PSF must be sampled at least as finely as the cell grid")
    threshold = float(encircled_energy)
    if not math.isfinite(threshold) or not (0.0 < threshold <= 1.0):
        raise ValueError("encircled_energy must be finite and in (0, 1]")

    tent_y = _cell_average_tent_matrix(
        psf.axis_m,
        psf.sample_spacing_m,
        spacing[0],
    )
    tent_x = _cell_average_tent_matrix(
        psf.axis_m,
        psf.sample_spacing_m,
        spacing[1],
    )
    cell_area = spacing[0] * spacing[1]
    mass = np.asarray(tent_y @ psf.density @ tent_x.T / cell_area, dtype=np.float64)
    mass[mass < 0.0] = 0.0
    represented_energy = float(mass.sum(dtype=np.float64))
    if not math.isfinite(represented_energy) or represented_energy <= 0.0:
        raise RuntimeError("cell-average transfer integration produced invalid energy")
    if not math.isclose(represented_energy, psf.energy, rel_tol=5e-12, abs_tol=5e-13):
        raise RuntimeError("cell-average transfer did not cover the continuous PSF support")
    mass /= represented_energy
    cropped, retained = _crop_encircled_energy(mass, spacing, threshold)
    return CellAverageTransferKernel(
        values=cropped,
        sample_spacing_m=spacing,
        retained_energy=retained,
        requested_encircled_energy=threshold,
        source_identity=psf.model_identity,
    )


__all__ = [
    "CellAverageTransferKernel",
    "PSFQuadratureKernel",
    "collapse_cell_average_transfer",
    "sample_continuous_psf",
]
