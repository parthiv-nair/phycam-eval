"""Pre-detector mechanism matching for common-neutral comparator studies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .._canonical import canonical_sha256, freeze_json_value, json_value
from ..capture import LDRCaptureSeverity, build_ldr_pipeline
from ..eval.mtf import MTFCurve, first_downward_crossing, kernel_axis_mtf
from ..optics.defocus import DefocusModel
from ..optics.sampling import collapse_cell_average_transfer
from ..profiles import CameraProfile
from .gaussian import GaussianComparatorConfig, gaussian_kernel
from .transfer_families import (
    QuadraticCosineComparatorConfig,
    SampledIncoherentComparatorConfig,
    sampled_incoherent_kernel,
)

FloatArray = NDArray[np.float64]

_NATIVE_ACTIVE_SENSOR_ROI_SOURCE_ADAPTER = "native_active_sensor_roi_v1"


def _validate_match_profile_contract(profile: CameraProfile) -> None:
    """Fail closed unless the profile executes the matched transfer contract."""

    source_adapter = profile.fixed_parameters.get("source_adapter")
    if source_adapter != _NATIVE_ACTIVE_SENSOR_ROI_SOURCE_ADAPTER:
        raise ValueError(
            "common-neutral mechanism matching requires "
            f"source_adapter={_NATIVE_ACTIVE_SENSOR_ROI_SOURCE_ADAPTER!r}; "
            "matched-window profiles require an explicit source geometry and "
            "cannot use the equal-grid native-ROI match"
        )
    if profile.optics.boundary_policy != "reflect":
        raise ValueError(
            "common-neutral mechanism matching requires the profile's "
            "whole-sample reflect boundary because the quadratic DCT-I "
            "comparator has no zero- or constant-boundary implementation"
        )


def luminance_weighted_kernel(
    model: DefocusModel,
    luminance_weights: Sequence[float],
) -> FloatArray:
    """Collapse exact equal-grid RGB formation kernels under luminance weights.

    Each channel includes both the declared piecewise-constant native source
    cell and the target photosite average.  Using the same transfer as native
    LDR capture keeps pre-detector MTF matching tied to the executed operator,
    rather than to the older single-aperture point-sample diagnostic kernel.
    """

    if not isinstance(model, DefocusModel):
        raise TypeError("model must be a DefocusModel")
    weights = np.asarray(luminance_weights, dtype=np.float64)
    if weights.shape != (len(model.channels),) or not np.all(np.isfinite(weights)):
        raise ValueError("luminance_weights must contain one finite value per channel")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("luminance_weights must be nonnegative with positive sum")
    weights = weights / weights.sum(dtype=np.float64)
    channel_kernels = tuple(
        collapse_cell_average_transfer(
            channel.psf,
            model.config.pixel_pitch_m,
            encircled_energy=model.config.encircled_energy,
        ).values
        for channel in model.channels
    )
    size = max(values.shape[0] for values in channel_kernels)
    result = np.zeros((size, size), dtype=np.float64)
    for weight, values in zip(weights, channel_kernels, strict=True):
        before = (size - values.shape[0]) // 2
        after = size - values.shape[0] - before
        result += float(weight) * np.pad(values, ((before, after), (before, after)))
    result /= result.sum(dtype=np.float64)
    result.setflags(write=False)
    return result


def _curve(kernel: FloatArray, *, sample_count: int = 8193) -> MTFCurve:
    return kernel_axis_mtf(kernel, sample_count=sample_count)


def _interpolate(curve: MTFCurve, frequency: float) -> float:
    return float(np.interp(frequency, curve.frequency, curve.mtf))


def _kernel_axis_magnitude_at_frequency(kernel: FloatArray, frequency: float) -> float:
    """Evaluate one implemented discrete kernel exactly at an axis frequency."""

    values = np.asarray(kernel, dtype=np.float64)
    line_spread = values.sum(axis=0, dtype=np.float64)
    positions = np.arange(line_spread.size, dtype=np.float64) - (line_spread.size - 1) / 2.0
    response = np.exp(-2j * np.pi * frequency * positions) @ line_spread
    return float(abs(response) / line_spread.sum(dtype=np.float64))


def _achieved_curve(
    neutral: MTFCurve,
    response: FloatArray,
    *,
    method: str,
) -> MTFCurve:
    values = neutral.mtf * np.abs(np.asarray(response, dtype=np.float64))
    values /= values[0]
    return MTFCurve(neutral.frequency, values, "cycles/pixel", method)


@dataclass(frozen=True, slots=True)
class MechanismMatch:
    """One comparator fixed against a physical luminance-MTF50 target."""

    comparator_family: str
    target_edge_waves_ref: float
    target_mtf50_cycles_per_pixel: float
    neutral_mtf_at_target: float
    config: Mapping[str, Any]
    config_sha256: str
    achieved_mtf50_cycles_per_pixel: float
    relative_match_error: float
    camera_profile_sha256: str
    neutral_model_sha256: str
    target_model_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.comparator_family, str) or not self.comparator_family:
            raise ValueError("comparator_family must be nonempty")
        for name in (
            "target_edge_waves_ref",
            "target_mtf50_cycles_per_pixel",
            "neutral_mtf_at_target",
            "achieved_mtf50_cycles_per_pixel",
            "relative_match_error",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        config = json_value(self.config)
        if canonical_sha256(config) != self.config_sha256:
            raise ValueError("config_sha256 does not match config")
        frozen = freeze_json_value(config)
        if not isinstance(frozen, Mapping):
            raise TypeError("config must be a mapping")
        object.__setattr__(self, "config", frozen)
        for name in (
            "camera_profile_sha256",
            "neutral_model_sha256",
            "target_model_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        response_match_scope = (
            "continuous_design_curve_exact_at_shape_dependent_dct_i_eigenfrequencies"
            if self.comparator_family == "adapted_quadratic_cosine"
            else "exact_dtft_of_executed_finite_shift_invariant_kernel"
        )
        payload = {
            "schema_version": 4,
            "record_type": "mechanism_mtf50_match",
            "matching_contract": {
                "criterion": "rec709_luminance_weighted_first_downward_mtf50",
                "frequency_unit": "cycles/pixel",
                "common_neutral_branch": "physical_W0_linear_rgb_optical",
                "comparator_insertion": "after_common_neutral_before_gamut_tone_encode",
                "physical_formation_operator": (
                    "exact_equal_grid_piecewise_constant_source_x_photosite_transfer_v1"
                ),
                "boundary_harmonization": ("profile_reflect_whole_sample_spatial_or_dct_i_v2"),
                "parameter_match_scope": (
                    "finite_kernel_dtft_or_continuous_dct_i_eigenvalue_design_v1"
                ),
                "dct_i_finite_shape_acceptance": (
                    "separate_complete_publication_axis_dimension_envelope_required"
                ),
            },
            "comparator_family": self.comparator_family,
            "target_edge_waves_ref": self.target_edge_waves_ref,
            "target_mtf50_cycles_per_pixel": self.target_mtf50_cycles_per_pixel,
            "neutral_mtf_at_target": self.neutral_mtf_at_target,
            "config": json_value(self.config),
            "config_sha256": self.config_sha256,
            "achieved_mtf50_cycles_per_pixel": self.achieved_mtf50_cycles_per_pixel,
            "relative_match_error": self.relative_match_error,
            "response_match_scope": response_match_scope,
            "camera_profile_sha256": self.camera_profile_sha256,
            "neutral_model_sha256": self.neutral_model_sha256,
            "target_model_sha256": self.target_model_sha256,
        }
        return {**payload, "match_sha256": canonical_sha256(payload)}

    @property
    def match_sha256(self) -> str:
        return self.to_dict()["match_sha256"]


def _record(
    *,
    family: str,
    waves: float,
    target_frequency: float,
    neutral_at_target: float,
    config: Mapping[str, Any],
    achieved: float,
    profile: CameraProfile,
    neutral_model: DefocusModel,
    target_model: DefocusModel,
) -> MechanismMatch:
    return MechanismMatch(
        comparator_family=family,
        target_edge_waves_ref=waves,
        target_mtf50_cycles_per_pixel=target_frequency,
        neutral_mtf_at_target=neutral_at_target,
        config=config,
        config_sha256=canonical_sha256(config),
        achieved_mtf50_cycles_per_pixel=achieved,
        relative_match_error=abs(achieved - target_frequency) / target_frequency,
        camera_profile_sha256=profile.profile_hash,
        neutral_model_sha256=neutral_model.cache_key,
        target_model_sha256=target_model.cache_key,
    )


def match_common_neutral_comparators(
    profile: CameraProfile,
    edge_waves_ref: float,
    *,
    relative_tolerance: float = 0.005,
    sampled_grid_size: int = 257,
    sampled_encircled_energy: float = 0.999,
) -> tuple[MechanismMatch, MechanismMatch, MechanismMatch]:
    """Match Gaussian and two adapted transfer families to physical MTF50."""

    if not isinstance(profile, CameraProfile):
        raise TypeError("profile must be a CameraProfile")
    waves = float(edge_waves_ref)
    tolerance = float(relative_tolerance)
    if not math.isfinite(waves) or waves <= 0.0:
        raise ValueError("edge_waves_ref must be finite and positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    _validate_match_profile_contract(profile)
    _, neutral_model = build_ldr_pipeline(profile, LDRCaptureSeverity())
    _, target_model = build_ldr_pipeline(
        profile,
        LDRCaptureSeverity(edge_waves_ref=waves),
    )
    weights = profile.isp.output_luminance_coefficients
    neutral_curve = _curve(luminance_weighted_kernel(neutral_model, weights))
    target_curve = _curve(luminance_weighted_kernel(target_model, weights))
    target_frequency = first_downward_crossing(target_curve.frequency, target_curve.mtf)
    if not math.isfinite(target_frequency):
        raise ValueError("physical target MTF50 is right-censored at image Nyquist")
    neutral_at_target = _interpolate(neutral_curve, target_frequency)
    required_response = 0.5 / neutral_at_target
    if not 0.0 < required_response < 1.0:
        raise ValueError("target is not below the common-neutral MTF at its MTF50")

    # Match the finite sampled kernel that is actually executed, rather than
    # its continuous-Gaussian idealization.  The distinction is material for
    # sub-pixel sigma: at W=0.5 for the declared native profile, fitting the
    # continuous formula misses the implemented MTF50 by roughly 9.5%.
    def gaussian_residual(candidate: float) -> float:
        config = GaussianComparatorConfig(candidate, profile.optics.boundary_policy)
        response = _kernel_axis_magnitude_at_frequency(
            gaussian_kernel(config),
            target_frequency,
        )
        return neutral_at_target * response - 0.5

    gaussian_lower = 0.0
    gaussian_lower_value = gaussian_residual(gaussian_lower)
    gaussian_upper = 0.25
    gaussian_upper_value = gaussian_residual(gaussian_upper)
    while gaussian_upper_value > 0.0 and gaussian_upper < 512.0:
        gaussian_lower, gaussian_lower_value = gaussian_upper, gaussian_upper_value
        gaussian_upper *= 2.0
        gaussian_upper_value = gaussian_residual(gaussian_upper)
    if gaussian_lower_value <= 0.0 or gaussian_upper_value > 0.0:
        raise ValueError("finite sampled Gaussian match has no first bracket below sigma=512")
    sigma = float(
        brentq(
            gaussian_residual,
            gaussian_lower,
            gaussian_upper,
            xtol=1e-12,
            rtol=1e-13,
        )
    )
    gaussian = GaussianComparatorConfig(sigma, profile.optics.boundary_policy)
    gaussian_curve = _curve(gaussian_kernel(gaussian))
    gaussian_response = np.interp(
        neutral_curve.frequency,
        gaussian_curve.frequency,
        gaussian_curve.mtf,
    )
    gaussian_achieved = first_downward_crossing(
        neutral_curve.frequency,
        _achieved_curve(neutral_curve, gaussian_response, method="neutral-times-gaussian").mtf,
    )

    alpha = math.acos(required_response) / (2.0 * target_frequency**2)
    quadratic = QuadraticCosineComparatorConfig(alpha)
    quadratic_response = np.cos(2.0 * alpha * np.square(neutral_curve.frequency))
    quadratic_achieved = first_downward_crossing(
        neutral_curve.frequency,
        _achieved_curve(
            neutral_curve,
            quadratic_response,
            method="neutral-times-adapted-quadratic-cosine",
        ).mtf,
    )

    def sampled_residual(candidate: float) -> float:
        config = SampledIncoherentComparatorConfig(
            candidate,
            boundary=profile.optics.boundary_policy,
            pupil_grid_size=sampled_grid_size,
            encircled_energy=sampled_encircled_energy,
        )
        comparator_curve = _curve(sampled_incoherent_kernel(config))
        return neutral_at_target * _interpolate(comparator_curve, target_frequency) - 0.5

    lower = 0.0
    lower_value = sampled_residual(lower)
    upper = 0.25
    upper_value = sampled_residual(upper)
    while upper_value > 0.0 and upper < 512.0:
        lower, lower_value = upper, upper_value
        upper += 0.25
        upper_value = sampled_residual(upper)
    if lower_value <= 0.0 or upper_value > 0.0:
        raise ValueError("sampled-incoherent match has no first monotone bracket below alpha=512")
    sampled_alpha = float(brentq(sampled_residual, lower, upper, xtol=1e-12, rtol=1e-13))
    sampled = SampledIncoherentComparatorConfig(
        sampled_alpha,
        boundary=profile.optics.boundary_policy,
        pupil_grid_size=sampled_grid_size,
        encircled_energy=sampled_encircled_energy,
    )
    sampled_curve = _curve(sampled_incoherent_kernel(sampled))
    sampled_response = np.interp(
        neutral_curve.frequency,
        sampled_curve.frequency,
        sampled_curve.mtf,
    )
    sampled_achieved = first_downward_crossing(
        neutral_curve.frequency,
        _achieved_curve(
            neutral_curve,
            sampled_response,
            method="neutral-times-adapted-sampled-incoherent",
        ).mtf,
    )

    matches = (
        _record(
            family="gaussian",
            waves=waves,
            target_frequency=target_frequency,
            neutral_at_target=neutral_at_target,
            config=gaussian.to_dict(),
            achieved=gaussian_achieved,
            profile=profile,
            neutral_model=neutral_model,
            target_model=target_model,
        ),
        _record(
            family="adapted_quadratic_cosine",
            waves=waves,
            target_frequency=target_frequency,
            neutral_at_target=neutral_at_target,
            config=quadratic.to_dict(),
            achieved=quadratic_achieved,
            profile=profile,
            neutral_model=neutral_model,
            target_model=target_model,
        ),
        _record(
            family="adapted_sampled_incoherent",
            waves=waves,
            target_frequency=target_frequency,
            neutral_at_target=neutral_at_target,
            config=sampled.to_dict(),
            achieved=sampled_achieved,
            profile=profile,
            neutral_model=neutral_model,
            target_model=target_model,
        ),
    )
    failures = [value for value in matches if value.relative_match_error > tolerance]
    if failures:
        detail = ", ".join(
            f"{value.comparator_family}={value.relative_match_error:.3%}" for value in failures
        )
        raise RuntimeError(f"mechanism match exceeds relative tolerance: {detail}")
    return matches


def mechanism_match_grid(
    profile: CameraProfile,
    edge_waves: Sequence[float],
    **kwargs: Any,
) -> tuple[MechanismMatch, ...]:
    """Build an ordered, deterministic match grid for declared severities."""

    if isinstance(edge_waves, (str, bytes)) or not isinstance(edge_waves, Sequence):
        raise TypeError("edge_waves must be an ordered sequence")
    values = tuple(float(value) for value in edge_waves)
    if not values or len(values) != len(set(values)):
        raise ValueError("edge_waves must be nonempty and unique")
    return tuple(
        match
        for value in values
        for match in match_common_neutral_comparators(profile, value, **kwargs)
    )


__all__ = [
    "MechanismMatch",
    "luminance_weighted_kernel",
    "match_common_neutral_comparators",
    "mechanism_match_grid",
]
