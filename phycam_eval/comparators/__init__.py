"""Explicitly nonphysical comparators for mechanism-matched studies."""

from .capture import (
    LDRComparatorConfig,
    LDRComparatorResult,
    build_ldr_comparator_pipeline,
    comparator_config_from_dict,
    render_ldr_comparator,
)
from .gaussian import (
    GaussianComparatorConfig,
    GaussianMatchCriterion,
    MatchedGaussian,
    apply_gaussian_comparator,
    gaussian_kernel,
    gaussian_mtf,
    match_gaussian_to_kernel,
    sigma_for_encircled_energy,
    sigma_for_mtf50,
)
from .matching import (
    MechanismMatch,
    luminance_weighted_kernel,
    match_common_neutral_comparators,
    mechanism_match_grid,
)
from .transfer_families import (
    QuadraticCosineComparatorConfig,
    SampledIncoherentComparatorConfig,
    apply_quadratic_cosine_comparator,
    apply_sampled_incoherent_comparator,
    quadratic_cosine_response,
    sampled_incoherent_kernel,
    sampled_incoherent_retained_energy,
)

__all__ = [
    "GaussianComparatorConfig",
    "GaussianMatchCriterion",
    "LDRComparatorConfig",
    "LDRComparatorResult",
    "MatchedGaussian",
    "MechanismMatch",
    "QuadraticCosineComparatorConfig",
    "SampledIncoherentComparatorConfig",
    "apply_gaussian_comparator",
    "apply_quadratic_cosine_comparator",
    "apply_sampled_incoherent_comparator",
    "build_ldr_comparator_pipeline",
    "comparator_config_from_dict",
    "gaussian_kernel",
    "gaussian_mtf",
    "luminance_weighted_kernel",
    "match_common_neutral_comparators",
    "match_gaussian_to_kernel",
    "mechanism_match_grid",
    "quadratic_cosine_response",
    "render_ldr_comparator",
    "sampled_incoherent_kernel",
    "sampled_incoherent_retained_energy",
    "sigma_for_encircled_energy",
    "sigma_for_mtf50",
]
