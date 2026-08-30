"""Scientific checks for the three mechanism-matched comparator families."""

from __future__ import annotations

import numpy as np
import pytest

from phycam_eval.capture import LDRCaptureSeverity, build_ldr_pipeline
from phycam_eval.comparators import (
    GaussianComparatorConfig,
    QuadraticCosineComparatorConfig,
    SampledIncoherentComparatorConfig,
    apply_quadratic_cosine_comparator,
    apply_sampled_incoherent_comparator,
    gaussian_kernel,
    luminance_weighted_kernel,
    match_common_neutral_comparators,
    quadratic_cosine_response,
    sampled_incoherent_kernel,
)
from phycam_eval.eval.mtf import first_downward_crossing, kernel_axis_mtf
from phycam_eval.reference_profiles import synthetic_coco_ldr_native_profile


@pytest.mark.parametrize("waves", (0.5, 1.5, 3.0))
def test_all_comparator_families_match_executed_physical_mtf50(waves: float) -> None:
    matches = match_common_neutral_comparators(synthetic_coco_ldr_native_profile(), waves)
    assert [match.comparator_family for match in matches] == [
        "gaussian",
        "adapted_quadratic_cosine",
        "adapted_sampled_incoherent",
    ]
    assert len({match.target_mtf50_cycles_per_pixel for match in matches}) == 1
    assert all(match.relative_match_error < 0.005 for match in matches)


def test_gaussian_match_is_measured_on_the_finite_executed_kernel() -> None:
    profile = synthetic_coco_ldr_native_profile()
    match = match_common_neutral_comparators(profile, 0.5)[0]
    config = GaussianComparatorConfig(
        sigma_pixels=match.config["sigma_pixels"],
        boundary=match.config["boundary"],
        truncate_sigma=match.config["truncate_sigma"],
        implementation_id=match.config["implementation_id"],
    )
    _, neutral = build_ldr_pipeline(profile, LDRCaptureSeverity())
    neutral_mtf = kernel_axis_mtf(
        luminance_weighted_kernel(neutral, profile.isp.output_luminance_coefficients),
        sample_count=8193,
    )
    gaussian_mtf = kernel_axis_mtf(gaussian_kernel(config), sample_count=8193)
    achieved = first_downward_crossing(
        neutral_mtf.frequency,
        neutral_mtf.mtf * gaussian_mtf.mtf,
    )
    assert achieved == pytest.approx(match.target_mtf50_cycles_per_pixel, rel=1e-7)


def test_adapted_transfer_families_preserve_dc_and_expected_sign() -> None:
    image = np.random.default_rng(3).random((17, 23, 3))
    np.testing.assert_array_equal(
        apply_quadratic_cosine_comparator(image, QuadraticCosineComparatorConfig(0.0)),
        image,
    )
    np.testing.assert_array_equal(
        apply_sampled_incoherent_comparator(image, SampledIncoherentComparatorConfig(0.0)),
        image,
    )

    signed = QuadraticCosineComparatorConfig(20.0)
    response = quadratic_cosine_response((19, 25), signed)
    assert response[0, 0] == 1.0
    assert np.min(response) < 0.0
    constant = np.full((19, 25, 3), 0.37)
    np.testing.assert_allclose(
        apply_quadratic_cosine_comparator(constant, signed),
        constant,
        atol=3e-15,
    )

    incoherent = SampledIncoherentComparatorConfig(12.0, encircled_energy=0.999)
    kernel = sampled_incoherent_kernel(incoherent)
    assert np.all(kernel >= 0.0)
    assert float(kernel.sum()) == pytest.approx(1.0, abs=3e-15)
    np.testing.assert_allclose(
        apply_sampled_incoherent_comparator(constant, incoherent),
        constant,
        atol=2e-15,
    )
