"""Paper-facing numerical acceptance checks without artifact-bundle machinery."""

from __future__ import annotations

import math

import pytest

from phycam_eval.reference_profiles import (
    synthetic_coco_ldr_native_profile,
    synthetic_coco_ldr_native_replication_profile,
)
from phycam_eval.validation_evidence import ValidationCriteria, build_validation_evidence


def test_primary_and_replication_profiles_preserve_the_declared_sampling_change() -> None:
    primary = synthetic_coco_ldr_native_profile()
    replication = synthetic_coco_ldr_native_replication_profile()
    primary_sampling = (
        primary.optics.f_number
        * primary.optics.reference_wavelength_m
        / primary.sensor.pixel_pitch_m
    )
    replication_sampling = (
        replication.optics.f_number
        * replication.optics.reference_wavelength_m
        / replication.sensor.pixel_pitch_m
    )
    assert primary.sensor.sensor_shape_pixels == (640, 640)
    assert replication.sensor.sensor_shape_pixels == (640, 640)
    assert primary.fixed_parameters["source_adapter"] == "native_active_sensor_roi_v1"
    assert replication.fixed_parameters["source_adapter"] == "native_active_sensor_roi_v1"
    assert primary.optics.f_number == 4.0
    assert primary.sensor.pixel_pitch_m == 4e-6
    assert replication.optics.f_number == 5.6
    assert replication.sensor.pixel_pitch_m == 2.8e-6
    for profile in (primary, replication):
        assert profile.optics.reference_wavelength_m == 550e-9
        assert dict(profile.optics.channel_wavelengths_m) == {
            "R": 620e-9,
            "G": 540e-9,
            "B": 460e-9,
        }
        assert profile.optics.pupil_grid_size == 65
        assert profile.optics.pupil_q_max == 1.2
        assert profile.optics.pupil_fft_size == 257
        assert profile.optics.psf_energy_fraction == 0.995
        assert profile.optics.boundary_policy == "reflect"
    assert math.isclose(replication_sampling, 2.0 * primary_sampling, rel_tol=1e-15)


def test_validation_criteria_are_the_paper_acceptance_boundaries() -> None:
    criteria = ValidationCriteria()
    assert criteria.flux_absolute_error_max == 5e-12
    assert criteria.symmetry_relative_linf_max == 5e-12
    assert criteria.convergence_declared_refined_mtf_linf_max == 0.005
    assert criteria.convergence_declared_refined_mtf50_relative_error_max == 0.01
    assert criteria.implemented_match_mtf50_relative_error_max == 0.005
    assert criteria.independent_tent_quadrature_linf_max == 2e-8
    assert criteria.independent_equal_grid_formation_linf_max == 5e-13
    assert criteria.analytic_airy_center_relative_error_max == 1e-3
    assert criteria.analytic_airy_normalized_intensity_linf_max == 5e-4
    assert criteria.analytic_circular_aperture_mtf_linf_max == 1.2e-3


@pytest.mark.parametrize(
    "profile_factory",
    (synthetic_coco_ldr_native_profile, synthetic_coco_ldr_native_replication_profile),
)
def test_declared_profiles_pass_independent_optics_and_comparator_oracles(
    profile_factory,
) -> None:
    report = build_validation_evidence(
        profile_factory(),
        [0.0, 0.5],
        dct_axis_dimensions=(145, 640),
        frequency_sample_count=257,
    )
    assert report["summary"]["all_passed"] is True
    assert report["analytic_zero_defocus_validation"]["passed"] is True
    assert report["independent_equal_grid_formation_validation"]["passed"] is True
    assert all(
        severity["convergence"]["passed"]
        and all(
            channel["independent_cell_average_transfer_oracle"]["passed"]
            for channel in severity["channels"]
        )
        for severity in report["physical_validation"]["severities"]
    )
    assert all(record["passed"] for record in report["comparator_validation"]["records"])
