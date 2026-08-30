"""Direct equation and numerical checks for the Fourier-optics model."""

from __future__ import annotations

import numpy as np
import pytest

from phycam_eval.optics.convolution import convolve_spatially_invariant
from phycam_eval.optics.focus import (
    airy_first_zero_radius_m,
    ideal_defocus_on_axis_intensity_ratio,
    paraxial_circle_of_confusion_diameter_m,
    paraxial_edge_waves_from_conjugates,
    paraxial_edge_waves_from_image_plane_offset,
    paraxial_image_plane_offset_from_edge_waves,
    thin_lens_image_distance_m,
)
from phycam_eval.optics.psf import ContinuousPSF, airy_psf_density, psf_to_otf, pupil_to_psf
from phycam_eval.optics.pupil import (
    PupilSampling,
    complex_pupil,
    wavelength_scaled_defocus,
)
from phycam_eval.optics.sampling import collapse_cell_average_transfer, sample_continuous_psf

WAVELENGTH_M = 550e-9
F_NUMBER = 4.0


def _pupil_psf(
    sampling: PupilSampling,
    *,
    edge_waves: float = 0.0,
    wavelength_m: float = WAVELENGTH_M,
    f_number: float = F_NUMBER,
) -> ContinuousPSF:
    return pupil_to_psf(
        complex_pupil(sampling, edge_waves),
        sampling,
        wavelength_m,
        f_number,
        edge_waves=edge_waves,
    )


def _gaussian_psf() -> ContinuousPSF:
    spacing = 0.0625e-6
    axis = np.arange(-160, 161, dtype=np.float64) * spacing
    y_m, x_m = np.meshgrid(axis, axis, indexing="ij")
    density = np.exp(-(x_m * x_m + y_m * y_m) / (2.0 * (1.15e-6) ** 2))
    density /= density.sum(dtype=np.float64) * spacing**2
    return ContinuousPSF(
        density=density,
        axis_m=axis,
        sample_spacing_m=spacing,
        wavelength_m=WAVELENGTH_M,
        f_number=F_NUMBER,
        delta_q=0.01,
        edge_waves=0.0,
        model_identity="analytic-gaussian-fixture",
    )


def test_clear_pupil_matches_analytic_airy_psf_and_mtf() -> None:
    sampling = PupilSampling(base_size=257, q_max=1.2, fft_size=1025)
    psf = _pupil_psf(sampling)
    center = psf.density.shape[0] // 2
    expected_spacing = 2.0 * WAVELENGTH_M * F_NUMBER / (sampling.fft_size * sampling.delta_q)
    assert psf.sample_spacing_m == pytest.approx(expected_spacing, rel=1e-15)
    assert psf.energy == pytest.approx(1.0, rel=5e-13, abs=5e-13)
    assert np.min(psf.density) >= 0.0

    numerical = psf.density[center]
    analytic = airy_psf_density(np.abs(psf.axis_m), WAVELENGTH_M, F_NUMBER)
    main_lobes = np.abs(psf.axis_m) <= 3.0 * WAVELENGTH_M * F_NUMBER
    np.testing.assert_allclose(
        numerical[main_lobes] / numerical[center],
        analytic[main_lobes] / analytic[center],
        rtol=0.0,
        atol=2e-4,
    )
    assert numerical[center] == pytest.approx(analytic[center], rel=1.5e-3)

    otf = psf_to_otf(psf)
    assert otf.values[center, center] == 1.0 + 0.0j
    rho = np.abs(otf.frequency_axis_cpm) * WAVELENGTH_M * F_NUMBER
    expected_mtf = np.zeros_like(rho)
    in_band = rho <= 1.0
    value = rho[in_band]
    expected_mtf[in_band] = 2.0 / np.pi * (np.arccos(value) - value * np.sqrt(1.0 - value * value))
    np.testing.assert_allclose(
        otf.values[center, rho <= 0.9].real,
        expected_mtf[rho <= 0.9],
        rtol=0.0,
        atol=1.2e-3,
    )


def test_wavelength_phase_and_psf_scale_follow_the_declared_equations() -> None:
    reference_waves = 0.75
    assert wavelength_scaled_defocus(reference_waves, WAVELENGTH_M, 450e-9) == pytest.approx(
        reference_waves * WAVELENGTH_M / 450e-9
    )
    assert wavelength_scaled_defocus(reference_waves, WAVELENGTH_M, 650e-9) == pytest.approx(
        reference_waves * WAVELENGTH_M / 650e-9
    )

    sampling = PupilSampling(base_size=65, q_max=1.2, fft_size=257)
    pupil = complex_pupil(sampling, reference_waves)
    q_x, q_y = sampling.coordinates()
    radius_squared = q_x * q_x + q_y * q_y
    edge = np.unravel_index(
        np.argmin(np.where(radius_squared <= 1.0, np.abs(radius_squared - 1.0), np.inf)),
        pupil.shape,
    )
    center = sampling.base_size // 2
    observed_phase = np.angle(pupil[edge] / pupil[center, center])
    assert observed_phase == pytest.approx(np.angle(np.exp(2j * np.pi * reference_waves)), abs=3e-2)

    wavelength_a, f_number_a = 450e-9, 2.8
    wavelength_b, f_number_b = 650e-9, 5.6
    psf_a = _pupil_psf(sampling, edge_waves=0.6, wavelength_m=wavelength_a, f_number=f_number_a)
    psf_b = _pupil_psf(sampling, edge_waves=0.6, wavelength_m=wavelength_b, f_number=f_number_b)
    np.testing.assert_allclose(
        psf_a.axis_m / (wavelength_a * f_number_a),
        psf_b.axis_m / (wavelength_b * f_number_b),
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        psf_a.density * (wavelength_a * f_number_a) ** 2,
        psf_b.density * (wavelength_b * f_number_b) ** 2,
        rtol=2e-14,
        atol=2e-14,
    )


def test_focus_conversion_and_defocus_signature_match_analytic_results() -> None:
    object_distance = 2.0
    focal_length = 50e-3
    image_distance = thin_lens_image_distance_m(
        focal_length_m=focal_length,
        object_distance_m=object_distance,
    )
    pupil_radius = image_distance / (2.0 * F_NUMBER)
    assert paraxial_edge_waves_from_conjugates(
        object_distance_m=object_distance,
        image_distance_m=image_distance,
        focal_length_m=focal_length,
        pupil_radius_m=pupil_radius,
        wavelength_m=WAVELENGTH_M,
    ) == pytest.approx(0.0, abs=2e-13)
    small_offset = 1e-6
    exact_waves = paraxial_edge_waves_from_conjugates(
        object_distance_m=object_distance,
        image_distance_m=image_distance + small_offset,
        focal_length_m=focal_length,
        pupil_radius_m=pupil_radius,
        wavelength_m=WAVELENGTH_M,
    )
    small_offset_waves = paraxial_edge_waves_from_image_plane_offset(
        small_offset,
        f_number=F_NUMBER,
        wavelength_m=WAVELENGTH_M,
    )
    assert exact_waves == pytest.approx(
        small_offset_waves * image_distance / (image_distance + small_offset),
        rel=2e-10,
    )

    one_wave_offset = paraxial_image_plane_offset_from_edge_waves(
        1.0, f_number=F_NUMBER, wavelength_m=WAVELENGTH_M
    )
    assert one_wave_offset == pytest.approx(-70.4e-6)
    assert paraxial_circle_of_confusion_diameter_m(
        one_wave_offset, f_number=F_NUMBER
    ) == pytest.approx(17.6e-6)
    for waves in (-1.0, 0.0, 0.5, 1.0):
        offset = paraxial_image_plane_offset_from_edge_waves(
            waves, f_number=F_NUMBER, wavelength_m=WAVELENGTH_M
        )
        recovered = paraxial_edge_waves_from_image_plane_offset(
            offset, f_number=F_NUMBER, wavelength_m=WAVELENGTH_M
        )
        assert recovered == pytest.approx(waves, abs=2e-16)

    sampling = PupilSampling(257, 1.2, 1025)
    reference = _pupil_psf(sampling)
    center = reference.density.shape[0] // 2
    for waves in (0.0, 0.5, 1.0, 1.5):
        defocused = _pupil_psf(sampling, edge_waves=waves)
        observed = defocused.density[center, center] / reference.density[center, center]
        predicted = ideal_defocus_on_axis_intensity_ratio(waves)
        if waves == 1.0:
            assert observed < 2e-6
        else:
            assert observed == pytest.approx(predicted, abs=3e-4)

    positive = _pupil_psf(PupilSampling(65, 1.2, 257), edge_waves=1.25).density
    negative = _pupil_psf(PupilSampling(65, 1.2, 257), edge_waves=-1.25).density
    np.testing.assert_allclose(positive / positive.max(), negative / negative.max(), atol=2e-14)
    assert airy_first_zero_radius_m(f_number=F_NUMBER, wavelength_m=WAVELENGTH_M) == pytest.approx(
        1.2196698912665045 * WAVELENGTH_M * F_NUMBER
    )


def test_psf_sampling_converges_and_cell_average_matches_independent_quadrature() -> None:
    psf = _gaussian_psf()
    y_m, x_m = np.meshgrid(psf.axis_m, psf.axis_m, indexing="ij")
    reference_moment = float(
        np.sum(psf.density * (x_m * x_m + y_m * y_m), dtype=np.float64) * psf.sample_spacing_m**2
    )

    def moment(spacing: float) -> float:
        kernel = sample_continuous_psf(psf, spacing)
        radius_squared = kernel.axis_y_m[:, None] ** 2 + kernel.axis_x_m[None, :] ** 2
        return float(np.sum(kernel.values * radius_squared, dtype=np.float64))

    errors = [abs(moment(spacing) - reference_moment) for spacing in (1e-6, 0.5e-6, 0.25e-6)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] / reference_moment < 0.005

    pitch_m = 0.83e-6
    transfer = collapse_cell_average_transfer(psf, pitch_m)
    subdivisions = 16
    subcell = (
        (np.arange(subdivisions, dtype=np.float64) + 0.5) / subdivisions - 0.5
    ) * psf.sample_spacing_m
    nodes = psf.axis_m[:, None] + subcell[None, :]
    tent = np.maximum(pitch_m - np.abs(nodes[None, :, :] - transfer.axis_x_m[:, None, None]), 0.0)
    integrated = tent.sum(axis=2, dtype=np.float64) * (psf.sample_spacing_m / subdivisions)
    independent = integrated @ psf.density @ integrated.T / pitch_m**2
    independent /= independent.sum(dtype=np.float64)
    assert transfer.energy == pytest.approx(1.0, abs=5e-13)
    np.testing.assert_allclose(transfer.values, independent, rtol=0.0, atol=2e-6)


def test_fft_convolution_matches_direct_boundary_and_has_no_wraparound() -> None:
    image = np.random.default_rng(7).random((19, 23))
    kernel = np.array(
        [[0.01, 0.04, 0.01], [0.04, 0.80, 0.04], [0.01, 0.04, 0.01]],
        dtype=np.float64,
    )
    fft_result = convolve_spatially_invariant(image, kernel, boundary="reflect", method="fft")
    direct = convolve_spatially_invariant(image, kernel, boundary="reflect", method="direct")
    np.testing.assert_allclose(fft_result, direct, rtol=2e-13, atol=2e-13)

    impulse = np.zeros((21, 21), dtype=np.float64)
    impulse[0, 0] = 1.0
    blurred = convolve_spatially_invariant(impulse, kernel, boundary="zero", method="fft")
    assert np.max(np.abs(blurred[(-1,), :])) < 1e-14
    assert np.max(np.abs(blurred[:, (-1,)])) < 1e-14
    constant = np.full((11, 13), 0.37)
    np.testing.assert_allclose(
        convolve_spatially_invariant(constant, kernel, boundary="reflect"),
        constant,
        atol=3e-16,
    )
