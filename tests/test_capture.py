"""Small end-to-end checks for the public LDR and forward-RAW paths."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from phycam_eval.capture import LDRCaptureSeverity, render_ldr
from phycam_eval.color import srgb_encode
from phycam_eval.domains import Domain
from phycam_eval.forward_capture import ForwardCaptureCondition, render_forward
from phycam_eval.reference_profiles import (
    REPRESENTATIVE_RGB_ADAPTER_ID,
    synthetic_coco_ldr_native_profile,
    synthetic_forward_profile,
)
from phycam_eval.sensor.exposure import ExposurePolicy
from phycam_eval.source_grid import GridGeometry


def _forward_fixture():
    base = synthetic_forward_profile()
    fixed = dict(base.fixed_parameters)
    forward_source = dict(fixed["forward_source"])
    forward_source["minimum_samples_per_pixel"] = 2.0
    fixed["forward_source"] = forward_source
    profile = replace(
        base,
        optics=replace(
            base.optics,
            pupil_grid_size=33,
            pupil_fft_size=129,
            psf_energy_fraction=0.99,
        ),
        sensor=replace(
            base.sensor,
            reference_electron_budget_electrons=1024.0,
            full_well_capacity_electrons=4096.0,
            read_noise_rms_electrons=0.0,
            dark_current_electrons_per_second=0.0,
            quantum_efficiency={"R": 1.0, "G": 1.0, "B": 1.0},
            base_conversion_gain_dn_per_electron=1.0,
            black_level_dn=64.0,
            adc_bit_depth=14,
        ),
        fixed_parameters=fixed,
    )
    height, width = profile.sensor.sensor_shape_pixels
    geometry = GridGeometry.square_pixels(height * 2, width * 2, 2e-6)
    return profile, geometry


def test_native_ldr_path_preserves_neutral_and_applies_static_defocus() -> None:
    profile = synthetic_coco_ldr_native_profile()
    color = np.array([0.17, 0.43, 0.79])
    constant = np.broadcast_to(color, (23, 41, 3)).copy()
    neutral = render_ldr(constant, profile, image_id="neutral")

    assert neutral.output_frame.shape == constant.shape
    np.testing.assert_allclose(neutral.output_frame.array, constant, rtol=0.0, atol=3e-15)
    assert [stage.name for stage in neutral.pipeline.stages] == [
        "inverse_srgb",
        "source_grid",
        "optics_readout",
        "ldr_rgb_adapter",
        "pre_tone_gamut",
        "global_tone",
        "post_tone_gamut",
        "srgb_encode",
    ]
    contract = neutral.provenance["physical_contract"]
    assert contract["pixel_integration_count"] == 1
    assert contract["optical_representation"] == "exact_equal_grid_cell_average_transfer_v1"

    y, x = np.indices((23, 41), dtype=np.float64)
    scene = np.stack((x / 40.0, y / 22.0, ((3 * x + 5 * y) % 17) / 16.0), axis=-1)
    defocused = render_ldr(
        scene,
        profile,
        LDRCaptureSeverity(edge_waves_ref=1.0),
        image_id="defocused",
    )
    baseline = render_ldr(scene, profile, image_id="modeled-neutral")
    assert np.max(np.abs(defocused.output_frame.array - baseline.output_frame.array)) > 1e-3


def test_noiseless_forward_capture_matches_sensor_equations() -> None:
    profile, geometry = _forward_fixture()
    color = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    scene = np.broadcast_to(color, (geometry.height, geometry.width, 3)).copy()
    result = render_forward(
        scene,
        source_geometry=geometry,
        profile=profile,
        spectral_adapter_id=REPRESENTATIVE_RGB_ADAPTER_ID,
        condition=ForwardCaptureCondition(stochastic=False),
        image_id="constant",
    )

    np.testing.assert_allclose(
        result.frame_at(Domain.PHOTOSITE_EXPECTATION).array[:2, :2],
        [[256.0, 512.0], [512.0, 768.0]],
        atol=2e-12,
    )
    np.testing.assert_array_equal(
        result.frame_at(Domain.RAW_ADC_DN).array[:2, :2],
        [[320, 576], [576, 832]],
    )
    np.testing.assert_allclose(
        result.output_frame.array,
        np.broadcast_to(srgb_encode(color), result.output_frame.shape),
        atol=2e-15,
    )
    assert result.provenance["physical_contract"]["pixel_integration_count"] == 1


def test_forward_defocus_sign_symmetry_and_best_focus_difference() -> None:
    profile, geometry = _forward_fixture()
    scene = np.zeros((geometry.height, geometry.width, 3), dtype=np.float64)
    scene[geometry.height // 2, geometry.width // 2] = 1.0
    kwargs = {
        "source_geometry": geometry,
        "profile": profile,
        "spectral_adapter_id": REPRESENTATIVE_RGB_ADAPTER_ID,
    }
    best = render_forward(scene, condition=ForwardCaptureCondition(stochastic=False), **kwargs)
    positive = render_forward(
        scene,
        condition=ForwardCaptureCondition(edge_waves_ref=1.0, stochastic=False),
        **kwargs,
    )
    negative = render_forward(
        scene,
        condition=ForwardCaptureCondition(edge_waves_ref=-1.0, stochastic=False),
        **kwargs,
    )
    positive_expectation = positive.frame_at(Domain.PHOTOSITE_EXPECTATION).array
    negative_expectation = negative.frame_at(Domain.PHOTOSITE_EXPECTATION).array
    np.testing.assert_allclose(positive_expectation, negative_expectation, atol=2e-12)
    assert not np.allclose(
        best.frame_at(Domain.PHOTOSITE_EXPECTATION).array,
        positive_expectation,
    )


def test_shutter_time_changes_formation_exposure_and_dark_charge() -> None:
    profile, geometry = _forward_fixture()
    profile = replace(
        profile,
        sensor=replace(profile.sensor, dark_current_electrons_per_second=100.0),
    )
    scene = np.zeros((geometry.height, geometry.width, 3), dtype=np.float64)
    kwargs = {
        "source_geometry": geometry,
        "profile": profile,
        "spectral_adapter_id": REPRESENTATIVE_RGB_ADAPTER_ID,
    }
    fixed = render_forward(
        scene,
        condition=ForwardCaptureCondition(
            photon_loss_stops=2.0,
            exposure_policy=ExposurePolicy.FIXED_DURATION_ATTENUATION,
            stochastic=False,
        ),
        **kwargs,
    )
    shutter = render_forward(
        scene,
        condition=ForwardCaptureCondition(
            photon_loss_stops=2.0,
            exposure_policy=ExposurePolicy.SHUTTER_TIME,
            stochastic=False,
        ),
        **kwargs,
    )
    exposure = profile.readout.exposure_time_s
    np.testing.assert_allclose(fixed.frame_at(Domain.ELECTRONS).array, 100.0 * exposure)
    np.testing.assert_allclose(shutter.frame_at(Domain.ELECTRONS).array, 25.0 * exposure)
