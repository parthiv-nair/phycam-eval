"""Disclosed synthetic benchmark profiles for examples and numerical checks.

These are numerical conventions, not measurements of a named camera.  Their
calibration strings say so explicitly; hardware claims require replacing them
with a measured profile while retaining the same immutable schema.
"""

from __future__ import annotations

from .domains import ColorSpace, DataMode
from .profiles import (
    CameraProfile,
    ISPProfile,
    OpticsProfile,
    ReadoutProfile,
    SensorProfile,
)

SYNTHETIC_CALIBRATION_REFERENCE = (
    "phycam synthetic benchmark convention v1; not hardware calibrated"
)
REPRESENTATIVE_RGB_ADAPTER_ID = "phycam-representative-rgb-wavelengths-v1"


def _optics() -> OpticsProfile:
    return OpticsProfile(
        f_number=4.0,
        reference_wavelength_m=550e-9,
        channel_wavelengths_m={"R": 620e-9, "G": 540e-9, "B": 460e-9},
        pupil_grid_size=65,
        pupil_q_max=1.2,
        pupil_fft_size=257,
        psf_energy_fraction=0.995,
        boundary_policy="reflect",
    )


def _sensor(*, raw: bool) -> SensorProfile:
    return SensorProfile(
        sensor_shape_pixels=(8, 10),
        pixel_pitch_m=4e-6,
        reference_electron_budget_electrons=4096.0,
        full_well_capacity_electrons=16_000.0,
        read_noise_rms_electrons=2.2,
        dark_current_electrons_per_second=0.15,
        quantum_efficiency={"R": 0.48, "G": 0.60, "B": 0.42},
        base_conversion_gain_dn_per_electron=0.25,
        analog_gain=1.0,
        digital_gain=1.0,
        black_level_dn=256.0,
        adc_bit_depth=14,
        cfa_pattern="RGGB" if raw else None,
    )


def _readout() -> ReadoutProfile:
    height = 8
    line_time_s = 10e-6
    exposure_s = 5e-3
    reference_s = 0.5 * (height - 1) * line_time_s + 0.5 * exposure_s
    return ReadoutProfile(
        frame_start_time_s=0.0,
        line_time_s=line_time_s,
        exposure_time_s=exposure_s,
        reference_time_s=reference_s,
        annotation_time_s=reference_s,
        quadrature_order=3,
    )


def _isp() -> ISPProfile:
    return ISPProfile(
        white_balance_gains=(1.0, 1.0, 1.0),
        camera_to_output_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        output_luminance_coefficients=(0.2126, 0.7152, 0.0722),
        pre_tone_gamut_policy="clip_negative",
        tone_pivot=0.18,
        post_tone_gamut_policy="clip_unit_range",
        output_color_space=ColorSpace.LINEAR_SRGB,
    )


def synthetic_ldr_profile() -> CameraProfile:
    """Return the fixed LDR re-degradation benchmark convention."""

    return CameraProfile(
        name="PhyCam synthetic LDR benchmark v1",
        data_mode=DataMode.LDR_REDEGRADATION,
        optics=_optics(),
        sensor=_sensor(raw=False),
        readout=_readout(),
        isp=_isp(),
        calibration_reference=SYNTHETIC_CALIBRATION_REFERENCE,
        fixed_parameters={
            "source_adapter": "matched_sensor_window_cell_average_v1",
            "camera_intrinsics_px": {
                "fx_px": 15.0,
                "fy_px": 15.0,
                "cx_px": 4.5,
                "cy_px": 3.5,
                "skew_px": 0.0,
            },
        },
    )


def synthetic_coco_ldr_pilot_profile() -> CameraProfile:
    """Return the disclosed 120-by-160 native-COCO LDR pilot convention.

    This is still a synthetic numerical profile, not a measured camera. Its
    lower sensor resolution keeps the authoritative Python formation path
    usable for a small detector pilot while retaining native-first decoding.
    """

    height, width = 120, 160
    line_time_s = 10e-6
    exposure_s = 5e-3
    reference_s = 0.5 * (height - 1) * line_time_s + 0.5 * exposure_s
    return CameraProfile(
        name="PhyCam synthetic native-COCO LDR pilot 120x160 v1",
        data_mode=DataMode.LDR_REDEGRADATION,
        optics=_optics(),
        sensor=SensorProfile(
            **{
                **_sensor(raw=False).to_dict(),
                "sensor_shape_pixels": (height, width),
            }
        ),
        readout=ReadoutProfile(
            frame_start_time_s=0.0,
            line_time_s=line_time_s,
            exposure_time_s=exposure_s,
            reference_time_s=reference_s,
            annotation_time_s=reference_s,
            quadrature_order=3,
        ),
        isp=_isp(),
        calibration_reference=SYNTHETIC_CALIBRATION_REFERENCE,
        fixed_parameters={
            "source_adapter": "matched_sensor_window_cell_average_v1",
            "camera_intrinsics_px": {
                "fx_px": 200.0,
                "fy_px": 200.0,
                "cx_px": 79.5,
                "cy_px": 59.5,
                "skew_px": 0.0,
            },
            "intended_use": "small native-COCO detector pilot; not hardware evidence",
        },
    )


def synthetic_coco_ldr_native_profile() -> CameraProfile:
    """Return the native-aspect 640-square active-ROI COCO convention.

    COCO val2017 stores every image within a 640-by-640 envelope. Each image is
    treated as an active sensor ROI with one stored cell per 4-micrometre
    photosite, preserving its stored aspect ratio and avoiding a camera-stage
    resize. This remains a synthetic re-degradation convention, not a measured
    camera profile.
    """

    height = width = 640
    exposure_s = 5e-3
    return CameraProfile(
        name="PhyCam synthetic native-aspect COCO LDR active-ROI v1",
        data_mode=DataMode.LDR_REDEGRADATION,
        optics=_optics(),
        sensor=SensorProfile(
            **{
                **_sensor(raw=False).to_dict(),
                "sensor_shape_pixels": (height, width),
            }
        ),
        readout=ReadoutProfile(
            frame_start_time_s=0.0,
            line_time_s=0.0,
            exposure_time_s=exposure_s,
            reference_time_s=0.5 * exposure_s,
            annotation_time_s=0.5 * exposure_s,
            quadrature_order=3,
        ),
        isp=_isp(),
        calibration_reference=SYNTHETIC_CALIBRATION_REFERENCE,
        fixed_parameters={
            "source_adapter": "native_active_sensor_roi_v1",
            "active_sensor_roi": {
                "maximum_shape_pixels": [height, width],
                "origin_policy": "upper_left",
                "stored_sample_to_photosite": "one_to_one",
            },
            "intended_use": (
                "full native-aspect COCO synthetic LDR optics benchmark; not hardware evidence"
            ),
        },
    )


def synthetic_coco_ldr_native_replication_profile() -> CameraProfile:
    """Return the stopped-down, fine-pitch native-COCO replication profile.

    This controlled synthetic replication doubles the dimensionless optical
    sampling ratio ``f_number * reference_wavelength / pixel_pitch`` relative
    to :func:`synthetic_coco_ldr_native_profile`: F/5.6 and 2.8-micrometre
    photosites replace F/4 and 4-micrometre photosites.  The resulting Airy
    first-zero diameter is approximately 2.68 pixels instead of 1.34 pixels,
    moving the reference optics from a marginally sampled to a resolved
    diffraction regime.  All other optical, sensor-response, readout, and ISP
    conventions are held fixed.  This is not a measured camera profile.
    """

    height = width = 640
    exposure_s = 5e-3
    optics = OpticsProfile(
        **{
            **_optics().to_dict(),
            "f_number": 5.6,
        }
    )
    sensor = SensorProfile(
        **{
            **_sensor(raw=False).to_dict(),
            "sensor_shape_pixels": (height, width),
            "pixel_pitch_m": 2.8e-6,
        }
    )
    return CameraProfile(
        name="PhyCam synthetic native-aspect COCO LDR fine-pitch replication v1",
        data_mode=DataMode.LDR_REDEGRADATION,
        optics=optics,
        sensor=sensor,
        readout=ReadoutProfile(
            frame_start_time_s=0.0,
            line_time_s=0.0,
            exposure_time_s=exposure_s,
            reference_time_s=0.5 * exposure_s,
            annotation_time_s=0.5 * exposure_s,
            quadrature_order=3,
        ),
        isp=_isp(),
        calibration_reference=SYNTHETIC_CALIBRATION_REFERENCE,
        fixed_parameters={
            "source_adapter": "native_active_sensor_roi_v1",
            "active_sensor_roi": {
                "maximum_shape_pixels": [height, width],
                "origin_policy": "upper_left",
                "stored_sample_to_photosite": "one_to_one",
            },
            "controlled_replication": {
                "reference_profile": "synthetic_coco_ldr_native_profile",
                "varied_parameters": ["optics.f_number", "sensor.pixel_pitch_m"],
                "held_fixed": ("all other optical, sensor-response, readout, and ISP conventions"),
                "optical_sampling_ratio_multiplier": 2.0,
            },
            "intended_use": (
                "full native-aspect COCO synthetic LDR optical-sampling replication; "
                "not hardware evidence"
            ),
        },
    )


def synthetic_forward_profile() -> CameraProfile:
    """Return the fixed oversampled RAW forward benchmark convention."""

    optics = OpticsProfile(
        **{
            **_optics().to_dict(),
            "pupil_fft_size": 1025,
        }
    )

    return CameraProfile(
        name="PhyCam synthetic forward RAW benchmark v1",
        data_mode=DataMode.FORWARD_CAMERA_VALIDATION,
        optics=optics,
        sensor=_sensor(raw=True),
        readout=_readout(),
        isp=_isp(),
        calibration_reference=SYNTHETIC_CALIBRATION_REFERENCE,
        fixed_parameters={
            "forward_source": {
                "grid_policy": "matched_sensor_window_cell_average_v1",
                "spectral_adapter_id": REPRESENTATIVE_RGB_ADAPTER_ID,
                "minimum_samples_per_pixel": 16.0,
                "sampling_validation": "smooth_cosine_direct_psf_fourier_oracle_v2",
                "sampling_validation_max_abs_error": 2.5e-3,
                "sampling_validation_certification": (
                    "candidate_error_plus_oracle_refinement_delta"
                ),
                "sampling_validation_oracle_pupil_grid_size": 129,
                "sampling_validation_oracle_fft_sizes": [513, 1025],
                "sampling_validation_max_oracle_refinement_delta": 2.0e-4,
                "sampling_validation_rejected_samples_per_pixel": 15,
            },
            "camera_intrinsics_px": {
                "fx_px": 15.0,
                "fy_px": 15.0,
                "cx_px": 4.5,
                "cy_px": 3.5,
                "skew_px": 0.0,
            },
        },
    )


def synthetic_forward_source_geometry():
    """Return the independently gated sixteen-samples-per-pixel-axis lattice."""

    from .source_grid import GridGeometry

    return GridGeometry.square_pixels(128, 160, 4e-6 / 16.0)


__all__ = [
    "REPRESENTATIVE_RGB_ADAPTER_ID",
    "SYNTHETIC_CALIBRATION_REFERENCE",
    "synthetic_coco_ldr_native_profile",
    "synthetic_coco_ldr_native_replication_profile",
    "synthetic_coco_ldr_pilot_profile",
    "synthetic_forward_profile",
    "synthetic_forward_source_geometry",
    "synthetic_ldr_profile",
]
