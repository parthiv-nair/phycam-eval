"""End-to-end rendered-camera builders for declared LDR re-degradation.

This path is intentionally explicit about its approximation: display RGB is
decoded into a linear-light proxy and interpreted as a piecewise-constant
source reconstruction over the fixed synthetic sensor window.  Continuous
optics and optional motion are evaluated before target photosite area is
integrated exactly once.  Bayer mosaicing and sensor noise remain bypassed;
the quantitative RAW path lives in :mod:`phycam_eval.forward_capture`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike

from ._canonical import canonical_sha256, freeze_json_value
from .boundary import BoundaryContract
from .color import SRGB_REC709_D65, OutputColorimetry, srgb_decode
from .domains import ColorSpace, DataMode, Domain
from .formation import render_joint_photosite_exposure
from .frame import Frame, FrameMetadata
from .isp.encode import srgb_encode
from .isp.gamut import apply_post_tone_gamut, apply_pre_tone_gamut
from .isp.tone import apply_global_tone
from .optics.defocus import DefocusConfig, DefocusModel, build_defocus_model
from .optics.pupil import PupilSampling
from .optics.sampling import collapse_cell_average_transfer, sample_continuous_psf
from .pipeline import CameraPipeline, StageSpec
from .profiles import CameraProfile
from .readout.motion import CameraIntrinsics, ConstantAngularVelocity
from .readout.timing import ReadoutTiming
from .source_grid import GridGeometry

_LDR_MATCHED_WINDOW_SOURCE_ADAPTER = "matched_sensor_window_cell_average_v1"
_LDR_NATIVE_ROI_SOURCE_ADAPTER = "native_active_sensor_roi_v1"
_LDR_SOURCE_ADAPTERS = frozenset(
    {_LDR_MATCHED_WINDOW_SOURCE_ADAPTER, _LDR_NATIVE_ROI_SOURCE_ADAPTER}
)


@dataclass(frozen=True, slots=True)
class LDRCaptureSeverity:
    """Controlled coordinates for a rendered LDR capture condition."""

    edge_waves_ref: float = 0.0
    angular_velocity_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tone_stop_ratio: float = 1.0

    def __post_init__(self) -> None:
        edge = float(self.edge_waves_ref)
        omega = tuple(float(value) for value in self.angular_velocity_rad_s)
        rho = float(self.tone_stop_ratio)
        if not np.isfinite(edge):
            raise ValueError("edge_waves_ref must be finite")
        if len(omega) != 3 or not all(np.isfinite(value) for value in omega):
            raise ValueError("angular_velocity_rad_s must contain three finite values")
        if not np.isfinite(rho) or not (0.0 < rho <= 1.0):
            raise ValueError("tone_stop_ratio must be in (0, 1]")
        object.__setattr__(self, "edge_waves_ref", edge)
        object.__setattr__(self, "angular_velocity_rad_s", omega)
        object.__setattr__(self, "tone_stop_ratio", rho)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_waves_ref": self.edge_waves_ref,
            "angular_velocity_rad_s": list(self.angular_velocity_rad_s),
            "tone_stop_ratio": self.tone_stop_ratio,
        }

    @property
    def condition_hash(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class LDRCaptureResult:
    input_frame: Frame
    output_frame: Frame
    pipeline: CameraPipeline
    severity: LDRCaptureSeverity
    defocus_model: DefocusModel
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        frozen = freeze_json_value(self.provenance)
        if not isinstance(frozen, Mapping):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "provenance", frozen)


def make_ldr_input_frame(image_srgb: ArrayLike, *, image_id: str = "unidentified") -> Frame:
    values = np.asarray(image_srgb)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("image_srgb must have shape (H, W, 3)")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("image_srgb must use a floating-point dtype")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("image_srgb must be nonempty and finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("image_srgb values must lie in [0, 1]")
    metadata = FrameMetadata(
        units="srgb_code_value",
        data_mode=DataMode.LDR_REDEGRADATION,
        channel_names=("R", "G", "B"),
        attributes={"image_id": str(image_id), "input_interpretation": "display_srgb"},
    )
    return Frame(values, Domain.DISPLAY_RGB, ColorSpace.SRGB, metadata)


def _metadata(
    frame: Frame,
    *,
    units: str,
    sample_spacing_m: tuple[float, float] | None = None,
    sensor_window_m: tuple[float, float, float, float] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> FrameMetadata:
    payload = frame.metadata.to_dict()
    payload["units"] = units
    if sample_spacing_m is not None:
        payload["sample_spacing_m"] = list(sample_spacing_m)
        payload["sensor_origin_m"] = [0.0, 0.0]
    if sensor_window_m is not None:
        payload["sensor_window_m"] = list(sensor_window_m)
    merged = dict(payload.get("attributes", {}))
    if attributes:
        merged.update(attributes)
    payload["attributes"] = merged
    return FrameMetadata.from_dict(payload)


def _stage(
    *,
    name: str,
    source: Domain,
    target: Domain,
    input_units: str,
    output_units: str,
    implementation_id: str,
    neutral: str,
    operation,
) -> StageSpec:
    return StageSpec(
        name=name,
        input_domain=source,
        output_domain=target,
        input_units=input_units,
        output_units=output_units,
        deterministic=True,
        implementation_id=implementation_id,
        neutral_condition=neutral,
        operation=operation,
    )


def _defocus(
    profile: CameraProfile,
    severity: LDRCaptureSeverity,
) -> tuple[DefocusConfig, DefocusModel]:
    wavelengths = profile.optics.channel_wavelengths_m
    missing = [channel for channel in ("R", "G", "B") if channel not in wavelengths]
    if missing:
        raise ValueError(f"optics profile is missing RGB wavelengths: {', '.join(missing)}")
    sampling = PupilSampling(
        profile.optics.pupil_grid_size,
        profile.optics.pupil_q_max,
        profile.optics.pupil_fft_size,
    )
    config = DefocusConfig(
        f_number=profile.optics.f_number,
        pixel_pitch_m=profile.sensor.pixel_pitch_m,
        wavelengths_m=tuple(wavelengths[channel] for channel in ("R", "G", "B")),
        reference_wavelength_m=profile.optics.reference_wavelength_m,
        edge_waves_ref=severity.edge_waves_ref,
        pupil_sampling=sampling,
        encircled_energy=profile.optics.psf_energy_fraction,
    )
    return config, build_defocus_model(config)


def _intrinsics(profile: CameraProfile) -> CameraIntrinsics:
    value = profile.fixed_parameters.get("camera_intrinsics_px")
    if not isinstance(value, Mapping):
        raise ValueError("nonzero angular motion requires fixed_parameters.camera_intrinsics_px")
    required = ("fx_px", "fy_px", "cx_px", "cy_px")
    if any(name not in value for name in required):
        raise ValueError("camera_intrinsics_px is missing a required field")
    return CameraIntrinsics(
        fx_px=value["fx_px"],
        fy_px=value["fy_px"],
        cx_px=value["cx_px"],
        cy_px=value["cy_px"],
        skew_px=value.get("skew_px", 0.0),
    )


def _validate_ldr_profile_contract(profile: CameraProfile) -> None:
    adapter = profile.fixed_parameters.get("source_adapter")
    if adapter not in _LDR_SOURCE_ADAPTERS:
        expected = ", ".join(repr(value) for value in sorted(_LDR_SOURCE_ADAPTERS))
        raise ValueError(f"fixed_parameters.source_adapter must be one of {expected}")
    if not np.allclose(
        profile.isp.white_balance_gains,
        (1.0, 1.0, 1.0),
        rtol=0.0,
        atol=0.0,
    ) or not np.allclose(
        profile.isp.camera_to_output_matrix,
        np.eye(3),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "the LDR proxy tier bypasses camera unprocessing and therefore "
            "requires identity white balance and camera-to-output matrix"
        )


def build_ldr_pipeline(
    profile: CameraProfile,
    severity: LDRCaptureSeverity = LDRCaptureSeverity(),
    *,
    active_sensor_shape: tuple[int, int] | None = None,
) -> tuple[CameraPipeline, DefocusModel]:
    """Build the deterministic LDR graph and its cached physical defocus model."""

    if not isinstance(profile, CameraProfile):
        raise TypeError("profile must be a CameraProfile")
    if profile.data_mode is not DataMode.LDR_REDEGRADATION:
        raise ValueError("profile must use LDR_REDEGRADATION mode")
    if not isinstance(severity, LDRCaptureSeverity):
        raise TypeError("severity must be an LDRCaptureSeverity")
    _validate_ldr_profile_contract(profile)
    if profile.isp.output_color_space is not ColorSpace.LINEAR_SRGB:
        raise ValueError("the first rendered path requires linear-sRGB output")
    if not np.allclose(
        profile.isp.output_luminance_coefficients,
        SRGB_REC709_D65.luminance_weights,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("linear-sRGB output requires Rec.709 luminance coefficients")
    _, model = _defocus(profile, severity)
    envelope_h, envelope_w = profile.sensor.sensor_shape_pixels
    source_adapter = profile.fixed_parameters["source_adapter"]
    if active_sensor_shape is None:
        target_h, target_w = envelope_h, envelope_w
    else:
        if (
            isinstance(active_sensor_shape, (str, bytes))
            or len(active_sensor_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in active_sensor_shape
            )
        ):
            raise TypeError("active_sensor_shape must contain integer height and width")
        target_h, target_w = (int(value) for value in active_sensor_shape)
        if target_h <= 0 or target_w <= 0:
            raise ValueError("active_sensor_shape dimensions must be positive")
        if source_adapter != _LDR_NATIVE_ROI_SOURCE_ADAPTER:
            if (target_h, target_w) != (envelope_h, envelope_w):
                raise ValueError("active_sensor_shape is only variable for the native ROI adapter")
        elif target_h > envelope_h or target_w > envelope_w:
            raise ValueError("active sensor ROI exceeds the declared sensor envelope")
    pitch = profile.sensor.pixel_pitch_m
    target_geometry = GridGeometry.square_pixels(target_h, target_w, pitch)
    boundary = BoundaryContract(
        profile.optics.boundary_policy,
        profile.optics.boundary_constant_value,
    )
    readout = profile.readout
    timing = ReadoutTiming(
        height=target_h,
        line_time_s=readout.line_time_s,
        exposure_s=readout.exposure_time_s,
        frame_start_s=readout.frame_start_time_s,
        reference_time_s=readout.reference_time_s,
        annotation_time_s=readout.annotation_time_s,
    )
    omega_nonzero = any(value != 0.0 for value in severity.angular_velocity_rad_s)
    if omega_nonzero and source_adapter == _LDR_NATIVE_ROI_SOURCE_ADAPTER:
        raise ValueError("native active-ROI LDR rendering presently supports static optics only")
    intrinsics = _intrinsics(profile) if omega_nonzero else None
    if omega_nonzero:
        motion = ConstantAngularVelocity(
            severity.angular_velocity_rad_s,
            reference_time_s=readout.reference_time_s,
        )

        def homography_at_time(time_s: float) -> np.ndarray:
            return motion.homography(time_s, intrinsics)
    else:
        homography_at_time = None
    colorimetry = OutputColorimetry(
        name="profile-linear-sRGB/Rec.709-D65",
        primaries_xy=SRGB_REC709_D65.primaries_xy,
        white_point_xy=SRGB_REC709_D65.white_point_xy,
        luminance_weights=profile.isp.output_luminance_coefficients,
    )

    def decode(frame: Frame) -> Frame:
        values = srgb_decode(np.asarray(frame.array, dtype=np.float64))
        return frame.with_array(
            values,
            domain=Domain.LINEAR_RGB_PROXY,
            color_space=ColorSpace.LINEAR_SRGB,
            metadata=_metadata(frame, units="relative_linear_light"),
        )

    def map_source(frame: Frame) -> Frame:
        height, width = frame.shape[:2]
        if source_adapter == _LDR_NATIVE_ROI_SOURCE_ADAPTER:
            if (height, width) != (target_h, target_w):
                raise ValueError("native active-ROI input shape drifted from the declared ROI")
            source_geometry = GridGeometry.square_pixels(height, width, pitch)
        else:
            source_geometry = GridGeometry(
                height,
                width,
                (target_geometry.extent_m[0] / height, target_geometry.extent_m[1] / width),
            )
        metadata = _metadata(
            frame,
            units="relative_linear_light",
            sample_spacing_m=source_geometry.pixel_pitch_m,
            sensor_window_m=(0.0, 0.0, target_w * pitch, target_h * pitch),
            attributes={
                "source_grid_policy": source_adapter,
                "source_shape": [height, width],
                "sensor_shape": [target_h, target_w],
                "sensor_envelope_shape": [envelope_h, envelope_w],
                "source_reconstruction": "piecewise_constant_cell_average",
            },
        )
        return frame.with_array(
            frame.array,
            domain=Domain.SOURCE_GRID_LINEAR_RGB,
            color_space=ColorSpace.LINEAR_SRGB,
            metadata=metadata,
        )

    def optical_capture(frame: Frame) -> Frame:
        height, width = frame.shape[:2]
        if frame.metadata.sample_spacing_m is None:
            raise ValueError("source-grid metadata must declare sample spacing")
        source_geometry = GridGeometry(
            height,
            width,
            frame.metadata.sample_spacing_m,
            frame.metadata.sensor_origin_m or (0.0, 0.0),
        )
        if source_geometry == target_geometry:
            kernels = tuple(
                collapse_cell_average_transfer(
                    channel.psf,
                    source_geometry.pixel_pitch_m,
                    encircled_energy=profile.optics.psf_energy_fraction,
                )
                for channel in model.channels
            )
            optical_representation = "exact_equal_grid_cell_average_transfer_v1"
            pixel_integration_owner = (
                "collapsed_source_reconstruction_and_target_photosite_transfer"
            )
        else:
            kernels = tuple(
                sample_continuous_psf(
                    channel.psf,
                    source_geometry.pixel_pitch_m,
                    encircled_energy=profile.optics.psf_energy_fraction,
                )
                for channel in model.channels
            )
            optical_representation = "continuous_psf_quadrature"
            pixel_integration_owner = "target_photosite_area_resampler"
        values = render_joint_photosite_exposure(
            frame.array,
            source=source_geometry,
            sensor=target_geometry,
            timing=timing,
            optical_kernels=kernels,
            homography_at_time=homography_at_time,
            quadrature_order=readout.quadrature_order,
            optics_boundary=boundary.convolution_mode,
            warp_boundary=boundary.warp_mode,
            constant_value=boundary.constant_value,
        )
        scale = max(1.0, float(np.max(np.abs(values))))
        if float(np.min(values)) < -2e-12 * scale:
            raise FloatingPointError("physical formation produced negative irradiance")
        values = np.maximum(values, 0.0)
        return frame.with_array(
            values,
            domain=Domain.LINEAR_RGB_OPTICAL,
            color_space=ColorSpace.LINEAR_SRGB,
            metadata=_metadata(
                frame,
                units="relative_linear_light",
                sample_spacing_m=(pitch, pitch),
                sensor_window_m=(0.0, 0.0, target_w * pitch, target_h * pitch),
                attributes={
                    "optical_representation": optical_representation,
                    "pixel_integration_owner": pixel_integration_owner,
                    "pixel_integration_count": 1,
                    "boundary_contract": boundary.to_dict(),
                },
            ),
        )

    def retag_signed(frame: Frame) -> Frame:
        return frame.with_array(
            frame.array,
            domain=Domain.OUTPUT_LINEAR_RGB_SIGNED,
            color_space=ColorSpace.LINEAR_SRGB,
        )

    def pre_gamut(frame: Frame) -> Frame:
        values = apply_pre_tone_gamut(
            np.asarray(frame.array, dtype=np.float64),
            policy=profile.isp.pre_tone_gamut_policy,
        )
        return frame.with_array(values, domain=Domain.OUTPUT_LINEAR_RGB_NONNEGATIVE)

    def tone(frame: Frame) -> Frame:
        values = apply_global_tone(
            np.asarray(frame.array, dtype=np.float64),
            severity.tone_stop_ratio,
            pivot=profile.isp.tone_pivot,
            colorimetry=colorimetry,
        )
        return frame.with_array(values, domain=Domain.TONE_MAPPED_LINEAR_RGB)

    def post_gamut(frame: Frame) -> Frame:
        values = apply_post_tone_gamut(
            np.asarray(frame.array, dtype=np.float64),
            policy=profile.isp.post_tone_gamut_policy,
        )
        return frame.with_array(values, domain=Domain.OUTPUT_GAMUT_LINEAR_RGB)

    def encode(frame: Frame) -> Frame:
        values = srgb_encode(np.asarray(frame.array, dtype=np.float64))
        return frame.with_array(
            values,
            domain=Domain.DISPLAY_RGB,
            color_space=ColorSpace.SRGB,
            metadata=_metadata(frame, units="srgb_code_value"),
        )

    if source_adapter == _LDR_NATIVE_ROI_SOURCE_ADAPTER:
        formation_optical_policy = "exact_equal_grid_cell_average_transfer_v1"
        formation_pixel_integration_policy = (
            "collapsed_source_reconstruction_and_target_photosite_transfer"
        )
    else:
        formation_optical_policy = (
            "geometry_selected_equal_grid_cell_average_or_strictly_oversampled_psf_quadrature_v1"
        )
        formation_pixel_integration_policy = (
            "geometry_selected_exactly_once_target_photosite_integration_v1"
        )

    formation_identity = canonical_sha256(
        {
            "model": model.cache_key,
            "angular_velocity_rad_s": list(severity.angular_velocity_rad_s),
            "timing": profile.readout.to_dict(),
            "boundary": boundary.to_dict(),
            "source_adapter": source_adapter,
            "optical_representation_policy": formation_optical_policy,
            "pixel_integration_policy": formation_pixel_integration_policy,
        }
    )
    stages = (
        _stage(
            name="inverse_srgb",
            source=Domain.DISPLAY_RGB,
            target=Domain.LINEAR_RGB_PROXY,
            input_units="srgb_code_value",
            output_units="relative_linear_light",
            implementation_id="color.srgb_decode.v1",
            neutral="exact inverse sRGB",
            operation=decode,
        ),
        _stage(
            name="source_grid",
            source=Domain.LINEAR_RGB_PROXY,
            target=Domain.SOURCE_GRID_LINEAR_RGB,
            input_units="relative_linear_light",
            output_units="relative_linear_light",
            implementation_id=(f"source.{source_adapter}:{target_h}x{target_w}"),
            neutral=(
                "one source cell per active sensor photosite"
                if source_adapter == _LDR_NATIVE_ROI_SOURCE_ADAPTER
                else "same-window cell-average reconstruction"
            ),
            operation=map_source,
        ),
        _stage(
            name="optics_readout",
            source=Domain.SOURCE_GRID_LINEAR_RGB,
            target=Domain.LINEAR_RGB_OPTICAL,
            input_units="relative_linear_light",
            output_units="relative_linear_light",
            implementation_id=f"formation.continuous_ldr.v2:{formation_identity}",
            neutral="W=0 finite aperture and zero angular velocity",
            operation=optical_capture,
        ),
        _stage(
            name="ldr_rgb_adapter",
            source=Domain.LINEAR_RGB_OPTICAL,
            target=Domain.OUTPUT_LINEAR_RGB_SIGNED,
            input_units="relative_linear_light",
            output_units="relative_linear_light",
            implementation_id="ldr.rgb_adapter.identity.v1",
            neutral="identity",
            operation=retag_signed,
        ),
        _stage(
            name="pre_tone_gamut",
            source=Domain.OUTPUT_LINEAR_RGB_SIGNED,
            target=Domain.OUTPUT_LINEAR_RGB_NONNEGATIVE,
            input_units="relative_linear_light",
            output_units="relative_linear_light",
            implementation_id=f"gamut.pre.v1:{profile.isp.pre_tone_gamut_policy}",
            neutral="no negative channels",
            operation=pre_gamut,
        ),
        _stage(
            name="global_tone",
            source=Domain.OUTPUT_LINEAR_RGB_NONNEGATIVE,
            target=Domain.TONE_MAPPED_LINEAR_RGB,
            input_units="relative_linear_light",
            output_units="relative_linear_light",
            implementation_id=f"tone.stop_ratio.v1:{severity.tone_stop_ratio.hex()}",
            neutral="rho=1",
            operation=tone,
        ),
        _stage(
            name="post_tone_gamut",
            source=Domain.TONE_MAPPED_LINEAR_RGB,
            target=Domain.OUTPUT_GAMUT_LINEAR_RGB,
            input_units="relative_linear_light",
            output_units="relative_linear_light",
            implementation_id=f"gamut.post.v1:{profile.isp.post_tone_gamut_policy}",
            neutral="already in unit output gamut",
            operation=post_gamut,
        ),
        _stage(
            name="srgb_encode",
            source=Domain.OUTPUT_GAMUT_LINEAR_RGB,
            target=Domain.DISPLAY_RGB,
            input_units="relative_linear_light",
            output_units="srgb_code_value",
            implementation_id="color.srgb_encode.v1",
            neutral="exact sRGB encode",
            operation=encode,
        ),
    )
    return CameraPipeline(stages, profile=profile), model


def render_ldr(
    image_srgb: ArrayLike,
    profile: CameraProfile,
    severity: LDRCaptureSeverity = LDRCaptureSeverity(),
    *,
    image_id: str = "unidentified",
) -> LDRCaptureResult:
    input_frame = make_ldr_input_frame(image_srgb, image_id=image_id)
    active_shape = (
        input_frame.shape[:2]
        if profile.fixed_parameters.get("source_adapter") == _LDR_NATIVE_ROI_SOURCE_ADAPTER
        else None
    )
    pipeline, model = build_ldr_pipeline(
        profile,
        severity,
        active_sensor_shape=active_shape,
    )
    output_frame, provenance = pipeline.run_with_provenance(input_frame)
    optical_representation = output_frame.metadata.attributes["optical_representation"]
    pixel_integration_owner = output_frame.metadata.attributes["pixel_integration_owner"]
    provenance["capture_condition"] = severity.to_dict()
    provenance["capture_condition_sha256"] = severity.condition_hash
    provenance["physical_contract"] = {
        "input_tier": "display_srgb_linear_proxy",
        "source_adapter": profile.fixed_parameters.get("source_adapter"),
        "optical_representation": optical_representation,
        "pixel_integration_owner": pixel_integration_owner,
        "pixel_integration_count": 1,
        "cfa_enabled": False,
        "sensor_electron_tier_enabled": False,
        "inactive_profile_components": [
            "sensor_electron_noise_gain_adc",
            "camera_white_balance_and_color_matrix",
        ],
        "claim_scope": "synthetic_ldr_redegradation",
    }
    return LDRCaptureResult(
        input_frame=input_frame,
        output_frame=output_frame,
        pipeline=pipeline,
        severity=severity,
        defocus_model=model,
        provenance=provenance,
    )
