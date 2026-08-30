"""Quantitative oversampled forward RAW capture and deterministic rendering.

Unlike :mod:`phycam_eval.capture`, this path never treats display RGB
as sensor channels.  Its input is explicitly camera-native, scene-linear RGB
under a serialized representative-wavelength adapter.  Motion and continuous
optics are evaluated on the oversampled source lattice; target photosite area
is integrated exactly once before Bayer selection and electron sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike

from ._canonical import canonical_sha256, freeze_json_value
from .boundary import BoundaryContract
from .color import SRGB_REC709_D65, OutputColorimetry
from .domains import ColorSpace, DataMode, Domain
from .formation import render_joint_photosite_exposure
from .frame import Frame, FrameMetadata
from .isp.color import ColorTransform, apply_color_transform
from .isp.demosaic import demosaic_bilinear
from .isp.encode import srgb_encode
from .isp.gamut import apply_post_tone_gamut, apply_pre_tone_gamut
from .isp.tone import apply_global_tone
from .optics.defocus import DefocusConfig, DefocusModel, build_defocus_model
from .optics.pupil import PupilSampling
from .optics.sampling import (
    CellAverageTransferKernel,
    PSFQuadratureKernel,
    collapse_cell_average_transfer,
    sample_continuous_psf,
)
from .pipeline import CameraPipeline, StageSpec
from .profiles import CameraProfile
from .readout.motion import CameraIntrinsics, ConstantAngularVelocity
from .readout.timing import ReadoutTiming
from .sensor.adc import (
    ADCProfile,
    GainSetting,
    black_subtract_and_normalize,
    quantize_adc,
    validate_headroom,
)
from .sensor.cfa import BayerPattern, cfa_channel_indices
from .sensor.exposure import ExposurePolicy, ExposureSetting, expected_photoelectrons
from .sensor.noise import RNGKey, StatelessRNG, capture_electrons
from .sensor.pipeline import GainPolicy
from .source_grid import GridGeometry

_FORWARD_GRID_POLICY = "matched_sensor_window_cell_average_v1"
_FORWARD_NOISE_SOURCE = "forward_sensor_capture"
ForwardOpticalKernel = PSFQuadratureKernel | CellAverageTransferKernel


@dataclass(frozen=True, slots=True)
class ForwardCaptureCondition:
    """Complete controlled condition and stochastic realization identity."""

    edge_waves_ref: float = 0.0
    angular_velocity_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    photon_loss_stops: float = 0.0
    exposure_policy: ExposurePolicy = ExposurePolicy.FIXED_DURATION_ATTENUATION
    gain_policy: GainPolicy = GainPolicy.FIXED_PROFILE
    tone_stop_ratio: float = 1.0
    stochastic: bool = True
    seed: int = 0
    realization: int = 0
    coupling_id: str | None = None

    def __post_init__(self) -> None:
        edge = float(self.edge_waves_ref)
        omega = tuple(float(value) for value in self.angular_velocity_rad_s)
        stops = float(self.photon_loss_stops)
        rho = float(self.tone_stop_ratio)
        if not np.isfinite(edge):
            raise ValueError("edge_waves_ref must be finite")
        if len(omega) != 3 or not all(np.isfinite(value) for value in omega):
            raise ValueError("angular_velocity_rad_s must contain three finite values")
        if not np.isfinite(stops) or stops < 0.0:
            raise ValueError("photon_loss_stops must be finite and nonnegative")
        if not np.isfinite(rho) or not (0.0 < rho <= 1.0):
            raise ValueError("tone_stop_ratio must lie in (0, 1]")
        if not isinstance(self.stochastic, (bool, np.bool_)):
            raise TypeError("stochastic must be bool")
        for name in ("seed", "realization"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))
        coupling = self.coupling_id
        if coupling is not None:
            if not isinstance(coupling, str) or not coupling:
                raise ValueError("coupling_id must be a nonempty string or None")
            object.__setattr__(self, "coupling_id", coupling)
        object.__setattr__(self, "edge_waves_ref", edge)
        object.__setattr__(self, "angular_velocity_rad_s", omega)
        object.__setattr__(self, "photon_loss_stops", stops)
        object.__setattr__(self, "exposure_policy", ExposurePolicy(self.exposure_policy))
        object.__setattr__(self, "gain_policy", GainPolicy(self.gain_policy))
        object.__setattr__(self, "tone_stop_ratio", rho)
        object.__setattr__(self, "stochastic", bool(self.stochastic))

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_waves_ref": self.edge_waves_ref,
            "angular_velocity_rad_s": list(self.angular_velocity_rad_s),
            "photon_loss_stops": self.photon_loss_stops,
            "exposure_policy": self.exposure_policy.value,
            "gain_policy": self.gain_policy.value,
            "tone_stop_ratio": self.tone_stop_ratio,
            "stochastic": self.stochastic,
            "seed": self.seed,
            "realization": self.realization,
            "coupling_id": self.coupling_id,
        }

    @property
    def condition_hash(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ForwardCaptureResult:
    input_frame: Frame
    output_frame: Frame
    trace: tuple[Frame, ...]
    pipeline: CameraPipeline
    condition: ForwardCaptureCondition
    defocus_model: DefocusModel
    optical_kernels: tuple[ForwardOpticalKernel, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        trace = tuple(self.trace)
        if not trace or trace[0] is not self.input_frame or trace[-1] is not self.output_frame:
            raise ValueError("trace must begin at input_frame and end at output_frame")
        object.__setattr__(self, "trace", trace)
        frozen = freeze_json_value(self.provenance)
        if not isinstance(frozen, MappingProxyType):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "provenance", frozen)

    def frame_at(self, domain: Domain) -> Frame:
        """Return the unique traced frame carrying ``domain``."""

        matches = [frame for frame in self.trace if frame.domain is Domain(domain)]
        if len(matches) != 1:
            raise ValueError(f"trace contains {len(matches)} frames for {Domain(domain).value}")
        return matches[0]


def _geometry_dict(geometry: GridGeometry) -> dict[str, Any]:
    return {
        "shape": [geometry.height, geometry.width],
        "pixel_pitch_m": list(geometry.pixel_pitch_m),
        "origin_m": list(geometry.origin_m),
        "bounds_m": list(geometry.bounds_m),
    }


def _sensor_window_metadata(geometry: GridGeometry) -> tuple[float, float, float, float]:
    extent_y, extent_x = geometry.extent_m
    return (geometry.origin_m[1], geometry.origin_m[0], extent_x, extent_y)


def make_forward_input_frame(
    scene_camera_rgb: ArrayLike,
    *,
    geometry: GridGeometry,
    spectral_adapter_id: str,
    image_id: str = "unidentified",
) -> Frame:
    """Declare an oversampled, nonnegative camera-native scene-linear input."""

    if not isinstance(geometry, GridGeometry):
        raise TypeError("geometry must be a GridGeometry")
    values = np.asarray(scene_camera_rgb)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("scene_camera_rgb must have shape (H, W, 3)")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("scene_camera_rgb must use a floating-point dtype")
    if values.shape[:2] != (geometry.height, geometry.width):
        raise ValueError("scene array shape must match geometry")
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("scene_camera_rgb must be nonempty, finite, and nonnegative")
    if not isinstance(spectral_adapter_id, str) or not spectral_adapter_id:
        raise ValueError("spectral_adapter_id must be a nonempty string")
    metadata = FrameMetadata(
        units="relative_camera_linear_irradiance",
        data_mode=DataMode.FORWARD_CAMERA_VALIDATION,
        sample_spacing_m=geometry.pixel_pitch_m,
        sensor_origin_m=geometry.origin_m,
        sensor_window_m=_sensor_window_metadata(geometry),
        channel_names=("R", "G", "B"),
        attributes={
            "image_id": str(image_id),
            "spectral_adapter_id": spectral_adapter_id,
            "input_interpretation": "camera_native_representative_wavelength_rgb",
        },
    )
    return Frame(
        values,
        Domain.OVERSAMPLED_SCENE_LINEAR_WITH_DECLARED_SPECTRAL_ADAPTER,
        ColorSpace.CAMERA_NATIVE,
        metadata,
    )


def _metadata(
    frame: Frame,
    *,
    units: str,
    geometry: GridGeometry | None = None,
    channel_names: tuple[str, ...] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> FrameMetadata:
    payload = frame.metadata.to_dict()
    payload["units"] = units
    if geometry is not None:
        payload["sample_spacing_m"] = list(geometry.pixel_pitch_m)
        payload["sensor_origin_m"] = list(geometry.origin_m)
        payload["sensor_window_m"] = list(_sensor_window_metadata(geometry))
    if channel_names is not None:
        payload["channel_names"] = list(channel_names)
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


def _source_contract(profile: CameraProfile) -> tuple[str, float]:
    value = profile.fixed_parameters.get("forward_source")
    if not isinstance(value, Mapping):
        raise ValueError("fixed_parameters.forward_source must be declared")
    if value.get("grid_policy") != _FORWARD_GRID_POLICY:
        raise ValueError(f"forward_source.grid_policy must be {_FORWARD_GRID_POLICY!r}")
    adapter = value.get("spectral_adapter_id")
    if not isinstance(adapter, str) or not adapter:
        raise ValueError("forward_source.spectral_adapter_id must be nonempty")
    minimum = value.get("minimum_samples_per_pixel")
    if isinstance(minimum, bool):
        raise TypeError("minimum_samples_per_pixel must be a real number")
    try:
        minimum_value = float(minimum)
    except (TypeError, ValueError) as exc:
        raise TypeError("minimum_samples_per_pixel must be a real number") from exc
    if not np.isfinite(minimum_value) or minimum_value < 1.0:
        raise ValueError("minimum_samples_per_pixel must be finite and at least one")
    return adapter, minimum_value


def _intrinsics(profile: CameraProfile) -> CameraIntrinsics:
    value = profile.fixed_parameters.get("camera_intrinsics_px")
    if not isinstance(value, Mapping):
        raise ValueError("nonzero angular motion requires camera_intrinsics_px")
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


def _defocus(
    profile: CameraProfile,
    source_geometry: GridGeometry,
    sensor_geometry: GridGeometry,
    condition: ForwardCaptureCondition,
) -> tuple[DefocusConfig, DefocusModel, tuple[ForwardOpticalKernel, ...]]:
    wavelengths = profile.optics.channel_wavelengths_m
    missing = [channel for channel in "RGB" if channel not in wavelengths]
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
        wavelengths_m=tuple(wavelengths[channel] for channel in "RGB"),
        reference_wavelength_m=profile.optics.reference_wavelength_m,
        edge_waves_ref=condition.edge_waves_ref,
        pupil_sampling=sampling,
        encircled_energy=profile.optics.psf_energy_fraction,
    )
    model = build_defocus_model(config)
    if source_geometry == sensor_geometry:
        kernels = tuple(
            collapse_cell_average_transfer(
                channel.psf,
                source_geometry.pixel_pitch_m,
                encircled_energy=profile.optics.psf_energy_fraction,
            )
            for channel in model.channels
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
    return config, model, kernels


def _operating_point(
    profile: CameraProfile,
    exposure: ExposureSetting,
    policy: GainPolicy,
) -> tuple[GainSetting, ADCProfile]:
    sensor = profile.sensor
    adc = ADCProfile(
        conversion_dn_per_electron=sensor.base_conversion_gain_dn_per_electron,
        black_level_dn=sensor.black_level_dn,
        bit_depth=sensor.adc_bit_depth,
    )
    fixed = GainSetting(sensor.analog_gain, sensor.digital_gain)
    profile_headroom = validate_headroom(
        reference_electrons=sensor.reference_electron_budget_electrons,
        full_well_electrons=sensor.full_well_capacity_electrons,
        gain=fixed,
        profile=adc,
    )
    tolerance = 1e-12 * sensor.reference_electron_budget_electrons
    if float(profile_headroom["neutral_headroom_electrons"]) < -tolerance:
        raise ValueError("sensor profile has negative neutral headroom")
    gain = (
        fixed
        if policy is GainPolicy.FIXED_PROFILE
        else GainSetting.compensate(
            exposure.photon_budget_factor,
            analog_gain=sensor.analog_gain,
        )
    )
    active = validate_headroom(
        reference_electrons=(
            exposure.photon_budget_factor * sensor.reference_electron_budget_electrons
        ),
        full_well_electrons=sensor.full_well_capacity_electrons,
        gain=gain,
        profile=adc,
    )
    if float(active["neutral_headroom_electrons"]) < -tolerance:
        raise ValueError("selected gain policy has negative active neutral headroom")
    return gain, adc


def build_forward_pipeline(
    profile: CameraProfile,
    source_geometry: GridGeometry,
    condition: ForwardCaptureCondition = ForwardCaptureCondition(),
) -> tuple[CameraPipeline, DefocusModel, tuple[ForwardOpticalKernel, ...]]:
    """Build the quantitative forward RAW graph for one fixed source lattice."""

    if not isinstance(profile, CameraProfile):
        raise TypeError("profile must be a CameraProfile")
    if profile.data_mode is not DataMode.FORWARD_CAMERA_VALIDATION:
        raise ValueError("profile must use FORWARD_CAMERA_VALIDATION mode")
    if not isinstance(source_geometry, GridGeometry):
        raise TypeError("source_geometry must be a GridGeometry")
    if not isinstance(condition, ForwardCaptureCondition):
        raise TypeError("condition must be a ForwardCaptureCondition")
    if profile.calibration_reference is None:
        raise ValueError("forward-camera profiles require a calibration_reference")
    if profile.sensor.cfa_pattern is None:
        raise ValueError("forward RAW capture requires a Bayer CFA pattern")
    try:
        cfa_pattern = BayerPattern(profile.sensor.cfa_pattern)
    except ValueError as exc:
        raise ValueError("sensor cfa_pattern must be a supported Bayer pattern") from exc
    if profile.isp.output_color_space is not ColorSpace.LINEAR_SRGB:
        raise ValueError("the first forward renderer requires linear-sRGB output")
    if not np.allclose(
        profile.isp.output_luminance_coefficients,
        SRGB_REC709_D65.luminance_weights,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("linear-sRGB output requires Rec.709 luminance coefficients")

    adapter_id, minimum_oversampling = _source_contract(profile)
    sensor_h, sensor_w = profile.sensor.sensor_shape_pixels
    pitch = profile.sensor.pixel_pitch_m
    sensor_geometry = GridGeometry.square_pixels(sensor_h, sensor_w, pitch)
    if not np.allclose(
        source_geometry.bounds_m,
        sensor_geometry.bounds_m,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("source geometry must cover the fixed sensor window")
    samples_per_pixel = (
        sensor_geometry.pixel_pitch_m[0] / source_geometry.pixel_pitch_m[0],
        sensor_geometry.pixel_pitch_m[1] / source_geometry.pixel_pitch_m[1],
    )
    if min(samples_per_pixel) + 1e-12 < minimum_oversampling:
        raise ValueError("source grid does not meet forward_source.minimum_samples_per_pixel")

    boundary = BoundaryContract(
        profile.optics.boundary_policy,
        profile.optics.boundary_constant_value,
    )
    exposure = ExposureSetting(
        condition.photon_loss_stops,
        profile.readout.exposure_time_s,
        condition.exposure_policy,
    )
    timing = ReadoutTiming(
        height=sensor_h,
        line_time_s=profile.readout.line_time_s,
        exposure_s=exposure.exposure_s,
        frame_start_s=profile.readout.frame_start_time_s,
        reference_time_s=profile.readout.reference_time_s,
        annotation_time_s=profile.readout.annotation_time_s,
    )
    omega_nonzero = any(value != 0.0 for value in condition.angular_velocity_rad_s)
    if omega_nonzero:
        intrinsics = _intrinsics(profile)
        motion = ConstantAngularVelocity(
            condition.angular_velocity_rad_s,
            reference_time_s=profile.readout.reference_time_s,
        )

        def homography_at_time(time_s: float) -> np.ndarray:
            return motion.homography(time_s, intrinsics)
    else:
        homography_at_time = None

    _, model, optical_kernels = _defocus(
        profile,
        source_geometry,
        sensor_geometry,
        condition,
    )
    equal_grid_transfer = isinstance(optical_kernels[0], CellAverageTransferKernel)
    optical_representation = (
        "exact_equal_grid_cell_average_transfer_v1"
        if equal_grid_transfer
        else "continuous_psf_quadrature"
    )
    pixel_integration_owner = (
        "collapsed_source_reconstruction_and_target_photosite_transfer"
        if equal_grid_transfer
        else "target_photosite_area_resampler"
    )
    gain, adc_profile = _operating_point(profile, exposure, condition.gain_policy)
    quantum_efficiency = np.asarray(
        [profile.sensor.quantum_efficiency.get(channel, np.nan) for channel in "RGB"],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(quantum_efficiency)):
        raise ValueError("sensor quantum_efficiency must declare R, G, and B")
    channel_map = cfa_channel_indices((sensor_h, sensor_w), cfa_pattern)
    colorimetry = OutputColorimetry(
        name="profile-linear-sRGB/Rec.709-D65",
        primaries_xy=SRGB_REC709_D65.primaries_xy,
        white_point_xy=SRGB_REC709_D65.white_point_xy,
        luminance_weights=profile.isp.output_luminance_coefficients,
    )
    color_transform = ColorTransform(
        white_balance=profile.isp.white_balance_gains,
        camera_to_output=profile.isp.camera_to_output_matrix,
        output_space=profile.isp.output_color_space.value,
    )
    source_identity = canonical_sha256(_geometry_dict(source_geometry))
    formation_identity = canonical_sha256(
        {
            "model": model.cache_key,
            "source_geometry": _geometry_dict(source_geometry),
            "sensor_geometry": _geometry_dict(sensor_geometry),
            "edge_waves_ref": condition.edge_waves_ref,
            "angular_velocity_rad_s": list(condition.angular_velocity_rad_s),
            "exposure_s": exposure.exposure_s,
            "line_time_s": profile.readout.line_time_s,
            "quadrature_order": profile.readout.quadrature_order,
            "boundary": boundary.to_dict(),
            "pixel_integration_owner": pixel_integration_owner,
        }
    )
    expectation_identity = canonical_sha256(
        {
            "photon_loss_stops": condition.photon_loss_stops,
            "exposure_policy": condition.exposure_policy.value,
            "reference_electrons": profile.sensor.reference_electron_budget_electrons,
            "quantum_efficiency": quantum_efficiency.tolist(),
            "cfa_pattern": cfa_pattern.value,
        }
    )
    default_coupling_id = canonical_sha256(
        {
            "camera_profile_sha256": profile.profile_hash,
            "condition": condition.to_dict(),
            "source_geometry": _geometry_dict(source_geometry),
            "formation_identity": formation_identity,
        }
    )
    coupling_id = condition.coupling_id or default_coupling_id

    def form_expectation(frame: Frame) -> Frame:
        if frame.color_space is not ColorSpace.CAMERA_NATIVE:
            raise ValueError("forward input must be camera-native RGB")
        if frame.metadata.attributes.get("spectral_adapter_id") != adapter_id:
            raise ValueError("input spectral adapter does not match the camera profile")
        if frame.shape[:2] != (source_geometry.height, source_geometry.width):
            raise ValueError("input shape does not match pipeline source_geometry")
        if frame.metadata.sample_spacing_m != source_geometry.pixel_pitch_m:
            raise ValueError("input sample spacing does not match source_geometry")
        if frame.metadata.sensor_origin_m != source_geometry.origin_m:
            raise ValueError("input origin does not match source_geometry")
        formed_rgb = render_joint_photosite_exposure(
            frame.array,
            source=source_geometry,
            sensor=sensor_geometry,
            timing=timing,
            optical_kernels=optical_kernels,
            homography_at_time=homography_at_time,
            quadrature_order=profile.readout.quadrature_order,
            optics_boundary=boundary.convolution_mode,
            warp_boundary=boundary.warp_mode,
            constant_value=boundary.constant_value,
        )
        scale = max(1.0, float(np.max(np.abs(formed_rgb))))
        minimum = float(np.min(formed_rgb))
        if minimum < -2e-12 * scale:
            raise FloatingPointError("physical formation produced negative irradiance")
        formed_rgb = np.maximum(formed_rgb, 0.0)
        qe_weighted = formed_rgb * quantum_efficiency[None, None, :]
        normalized_photosites = np.take_along_axis(qe_weighted, channel_map[..., None], axis=-1)[
            ..., 0
        ]
        expectation = expected_photoelectrons(
            normalized_photosites,
            reference_electrons=profile.sensor.reference_electron_budget_electrons,
            exposure=exposure,
        )
        return frame.with_array(
            expectation,
            domain=Domain.PHOTOSITE_EXPECTATION,
            color_space=ColorSpace.RAW_MOSAIC,
            metadata=_metadata(
                frame,
                units="electrons_expected",
                geometry=sensor_geometry,
                channel_names=(),
                attributes={
                    "spectral_adapter_id": adapter_id,
                    "grid_policy": _FORWARD_GRID_POLICY,
                    "source_geometry": _geometry_dict(source_geometry),
                    "sensor_geometry": _geometry_dict(sensor_geometry),
                    "optical_representation": optical_representation,
                    "pixel_integration_owner": pixel_integration_owner,
                    "pixel_integration_count": 1,
                    "cfa_pattern": cfa_pattern.value,
                    "quantum_efficiency_rgb": quantum_efficiency.tolist(),
                    "exposure_policy": exposure.policy.value,
                    "photon_budget_factor": exposure.photon_budget_factor,
                    "exposure_s": exposure.exposure_s,
                },
            ),
        )

    def sample_electrons(frame: Frame) -> Frame:
        expectation = np.asarray(frame.array, dtype=np.float64)
        expected_dark = profile.sensor.dark_current_electrons_per_second * exposure.exposure_s
        image_id = str(frame.metadata.attributes.get("image_id", "unidentified"))
        key = RNGKey(
            profile_hash=profile.profile_hash,
            seed=condition.seed,
            image_id=image_id,
            coupling_id=coupling_id,
            realization=condition.realization,
            noise_source=_FORWARD_NOISE_SOURCE,
        )
        counters = np.arange(expectation.size, dtype=np.uint64).reshape(expectation.shape)
        if condition.stochastic:
            captured = capture_electrons(
                expectation,
                full_well_electrons=profile.sensor.full_well_capacity_electrons,
                read_noise_electrons=profile.sensor.read_noise_rms_electrons,
                expected_dark_electrons=expected_dark,
                rng=StatelessRNG(),
                key=key,
                counters=counters,
            )
            electrons = captured.electrons
            saturated_count = int(
                np.count_nonzero(
                    captured.well_electrons >= profile.sensor.full_well_capacity_electrons
                )
            )
        else:
            well = np.minimum(
                expectation + expected_dark,
                profile.sensor.full_well_capacity_electrons,
            )
            electrons = well
            saturated_count = int(
                np.count_nonzero(well >= profile.sensor.full_well_capacity_electrons)
            )
        return frame.with_array(
            electrons,
            domain=Domain.ELECTRONS,
            color_space=ColorSpace.RAW_MOSAIC,
            metadata=_metadata(
                frame,
                units="electrons",
                attributes={
                    "stochastic": condition.stochastic,
                    "rng_algorithm": StatelessRNG.algorithm,
                    "rng_profile_hash": key.profile_hash,
                    "rng_seed": key.seed,
                    "rng_coupling_id": key.coupling_id,
                    "rng_realization": key.realization,
                    "expected_dark_electrons": expected_dark,
                    "full_well_capacity_electrons": (profile.sensor.full_well_capacity_electrons),
                    "full_well_saturated_count": saturated_count,
                },
            ),
        )

    def convert_adc(frame: Frame) -> Frame:
        adc_dn = quantize_adc(frame.array, gain=gain, profile=adc_profile)
        return frame.with_array(
            adc_dn,
            domain=Domain.RAW_ADC_DN,
            color_space=ColorSpace.RAW_MOSAIC,
            metadata=_metadata(
                frame,
                units="DN",
                attributes={
                    "analog_gain": gain.analog_gain,
                    "digital_gain": gain.digital_gain,
                    "gain_policy": condition.gain_policy.value,
                    "black_level_dn": adc_profile.black_level_dn,
                    "adc_bit_depth": adc_profile.bit_depth,
                    "adc_maximum_dn": adc_profile.maximum_dn,
                    "adc_saturated_count": int(np.count_nonzero(adc_dn >= adc_profile.maximum_dn)),
                    "adc_rounding": "ties_to_even",
                },
            ),
        )

    def normalize_raw(frame: Frame) -> Frame:
        signed = black_subtract_and_normalize(
            frame.array,
            gain=gain,
            profile=adc_profile,
            reference_electrons=profile.sensor.reference_electron_budget_electrons,
        )
        return frame.with_array(
            signed,
            domain=Domain.SIGNED_CAMERA_LINEAR,
            color_space=ColorSpace.RAW_MOSAIC,
            metadata=_metadata(frame, units="relative_camera_linear"),
        )

    def demosaic(frame: Frame) -> Frame:
        values = demosaic_bilinear(frame.array, cfa_pattern, boundary="mirror")
        return frame.with_array(
            values,
            domain=Domain.CAMERA_LINEAR_RGB,
            color_space=ColorSpace.CAMERA_LINEAR_RGB,
            metadata=_metadata(
                frame,
                units="relative_camera_linear",
                channel_names=("R", "G", "B"),
                attributes={"demosaic": "fixed_normalized_bilinear_v1"},
            ),
        )

    def color(frame: Frame) -> Frame:
        values = apply_color_transform(frame.array, color_transform)
        return frame.with_array(
            values,
            domain=Domain.OUTPUT_LINEAR_RGB_SIGNED,
            color_space=ColorSpace.LINEAR_SRGB,
            metadata=_metadata(frame, units="relative_output_linear_light"),
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
            condition.tone_stop_ratio,
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

    stages = (
        _stage(
            name="joint_formation_and_photosite_expectation",
            source=Domain.OVERSAMPLED_SCENE_LINEAR_WITH_DECLARED_SPECTRAL_ADAPTER,
            target=Domain.PHOTOSITE_EXPECTATION,
            input_units="relative_camera_linear_irradiance",
            output_units="electrons_expected",
            implementation_id=(f"formation.continuous.v1:{formation_identity}:{source_identity}"),
            neutral="W=0, zero motion, fixed finite aperture",
            operation=form_expectation,
        ),
        _stage(
            name="electron_capture",
            source=Domain.PHOTOSITE_EXPECTATION,
            target=Domain.ELECTRONS,
            input_units="electrons_expected",
            output_units="electrons",
            implementation_id=(
                "sensor.electrons.v2:"
                + canonical_sha256(
                    {
                        "expectation": expectation_identity,
                        "rng_algorithm": StatelessRNG.algorithm,
                        "condition": {
                            "stochastic": condition.stochastic,
                            "seed": condition.seed,
                            "realization": condition.realization,
                            "coupling_id": coupling_id,
                        },
                        "full_well": profile.sensor.full_well_capacity_electrons,
                        "read_noise": profile.sensor.read_noise_rms_electrons,
                        "dark_current": profile.sensor.dark_current_electrons_per_second,
                    }
                )
            ),
            neutral="declared realization at zero photon loss",
            operation=sample_electrons,
        ),
        _stage(
            name="analog_gain_adc",
            source=Domain.ELECTRONS,
            target=Domain.RAW_ADC_DN,
            input_units="electrons",
            output_units="DN",
            implementation_id=(
                "sensor.adc.v1:"
                + canonical_sha256(
                    {
                        "gain": [gain.analog_gain, gain.digital_gain],
                        "conversion": adc_profile.conversion_dn_per_electron,
                        "black": adc_profile.black_level_dn,
                        "bits": adc_profile.bit_depth,
                    }
                )
            ),
            neutral="profile gain and ties-to-even quantization",
            operation=convert_adc,
        ),
        _stage(
            name="signed_black_subtraction",
            source=Domain.RAW_ADC_DN,
            target=Domain.SIGNED_CAMERA_LINEAR,
            input_units="DN",
            output_units="relative_camera_linear",
            implementation_id="sensor.black_normalize.v1",
            neutral="signed black subtraction without shadow clipping",
            operation=normalize_raw,
        ),
        _stage(
            name="fixed_bilinear_demosaic",
            source=Domain.SIGNED_CAMERA_LINEAR,
            target=Domain.CAMERA_LINEAR_RGB,
            input_units="relative_camera_linear",
            output_units="relative_camera_linear",
            implementation_id=f"isp.demosaic.bilinear.v1:{cfa_pattern.value}:mirror",
            neutral="fixed non-content-adaptive interpolation",
            operation=demosaic,
        ),
        _stage(
            name="white_balance_and_color",
            source=Domain.CAMERA_LINEAR_RGB,
            target=Domain.OUTPUT_LINEAR_RGB_SIGNED,
            input_units="relative_camera_linear",
            output_units="relative_output_linear_light",
            implementation_id=(
                "isp.color.fixed.v1:"
                + canonical_sha256(
                    {
                        "white_balance": list(profile.isp.white_balance_gains),
                        "matrix": [list(row) for row in profile.isp.camera_to_output_matrix],
                        "output": profile.isp.output_color_space.value,
                    }
                )
            ),
            neutral="profile white balance and camera-to-output transform",
            operation=color,
        ),
        _stage(
            name="pre_tone_gamut",
            source=Domain.OUTPUT_LINEAR_RGB_SIGNED,
            target=Domain.OUTPUT_LINEAR_RGB_NONNEGATIVE,
            input_units="relative_output_linear_light",
            output_units="relative_output_linear_light",
            implementation_id=f"gamut.pre.v1:{profile.isp.pre_tone_gamut_policy}",
            neutral="no negative output channels",
            operation=pre_gamut,
        ),
        _stage(
            name="global_tone",
            source=Domain.OUTPUT_LINEAR_RGB_NONNEGATIVE,
            target=Domain.TONE_MAPPED_LINEAR_RGB,
            input_units="relative_output_linear_light",
            output_units="relative_output_linear_light",
            implementation_id=f"tone.stop_ratio.v1:{condition.tone_stop_ratio.hex()}",
            neutral="rho=1",
            operation=tone,
        ),
        _stage(
            name="post_tone_gamut",
            source=Domain.TONE_MAPPED_LINEAR_RGB,
            target=Domain.OUTPUT_GAMUT_LINEAR_RGB,
            input_units="relative_output_linear_light",
            output_units="relative_output_linear_light",
            implementation_id=f"gamut.post.v1:{profile.isp.post_tone_gamut_policy}",
            neutral="already in unit output gamut",
            operation=post_gamut,
        ),
        _stage(
            name="srgb_encode",
            source=Domain.OUTPUT_GAMUT_LINEAR_RGB,
            target=Domain.DISPLAY_RGB,
            input_units="relative_output_linear_light",
            output_units="srgb_code_value",
            implementation_id="color.srgb_encode.v1",
            neutral="exact sRGB encode",
            operation=encode,
        ),
    )
    return CameraPipeline(stages, profile=profile), model, optical_kernels


def render_forward(
    scene_camera_rgb: ArrayLike,
    *,
    source_geometry: GridGeometry,
    profile: CameraProfile,
    spectral_adapter_id: str,
    condition: ForwardCaptureCondition = ForwardCaptureCondition(),
    image_id: str = "unidentified",
) -> ForwardCaptureResult:
    """Run one fully traced forward-camera condition and realization."""

    input_frame = make_forward_input_frame(
        scene_camera_rgb,
        geometry=source_geometry,
        spectral_adapter_id=spectral_adapter_id,
        image_id=image_id,
    )
    pipeline, model, kernels = build_forward_pipeline(
        profile,
        source_geometry,
        condition,
    )
    equal_grid_transfer = isinstance(kernels[0], CellAverageTransferKernel)
    trace, provenance = pipeline.run_trace_with_provenance(input_frame)
    provenance.update(
        {
            "capture_condition": condition.to_dict(),
            "capture_condition_sha256": condition.condition_hash,
            "source_geometry": _geometry_dict(source_geometry),
            "stage_boundaries": [frame.descriptor() for frame in trace],
            "physical_contract": {
                "input_tier": (
                    "equal_grid_camera_native_representative_wavelength_rgb"
                    if equal_grid_transfer
                    else "oversampled_camera_native_representative_wavelength_rgb"
                ),
                "optical_representation": (
                    "exact_equal_grid_cell_average_transfer_v1"
                    if equal_grid_transfer
                    else "continuous_psf_quadrature"
                ),
                "pixel_integration_owner": (
                    "collapsed_source_reconstruction_and_target_photosite_transfer"
                    if equal_grid_transfer
                    else "target_photosite_area_resampler"
                ),
                "pixel_integration_count": 1,
                "cfa_enabled": True,
                "sensor_noise_domain": "electrons",
                "signed_shadow_values_preserved_until_pre_tone_gamut": True,
            },
        }
    )
    return ForwardCaptureResult(
        input_frame=trace[0],
        output_frame=trace[-1],
        trace=trace,
        pipeline=pipeline,
        condition=condition,
        defocus_model=model,
        optical_kernels=kernels,
        provenance=provenance,
    )


__all__ = [
    "ForwardCaptureCondition",
    "ForwardCaptureResult",
    "build_forward_pipeline",
    "make_forward_input_frame",
    "render_forward",
]
