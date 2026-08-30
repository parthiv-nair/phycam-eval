"""Common-neutral LDR camera branches for mechanism-matched comparators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias

from numpy.typing import ArrayLike

from .._canonical import canonical_sha256, freeze_json_value, json_value
from ..boundary import BoundaryContract
from ..capture import LDRCaptureSeverity, build_ldr_pipeline, make_ldr_input_frame
from ..domains import DataMode, Domain
from ..frame import Frame
from ..optics.defocus import DefocusModel
from ..pipeline import CameraPipeline, StageSpec
from ..profiles import CameraProfile
from .gaussian import GaussianComparatorConfig, apply_gaussian_comparator
from .transfer_families import (
    QuadraticCosineComparatorConfig,
    SampledIncoherentComparatorConfig,
    apply_quadratic_cosine_comparator,
    apply_sampled_incoherent_comparator,
)

LDRComparatorConfig: TypeAlias = (
    GaussianComparatorConfig | QuadraticCosineComparatorConfig | SampledIncoherentComparatorConfig
)


def comparator_config_from_dict(record: Mapping[str, Any]) -> LDRComparatorConfig:
    """Reconstruct one canonical comparator configuration fail-closed."""

    if not isinstance(record, Mapping):
        raise TypeError("comparator config record must be a mapping")
    value = json_value(record)
    operator = value.get("operator")
    if operator == "gaussian_comparator":
        config: LDRComparatorConfig = GaussianComparatorConfig(
            sigma_pixels=value.get("sigma_pixels"),
            boundary=value.get("boundary"),
            truncate_sigma=value.get("truncate_sigma"),
            implementation_id=value.get("implementation_id"),
        )
    elif operator == "adapted_quadratic_cosine":
        config = QuadraticCosineComparatorConfig(
            alpha=value.get("alpha"),
            implementation_id=value.get("implementation_id"),
        )
    elif operator == "adapted_sampled_incoherent_quadratic_pupil":
        config = SampledIncoherentComparatorConfig(
            alpha=value.get("alpha"),
            boundary=value.get("boundary"),
            pupil_grid_size=value.get("pupil_grid_size"),
            encircled_energy=value.get("encircled_energy"),
            implementation_id=value.get("implementation_id"),
        )
    else:
        raise ValueError("unsupported comparator operator")
    if config.to_dict() != value:
        raise ValueError("comparator config record contains unknown or noncanonical fields")
    return config


def _config_record(config: LDRComparatorConfig) -> dict[str, Any]:
    if not isinstance(
        config,
        (
            GaussianComparatorConfig,
            QuadraticCosineComparatorConfig,
            SampledIncoherentComparatorConfig,
        ),
    ):
        raise TypeError("config must be a supported LDR comparator configuration")
    return config.to_dict()


def _application_record(
    profile: CameraProfile,
    config: LDRComparatorConfig,
) -> dict[str, Any]:
    """Bind a comparator configuration to its complete boundary application."""

    boundary = BoundaryContract(
        profile.optics.boundary_policy,
        profile.optics.boundary_constant_value,
    )
    return {
        "config": _config_record(config),
        "boundary_contract": boundary.to_dict(),
    }


def _validate_boundary(profile: CameraProfile, config: LDRComparatorConfig) -> None:
    boundary = profile.optics.boundary_policy
    if isinstance(config, QuadraticCosineComparatorConfig):
        if boundary != "reflect":
            raise ValueError("the DCT-even quadratic comparator requires profile reflect boundary")
    elif config.boundary != boundary:
        raise ValueError("comparator boundary must match the camera profile boundary")


def build_ldr_comparator_pipeline(
    profile: CameraProfile,
    config: LDRComparatorConfig,
    *,
    active_sensor_shape: tuple[int, int] | None = None,
) -> tuple[CameraPipeline, DefocusModel]:
    """Insert one digital comparator after W=0 optics and before gamut/tone."""

    if not isinstance(profile, CameraProfile):
        raise TypeError("profile must be a CameraProfile")
    if profile.data_mode is not DataMode.LDR_REDEGRADATION:
        raise ValueError("comparator capture requires an LDR_REDEGRADATION profile")
    _validate_boundary(profile, config)
    application_record = _application_record(profile, config)
    neutral, model = build_ldr_pipeline(
        profile,
        LDRCaptureSeverity(),
        active_sensor_shape=active_sensor_shape,
    )
    constant_value = profile.optics.boundary_constant_value

    def apply(frame: Frame) -> Frame:
        if isinstance(config, GaussianComparatorConfig):
            values = apply_gaussian_comparator(
                frame.array,
                config,
                constant_value=constant_value,
            )
        elif isinstance(config, QuadraticCosineComparatorConfig):
            values = apply_quadratic_cosine_comparator(frame.array, config)
        else:
            values = apply_sampled_incoherent_comparator(
                frame.array,
                config,
                constant_value=constant_value,
            )
        return frame.with_array(values, domain=Domain.OUTPUT_LINEAR_RGB_SIGNED)

    stage = StageSpec(
        name="mechanism_matched_comparator",
        input_domain=Domain.LINEAR_RGB_OPTICAL,
        output_domain=Domain.OUTPUT_LINEAR_RGB_SIGNED,
        input_units="relative_linear_light",
        output_units="relative_linear_light",
        deterministic=True,
        implementation_id=f"comparator:{canonical_sha256(application_record)}",
        neutral_condition="comparator coordinate zero",
        operation=apply,
    )
    # Stages 0:3 produce the common W=0 LINEAR_RGB_OPTICAL branch. The
    # comparator assumes the signed-output adapter's semantic transition, so
    # the original identity adapter at index 3 is deliberately replaced.
    stages = (*neutral.stages[:3], stage, *neutral.stages[4:])
    return CameraPipeline(stages, profile=profile), model


@dataclass(frozen=True, slots=True)
class LDRComparatorResult:
    """One rendered common-neutral comparator capture and its provenance."""

    input_frame: Frame
    output_frame: Frame
    pipeline: CameraPipeline
    config: LDRComparatorConfig
    neutral_defocus_model: DefocusModel
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        frozen = freeze_json_value(self.provenance)
        if not isinstance(frozen, Mapping):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "provenance", frozen)


def render_ldr_comparator(
    image_srgb: ArrayLike,
    profile: CameraProfile,
    config: LDRComparatorConfig,
    *,
    image_id: str = "unidentified",
) -> LDRComparatorResult:
    """Render one comparator arm from the common modeled-neutral optical branch."""

    input_frame = make_ldr_input_frame(image_srgb, image_id=image_id)
    active_shape = (
        input_frame.shape[:2]
        if profile.fixed_parameters.get("source_adapter") == "native_active_sensor_roi_v1"
        else None
    )
    pipeline, model = build_ldr_comparator_pipeline(
        profile,
        config,
        active_sensor_shape=active_shape,
    )
    output_frame, provenance = pipeline.run_with_provenance(input_frame)
    condition = {
        "common_neutral": {"physical_defocus_edge_waves_ref": 0.0},
        "comparator": _config_record(config),
        "comparator_boundary_contract": _application_record(profile, config)["boundary_contract"],
    }
    provenance["capture_condition"] = condition
    provenance["capture_condition_sha256"] = canonical_sha256(condition)
    optical_representation = output_frame.metadata.attributes.get("optical_representation")
    pixel_integration_owner = output_frame.metadata.attributes.get("pixel_integration_owner")
    if not isinstance(optical_representation, str) or not isinstance(pixel_integration_owner, str):
        raise RuntimeError("common-neutral optical formation metadata is incomplete")
    provenance["physical_contract"] = {
        "input_tier": "display_srgb_linear_proxy",
        "source_adapter": profile.fixed_parameters.get("source_adapter"),
        "common_neutral_branch": "W=0 LINEAR_RGB_OPTICAL",
        "comparator_insertion_boundary": "before_pre_tone_gamut",
        "comparator_is_physical_model": False,
        "common_neutral_optical_representation": optical_representation,
        "pixel_integration_owner": pixel_integration_owner,
        "pixel_integration_count": 1,
        "claim_scope": "synthetic_ldr_redegradation_mechanism_comparator",
    }
    return LDRComparatorResult(
        input_frame=input_frame,
        output_frame=output_frame,
        pipeline=pipeline,
        config=config,
        neutral_defocus_model=model,
        provenance=provenance,
    )


__all__ = [
    "LDRComparatorConfig",
    "LDRComparatorResult",
    "build_ldr_comparator_pipeline",
    "comparator_config_from_dict",
    "render_ldr_comparator",
]
