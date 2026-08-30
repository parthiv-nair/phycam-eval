"""Content-addressed physical experiment conditions.

The records in this module describe camera conditions, not detector settings.
Every identity is bound to one immutable camera profile, one data mode, and an
ordered set of stochastic realization IDs.  Public coordinates carry fixed
units through their family type; callers cannot relabel a numeric grid with an
arbitrary unit string.
"""

from __future__ import annotations

import math
import numbers
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence, TypeAlias

import numpy as np

from .._canonical import (
    canonical_sha256,
    finite_float,
    freeze_json_value,
    json_value,
    nfc_string,
)
from ..capture import LDRCaptureSeverity
from ..domains import DataMode
from ..forward_capture import ForwardCaptureCondition
from ..profiles import CameraProfile
from ..sensor.exposure import ExposurePolicy
from ..sensor.pipeline import GainPolicy


class ConditionFamily(str, Enum):
    """Supported physical sweep axes in normative camera order."""

    ROLLING_ROTATION = "rolling_rotation"
    PHYSICAL_DEFOCUS = "physical_defocus"
    PHOTON_LOSS = "photon_loss"
    TONE_STOP_RATIO = "tone_stop_ratio"
    FACTORIAL = "factorial"
    BASELINE = "baseline"
    MECHANISM_COMPARATOR = "mechanism_comparator"


class BaselineKind(str, Enum):
    """Semantically distinct experiment baselines."""

    UNTOUCHED_INPUT = "untouched_input"
    MODELED_NEUTRAL = "modeled_neutral"


# Projection/motion precedes optics; sensor exposure/noise precedes ISP tone.
PHYSICAL_FAMILY_ORDER: tuple[ConditionFamily, ...] = (
    ConditionFamily.ROLLING_ROTATION,
    ConditionFamily.PHYSICAL_DEFOCUS,
    ConditionFamily.PHOTON_LOSS,
    ConditionFamily.TONE_STOP_RATIO,
)
_FAMILY_RANK = {family: rank for rank, family in enumerate(PHYSICAL_FAMILY_ORDER)}

_MECHANISM_MATCH_V4_KEYS = {
    "schema_version",
    "record_type",
    "matching_contract",
    "comparator_family",
    "target_edge_waves_ref",
    "target_mtf50_cycles_per_pixel",
    "neutral_mtf_at_target",
    "config",
    "config_sha256",
    "achieved_mtf50_cycles_per_pixel",
    "relative_match_error",
    "response_match_scope",
    "camera_profile_sha256",
    "neutral_model_sha256",
    "target_model_sha256",
    "match_sha256",
}
_MECHANISM_MATCHING_CONTRACT_V4 = {
    "criterion": "rec709_luminance_weighted_first_downward_mtf50",
    "frequency_unit": "cycles/pixel",
    "common_neutral_branch": "physical_W0_linear_rgb_optical",
    "comparator_insertion": "after_common_neutral_before_gamut_tone_encode",
    "physical_formation_operator": (
        "exact_equal_grid_piecewise_constant_source_x_photosite_transfer_v1"
    ),
    "boundary_harmonization": "profile_reflect_whole_sample_spatial_or_dct_i_v2",
    "parameter_match_scope": "finite_kernel_dtft_or_continuous_dct_i_eigenvalue_design_v1",
    "dct_i_finite_shape_acceptance": (
        "separate_complete_publication_axis_dimension_envelope_required"
    ),
}
_MATCH_FAMILY_OPERATOR = {
    "gaussian": "gaussian_comparator",
    "adapted_quadratic_cosine": "adapted_quadratic_cosine",
    "adapted_sampled_incoherent": "adapted_sampled_incoherent_quadratic_pupil",
}
_MATCH_FAMILY_RESPONSE_SCOPE = {
    "gaussian": "exact_dtft_of_executed_finite_shift_invariant_kernel",
    "adapted_quadratic_cosine": (
        "continuous_design_curve_exact_at_shape_dependent_dct_i_eigenfrequencies"
    ),
    "adapted_sampled_incoherent": ("exact_dtft_of_executed_finite_shift_invariant_kernel"),
}


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    digest = nfc_string(value, field_name=field_name)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


def _realization_ids(values: Iterable[int]) -> tuple[int, ...]:
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError("realization_ids must be an iterable of integers") from exc
    if not items:
        raise ValueError("realization_ids must not be empty")
    result: list[int] = []
    for index, value in enumerate(items):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (numbers.Integral, np.integer)
        ):
            raise TypeError(f"realization_ids[{index}] must be an integer")
        item = int(value)
        if item < 0:
            raise ValueError(f"realization_ids[{index}] must be nonnegative")
        if item in result:
            raise ValueError("realization_ids must be unique and ordered explicitly")
        result.append(item)
    return tuple(result)


def _axis(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("rotation_axis must contain exactly three values")
    raw = tuple(
        finite_float(value, field_name=f"rotation_axis[{index}]")
        for index, value in enumerate(values)
    )
    norm = math.sqrt(sum(value * value for value in raw))
    if norm == 0.0:
        raise ValueError("rotation_axis must have nonzero length")
    if math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=4.0 * np.finfo(float).eps):
        return raw
    return tuple(  # type: ignore[return-value]
        0.0 if value == 0.0 else value / norm for value in raw
    )


@dataclass(frozen=True, slots=True)
class ConditionBinding:
    """Profile, mode, and ordered realization selection shared by a plan."""

    fixed_profile_sha256: str
    data_mode: DataMode
    realization_ids: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixed_profile_sha256",
            _sha256(self.fixed_profile_sha256, field_name="fixed_profile_sha256"),
        )
        object.__setattr__(self, "data_mode", DataMode(self.data_mode))
        object.__setattr__(self, "realization_ids", _realization_ids(self.realization_ids))

    @classmethod
    def from_profile(
        cls,
        profile: CameraProfile,
        realization_ids: Iterable[int] = (0,),
    ) -> "ConditionBinding":
        if not isinstance(profile, CameraProfile):
            raise TypeError("profile must be a CameraProfile")
        return cls(profile.profile_hash, profile.data_mode, tuple(realization_ids))

    def assert_profile(self, profile: CameraProfile) -> None:
        """Fail if ``profile`` is not the exact profile bound by this record."""

        if not isinstance(profile, CameraProfile):
            raise TypeError("profile must be a CameraProfile")
        if profile.profile_hash != self.fixed_profile_sha256:
            raise ValueError("condition is bound to a different camera profile")
        if profile.data_mode is not self.data_mode:
            raise ValueError("condition data mode does not match camera profile")

    def to_dict(self) -> dict[str, Any]:
        selection = {
            "ordered_realization_ids": list(self.realization_ids),
        }
        return {
            "fixed_profile_sha256": self.fixed_profile_sha256,
            "data_mode": self.data_mode.value,
            **selection,
            "realization_selection_sha256": canonical_sha256(selection),
        }

    @property
    def binding_hash(self) -> str:
        return canonical_sha256(self.to_dict())


class ExperimentCondition(ABC):
    """Common immutable/content-addressed condition interface."""

    __slots__ = ()

    binding: ConditionBinding

    @property
    @abstractmethod
    def family(self) -> ConditionFamily:
        raise NotImplementedError

    @property
    def fixed_profile_sha256(self) -> str:
        return self.binding.fixed_profile_sha256

    @property
    def data_mode(self) -> DataMode:
        return self.binding.data_mode

    @property
    def realization_ids(self) -> tuple[int, ...]:
        return self.binding.realization_ids

    @abstractmethod
    def identity_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def condition_hash(self) -> str:
        return canonical_sha256(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        return {**payload, "condition_sha256": canonical_sha256(payload)}


def _base_payload(
    condition: ExperimentCondition,
    *,
    coordinate_name: str,
    coordinate_unit: str,
    coordinate_value: float,
    fixed_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "record_type": "physical_experiment_condition",
        "family": condition.family.value,
        **condition.binding.to_dict(),
        "coordinate": {
            "name": coordinate_name,
            "unit": coordinate_unit,
            "value": coordinate_value,
        },
    }
    if fixed_parameters:
        payload["fixed_parameters"] = fixed_parameters
    return payload


@dataclass(frozen=True, slots=True)
class PhysicalDefocusCondition(ExperimentCondition):
    """One physical defocus coordinate in reference-edge waves."""

    binding: ConditionBinding
    edge_waves_ref: float

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ConditionBinding):
            raise TypeError("binding must be a ConditionBinding")
        object.__setattr__(
            self,
            "edge_waves_ref",
            finite_float(self.edge_waves_ref, field_name="edge_waves_ref"),
        )

    @property
    def family(self) -> ConditionFamily:
        return ConditionFamily.PHYSICAL_DEFOCUS

    @property
    def coordinate_value(self) -> float:
        return self.edge_waves_ref

    def identity_payload(self) -> dict[str, Any]:
        return _base_payload(
            self,
            coordinate_name="edge_waves_ref",
            coordinate_unit="waves_at_reference_wavelength",
            coordinate_value=self.edge_waves_ref,
        )


@dataclass(frozen=True, slots=True)
class RollingRotationCondition(ExperimentCondition):
    """One angular-speed magnitude about one fixed unit rotation axis."""

    binding: ConditionBinding
    angular_speed_rad_s: float
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ConditionBinding):
            raise TypeError("binding must be a ConditionBinding")
        speed = finite_float(self.angular_speed_rad_s, field_name="angular_speed_rad_s")
        if speed < 0.0:
            raise ValueError("angular_speed_rad_s is a magnitude and must be nonnegative")
        object.__setattr__(self, "angular_speed_rad_s", speed)
        object.__setattr__(self, "rotation_axis", _axis(self.rotation_axis))

    @classmethod
    def from_angular_velocity(
        cls,
        binding: ConditionBinding,
        angular_velocity_rad_s: Sequence[float],
        *,
        zero_axis: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> "RollingRotationCondition":
        if len(angular_velocity_rad_s) != 3:
            raise ValueError("angular_velocity_rad_s must contain exactly three values")
        vector = tuple(
            finite_float(value, field_name=f"angular_velocity_rad_s[{index}]")
            for index, value in enumerate(angular_velocity_rad_s)
        )
        magnitude = math.sqrt(sum(value * value for value in vector))
        axis = _axis(zero_axis if magnitude == 0.0 else vector)
        return cls(binding, magnitude, axis)

    @property
    def family(self) -> ConditionFamily:
        return ConditionFamily.ROLLING_ROTATION

    @property
    def coordinate_value(self) -> float:
        return self.angular_speed_rad_s

    @property
    def angular_velocity_rad_s(self) -> tuple[float, float, float]:
        return tuple(
            0.0 if self.angular_speed_rad_s == 0.0 else self.angular_speed_rad_s * value
            for value in self.rotation_axis
        )  # type: ignore[return-value]

    def identity_payload(self) -> dict[str, Any]:
        return _base_payload(
            self,
            coordinate_name="angular_speed",
            coordinate_unit="rad/s",
            coordinate_value=self.angular_speed_rad_s,
            fixed_parameters={"rotation_axis": list(self.rotation_axis)},
        )


@dataclass(frozen=True, slots=True)
class PhotonLossCondition(ExperimentCondition):
    """One photon-budget loss with explicit exposure and gain mechanisms."""

    binding: ConditionBinding
    photon_loss_stops: float
    exposure_policy: ExposurePolicy = ExposurePolicy.FIXED_DURATION_ATTENUATION
    gain_policy: GainPolicy = GainPolicy.FIXED_PROFILE

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ConditionBinding):
            raise TypeError("binding must be a ConditionBinding")
        if self.binding.data_mode is not DataMode.FORWARD_CAMERA_VALIDATION:
            raise ValueError("photon-loss conditions require FORWARD_CAMERA_VALIDATION mode")
        stops = finite_float(self.photon_loss_stops, field_name="photon_loss_stops")
        if stops < 0.0:
            raise ValueError("photon_loss_stops must be nonnegative")
        object.__setattr__(self, "photon_loss_stops", stops)
        object.__setattr__(self, "exposure_policy", ExposurePolicy(self.exposure_policy))
        object.__setattr__(self, "gain_policy", GainPolicy(self.gain_policy))

    @property
    def family(self) -> ConditionFamily:
        return ConditionFamily.PHOTON_LOSS

    @property
    def coordinate_value(self) -> float:
        return self.photon_loss_stops

    def identity_payload(self) -> dict[str, Any]:
        return _base_payload(
            self,
            coordinate_name="photon_loss",
            coordinate_unit="stops",
            coordinate_value=self.photon_loss_stops,
            fixed_parameters={
                "exposure_policy": self.exposure_policy.value,
                "gain_policy": self.gain_policy.value,
            },
        )


@dataclass(frozen=True, slots=True)
class ToneStopRatioCondition(ExperimentCondition):
    """One global tone-curve output-stops/input-stop ratio."""

    binding: ConditionBinding
    tone_stop_ratio: float

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ConditionBinding):
            raise TypeError("binding must be a ConditionBinding")
        ratio = finite_float(self.tone_stop_ratio, field_name="tone_stop_ratio")
        if not 0.0 < ratio <= 1.0:
            raise ValueError("tone_stop_ratio must lie in (0, 1]")
        object.__setattr__(self, "tone_stop_ratio", ratio)

    @property
    def family(self) -> ConditionFamily:
        return ConditionFamily.TONE_STOP_RATIO

    @property
    def coordinate_value(self) -> float:
        return self.tone_stop_ratio

    def identity_payload(self) -> dict[str, Any]:
        return _base_payload(
            self,
            coordinate_name="tone_stop_ratio",
            coordinate_unit="output_stops/input_stop",
            coordinate_value=self.tone_stop_ratio,
        )


OneKnobCondition: TypeAlias = (
    PhysicalDefocusCondition
    | RollingRotationCondition
    | PhotonLossCondition
    | ToneStopRatioCondition
)


@dataclass(frozen=True, slots=True)
class BaselineCondition(ExperimentCondition):
    """Untouched-input or modeled-neutral baseline identity."""

    binding: ConditionBinding
    kind: BaselineKind

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ConditionBinding):
            raise TypeError("binding must be a ConditionBinding")
        object.__setattr__(self, "kind", BaselineKind(self.kind))

    @property
    def family(self) -> ConditionFamily:
        return ConditionFamily.BASELINE

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "record_type": "physical_experiment_baseline",
            "family": self.family.value,
            "baseline_type": self.kind.value,
            **self.binding.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class MechanismComparatorCondition(ExperimentCondition):
    """One pre-declared comparator arm bound to a mechanism-match record."""

    binding: ConditionBinding
    match: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ConditionBinding):
            raise TypeError("binding must be a ConditionBinding")
        if self.binding.data_mode is not DataMode.LDR_REDEGRADATION:
            raise ValueError("mechanism comparator conditions require LDR_REDEGRADATION mode")
        if not isinstance(self.match, Mapping):
            raise TypeError("match must be a mechanism-match mapping")
        record = json_value(self.match)
        if record.get("schema_version") != 4 or record.get("record_type") != (
            "mechanism_mtf50_match"
        ):
            raise ValueError("match must be a schema-v4 mechanism_mtf50_match record")
        if set(record) != _MECHANISM_MATCH_V4_KEYS:
            raise ValueError("schema-v4 mechanism match has missing or unknown fields")
        if record.get("matching_contract") != _MECHANISM_MATCHING_CONTRACT_V4:
            raise ValueError("mechanism match uses a different formation or matching contract")
        supplied = record.get("match_sha256")
        payload = {key: value for key, value in record.items() if key != "match_sha256"}
        if supplied != canonical_sha256(payload):
            raise ValueError("match_sha256 does not match the mechanism-match payload")
        if record.get("camera_profile_sha256") != self.binding.fixed_profile_sha256:
            raise ValueError("mechanism match is bound to a different camera profile")
        family = record.get("comparator_family")
        if family not in _MATCH_FAMILY_OPERATOR:
            raise ValueError("mechanism match uses an unsupported comparator family")
        config = record.get("config")
        if not isinstance(config, Mapping) or record.get("config_sha256") != canonical_sha256(
            config
        ):
            raise ValueError("mechanism match has an invalid comparator config identity")
        if config.get("operator") != _MATCH_FAMILY_OPERATOR[family]:
            raise ValueError("mechanism match family and comparator operator disagree")
        if record.get("response_match_scope") != _MATCH_FAMILY_RESPONSE_SCOPE[family]:
            raise ValueError("mechanism match family and response scope disagree")
        for name in (
            "target_edge_waves_ref",
            "target_mtf50_cycles_per_pixel",
            "neutral_mtf_at_target",
            "achieved_mtf50_cycles_per_pixel",
            "relative_match_error",
        ):
            value = finite_float(record.get(name), field_name=f"mechanism match {name}")
            if name == "relative_match_error":
                if value < 0.0:
                    raise ValueError("mechanism match relative error must be nonnegative")
            elif value <= 0.0:
                raise ValueError(f"mechanism match {name} must be positive")
        for name in (
            "camera_profile_sha256",
            "neutral_model_sha256",
            "target_model_sha256",
            "config_sha256",
            "match_sha256",
        ):
            _sha256(record.get(name), field_name=f"mechanism match {name}")
        frozen = freeze_json_value(record)
        if not isinstance(frozen, Mapping):
            raise TypeError("match must be a mapping")
        object.__setattr__(self, "match", frozen)

    @property
    def family(self) -> ConditionFamily:
        return ConditionFamily.MECHANISM_COMPARATOR

    @property
    def comparator_family(self) -> str:
        return str(self.match["comparator_family"])

    @property
    def target_edge_waves_ref(self) -> float:
        return float(self.match["target_edge_waves_ref"])

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "record_type": "mechanism_comparator_condition",
            "family": self.family.value,
            **self.binding.to_dict(),
            "mechanism_match": json_value(self.match),
        }


@dataclass(frozen=True, slots=True)
class FactorialCondition(ExperimentCondition):
    """One physically ordered combination of distinct one-knob conditions."""

    components: tuple[OneKnobCondition, ...]

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components:
            raise ValueError("factorial condition must contain at least one component")
        if not all(
            isinstance(
                component,
                (
                    PhysicalDefocusCondition,
                    RollingRotationCondition,
                    PhotonLossCondition,
                    ToneStopRatioCondition,
                ),
            )
            for component in components
        ):
            raise TypeError("factorial components must be one-knob conditions")
        bindings = {component.binding for component in components}
        if len(bindings) != 1:
            raise ValueError("factorial components must share the exact condition binding")
        families = tuple(component.family for component in components)
        if len(set(families)) != len(families):
            raise ValueError("factorial components must use distinct condition families")
        ranks = tuple(_FAMILY_RANK[family] for family in families)
        if ranks != tuple(sorted(ranks)):
            expected = " -> ".join(family.value for family in PHYSICAL_FAMILY_ORDER)
            raise ValueError(f"factorial components must follow physical order: {expected}")
        object.__setattr__(self, "components", components)

    @property
    def binding(self) -> ConditionBinding:
        return self.components[0].binding

    @property
    def family(self) -> ConditionFamily:
        return ConditionFamily.FACTORIAL

    @property
    def active_families(self) -> tuple[ConditionFamily, ...]:
        return tuple(component.family for component in self.components)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "record_type": "physical_factorial_condition",
            "family": self.family.value,
            **self.binding.to_dict(),
            "physical_stage_order": [family.value for family in self.active_families],
            "components": [component.to_dict() for component in self.components],
        }


ConvertibleCondition: TypeAlias = (
    OneKnobCondition | BaselineCondition | MechanismComparatorCondition | FactorialCondition
)


def _components(condition: ConvertibleCondition) -> tuple[OneKnobCondition, ...]:
    if isinstance(condition, FactorialCondition):
        return condition.components
    if isinstance(
        condition,
        (
            PhysicalDefocusCondition,
            RollingRotationCondition,
            PhotonLossCondition,
            ToneStopRatioCondition,
        ),
    ):
        return (condition,)
    if isinstance(condition, BaselineCondition):
        if condition.kind is BaselineKind.UNTOUCHED_INPUT:
            raise ValueError("untouched-input baseline bypasses the camera graph")
        return ()
    if isinstance(condition, MechanismComparatorCondition):
        raise ValueError("mechanism comparator conditions require the comparator capture branch")
    raise TypeError("condition is not a convertible physical experiment condition")


def to_ldr_capture_severity(
    condition: ConvertibleCondition,
    *,
    profile: CameraProfile,
) -> LDRCaptureSeverity:
    """Convert a valid LDR condition into the rendered-camera API record."""

    if not isinstance(condition, ExperimentCondition):
        raise TypeError("condition must be an ExperimentCondition")
    condition.binding.assert_profile(profile)
    if condition.data_mode is not DataMode.LDR_REDEGRADATION:
        raise ValueError("condition is not bound to LDR_REDEGRADATION mode")
    edge = 0.0
    omega = (0.0, 0.0, 0.0)
    tone = 1.0
    for component in _components(condition):
        if isinstance(component, PhysicalDefocusCondition):
            edge = component.edge_waves_ref
        elif isinstance(component, RollingRotationCondition):
            omega = component.angular_velocity_rad_s
        elif isinstance(component, ToneStopRatioCondition):
            tone = component.tone_stop_ratio
        else:
            raise ValueError("photon-loss conditions are unavailable in the LDR tier")
    return LDRCaptureSeverity(edge, omega, tone)


def _selected_realization(
    binding: ConditionBinding,
    realization_id: int | None,
) -> int:
    if realization_id is None:
        if len(binding.realization_ids) != 1:
            raise ValueError("realization_id is required when multiple IDs are bound")
        return binding.realization_ids[0]
    if isinstance(realization_id, (bool, np.bool_)) or not isinstance(
        realization_id, (numbers.Integral, np.integer)
    ):
        raise TypeError("realization_id must be an integer")
    selected = int(realization_id)
    if selected not in binding.realization_ids:
        raise ValueError("realization_id was not declared in the condition binding")
    return selected


def to_forward_capture_condition(
    condition: ConvertibleCondition,
    *,
    profile: CameraProfile,
    realization_id: int | None = None,
    stochastic: bool = True,
    seed: int = 0,
    coupling_id: str | None = None,
) -> ForwardCaptureCondition:
    """Convert a valid forward condition into one stochastic realization."""

    if not isinstance(condition, ExperimentCondition):
        raise TypeError("condition must be an ExperimentCondition")
    condition.binding.assert_profile(profile)
    if condition.data_mode is not DataMode.FORWARD_CAMERA_VALIDATION:
        raise ValueError("condition is not bound to FORWARD_CAMERA_VALIDATION mode")
    realization = _selected_realization(condition.binding, realization_id)
    edge = 0.0
    omega = (0.0, 0.0, 0.0)
    loss = 0.0
    exposure = ExposurePolicy.FIXED_DURATION_ATTENUATION
    gain = GainPolicy.FIXED_PROFILE
    tone = 1.0
    for component in _components(condition):
        if isinstance(component, PhysicalDefocusCondition):
            edge = component.edge_waves_ref
        elif isinstance(component, RollingRotationCondition):
            omega = component.angular_velocity_rad_s
        elif isinstance(component, PhotonLossCondition):
            loss = component.photon_loss_stops
            exposure = component.exposure_policy
            gain = component.gain_policy
        elif isinstance(component, ToneStopRatioCondition):
            tone = component.tone_stop_ratio
    rng_coupling = condition.condition_hash if coupling_id is None else coupling_id
    return ForwardCaptureCondition(
        edge_waves_ref=edge,
        angular_velocity_rad_s=omega,
        photon_loss_stops=loss,
        exposure_policy=exposure,
        gain_policy=gain,
        tone_stop_ratio=tone,
        stochastic=stochastic,
        seed=seed,
        realization=realization,
        coupling_id=rng_coupling,
    )


def _binding_from_record(record: Mapping[str, Any]) -> ConditionBinding:
    ids = record.get("ordered_realization_ids")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
        raise TypeError("ordered_realization_ids must be an array")
    binding = ConditionBinding(
        record.get("fixed_profile_sha256"),
        record.get("data_mode"),
        tuple(ids),
    )
    expected_selection_hash = binding.to_dict()["realization_selection_sha256"]
    if record.get("realization_selection_sha256") != expected_selection_hash:
        raise ValueError("realization_selection_sha256 does not match realization IDs")
    return binding


def condition_from_dict(record: Mapping[str, Any]) -> ConvertibleCondition:
    """Validate and reconstruct a serialized condition, including fixed units."""

    if not isinstance(record, Mapping):
        raise TypeError("condition record must be a mapping")
    if record.get("schema_version") != 2:
        raise ValueError("condition record requires schema_version 2")
    supplied_hash = record.get("condition_sha256")
    if not isinstance(supplied_hash, str):
        raise ValueError("condition record is missing condition_sha256")
    payload = {key: value for key, value in record.items() if key != "condition_sha256"}
    if canonical_sha256(payload) != supplied_hash:
        raise ValueError("condition_sha256 does not match condition payload")
    binding = _binding_from_record(payload)
    record_type = payload.get("record_type")
    family = ConditionFamily(payload.get("family"))
    if record_type == "physical_experiment_baseline":
        if family is not ConditionFamily.BASELINE:
            raise ValueError("baseline record must use family='baseline'")
        result: ConvertibleCondition = BaselineCondition(
            binding, BaselineKind(payload.get("baseline_type"))
        )
    elif record_type == "mechanism_comparator_condition":
        if family is not ConditionFamily.MECHANISM_COMPARATOR:
            raise ValueError("mechanism comparator record has the wrong family")
        result = MechanismComparatorCondition(binding, payload.get("mechanism_match"))
    elif record_type == "physical_experiment_condition":
        coordinate = payload.get("coordinate")
        if not isinstance(coordinate, Mapping):
            raise TypeError("condition coordinate must be a mapping")
        name = coordinate.get("name")
        unit = coordinate.get("unit")
        value = coordinate.get("value")
        fixed = payload.get("fixed_parameters")
        if family is ConditionFamily.PHYSICAL_DEFOCUS:
            if (name, unit) != (
                "edge_waves_ref",
                "waves_at_reference_wavelength",
            ):
                raise ValueError("physical defocus coordinate name/unit mismatch")
            if fixed is not None:
                raise ValueError("physical defocus condition has unknown fixed parameters")
            result = PhysicalDefocusCondition(binding, value)
        elif family is ConditionFamily.ROLLING_ROTATION:
            if (name, unit) != ("angular_speed", "rad/s"):
                raise ValueError("rolling rotation coordinate name/unit mismatch")
            if not isinstance(fixed, Mapping) or set(fixed) != {"rotation_axis"}:
                raise ValueError("rolling rotation requires only rotation_axis")
            result = RollingRotationCondition(binding, value, fixed["rotation_axis"])
        elif family is ConditionFamily.PHOTON_LOSS:
            if (name, unit) != ("photon_loss", "stops"):
                raise ValueError("photon loss coordinate name/unit mismatch")
            if not isinstance(fixed, Mapping) or set(fixed) != {
                "exposure_policy",
                "gain_policy",
            }:
                raise ValueError("photon loss requires explicit exposure and gain policies")
            result = PhotonLossCondition(
                binding,
                value,
                fixed["exposure_policy"],
                fixed["gain_policy"],
            )
        elif family is ConditionFamily.TONE_STOP_RATIO:
            if (name, unit) != (
                "tone_stop_ratio",
                "output_stops/input_stop",
            ):
                raise ValueError("tone stop ratio coordinate name/unit mismatch")
            if fixed is not None:
                raise ValueError("tone condition has unknown fixed parameters")
            result = ToneStopRatioCondition(binding, value)
        else:
            raise ValueError("record does not contain a one-knob condition family")
    elif record_type == "physical_factorial_condition":
        if family is not ConditionFamily.FACTORIAL:
            raise ValueError("factorial record must use family='factorial'")
        raw_components = payload.get("components")
        if not isinstance(raw_components, Sequence) or isinstance(
            raw_components, (str, bytes, bytearray)
        ):
            raise TypeError("factorial components must be an array")
        components = tuple(condition_from_dict(item) for item in raw_components)
        if not all(
            isinstance(
                item,
                (
                    PhysicalDefocusCondition,
                    RollingRotationCondition,
                    PhotonLossCondition,
                    ToneStopRatioCondition,
                ),
            )
            for item in components
        ):
            raise TypeError("factorial record may contain only one-knob conditions")
        result = FactorialCondition(components)  # type: ignore[arg-type]
    else:
        raise ValueError("unknown physical experiment condition record_type")
    if result.identity_payload() != payload:
        raise ValueError("condition record has unknown or noncanonical fields")
    return result


__all__ = [
    "BaselineCondition",
    "BaselineKind",
    "ConditionBinding",
    "ConditionFamily",
    "ConvertibleCondition",
    "ExperimentCondition",
    "FactorialCondition",
    "MechanismComparatorCondition",
    "OneKnobCondition",
    "PHYSICAL_FAMILY_ORDER",
    "PhotonLossCondition",
    "PhysicalDefocusCondition",
    "RollingRotationCondition",
    "ToneStopRatioCondition",
    "condition_from_dict",
    "to_forward_capture_condition",
    "to_ldr_capture_severity",
]
