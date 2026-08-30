"""Stage specifications and physically ordered camera pipelines."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from ._canonical import canonical_sha256, nfc_string
from .domains import DataMode, Domain, domains_for_mode, is_legal_transition
from .frame import Frame
from .profiles import CameraProfile

StageOperation = Callable[[Frame], Frame]


def _required_string(value: str, *, field_name: str) -> str:
    result = nfc_string(value, field_name=field_name)
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


@dataclass(frozen=True, slots=True)
class StageSpec:
    """A declared, provenance-ready camera operation boundary.

    ``operation=None`` denotes a true identity and is accepted only when both
    domain and units remain unchanged.  A semantic/domain conversion must
    supply an operation that returns a newly tagged :class:`Frame`.
    """

    name: str
    input_domain: Domain
    output_domain: Domain
    input_units: str
    output_units: str
    deterministic: bool
    implementation_id: str
    neutral_condition: Optional[str]
    operation: Optional[StageOperation] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_string(self.name, field_name="name"))
        object.__setattr__(self, "input_domain", Domain(self.input_domain))
        object.__setattr__(self, "output_domain", Domain(self.output_domain))
        object.__setattr__(
            self,
            "input_units",
            _required_string(self.input_units, field_name="input_units"),
        )
        object.__setattr__(
            self,
            "output_units",
            _required_string(self.output_units, field_name="output_units"),
        )
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be bool")
        object.__setattr__(
            self,
            "implementation_id",
            _required_string(self.implementation_id, field_name="implementation_id"),
        )
        if self.neutral_condition is not None:
            object.__setattr__(
                self,
                "neutral_condition",
                _required_string(self.neutral_condition, field_name="neutral_condition"),
            )
        if self.operation is not None and not callable(self.operation):
            raise TypeError("operation must be callable or None")
        if self.operation is None and (
            self.input_domain is not self.output_domain or self.input_units != self.output_units
        ):
            raise ValueError("an operation is required when a stage changes domain or units")

    @classmethod
    def identity(
        cls,
        *,
        name: str,
        domain: Domain,
        units: str,
        implementation_id: str = "identity.python.v1",
    ) -> "StageSpec":
        """Build a deterministic numerical and semantic identity stage."""

        return cls(
            name=name,
            input_domain=domain,
            output_domain=domain,
            input_units=units,
            output_units=units,
            deterministic=True,
            implementation_id=implementation_id,
            neutral_condition="identity",
            operation=None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable stage description; executable code is excluded."""

        return {
            "name": self.name,
            "input_domain": self.input_domain.value,
            "output_domain": self.output_domain.value,
            "input_units": self.input_units,
            "output_units": self.output_units,
            "deterministic": self.deterministic,
            "implementation_id": self.implementation_id,
            "neutral_condition": self.neutral_condition,
        }


@dataclass(frozen=True, slots=True, init=False)
class CameraPipeline:
    """An immutable sequence of domain-checked physical camera stages."""

    data_mode: DataMode
    stages: tuple[StageSpec, ...]
    profile: Optional[CameraProfile]

    def __init__(
        self,
        stages: Sequence[StageSpec],
        *,
        data_mode: Optional[DataMode] = None,
        profile: Optional[CameraProfile] = None,
    ) -> None:
        if profile is not None and not isinstance(profile, CameraProfile):
            raise TypeError("profile must be a CameraProfile or None")
        if data_mode is None:
            if profile is None:
                raise TypeError("data_mode is required when profile is not supplied")
            resolved_mode = profile.data_mode
        else:
            resolved_mode = DataMode(data_mode)
        if profile is not None and profile.data_mode is not resolved_mode:
            raise ValueError("pipeline data_mode does not match the camera profile")

        resolved_stages = tuple(stages)
        if any(not isinstance(stage, StageSpec) for stage in resolved_stages):
            raise TypeError("every pipeline stage must be a StageSpec")

        legal_domains = domains_for_mode(resolved_mode)
        for index, stage in enumerate(resolved_stages):
            if stage.input_domain not in legal_domains or stage.output_domain not in legal_domains:
                raise ValueError(
                    f"stage {stage.name!r} uses a domain outside {resolved_mode.value!r}"
                )
            if not is_legal_transition(resolved_mode, stage.input_domain, stage.output_domain):
                raise ValueError(
                    f"stage {stage.name!r} declares illegal transition "
                    f"{stage.input_domain.value} -> {stage.output_domain.value} "
                    f"for {resolved_mode.value}"
                )
            if index:
                previous = resolved_stages[index - 1]
                if previous.output_domain is not stage.input_domain:
                    raise ValueError(
                        f"stage order mismatch: {previous.name!r} outputs "
                        f"{previous.output_domain.value}, but {stage.name!r} expects "
                        f"{stage.input_domain.value}"
                    )
                if previous.output_units != stage.input_units:
                    raise ValueError(
                        f"unit mismatch: {previous.name!r} outputs "
                        f"{previous.output_units!r}, but {stage.name!r} expects "
                        f"{stage.input_units!r}"
                    )

        object.__setattr__(self, "data_mode", resolved_mode)
        object.__setattr__(self, "stages", resolved_stages)
        object.__setattr__(self, "profile", profile)

    @property
    def input_domain(self) -> Optional[Domain]:
        return self.stages[0].input_domain if self.stages else None

    @property
    def output_domain(self) -> Optional[Domain]:
        return self.stages[-1].output_domain if self.stages else None

    @property
    def deterministic(self) -> bool:
        return all(stage.deterministic for stage in self.stages)

    @property
    def stage_graph_sha256(self) -> str:
        return canonical_sha256([stage.to_dict() for stage in self.stages])

    def run_trace(self, frame: Frame) -> tuple[Frame, ...]:
        """Execute and return the immutable input plus every stage boundary.

        The trace is the reference way to inspect physically meaningful
        intermediates (for example expected electrons, RAW DN, and signed
        camera-linear samples).  Stage operations still return only a
        :class:`Frame`, so no hidden mutable side channel is needed to expose
        diagnostics.
        """

        if not isinstance(frame, Frame):
            raise TypeError("pipeline input must be a Frame")
        if frame.metadata.data_mode is not self.data_mode:
            raise ValueError("input frame data mode does not match the pipeline")
        if not self.stages:
            return (frame,)
        first = self.stages[0]
        if frame.domain is not first.input_domain:
            raise ValueError(
                f"pipeline expects {first.input_domain.value}, got {frame.domain.value}"
            )
        if frame.metadata.units != first.input_units:
            raise ValueError(
                f"pipeline expects units {first.input_units!r}, got {frame.metadata.units!r}"
            )

        current = frame
        trace = [frame]
        for stage in self.stages:
            result = current if stage.operation is None else stage.operation(current)
            if not isinstance(result, Frame):
                raise TypeError(f"stage {stage.name!r} did not return a Frame")
            if result is current and (
                stage.input_domain is not stage.output_domain
                or stage.input_units != stage.output_units
            ):
                raise ValueError(
                    f"stage {stage.name!r} returned its input despite changing semantics"
                )
            if result.domain is not stage.output_domain:
                raise ValueError(
                    f"stage {stage.name!r} returned {result.domain.value}, expected "
                    f"{stage.output_domain.value}"
                )
            if result.metadata.units != stage.output_units:
                raise ValueError(
                    f"stage {stage.name!r} returned units "
                    f"{result.metadata.units!r}, expected {stage.output_units!r}"
                )
            if result.metadata.data_mode is not self.data_mode:
                raise ValueError(f"stage {stage.name!r} changed the data mode")
            current = result
            trace.append(current)
        return tuple(trace)

    def run(self, frame: Frame) -> Frame:
        """Execute all stages and return the validated final boundary."""

        return self.run_trace(frame)[-1]

    __call__ = run

    def stage_graph(self) -> list[dict[str, Any]]:
        """Return a fresh, JSON-compatible stage graph."""

        return [stage.to_dict() for stage in self.stages]

    def provenance_record(
        self,
        *,
        input_frame: Optional[Frame] = None,
        output_frame: Optional[Frame] = None,
        profile: Optional[CameraProfile] = None,
    ) -> dict[str, Any]:
        """Serialize schema-v2 profile, graph, and optional frame boundaries."""

        resolved_profile = self.profile if profile is None else profile
        if resolved_profile is None:
            raise ValueError("a CameraProfile is required for complete provenance")
        if resolved_profile.data_mode is not self.data_mode:
            raise ValueError("provenance profile data mode does not match pipeline")
        record: dict[str, Any] = {
            "schema_version": 2,
            "data_mode": self.data_mode.value,
            "camera_profile_sha256": resolved_profile.profile_hash,
            "camera_profile": resolved_profile.to_dict(),
            "stage_graph_sha256": self.stage_graph_sha256,
            "stage_graph": self.stage_graph(),
            "deterministic": self.deterministic,
            "input_frame": None if input_frame is None else input_frame.descriptor(),
            "output_frame": None if output_frame is None else output_frame.descriptor(),
        }
        return record

    def run_with_provenance(self, frame: Frame) -> tuple[Frame, dict[str, Any]]:
        """Execute and return a complete deterministic schema-v2 record."""

        output = self.run(frame)
        return output, self.provenance_record(input_frame=frame, output_frame=output)

    def run_trace_with_provenance(self, frame: Frame) -> tuple[tuple[Frame, ...], dict[str, Any]]:
        """Execute once and return all boundaries plus complete provenance."""

        trace = self.run_trace(frame)
        return trace, self.provenance_record(input_frame=trace[0], output_frame=trace[-1])

    def provenance_json(
        self,
        *,
        input_frame: Optional[Frame] = None,
        output_frame: Optional[Frame] = None,
        profile: Optional[CameraProfile] = None,
        indent: Optional[int] = 2,
    ) -> str:
        """Return stable human-readable JSON for a provenance record."""

        return json.dumps(
            self.provenance_record(
                input_frame=input_frame,
                output_frame=output_frame,
                profile=profile,
            ),
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )


__all__ = ["CameraPipeline", "StageOperation", "StageSpec"]
