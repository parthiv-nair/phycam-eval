"""Immutable, domain-tagged image arrays for camera-pipeline boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ._canonical import finite_float, freeze_json_value, json_value, nfc_string, positive_float
from .domains import ColorSpace, DataMode, Domain, domains_for_mode


def _optional_pair(
    value: Optional[Sequence[float]], *, field_name: str, positive: bool
) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    validator = positive_float if positive else finite_float
    return (
        validator(value[0], field_name=f"{field_name}[0]"),
        validator(value[1], field_name=f"{field_name}[1]"),
    )


def _optional_quad(
    value: Optional[Sequence[float]], *, field_name: str
) -> Optional[tuple[float, float, float, float]]:
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError(f"{field_name} must contain exactly four values")
    result = tuple(
        finite_float(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)
    )
    if result[2] <= 0.0 or result[3] <= 0.0:
        raise ValueError(f"{field_name} width and height must be positive")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Physical metadata that cannot be inferred safely from array shape.

    ``units`` and ``data_mode`` are deliberately required.  Spatial fields may
    be ``None`` only when the current boundary has no declared physical grid
    (for example, the original display-referred input).  A source-grid stage
    must populate them before an operator that needs metric sampling.
    """

    units: str
    data_mode: DataMode
    sample_spacing_m: Optional[tuple[float, float]] = None
    sensor_origin_m: Optional[tuple[float, float]] = None
    sensor_window_m: Optional[tuple[float, float, float, float]] = None
    channel_names: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        units = nfc_string(self.units, field_name="units")
        if not units:
            raise ValueError("units must not be empty")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "data_mode", DataMode(self.data_mode))
        object.__setattr__(
            self,
            "sample_spacing_m",
            _optional_pair(
                self.sample_spacing_m,
                field_name="sample_spacing_m",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "sensor_origin_m",
            _optional_pair(
                self.sensor_origin_m,
                field_name="sensor_origin_m",
                positive=False,
            ),
        )
        object.__setattr__(
            self,
            "sensor_window_m",
            _optional_quad(self.sensor_window_m, field_name="sensor_window_m"),
        )
        names = tuple(
            nfc_string(name, field_name=f"channel_names[{index}]")
            for index, name in enumerate(self.channel_names)
        )
        if any(not name for name in names):
            raise ValueError("channel_names must not contain empty strings")
        if len(set(names)) != len(names):
            raise ValueError("channel_names must be unique")
        object.__setattr__(self, "channel_names", names)

        frozen_attributes = freeze_json_value(self.attributes)
        if not isinstance(frozen_attributes, MappingProxyType):
            raise TypeError("attributes must be a mapping")
        object.__setattr__(self, "attributes", frozen_attributes)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata without exposing mutable internals."""

        return {
            "units": self.units,
            "data_mode": self.data_mode.value,
            "sample_spacing_m": json_value(self.sample_spacing_m),
            "sensor_origin_m": json_value(self.sensor_origin_m),
            "sensor_window_m": json_value(self.sensor_window_m),
            "channel_names": list(self.channel_names),
            "attributes": json_value(self.attributes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrameMetadata":
        """Construct metadata from its serialized representation."""

        return cls(
            units=value["units"],
            data_mode=DataMode(value["data_mode"]),
            sample_spacing_m=value.get("sample_spacing_m"),
            sensor_origin_m=value.get("sensor_origin_m"),
            sensor_window_m=value.get("sensor_window_m"),
            channel_names=tuple(value.get("channel_names", ())),
            attributes=value.get("attributes", {}),
        )

    def with_units(self, units: str) -> "FrameMetadata":
        """Return a copy carrying the units at a new stage boundary."""

        value = self.to_dict()
        value["units"] = units
        return type(self).from_dict(value)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Frame:
    """An immutable owned NumPy array with explicit semantic metadata.

    Construction always takes a defensive C-order copy.  The stored array is
    backed by immutable ``bytes`` rather than merely setting NumPy's writable
    flag, so callers cannot re-enable writes with ``setflags(write=True)``.
    A stage that changes samples or semantics constructs a new ``Frame``.
    """

    _array: np.ndarray
    domain: Domain
    color_space: ColorSpace
    metadata: FrameMetadata

    def __init__(
        self,
        array: np.ndarray,
        domain: Domain,
        color_space: ColorSpace,
        metadata: FrameMetadata,
    ) -> None:
        if not isinstance(metadata, FrameMetadata):
            raise TypeError("metadata must be a FrameMetadata instance")
        resolved_domain = Domain(domain)
        if resolved_domain not in domains_for_mode(metadata.data_mode):
            raise ValueError(
                f"domain {resolved_domain.value!r} is not legal for "
                f"mode {metadata.data_mode.value!r}"
            )

        source = np.asarray(array)
        if source.ndim < 2:
            raise ValueError("frame arrays must have at least two dimensions")
        if source.dtype.hasobject:
            raise TypeError("frame arrays cannot have object dtype")

        contiguous = np.array(source, copy=True, order="C", subok=False)
        immutable_storage = contiguous.tobytes(order="C")
        immutable = np.frombuffer(immutable_storage, dtype=contiguous.dtype).reshape(
            contiguous.shape
        )

        if metadata.channel_names:
            if immutable.ndim < 3:
                raise ValueError("channel_names require a channel dimension")
            if len(metadata.channel_names) != immutable.shape[-1]:
                raise ValueError("channel_names length must match the final array dimension")

        object.__setattr__(self, "_array", immutable)
        object.__setattr__(self, "domain", resolved_domain)
        object.__setattr__(self, "color_space", ColorSpace(color_space))
        object.__setattr__(self, "metadata", metadata)

    @property
    def array(self) -> np.ndarray:
        """Read-only owned samples; create a new frame to change them."""

        return self._array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    @property
    def ndim(self) -> int:
        return self._array.ndim

    def with_array(
        self,
        array: np.ndarray,
        *,
        domain: Optional[Domain] = None,
        color_space: Optional[ColorSpace] = None,
        metadata: Optional[FrameMetadata] = None,
    ) -> "Frame":
        """Construct the next immutable frame while retaining explicit tags."""

        return type(self)(
            array=array,
            domain=self.domain if domain is None else domain,
            color_space=self.color_space if color_space is None else color_space,
            metadata=self.metadata if metadata is None else metadata,
        )

    def descriptor(self) -> dict[str, Any]:
        """Return the serializable, non-pixel portion of the frame contract."""

        return {
            "domain": self.domain.value,
            "color_space": self.color_space.value,
            "shape": list(self.shape),
            "dtype": self.dtype.str,
            "metadata": self.metadata.to_dict(),
        }
