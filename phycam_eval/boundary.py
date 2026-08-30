"""Shared outside-frame continuation semantics for camera operators.

The numerical libraries used by the reference implementation do not attach
the same meaning to the string ``"reflect"``.  NumPy padding calls
whole-sample reflection ``"reflect"``, while ``scipy.ndimage`` calls the same
continuation ``"mirror"``.  This module gives profiles and stage builders one
physical contract and owns those backend-specific translations.

Geometry-changing valid convolution/cropping is deliberately absent.  It
cannot be represented by a padding string alone because it must also update
the sensor window, intrinsics, readout geometry, and frame metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from ._canonical import canonical_sha256

ConvolutionBoundary = Literal["reflect", "zero", "constant"]
WarpBoundary = Literal["mirror", "constant"]


class BoundaryMode(str, Enum):
    """Physical continuation outside the declared image window."""

    REFLECT = "reflect"
    ZERO = "zero"
    CONSTANT = "constant"


_GEOMETRY_CHANGING_MODES = frozenset({"valid", "crop", "valid_crop", "valid-crop", "valid crop"})


@dataclass(frozen=True, slots=True)
class BoundaryContract:
    """Immutable, content-addressable boundary continuation contract.

    ``reflect`` means whole-sample symmetry: the edge sample is not repeated.
    ``zero`` is zero radiance outside the frame.  ``constant`` uses the
    explicitly stored finite value, in the same units as the operated frame.

    Nonzero ``constant_value`` is meaningful only for ``constant``.  Rejecting
    it for the other modes ensures that numerically identical contracts cannot
    acquire different stage identities through an ignored parameter.
    """

    mode: BoundaryMode = BoundaryMode.REFLECT
    constant_value: float = 0.0

    def __post_init__(self) -> None:
        raw_mode = self.mode
        try:
            mode = BoundaryMode(raw_mode)
        except (TypeError, ValueError) as exc:
            normalized = str(raw_mode).strip().lower().replace("/", "_")
            if normalized in _GEOMETRY_CHANGING_MODES:
                raise ValueError(
                    "valid/crop boundary modes require geometry-aware crop "
                    "support and are not available in BoundaryContract"
                ) from exc
            choices = ", ".join(item.value for item in BoundaryMode)
            raise ValueError(
                f"unsupported boundary mode {raw_mode!r}; expected one of: {choices}"
            ) from exc

        if isinstance(self.constant_value, bool):
            raise TypeError("constant_value must be a finite real number")
        try:
            constant = float(self.constant_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("constant_value must be a finite real number") from exc
        if not math.isfinite(constant):
            raise ValueError("constant_value must be finite")
        if constant == 0.0:
            constant = 0.0
        if mode is not BoundaryMode.CONSTANT and constant != 0.0:
            raise ValueError("constant_value must be zero unless mode is 'constant'")

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "constant_value", constant)

    @classmethod
    def reflect(cls) -> "BoundaryContract":
        """Construct whole-sample reflection continuation."""

        return cls(BoundaryMode.REFLECT, 0.0)

    @classmethod
    def zero(cls) -> "BoundaryContract":
        """Construct zero-radiance continuation."""

        return cls(BoundaryMode.ZERO, 0.0)

    @classmethod
    def constant(cls, value: float) -> "BoundaryContract":
        """Construct continuation by an explicit finite constant."""

        return cls(BoundaryMode.CONSTANT, value)

    @property
    def convolution_mode(self) -> ConvolutionBoundary:
        """Boundary string expected by padded optical convolution."""

        if self.mode is BoundaryMode.REFLECT:
            return "reflect"
        if self.mode is BoundaryMode.ZERO:
            return "zero"
        return "constant"

    @property
    def warp_mode(self) -> WarpBoundary:
        """Boundary string with matching semantics for ``scipy.ndimage``."""

        if self.mode is BoundaryMode.REFLECT:
            # scipy.ndimage's `mirror` is whole-sample reflection, matching
            # numpy.pad(..., mode="reflect") used by optical convolution.
            return "mirror"
        return "constant"

    @property
    def warp_cval(self) -> float:
        """Outside-frame value supplied to ``scipy.ndimage`` warping."""

        return self.constant_value

    def convolution_kwargs(self) -> dict[str, str | float]:
        """Return fresh keyword arguments for optical convolution."""

        return {
            "boundary": self.convolution_mode,
            "constant_value": self.constant_value,
        }

    def warp_kwargs(self) -> dict[str, str | float]:
        """Return fresh keyword arguments for homography warping."""

        return {"boundary": self.warp_mode, "cval": self.warp_cval}

    def to_dict(self) -> dict[str, Any]:
        """Return the stable physical semantics used in stage identity."""

        return {
            "schema_version": 1,
            "mode": self.mode.value,
            "constant_value": self.constant_value,
            "reflection_convention": (
                "whole_sample" if self.mode is BoundaryMode.REFLECT else None
            ),
        }

    @property
    def sha256(self) -> str:
        """Canonical content hash suitable for a stage implementation ID."""

        return canonical_sha256(self.to_dict())


__all__ = [
    "BoundaryContract",
    "BoundaryMode",
    "ConvolutionBoundary",
    "WarpBoundary",
]
