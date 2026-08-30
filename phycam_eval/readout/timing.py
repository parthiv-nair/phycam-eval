"""Global- and rolling-shutter row timing contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ReadoutTiming:
    """Timing for a sensor whose rows expose sequentially.

    ``line_time_s == 0`` represents a global shutter.  Physical capture always
    requires a positive exposure duration; instantaneous geometry is exposed
    by a separate renderer as the ``exposure_s -> 0+`` limit.
    """

    height: int
    line_time_s: float
    exposure_s: float
    frame_start_s: float = 0.0
    reference_time_s: float | None = None
    annotation_time_s: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.height, bool) or not isinstance(self.height, (int, np.integer)):
            raise TypeError("height must be an integer")
        if int(self.height) <= 0:
            raise ValueError("height must be positive")
        object.__setattr__(self, "height", int(self.height))

        for name in ("line_time_s", "exposure_s", "frame_start_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.line_time_s < 0.0:
            raise ValueError("line_time_s must be nonnegative")
        if self.exposure_s <= 0.0:
            raise ValueError("exposure_s must be positive")

        default_reference = (
            self.frame_start_s + 0.5 * (self.height - 1) * self.line_time_s + 0.5 * self.exposure_s
        )
        reference = (
            default_reference if self.reference_time_s is None else float(self.reference_time_s)
        )
        annotation = reference if self.annotation_time_s is None else float(self.annotation_time_s)
        if not np.isfinite(reference) or not np.isfinite(annotation):
            raise ValueError("reference and annotation times must be finite")
        object.__setattr__(self, "reference_time_s", reference)
        object.__setattr__(self, "annotation_time_s", annotation)

    @property
    def is_global(self) -> bool:
        return self.line_time_s == 0.0

    @property
    def frame_end_s(self) -> float:
        return self.frame_start_s + (self.height - 1) * self.line_time_s + self.exposure_s

    def row_start_s(self, row: int) -> float:
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
            raise TypeError("row must be an integer")
        row = int(row)
        if row < 0 or row >= self.height:
            raise IndexError("row is outside the sensor")
        return self.frame_start_s + row * self.line_time_s

    def row_starts_s(self) -> NDArray[np.float64]:
        return self.frame_start_s + np.arange(self.height, dtype=np.float64) * self.line_time_s

    def row_midpoints_s(self) -> NDArray[np.float64]:
        return self.row_starts_s() + 0.5 * self.exposure_s
