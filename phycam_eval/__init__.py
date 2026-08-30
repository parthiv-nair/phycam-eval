"""Math-first physical camera forward models for robustness evaluation.

Python is the authoritative scientific backend.  The package exposes both a
clearly labeled LDR re-degradation approximation and an oversampled forward
RAW path whose optical, photosite, electron, ADC, and ISP boundaries are
domain checked and provenance ready.
"""

from .capture import LDRCaptureResult, LDRCaptureSeverity, render_ldr
from .forward_capture import (
    ForwardCaptureCondition,
    ForwardCaptureResult,
    render_forward,
)

__version__ = "0.1.0"

__all__ = [
    "ForwardCaptureCondition",
    "ForwardCaptureResult",
    "LDRCaptureResult",
    "LDRCaptureSeverity",
    "render_forward",
    "render_ldr",
]
