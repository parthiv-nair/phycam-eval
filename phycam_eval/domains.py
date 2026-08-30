"""Semantic array domains used by the physical camera stage graphs.

Domains describe what an array *means*, not merely its shape or dtype.  The
forward-camera and LDR re-degradation graphs intentionally have different
intermediate domains; shared output/ISP domains are represented by the same
enum members.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet


class DataMode(str, Enum):
    """Supported interpretations of an input image."""

    LDR_REDEGRADATION = "ldr_redegradation"
    FORWARD_CAMERA_VALIDATION = "forward_camera_validation"


class ColorSpace(str, Enum):
    """Declared color encoding or sensor interpretation of a frame."""

    NONE = "none"
    SRGB = "srgb"
    LINEAR_SRGB = "linear_srgb"
    SCENE_SPECTRAL = "scene_spectral"
    CAMERA_NATIVE = "camera_native"
    CAMERA_LINEAR_RGB = "camera_linear_rgb"
    OUTPUT_LINEAR_RGB = "output_linear_rgb"
    RAW_MOSAIC = "raw_mosaic"


class Domain(str, Enum):
    """Physical/semantic domain at a camera-pipeline boundary."""

    # Common rendered-output domains.
    DISPLAY_RGB = "display_rgb"
    OUTPUT_LINEAR_RGB_SIGNED = "output_linear_rgb_signed"
    OUTPUT_LINEAR_RGB_NONNEGATIVE = "output_linear_rgb_nonnegative"
    TONE_MAPPED_LINEAR_RGB = "tone_mapped_linear_rgb"
    OUTPUT_GAMUT_LINEAR_RGB = "output_gamut_linear_rgb"
    DETECTOR_INPUT = "detector_input"

    # LDR re-degradation path.
    LINEAR_RGB_PROXY = "linear_rgb_proxy"
    SOURCE_GRID_LINEAR_RGB = "source_grid_linear_rgb"
    LINEAR_RGB_OPTICAL = "linear_rgb_optical"
    LINEAR_RGB_PHOTOSITE_EXPECTATION = "linear_rgb_photosite_expectation"
    LINEAR_RGB_ELECTRONS = "linear_rgb_electrons"
    LINEAR_RGB_ADC_DN = "linear_rgb_adc_dn"
    SIGNED_CAMERA_LINEAR_RGB = "signed_camera_linear_rgb"

    # Forward RAW/HDR path.
    SCENE_SPECTRAL = "scene_spectral"
    OVERSAMPLED_SCENE_LINEAR_WITH_DECLARED_SPECTRAL_ADAPTER = (
        "oversampled_scene_linear_with_declared_spectral_adapter"
    )
    INSTANTANEOUS_SCENE_PROJECTION = "instantaneous_scene_projection"
    OPTICAL_SENSOR_IRRADIANCE = "optical_sensor_irradiance"
    PHOTOSITE_EXPECTATION = "photosite_expectation"
    ELECTRONS = "electrons"
    RAW_ADC_DN = "raw_adc_dn"
    SIGNED_CAMERA_LINEAR = "signed_camera_linear"
    CAMERA_LINEAR_RGB = "camera_linear_rgb"

    # Scientific diagnostics are valid array semantics but are never
    # legal Frame boundaries in a camera pipeline.
    OPTICAL_PUPIL_DIAGNOSTIC = "optical_pupil_diagnostic"
    OPTICAL_PSF_DIAGNOSTIC = "optical_psf_diagnostic"
    OPTICAL_OTF_DIAGNOSTIC = "optical_otf_diagnostic"
    OPTICAL_KERNEL_DIAGNOSTIC = "optical_kernel_diagnostic"
    RNG_DIAGNOSTIC = "rng_diagnostic"


DIAGNOSTIC_DOMAINS: FrozenSet[Domain] = frozenset(
    {
        Domain.OPTICAL_PUPIL_DIAGNOSTIC,
        Domain.OPTICAL_PSF_DIAGNOSTIC,
        Domain.OPTICAL_OTF_DIAGNOSTIC,
        Domain.OPTICAL_KERNEL_DIAGNOSTIC,
        Domain.RNG_DIAGNOSTIC,
    }
)


_SHARED_DOMAINS = frozenset(
    {
        Domain.DISPLAY_RGB,
        Domain.OUTPUT_LINEAR_RGB_SIGNED,
        Domain.OUTPUT_LINEAR_RGB_NONNEGATIVE,
        Domain.TONE_MAPPED_LINEAR_RGB,
        Domain.OUTPUT_GAMUT_LINEAR_RGB,
        Domain.DETECTOR_INPUT,
    }
)

LDR_DOMAINS: FrozenSet[Domain] = _SHARED_DOMAINS | frozenset(
    {
        Domain.LINEAR_RGB_PROXY,
        Domain.SOURCE_GRID_LINEAR_RGB,
        Domain.LINEAR_RGB_OPTICAL,
        Domain.LINEAR_RGB_PHOTOSITE_EXPECTATION,
        Domain.LINEAR_RGB_ELECTRONS,
        Domain.LINEAR_RGB_ADC_DN,
        Domain.SIGNED_CAMERA_LINEAR_RGB,
    }
)

FORWARD_CAMERA_DOMAINS: FrozenSet[Domain] = _SHARED_DOMAINS | frozenset(
    {
        Domain.SCENE_SPECTRAL,
        Domain.OVERSAMPLED_SCENE_LINEAR_WITH_DECLARED_SPECTRAL_ADAPTER,
        Domain.INSTANTANEOUS_SCENE_PROJECTION,
        Domain.OPTICAL_SENSOR_IRRADIANCE,
        Domain.PHOTOSITE_EXPECTATION,
        Domain.ELECTRONS,
        Domain.RAW_ADC_DN,
        Domain.SIGNED_CAMERA_LINEAR,
        Domain.CAMERA_LINEAR_RGB,
    }
)


# These edges encode the normative stage order.  Same-domain stages are
# separately allowed so that operations such as full-well clipping can retain
# a domain while still appearing explicitly in provenance.
LDR_TRANSITIONS: FrozenSet[tuple[Domain, Domain]] = frozenset(
    {
        (Domain.DISPLAY_RGB, Domain.LINEAR_RGB_PROXY),
        (Domain.LINEAR_RGB_PROXY, Domain.SOURCE_GRID_LINEAR_RGB),
        (Domain.SOURCE_GRID_LINEAR_RGB, Domain.LINEAR_RGB_OPTICAL),
        (Domain.LINEAR_RGB_OPTICAL, Domain.LINEAR_RGB_PHOTOSITE_EXPECTATION),
        (
            Domain.LINEAR_RGB_PHOTOSITE_EXPECTATION,
            Domain.LINEAR_RGB_ELECTRONS,
        ),
        (Domain.LINEAR_RGB_ELECTRONS, Domain.LINEAR_RGB_ADC_DN),
        (Domain.LINEAR_RGB_ADC_DN, Domain.SIGNED_CAMERA_LINEAR_RGB),
        (Domain.SIGNED_CAMERA_LINEAR_RGB, Domain.OUTPUT_LINEAR_RGB_SIGNED),
        # Optics-only studies and the default LDR tier may bypass the optional
        # electron approximation while preserving its explicit graph identity.
        (Domain.LINEAR_RGB_OPTICAL, Domain.OUTPUT_LINEAR_RGB_SIGNED),
        (
            Domain.OUTPUT_LINEAR_RGB_SIGNED,
            Domain.OUTPUT_LINEAR_RGB_NONNEGATIVE,
        ),
        (
            Domain.OUTPUT_LINEAR_RGB_NONNEGATIVE,
            Domain.TONE_MAPPED_LINEAR_RGB,
        ),
        (Domain.TONE_MAPPED_LINEAR_RGB, Domain.OUTPUT_GAMUT_LINEAR_RGB),
        (Domain.OUTPUT_GAMUT_LINEAR_RGB, Domain.DISPLAY_RGB),
        (Domain.DISPLAY_RGB, Domain.DETECTOR_INPUT),
    }
)

FORWARD_CAMERA_TRANSITIONS: FrozenSet[tuple[Domain, Domain]] = frozenset(
    {
        (Domain.SCENE_SPECTRAL, Domain.INSTANTANEOUS_SCENE_PROJECTION),
        (
            Domain.OVERSAMPLED_SCENE_LINEAR_WITH_DECLARED_SPECTRAL_ADAPTER,
            Domain.INSTANTANEOUS_SCENE_PROJECTION,
        ),
        (
            Domain.INSTANTANEOUS_SCENE_PROJECTION,
            Domain.OPTICAL_SENSOR_IRRADIANCE,
        ),
        (Domain.OPTICAL_SENSOR_IRRADIANCE, Domain.PHOTOSITE_EXPECTATION),
        # A joint projection/optics/exposure/CFA implementation is normative
        # and can expose only its final photosite expectation boundary.
        (Domain.SCENE_SPECTRAL, Domain.PHOTOSITE_EXPECTATION),
        (
            Domain.OVERSAMPLED_SCENE_LINEAR_WITH_DECLARED_SPECTRAL_ADAPTER,
            Domain.PHOTOSITE_EXPECTATION,
        ),
        (Domain.PHOTOSITE_EXPECTATION, Domain.ELECTRONS),
        (Domain.ELECTRONS, Domain.RAW_ADC_DN),
        (Domain.RAW_ADC_DN, Domain.SIGNED_CAMERA_LINEAR),
        (Domain.SIGNED_CAMERA_LINEAR, Domain.CAMERA_LINEAR_RGB),
        (Domain.CAMERA_LINEAR_RGB, Domain.OUTPUT_LINEAR_RGB_SIGNED),
        (
            Domain.OUTPUT_LINEAR_RGB_SIGNED,
            Domain.OUTPUT_LINEAR_RGB_NONNEGATIVE,
        ),
        (
            Domain.OUTPUT_LINEAR_RGB_NONNEGATIVE,
            Domain.TONE_MAPPED_LINEAR_RGB,
        ),
        (Domain.TONE_MAPPED_LINEAR_RGB, Domain.OUTPUT_GAMUT_LINEAR_RGB),
        (Domain.OUTPUT_GAMUT_LINEAR_RGB, Domain.DISPLAY_RGB),
        (Domain.DISPLAY_RGB, Domain.DETECTOR_INPUT),
    }
)


def domains_for_mode(mode: DataMode) -> FrozenSet[Domain]:
    """Return the domains that are legal in ``mode``."""

    mode = DataMode(mode)
    if mode is DataMode.LDR_REDEGRADATION:
        return LDR_DOMAINS
    return FORWARD_CAMERA_DOMAINS


def is_legal_transition(mode: DataMode, source: Domain, target: Domain) -> bool:
    """Return whether one declared stage transition is physically ordered."""

    mode = DataMode(mode)
    source = Domain(source)
    target = Domain(target)
    if source is target:
        return source in domains_for_mode(mode)
    transitions = (
        LDR_TRANSITIONS if mode is DataMode.LDR_REDEGRADATION else FORWARD_CAMERA_TRANSITIONS
    )
    return (source, target) in transitions
