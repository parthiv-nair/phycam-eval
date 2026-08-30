"""Paraxial conversions between physical focus error and pupil defocus waves.

The Fourier-optics core uses signed edge-to-center optical path difference in
waves.  Camera and bench measurements more often report an axial image-plane
offset, object distance, focal length, or circle-of-confusion diameter.  This
module makes the approximation and sign convention joining those quantities
explicit; it does not pretend that a photographic focus motor is a calibrated
focal-length actuator.
"""

from __future__ import annotations

import math


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def thin_lens_image_distance_m(
    *,
    focal_length_m: float,
    object_distance_m: float,
) -> float:
    """Return the real paraxial image distance for a thin lens in air.

    Distances are positive magnitudes under ``1/f = 1/s + 1/v``.  The object
    must lie beyond the focal plane so that the returned image is real.
    """

    focal_length = _positive_finite(focal_length_m, name="focal_length_m")
    object_distance = _positive_finite(object_distance_m, name="object_distance_m")
    if object_distance <= focal_length:
        raise ValueError("object_distance_m must exceed focal_length_m for a real image")
    return focal_length * object_distance / (object_distance - focal_length)


def paraxial_edge_waves_from_conjugates(
    *,
    object_distance_m: float,
    image_distance_m: float,
    focal_length_m: float,
    pupil_radius_m: float,
    wavelength_m: float,
) -> float:
    r"""Return signed pupil-edge defocus waves from exact paraxial geometry.

    Distances are positive magnitudes for a real on-axis object and image in
    air.  The residual vergence and edge-to-center optical-path difference are

    .. math::

       \epsilon = 1/u + 1/v_s - 1/f,\qquad
       W_\lambda = a^2\epsilon/(2\lambda).

    This is the exact thin-lens bridge used to define the paper's sign and
    unit convention; it is not a calibrated model of a compound camera lens.
    """

    object_distance = _positive_finite(object_distance_m, name="object_distance_m")
    image_distance = _positive_finite(image_distance_m, name="image_distance_m")
    focal_length = _positive_finite(focal_length_m, name="focal_length_m")
    pupil_radius = _positive_finite(pupil_radius_m, name="pupil_radius_m")
    wavelength = _positive_finite(wavelength_m, name="wavelength_m")
    residual_vergence = math.fsum(
        (1.0 / object_distance, 1.0 / image_distance, -1.0 / focal_length)
    )
    return pupil_radius * pupil_radius * residual_vergence / (2.0 * wavelength)


def image_plane_offset_from_focal_length_change_m(
    *,
    nominal_focal_length_m: float,
    perturbed_focal_length_m: float,
    object_distance_m: float,
) -> float:
    """Return sensor-plane minus perturbed best-focus distance in meters.

    The sensor is assumed fixed at the image distance of the nominal thin
    lens.  A positive result means the sensor lies behind the perturbed
    paraxial focus.  Real photographic lenses generally focus by moving lens
    groups, so this conversion is valid only when effective focal length is the
    physically calibrated actuator being perturbed.
    """

    nominal = thin_lens_image_distance_m(
        focal_length_m=nominal_focal_length_m,
        object_distance_m=object_distance_m,
    )
    perturbed = thin_lens_image_distance_m(
        focal_length_m=perturbed_focal_length_m,
        object_distance_m=object_distance_m,
    )
    return nominal - perturbed


def image_plane_offset_from_object_distance_change_m(
    *,
    focal_length_m: float,
    focused_object_distance_m: float,
    perturbed_object_distance_m: float,
) -> float:
    """Return sensor-plane minus perturbed best-focus distance in meters."""

    sensor_distance = thin_lens_image_distance_m(
        focal_length_m=focal_length_m,
        object_distance_m=focused_object_distance_m,
    )
    perturbed_focus = thin_lens_image_distance_m(
        focal_length_m=focal_length_m,
        object_distance_m=perturbed_object_distance_m,
    )
    return sensor_distance - perturbed_focus


def paraxial_edge_waves_from_image_plane_offset(
    image_plane_offset_m: float,
    *,
    f_number: float,
    wavelength_m: float,
) -> float:
    r"""Convert a small axial image-plane offset to signed pupil-edge waves.

    ``image_plane_offset_m`` is sensor position minus best-focus position.  In
    the declared Fresnel convention a sensor behind focus has negative pupil
    defocus:

    .. math:: W_{edge} \simeq -\Delta z/(8\lambda N^2).

    This is a scalar, paraxial, small-offset relation in air using the
    image-side f-number ``N``.  Only the magnitude affects the ideal circular
    pupil's radially symmetric incoherent PSF in this approximation.
    """

    offset = _finite(image_plane_offset_m, name="image_plane_offset_m")
    f_stop = _positive_finite(f_number, name="f_number")
    wavelength = _positive_finite(wavelength_m, name="wavelength_m")
    return -offset / (8.0 * wavelength * f_stop * f_stop)


def paraxial_image_plane_offset_from_edge_waves(
    edge_waves: float,
    *,
    f_number: float,
    wavelength_m: float,
) -> float:
    """Invert :func:`paraxial_edge_waves_from_image_plane_offset`."""

    waves = _finite(edge_waves, name="edge_waves")
    f_stop = _positive_finite(f_number, name="f_number")
    wavelength = _positive_finite(wavelength_m, name="wavelength_m")
    return -8.0 * wavelength * f_stop * f_stop * waves


def paraxial_circle_of_confusion_diameter_m(
    image_plane_offset_m: float,
    *,
    f_number: float,
) -> float:
    r"""Return the geometrical blur-disk diameter ``|Delta z|/N`` in meters.

    This ray-optical scale is a diagnostic for sufficiently large defocus; it
    is not a replacement for the diffraction PSF at small defocus.
    """

    offset = _finite(image_plane_offset_m, name="image_plane_offset_m")
    f_stop = _positive_finite(f_number, name="f_number")
    return abs(offset) / f_stop


def airy_first_zero_radius_m(*, f_number: float, wavelength_m: float) -> float:
    """Return the clear-circular-pupil Airy first-zero radius in meters."""

    f_stop = _positive_finite(f_number, name="f_number")
    wavelength = _positive_finite(wavelength_m, name="wavelength_m")
    return 1.2196698912665045 * wavelength * f_stop


def ideal_defocus_on_axis_intensity_ratio(edge_waves: float) -> float:
    r"""Return the analytic on-axis intensity ratio for pure quadratic defocus.

    For a uniformly illuminated clear circular pupil with phase
    ``2*pi*W*rho**2``, normalized on-axis intensity is
    ``(sin(pi W)/(pi W))**2 = sinc(W)**2``.  The limit at zero is exactly one.
    """

    waves = _finite(edge_waves, name="edge_waves")
    if waves == 0.0:
        return 1.0
    value = math.sin(math.pi * waves) / (math.pi * waves)
    return value * value


__all__ = [
    "airy_first_zero_radius_m",
    "ideal_defocus_on_axis_intensity_ratio",
    "image_plane_offset_from_focal_length_change_m",
    "image_plane_offset_from_object_distance_change_m",
    "paraxial_circle_of_confusion_diameter_m",
    "paraxial_edge_waves_from_conjugates",
    "paraxial_edge_waves_from_image_plane_offset",
    "paraxial_image_plane_offset_from_edge_waves",
    "thin_lens_image_distance_m",
]
