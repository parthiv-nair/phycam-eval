"""Optical pupil, PSF, OTF, focus conversion, and scene-grid sampling stages."""

from .focus import (
    airy_first_zero_radius_m,
    ideal_defocus_on_axis_intensity_ratio,
    image_plane_offset_from_focal_length_change_m,
    image_plane_offset_from_object_distance_change_m,
    paraxial_circle_of_confusion_diameter_m,
    paraxial_edge_waves_from_conjugates,
    paraxial_edge_waves_from_image_plane_offset,
    paraxial_image_plane_offset_from_edge_waves,
    thin_lens_image_distance_m,
)
from .sampling import (
    CellAverageTransferKernel,
    PSFQuadratureKernel,
    collapse_cell_average_transfer,
    sample_continuous_psf,
)

__all__ = [
    "CellAverageTransferKernel",
    "PSFQuadratureKernel",
    "airy_first_zero_radius_m",
    "collapse_cell_average_transfer",
    "ideal_defocus_on_axis_intensity_ratio",
    "image_plane_offset_from_focal_length_change_m",
    "image_plane_offset_from_object_distance_change_m",
    "paraxial_circle_of_confusion_diameter_m",
    "paraxial_edge_waves_from_conjugates",
    "paraxial_edge_waves_from_image_plane_offset",
    "paraxial_image_plane_offset_from_edge_waves",
    "sample_continuous_psf",
    "thin_lens_image_distance_m",
]
