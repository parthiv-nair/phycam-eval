"""Fixed white balance and camera-to-output linear color transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class ColorTransform:
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
    camera_to_output: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    output_space: str = "linear-srgb-d65"

    def __post_init__(self) -> None:
        white_balance = tuple(float(value) for value in self.white_balance)
        matrix = tuple(tuple(float(value) for value in row) for row in self.camera_to_output)
        if len(white_balance) != 3 or not all(
            np.isfinite(value) and value > 0.0 for value in white_balance
        ):
            raise ValueError("white_balance must contain three finite positive values")
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError("camera_to_output must be a 3 by 3 matrix")
        matrix_array = np.asarray(matrix, dtype=np.float64)
        if not np.all(np.isfinite(matrix_array)) or abs(np.linalg.det(matrix_array)) < 1e-12:
            raise ValueError("camera_to_output must be finite and invertible")
        output_space = str(self.output_space)
        if not output_space:
            raise ValueError("output_space must be nonempty")
        object.__setattr__(self, "white_balance", white_balance)
        object.__setattr__(self, "camera_to_output", matrix)
        object.__setattr__(self, "output_space", output_space)

    @property
    def matrix(self) -> NDArray[np.float64]:
        return np.asarray(self.camera_to_output, dtype=np.float64)


def apply_color_transform(
    camera_linear_rgb: ArrayLike,
    transform: ColorTransform,
) -> NDArray[np.float64]:
    """Apply white balance then a fixed 3 by 3 camera-to-output transform."""

    values = np.asarray(camera_linear_rgb, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("camera_linear_rgb must have a final RGB dimension")
    if not np.all(np.isfinite(values)):
        raise ValueError("camera_linear_rgb must be finite")
    balanced = values * np.asarray(transform.white_balance, dtype=np.float64)
    return np.einsum("...c,oc->...o", balanced, transform.matrix)
