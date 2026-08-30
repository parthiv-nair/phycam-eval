"""Calibrated pure-rotation motion used by the first readout reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    skew_px: float = 0.0

    def __post_init__(self) -> None:
        for name in ("fx_px", "fy_px", "cx_px", "cy_px", "skew_px"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("focal lengths must be positive")

    @property
    def matrix(self) -> NDArray[np.float64]:
        return np.array(
            [
                [self.fx_px, self.skew_px, self.cx_px],
                [0.0, self.fy_px, self.cy_px],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class ConstantAngularVelocity:
    """World-to-camera relative rotation with constant angular velocity."""

    omega_rad_s: tuple[float, float, float]
    reference_time_s: float = 0.0

    def __post_init__(self) -> None:
        omega = tuple(float(value) for value in self.omega_rad_s)
        if len(omega) != 3 or not all(np.isfinite(value) for value in omega):
            raise ValueError("omega_rad_s must contain three finite values")
        object.__setattr__(self, "omega_rad_s", omega)
        object.__setattr__(
            self, "reference_time_s", _finite(self.reference_time_s, "reference_time_s")
        )

    def relative_rotation(self, time_s: float) -> NDArray[np.float64]:
        rotation_vector = np.asarray(self.omega_rad_s, dtype=np.float64) * (
            _finite(time_s, "time_s") - self.reference_time_s
        )
        return rodrigues(rotation_vector)

    def homography(self, time_s: float, intrinsics: CameraIntrinsics) -> NDArray[np.float64]:
        calibration = intrinsics.matrix
        return calibration @ self.relative_rotation(time_s) @ np.linalg.inv(calibration)


def rodrigues(rotation_vector: ArrayLike) -> NDArray[np.float64]:
    """Convert an axis-angle vector in radians to a proper rotation matrix."""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation_vector must be a finite vector of shape (3,)")
    theta = float(np.linalg.norm(vector))
    skew = np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ],
        dtype=np.float64,
    )
    if theta < 1e-8:
        # The second-order term matters for orthogonality near zero.
        return np.eye(3) + skew + 0.5 * (skew @ skew)
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * skew
        + ((1.0 - np.cos(theta)) / theta**2) * (skew @ skew)
    )


def translation_homography(dx_px: float, dy_px: float) -> NDArray[np.float64]:
    """Return a reference-to-time image translation for regression fixtures."""

    return np.array(
        [[1.0, 0.0, _finite(dx_px, "dx_px")], [0.0, 1.0, _finite(dy_px, "dy_px")], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
