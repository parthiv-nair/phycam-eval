"""Deterministic detector-boundary preprocessing for rendered camera frames.

The physical camera graph ends at display-referred floating-point sRGB.  This
module owns the *separate* detector resize/letterbox operation and retags the
result as :class:`~phycam_eval.domains.Domain.DETECTOR_INPUT`.  Keeping
that boundary explicit prevents detector preprocessing from silently becoming
part of the camera model.

Boxes use continuous ``(x_min, y_min, x_max, y_max)`` image-edge coordinates.
The nominal aspect-preserving scale determines the integer resized raster.
Because rounding that raster can make the realized horizontal and vertical
scales differ by a tiny amount, both realized scales are recorded and used for
the exactly invertible box transform.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .._canonical import canonical_sha256, freeze_json_value, json_value, positive_int
from ..domains import ColorSpace, Domain
from ..frame import Frame, FrameMetadata
from .protocol import preprocessing_identity

_IMPLEMENTATION_ID = "letterbox.numpy.pixel_center_bilinear.v1"
_BOX_CONVENTION = "continuous_xyxy_image_edges"


def _shape2(value: Sequence[int], *, field_name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two integers")
    return (
        positive_int(value[0], field_name=f"{field_name}[0]"),
        positive_int(value[1], field_name=f"{field_name}[1]"),
    )


def _pad3(value: float | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, (bool, np.bool_)) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("pad_value must contain real numbers")
    if isinstance(value, numbers.Real):
        scalar = float(value)
        result = (scalar, scalar, scalar)
    else:
        array = np.asarray(value)
        if array.shape != (3,):
            raise ValueError("pad_value must be a scalar or contain exactly three values")
        if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
            raise TypeError("pad_value must contain real numbers")
        if np.issubdtype(array.dtype, np.bool_):
            raise TypeError("pad_value must contain real numbers")
        result = tuple(float(item) for item in array)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("pad_value must be finite")
    if any(item < 0.0 or item > 1.0 for item in result):
        raise ValueError("pad_value must lie in [0, 1]")
    return result  # type: ignore[return-value]


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = freeze_json_value(value)
    if not isinstance(frozen, MappingProxyType):
        raise TypeError("identity must be a mapping")
    return frozen


def _immutable_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True, slots=True)
class LetterboxConfig:
    """Fixed, content-addressed detector preprocessing configuration."""

    output_shape: tuple[int, int]
    pad_value: tuple[float, float, float] | float = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_shape",
            _shape2(self.output_shape, field_name="output_shape"),
        )
        object.__setattr__(self, "pad_value", _pad3(self.pad_value))

    def parameters(self) -> dict[str, Any]:
        """Return the complete numerical and semantic preprocessing contract."""

        return {
            "output_shape_hw": list(self.output_shape),
            "pad_value_srgb": list(self.pad_value),
            "resize": {
                "method": "bilinear",
                "coordinate_transform": "half_pixel_pixel_centers",
                "edge_handling": "clamp_to_edge",
                "integer_extent_rounding": "floor_positive_value_plus_half",
                "aspect_policy": "fit_inside_preserve_aspect",
            },
            "padding": "centered_top_left_floor",
            "input_contract": {
                "domain": Domain.DISPLAY_RGB.value,
                "color_space": ColorSpace.SRGB.value,
                "layout": "HWC_RGB",
                "sample_type": "floating_point",
                "range": [0.0, 1.0],
            },
            "output_contract": {
                "domain": Domain.DETECTOR_INPUT.value,
                "color_space": ColorSpace.SRGB.value,
                "layout": "HWC_RGB",
                "sample_type": "floating_point_no_uint8_quantization",
            },
            "box_convention": _BOX_CONVENTION,
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        """Immutable identity accepted by :func:`eval.protocol.preprocessing_identity`."""

        return _immutable_mapping(
            preprocessing_identity(
                name="aspect_preserving_letterbox",
                implementation_id=_IMPLEMENTATION_ID,
                parameters=self.parameters(),
            )
        )


@dataclass(frozen=True, slots=True)
class LetterboxGeometry:
    """Explicit resize and padding geometry for one source image."""

    input_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    output_shape: tuple[int, int]
    nominal_scale: float
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    def __post_init__(self) -> None:
        input_shape = _shape2(self.input_shape, field_name="input_shape")
        resized_shape = _shape2(self.resized_shape, field_name="resized_shape")
        output_shape = _shape2(self.output_shape, field_name="output_shape")
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "resized_shape", resized_shape)
        object.__setattr__(self, "output_shape", output_shape)
        for name in ("nominal_scale", "scale_x", "scale_y"):
            number = float(getattr(self, name))
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, number)
        for name in ("pad_left", "pad_top", "pad_right", "pad_bottom"):
            object.__setattr__(
                self,
                name,
                positive_int(getattr(self, name), field_name=name, allow_zero=True),
            )
        if resized_shape[0] + self.pad_top + self.pad_bottom != output_shape[0]:
            raise ValueError("vertical resize and padding do not fill output_shape")
        if resized_shape[1] + self.pad_left + self.pad_right != output_shape[1]:
            raise ValueError("horizontal resize and padding do not fill output_shape")
        expected_x = resized_shape[1] / input_shape[1]
        expected_y = resized_shape[0] / input_shape[0]
        if self.scale_x != expected_x or self.scale_y != expected_y:
            raise ValueError("realized scales must exactly match resized/input extents")

    @classmethod
    def fit(
        cls,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
    ) -> "LetterboxGeometry":
        """Compute centered fit-inside geometry using an explicit rounding rule."""

        input_h, input_w = _shape2(input_shape, field_name="input_shape")
        output_h, output_w = _shape2(output_shape, field_name="output_shape")
        nominal = min(output_h / input_h, output_w / input_w)
        resized_h = min(output_h, max(1, math.floor(input_h * nominal + 0.5)))
        resized_w = min(output_w, max(1, math.floor(input_w * nominal + 0.5)))
        remaining_h = output_h - resized_h
        remaining_w = output_w - resized_w
        top = remaining_h // 2
        left = remaining_w // 2
        return cls(
            input_shape=(input_h, input_w),
            resized_shape=(resized_h, resized_w),
            output_shape=(output_h, output_w),
            nominal_scale=nominal,
            scale_x=resized_w / input_w,
            scale_y=resized_h / input_h,
            pad_left=left,
            pad_top=top,
            pad_right=remaining_w - left,
            pad_bottom=remaining_h - top,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LetterboxGeometry":
        """Reconstruct and revalidate serialized evaluation geometry."""

        if not isinstance(value, Mapping):
            raise TypeError("letterbox geometry record must be a mapping")
        if value.get("box_convention") != _BOX_CONVENTION:
            raise ValueError("letterbox geometry box convention is unsupported")
        scale = value.get("realized_scale_xy")
        padding = value.get("padding_ltrb")
        if not isinstance(scale, Sequence) or isinstance(scale, (str, bytes)) or len(scale) != 2:
            raise ValueError("realized_scale_xy must contain two values")
        if (
            not isinstance(padding, Sequence)
            or isinstance(padding, (str, bytes))
            or len(padding) != 4
        ):
            raise ValueError("padding_ltrb must contain four values")
        return cls(
            input_shape=value.get("input_shape_hw"),
            resized_shape=value.get("resized_shape_hw"),
            output_shape=value.get("output_shape_hw"),
            nominal_scale=value.get("nominal_scale"),
            scale_x=scale[0],
            scale_y=scale[1],
            pad_left=padding[0],
            pad_top=padding[1],
            pad_right=padding[2],
            pad_bottom=padding[3],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_shape_hw": list(self.input_shape),
            "resized_shape_hw": list(self.resized_shape),
            "output_shape_hw": list(self.output_shape),
            "nominal_scale": self.nominal_scale,
            "realized_scale_xy": [self.scale_x, self.scale_y],
            "padding_ltrb": [
                self.pad_left,
                self.pad_top,
                self.pad_right,
                self.pad_bottom,
            ],
            "box_convention": _BOX_CONVENTION,
        }

    def _boxes(self, boxes_xyxy: ArrayLike, *, inverse: bool) -> np.ndarray:
        boxes = np.asarray(boxes_xyxy)
        if boxes.ndim == 0 or boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape (..., 4)")
        if boxes.dtype.hasobject or not np.issubdtype(boxes.dtype, np.number):
            raise TypeError("boxes must contain numeric values")
        transformed = np.array(boxes, dtype=np.float64, copy=True, order="C")
        if not np.all(np.isfinite(transformed)):
            raise ValueError("boxes must be finite")
        if np.any(transformed[..., 2] < transformed[..., 0]) or np.any(
            transformed[..., 3] < transformed[..., 1]
        ):
            raise ValueError("boxes must satisfy x_max >= x_min and y_max >= y_min")
        if inverse:
            transformed[..., (0, 2)] = (transformed[..., (0, 2)] - self.pad_left) / self.scale_x
            transformed[..., (1, 3)] = (transformed[..., (1, 3)] - self.pad_top) / self.scale_y
        else:
            transformed[..., (0, 2)] = transformed[..., (0, 2)] * self.scale_x + self.pad_left
            transformed[..., (1, 3)] = transformed[..., (1, 3)] * self.scale_y + self.pad_top
        return _immutable_array(transformed)

    def forward_boxes(self, boxes_xyxy: ArrayLike) -> np.ndarray:
        """Map source-image boxes into detector-input coordinates without clipping."""

        return self._boxes(boxes_xyxy, inverse=False)

    def inverse_boxes(self, boxes_xyxy: ArrayLike) -> np.ndarray:
        """Map detector-input boxes back into source-image coordinates."""

        return self._boxes(boxes_xyxy, inverse=True)

    # Descriptive aliases used by detector adapters.
    to_detector_boxes = forward_boxes
    to_source_boxes = inverse_boxes


@dataclass(frozen=True, slots=True)
class DetectorInput:
    """One immutable detector-ready frame plus its reversible geometry."""

    frame: Frame
    geometry: LetterboxGeometry
    preprocessing: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, Frame):
            raise TypeError("frame must be a Frame")
        if self.frame.domain is not Domain.DETECTOR_INPUT:
            raise ValueError("detector input frame must use DETECTOR_INPUT domain")
        if self.frame.color_space is not ColorSpace.SRGB:
            raise ValueError("detector input frame must use SRGB color space")
        if self.frame.shape[:2] != self.geometry.output_shape:
            raise ValueError("detector frame shape does not match letterbox geometry")
        if self.frame.ndim != 3 or self.frame.shape[-1] != 3:
            raise ValueError("detector input frame must have shape (H, W, 3)")
        if not np.issubdtype(self.frame.dtype, np.floating):
            raise TypeError("detector input must remain floating point")
        if not np.all(np.isfinite(self.frame.array)):
            raise ValueError("detector input samples must be finite")
        if np.any(self.frame.array < 0.0) or np.any(self.frame.array > 1.0):
            raise ValueError("detector input samples must lie in [0, 1]")
        identity = json_value(self.preprocessing)
        supplied_hash = identity.get("preprocessing_sha256")
        payload = {key: value for key, value in identity.items() if key != "preprocessing_sha256"}
        if supplied_hash != canonical_sha256(payload):
            raise ValueError("preprocessing_sha256 does not match preprocessing contract")
        object.__setattr__(self, "preprocessing", _immutable_mapping(identity))

    @property
    def array(self) -> np.ndarray:
        return self.frame.array


def bilinear_resize(image: ArrayLike, output_shape: Sequence[int]) -> np.ndarray:
    """Resize a finite floating HWC array with deterministic half-pixel bilinear sampling."""

    source = np.asarray(image)
    if source.ndim != 3:
        raise ValueError("image must have shape (H, W, C)")
    if source.shape[0] < 1 or source.shape[1] < 1 or source.shape[2] < 1:
        raise ValueError("image dimensions must be nonempty")
    if not np.issubdtype(source.dtype, np.floating):
        raise TypeError("bilinear resize requires floating-point samples")
    if not np.all(np.isfinite(source)):
        raise ValueError("image samples must be finite")
    output_h, output_w = _shape2(output_shape, field_name="output_shape")
    input_h, input_w = source.shape[:2]
    work_dtype = np.result_type(source.dtype, np.float32)
    values = np.asarray(source, dtype=work_dtype)
    if (input_h, input_w) == (output_h, output_w):
        return _immutable_array(np.array(values, copy=True, order="C"))

    y = (np.arange(output_h, dtype=np.float64) + 0.5) * (input_h / output_h) - 0.5
    x = (np.arange(output_w, dtype=np.float64) + 0.5) * (input_w / output_w) - 0.5
    np.clip(y, 0.0, input_h - 1.0, out=y)
    np.clip(x, 0.0, input_w - 1.0, out=x)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, input_h - 1)
    x1 = np.minimum(x0 + 1, input_w - 1)
    wy = np.asarray(y - y0, dtype=work_dtype)
    wx = np.asarray(x - x0, dtype=work_dtype)

    top = values[y0[:, None], x0[None, :], :] * (1.0 - wx[None, :, None])
    top += values[y0[:, None], x1[None, :], :] * wx[None, :, None]
    bottom = values[y1[:, None], x0[None, :], :] * (1.0 - wx[None, :, None])
    bottom += values[y1[:, None], x1[None, :], :] * wx[None, :, None]
    resized = top * (1.0 - wy[:, None, None]) + bottom * wy[:, None, None]
    return _immutable_array(np.asarray(resized, dtype=work_dtype, order="C"))


def letterbox_display_frame(frame: Frame, config: LetterboxConfig) -> DetectorInput:
    """Validate, resize, pad, and retag one rendered camera output."""

    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    if not isinstance(config, LetterboxConfig):
        raise TypeError("config must be a LetterboxConfig")
    if frame.domain is not Domain.DISPLAY_RGB or frame.color_space is not ColorSpace.SRGB:
        raise ValueError("detector preprocessing requires DISPLAY_RGB in SRGB")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("rendered camera frame must have shape (H, W, 3)")
    if not np.issubdtype(frame.dtype, np.floating):
        raise TypeError("rendered camera frame must use floating-point samples")
    if not np.all(np.isfinite(frame.array)):
        raise ValueError("rendered camera frame must contain only finite values")
    if np.any(frame.array < 0.0) or np.any(frame.array > 1.0):
        raise ValueError("rendered sRGB samples must lie in [0, 1]")

    geometry = LetterboxGeometry.fit(frame.shape[:2], config.output_shape)
    resized = bilinear_resize(frame.array, geometry.resized_shape)
    output = np.empty((*geometry.output_shape, 3), dtype=resized.dtype)
    output[...] = np.asarray(config.pad_value, dtype=resized.dtype)
    row_slice = slice(geometry.pad_top, geometry.pad_top + geometry.resized_shape[0])
    col_slice = slice(geometry.pad_left, geometry.pad_left + geometry.resized_shape[1])
    output[row_slice, col_slice, :] = resized

    metadata = frame.metadata.to_dict()
    metadata.update(
        {
            "units": "srgb_code_value",
            "sample_spacing_m": None,
            "sensor_origin_m": None,
            "sensor_window_m": None,
        }
    )
    attributes = dict(metadata.get("attributes", {}))
    attributes["detector_preprocessing"] = {
        "identity": json_value(config.identity),
        "geometry": geometry.to_dict(),
    }
    metadata["attributes"] = attributes
    output_frame = Frame(
        output,
        Domain.DETECTOR_INPUT,
        ColorSpace.SRGB,
        FrameMetadata.from_dict(metadata),
    )
    return DetectorInput(output_frame, geometry, config.identity)


# Short public spelling for callers that already know this is the display boundary.
letterbox = letterbox_display_frame


__all__ = [
    "DetectorInput",
    "LetterboxConfig",
    "LetterboxGeometry",
    "bilinear_resize",
    "letterbox",
    "letterbox_display_frame",
]
