"""Native-first COCO loading and detector-output coordinate recovery.

JPEGs are decoded at their stored dimensions into floating HWC sRGB. Camera
rendering therefore happens before any detector resize. Ground-truth boxes
remain in native image-edge coordinates, and detector boxes are mapped back
through the recorded letterbox geometry before COCO evaluation.
"""

from __future__ import annotations

import hashlib
import io
import json
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .._canonical import canonical_sha256, freeze_json_value, json_value
from .metrics import yolo_to_coco_category_ids
from .preprocess import LetterboxGeometry
from .protocol import image_selection_identity

_SPLITS = {"train2017", "val2017"}
_LABEL_SPACES = {"coco_sparse", "coco80_contiguous"}
_TRUSTED_LOADER_TOKEN = object()


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _immutable_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = freeze_json_value(value)
    if not isinstance(frozen, MappingProxyType):
        raise TypeError("identity must be a mapping")
    return frozen


def _decode_native_srgb_bytes(payload: bytes, *, source_name: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("native COCO decoding requires the optional Pillow dependency") from exc
    try:
        with Image.open(io.BytesIO(payload)) as source:
            rgb = source.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"failed to decode native image bytes: {source_name}") from exc
    if array.ndim != 3 or array.shape[-1] != 3:
        raise RuntimeError("RGB decoder returned an unexpected array shape")
    return _immutable_array(array)


def _artifact_from_bytes(payload: bytes, *, published_name: str) -> dict[str, Any]:
    return {
        "name": published_name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def decode_native_srgb(path: str | Path) -> np.ndarray:
    """Read and decode one image without resizing or boundary quantization."""

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"COCO image is missing: {image_path}")
    try:
        payload = image_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"failed to read native image: {image_path}") from exc
    return _decode_native_srgb_bytes(payload, source_name=str(image_path))


def _array_record(values: Sequence[object], *, dtype: np.dtype[Any]) -> np.ndarray:
    return _immutable_array(np.asarray(values, dtype=dtype))


@dataclass(frozen=True, slots=True)
class NativeCOCOSubset:
    """An ordered in-memory COCO subset with a portable content identity."""

    image_ids: tuple[int, ...]
    images: tuple[np.ndarray, ...]
    targets: tuple[Mapping[str, Any], ...]
    category_ids: tuple[int, ...]
    identity: Mapping[str, Any]
    _loader_token: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        image_ids = tuple(self.image_ids)
        images = tuple(self.images)
        targets = tuple(self.targets)
        categories = tuple(self.category_ids)
        if not image_ids or len(image_ids) != len(set(image_ids)):
            raise ValueError("image_ids must be nonempty and unique")
        if len(images) != len(image_ids) or len(targets) != len(image_ids):
            raise ValueError("images and targets must align one-to-one with image_ids")
        if not categories or len(categories) != len(set(categories)):
            raise ValueError("category_ids must be nonempty and unique")
        normalized_images: list[np.ndarray] = []
        normalized_targets: list[Mapping[str, Any]] = []
        for image_id, image, target in zip(image_ids, images, targets):
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError("native COCO images must have HWC RGB shape")
            if array.dtype != np.float32 or not np.all(np.isfinite(array)):
                raise ValueError("native COCO images must be finite float32 arrays")
            if np.any(array < 0.0) or np.any(array > 1.0):
                raise ValueError("native COCO images must lie in [0, 1]")
            if target.get("image_id") != image_id:
                raise ValueError("target image_id does not match the ordered image axis")
            normalized_images.append(_immutable_array(array))
            normalized_targets.append(target)
        identity = json_value(self.identity)
        supplied_hash = identity.get("dataset_sha256")
        payload = {key: value for key, value in identity.items() if key != "dataset_sha256"}
        if supplied_hash != canonical_sha256(payload):
            raise ValueError("dataset_sha256 does not match the embedded COCO subset identity")
        artifacts = identity.get("image_artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != len(image_ids):
            raise ValueError("COCO identity image artifacts must align with image_ids")
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping) or set(artifact) != {"name", "bytes", "sha256"}:
                raise ValueError(f"COCO image artifact {index} is noncanonical")
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "images", tuple(normalized_images))
        object.__setattr__(self, "targets", tuple(normalized_targets))
        object.__setattr__(self, "category_ids", categories)
        object.__setattr__(self, "identity", _immutable_mapping(identity))

    def image(self, image_id: int) -> np.ndarray:
        """Return the immutable native HWC sRGB array for one selected ID."""

        try:
            return self.images[self.image_ids.index(image_id)]
        except ValueError as exc:
            raise KeyError(f"image_id {image_id} is not in this COCO subset") from exc

    def target(self, image_id: int) -> Mapping[str, Any]:
        """Return the native-coordinate target record for one selected ID."""

        try:
            return self.targets[self.image_ids.index(image_id)]
        except ValueError as exc:
            raise KeyError(f"image_id {image_id} is not in this COCO subset") from exc

    def image_shape(self, image_id: int) -> tuple[int, int]:
        """Return the stored native height and width without copying samples."""

        return tuple(int(value) for value in self.image(image_id).shape[:2])

    @property
    def loader_attested(self) -> bool:
        """Whether the subset was constructed by the exact byte-verifying loader."""

        return self._loader_token is _TRUSTED_LOADER_TOKEN


@dataclass(frozen=True, slots=True)
class LazyNativeCOCOSubset:
    """An ordered COCO subset that decodes exactly one requested image at a time."""

    image_ids: tuple[int, ...]
    image_paths: tuple[Path, ...]
    image_shapes: tuple[tuple[int, int], ...]
    targets: tuple[Mapping[str, Any], ...]
    category_ids: tuple[int, ...]
    identity: Mapping[str, Any]
    _loader_token: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        image_ids = tuple(self.image_ids)
        paths = tuple(Path(value) for value in self.image_paths)
        shapes = tuple(tuple(int(item) for item in value) for value in self.image_shapes)
        targets = tuple(self.targets)
        categories = tuple(self.category_ids)
        if not image_ids or len(image_ids) != len(set(image_ids)):
            raise ValueError("image_ids must be nonempty and unique")
        if not (len(paths) == len(shapes) == len(targets) == len(image_ids)):
            raise ValueError("paths, shapes, and targets must align with image_ids")
        if not categories or len(categories) != len(set(categories)):
            raise ValueError("category_ids must be nonempty and unique")
        for image_id, path, shape, target in zip(image_ids, paths, shapes, targets):
            if not path.is_file():
                raise FileNotFoundError(f"COCO image is missing: {path}")
            if len(shape) != 2 or any(value <= 0 for value in shape):
                raise ValueError("native COCO image shapes must contain positive height and width")
            if target.get("image_id") != image_id:
                raise ValueError("target image_id does not match the ordered image axis")
        identity = json_value(self.identity)
        supplied_hash = identity.get("dataset_sha256")
        payload = {key: value for key, value in identity.items() if key != "dataset_sha256"}
        if supplied_hash != canonical_sha256(payload):
            raise ValueError("dataset_sha256 does not match the embedded COCO subset identity")
        artifacts = identity.get("image_artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != len(image_ids):
            raise ValueError("COCO identity image artifacts must align with image_ids")
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping) or set(artifact) != {"name", "bytes", "sha256"}:
                raise ValueError(f"COCO image artifact {index} is noncanonical")
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "image_paths", paths)
        object.__setattr__(self, "image_shapes", shapes)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "category_ids", categories)
        object.__setattr__(self, "identity", _immutable_mapping(identity))

    def _index(self, image_id: int) -> int:
        try:
            return self.image_ids.index(image_id)
        except ValueError as exc:
            raise KeyError(f"image_id {image_id} is not in this COCO subset") from exc

    def image(self, image_id: int) -> np.ndarray:
        """Hash and decode the same frozen native image bytes on demand."""

        index = self._index(image_id)
        path = self.image_paths[index]
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"failed to read COCO image {image_id}: {path}") from exc
        expected = self.identity["image_artifacts"][index]
        observed = _artifact_from_bytes(payload, published_name=str(expected["name"]))
        if observed != json_value(expected):
            raise RuntimeError(f"COCO image artifact bytes drifted for image {image_id}")
        image = _decode_native_srgb_bytes(payload, source_name=str(path))
        if image.shape[:2] != self.image_shapes[index]:
            raise RuntimeError(f"decoded dimensions drifted for COCO image {image_id}")
        return image

    def target(self, image_id: int) -> Mapping[str, Any]:
        """Return the native-coordinate target record for one selected ID."""

        return self.targets[self._index(image_id)]

    def image_shape(self, image_id: int) -> tuple[int, int]:
        """Return stored native dimensions without decoding the JPEG."""

        return self.image_shapes[self._index(image_id)]

    @property
    def loader_attested(self) -> bool:
        """Whether paths, targets, and identities came from the exact loader."""

        return self._loader_token is _TRUSTED_LOADER_TOKEN


NativeCOCODataset = NativeCOCOSubset | LazyNativeCOCOSubset


def _attest_loader_construction(dataset: NativeCOCODataset) -> NativeCOCODataset:
    """Mark only the exact loader's freshly validated result as trusted.

    The token is deliberately excluded from the public dataclass constructors.
    This also means :func:`dataclasses.replace` cannot accidentally transfer
    loader attestation to caller-supplied arrays, targets, paths, or identities.
    """

    object.__setattr__(dataset, "_loader_token", _TRUSTED_LOADER_TOKEN)
    return dataset


def _selection(
    available: Sequence[int],
    *,
    ordered_image_ids: Sequence[int] | None,
    max_images: int | None,
    image_offset: int,
) -> tuple[int, ...]:
    offset = _nonnegative_integer(image_offset, name="image_offset")
    if max_images is not None:
        maximum = _nonnegative_integer(max_images, name="max_images")
        if maximum == 0:
            raise ValueError("max_images must be positive when supplied")
    else:
        maximum = None
    if ordered_image_ids is not None:
        if isinstance(ordered_image_ids, (str, bytes)) or not isinstance(
            ordered_image_ids, Sequence
        ):
            raise TypeError("ordered_image_ids must be an ordered sequence")
        if offset or maximum is not None:
            raise ValueError(
                "explicit ordered_image_ids cannot be combined with max_images or image_offset"
            )
        selected = tuple(
            _nonnegative_integer(value, name="image_id") for value in ordered_image_ids
        )
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("ordered_image_ids must be nonempty and unique")
    else:
        selected = tuple(
            available[offset:] if maximum is None else available[offset : offset + maximum]
        )
        if not selected:
            raise ValueError("COCO selection is empty")
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"selected image IDs are absent from the annotation file: {unknown}")
    return selected


def load_native_coco_subset(
    coco_root: str | Path,
    *,
    split: str = "val2017",
    ordered_image_ids: Sequence[int] | None = None,
    max_images: int | None = 100,
    image_offset: int = 0,
    eager: bool = False,
) -> NativeCOCODataset:
    """Load an ordered COCO subset at stored image resolution.

    Pass ``max_images=None`` when supplying ``ordered_image_ids``. Every
    selected JPEG and the annotation file are hashed into ``dataset_sha256``.
    The default lazy representation retains paths and dimensions but decodes
    one image on demand. Set ``eager=True`` only for deliberately small tests.
    """

    if split not in _SPLITS:
        raise ValueError("split must be 'train2017' or 'val2017'")
    root = Path(coco_root)
    annotation_path = root / "annotations" / f"instances_{split}.json"
    image_directory = root / "images" / split
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotation file is missing: {annotation_path}")
    try:
        annotation_bytes = annotation_path.read_bytes()
        raw = json.loads(annotation_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read COCO annotations: {annotation_path}") from exc
    annotation_artifact = _artifact_from_bytes(
        annotation_bytes,
        published_name=f"annotations/instances_{split}.json",
    )
    if not isinstance(raw, Mapping):
        raise ValueError("COCO annotation root must be a mapping")
    raw_images = raw.get("images")
    raw_annotations = raw.get("annotations")
    raw_categories = raw.get("categories")
    if not all(isinstance(value, list) for value in (raw_images, raw_annotations, raw_categories)):
        raise ValueError("COCO annotations require images, annotations, and categories arrays")

    metadata: dict[int, Mapping[str, Any]] = {}
    for item in raw_images:
        if not isinstance(item, Mapping):
            raise ValueError("COCO image metadata entries must be mappings")
        image_id = _nonnegative_integer(item.get("id"), name="COCO image id")
        if image_id in metadata:
            raise ValueError("COCO image metadata contains duplicate IDs")
        metadata[image_id] = item
    selected = _selection(
        sorted(metadata),
        ordered_image_ids=ordered_image_ids,
        max_images=max_images,
        image_offset=image_offset,
    )
    selected_set = set(selected)
    annotations: dict[int, list[Mapping[str, Any]]] = {image_id: [] for image_id in selected}
    for item in raw_annotations:
        if not isinstance(item, Mapping):
            raise ValueError("COCO annotation entries must be mappings")
        image_id = item.get("image_id")
        if image_id in selected_set:
            annotations[int(image_id)].append(item)

    category_ids = tuple(
        sorted(
            _nonnegative_integer(item.get("id"), name="COCO category id")
            for item in raw_categories
            if isinstance(item, Mapping)
        )
    )
    if not category_ids or category_ids[0] == 0 or len(category_ids) != len(set(category_ids)):
        raise ValueError("COCO categories must contain unique positive IDs")
    category_set = set(category_ids)
    if not isinstance(eager, bool):
        raise TypeError("eager must be bool")
    images: list[np.ndarray] = []
    image_paths: list[Path] = []
    image_shapes: list[tuple[int, int]] = []
    targets: list[Mapping[str, Any]] = []
    image_artifacts: list[dict[str, Any]] = []
    filtered_annotations = 0
    for image_id in selected:
        item = metadata[image_id]
        file_name = item.get("file_name")
        if not isinstance(file_name, str) or not file_name or Path(file_name).name != file_name:
            raise ValueError("COCO file_name must be a plain nonempty filename")
        path = image_directory / file_name
        height = _nonnegative_integer(item.get("height"), name="COCO image height")
        width = _nonnegative_integer(item.get("width"), name="COCO image width")
        if height == 0 or width == 0:
            raise ValueError(f"COCO metadata has invalid dimensions for image {image_id}")
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "native COCO inspection requires the optional Pillow dependency"
            ) from exc
        try:
            encoded_bytes = path.read_bytes()
            with Image.open(io.BytesIO(encoded_bytes)) as encoded:
                decoded_shape = (int(encoded.height), int(encoded.width))
                encoded.verify()
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"failed to verify COCO image {image_id}: {path}") from exc
        if decoded_shape != (height, width):
            raise ValueError(f"decoded dimensions disagree with COCO metadata for image {image_id}")
        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        crowds: list[int] = []
        for annotation in annotations[image_id]:
            bbox = annotation.get("bbox")
            if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
                raise ValueError("COCO bbox must contain four values")
            try:
                x, y, box_width, box_height = (float(value) for value in bbox)
                area = float(annotation.get("area", box_width * box_height))
            except (TypeError, ValueError) as exc:
                raise ValueError("COCO bbox and area must be numeric") from exc
            if not np.all(np.isfinite((x, y, box_width, box_height, area))):
                raise ValueError("COCO bbox and area must be finite")
            if box_width <= 0.0 or box_height <= 0.0 or area <= 0.0:
                filtered_annotations += 1
                continue
            label = _nonnegative_integer(annotation.get("category_id"), name="category_id")
            if label not in category_set:
                raise ValueError("COCO annotation references an undeclared category")
            crowd = _nonnegative_integer(annotation.get("iscrowd", 0), name="iscrowd")
            if crowd not in {0, 1}:
                raise ValueError("COCO iscrowd must be 0 or 1")
            boxes.append([x, y, x + box_width, y + box_height])
            labels.append(label)
            areas.append(area)
            crowds.append(crowd)
        target = {
            "image_id": image_id,
            "boxes": _array_record(boxes, dtype=np.dtype(np.float64)).reshape((-1, 4)),
            "labels": _array_record(labels, dtype=np.dtype(np.int64)),
            "area": _array_record(areas, dtype=np.dtype(np.float64)),
            "iscrowd": _array_record(crowds, dtype=np.dtype(np.int64)),
        }
        image_paths.append(path)
        image_shapes.append((height, width))
        if eager:
            images.append(_decode_native_srgb_bytes(encoded_bytes, source_name=str(path)))
        targets.append(MappingProxyType(target))
        image_artifacts.append(
            _artifact_from_bytes(encoded_bytes, published_name=f"{split}/{file_name}")
        )

    selection = image_selection_identity(selected)
    payload = {
        "schema_version": 2,
        "record_type": "native_coco_subset",
        "dataset": "COCO",
        "split": split,
        "annotation_artifact": annotation_artifact,
        "ordered_image_ids": list(selected),
        "image_selection": selection,
        "image_artifacts": image_artifacts,
        "category_ids": list(category_ids),
        "filtered_nonpositive_annotations": filtered_annotations,
        "decode_contract": {
            "implementation_id": "pillow.rgb.native.float32.v1",
            "layout": "HWC_RGB",
            "range": [0.0, 1.0],
            "resize": None,
            "uint8_to_float": "exact_divide_by_255",
        },
        "target_contract": {
            "coordinate_space": "native_stored_image",
            "box_convention": "continuous_xyxy_image_edges",
            "area": "official_coco_annotation_area",
        },
    }
    identity = {**payload, "dataset_sha256": canonical_sha256(payload)}
    if eager:
        return _attest_loader_construction(
            NativeCOCOSubset(
                selected,
                tuple(images),
                tuple(targets),
                category_ids,
                identity,
            )
        )
    return _attest_loader_construction(
        LazyNativeCOCOSubset(
            selected,
            tuple(image_paths),
            tuple(image_shapes),
            tuple(targets),
            category_ids,
            identity,
        )
    )


def detector_output_to_native_prediction(
    *,
    image_id: int,
    detector_output: Mapping[str, Any],
    geometry: LetterboxGeometry,
    native_shape: Sequence[int] | None = None,
    label_space: str = "coco_sparse",
    realization_id: int | None = None,
) -> dict[str, Any]:
    """Recover one detector output into clipped native COCO coordinates."""

    resolved_image_id = _nonnegative_integer(image_id, name="image_id")
    if not isinstance(detector_output, Mapping):
        raise TypeError("detector_output must be a mapping")
    if not isinstance(geometry, LetterboxGeometry):
        raise TypeError("geometry must be a LetterboxGeometry")
    missing = {"boxes", "labels", "scores"}.difference(detector_output)
    if missing:
        raise ValueError(f"detector_output is missing keys: {sorted(missing)}")
    boxes = np.asarray(detector_output["boxes"])
    labels = np.asarray(detector_output["labels"])
    scores = np.asarray(detector_output["scores"], dtype=np.float64)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("detector boxes must have shape (N, 4)")
    if (
        labels.ndim != 1
        or scores.ndim != 1
        or len(labels) != len(boxes)
        or len(scores) != len(boxes)
    ):
        raise ValueError("detector labels and scores must have one entry per box")
    camera_boxes = np.array(geometry.inverse_boxes(boxes), copy=True)
    camera_height, camera_width = geometry.input_shape
    if native_shape is None:
        height, width = camera_height, camera_width
    else:
        if (
            isinstance(native_shape, (str, bytes))
            or not isinstance(native_shape, Sequence)
            or len(native_shape) != 2
        ):
            raise TypeError("native_shape must contain height and width")
        height = _nonnegative_integer(native_shape[0], name="native height")
        width = _nonnegative_integer(native_shape[1], name="native width")
        if height == 0 or width == 0:
            raise ValueError("native_shape dimensions must be positive")
    native = camera_boxes
    native[:, (0, 2)] *= width / camera_width
    native[:, (1, 3)] *= height / camera_height
    native[:, (0, 2)] = np.clip(native[:, (0, 2)], 0.0, float(width))
    native[:, (1, 3)] = np.clip(native[:, (1, 3)], 0.0, float(height))
    keep = (native[:, 2] > native[:, 0]) & (native[:, 3] > native[:, 1])
    native = native[keep]
    labels = labels[keep]
    scores = scores[keep]
    if label_space not in _LABEL_SPACES:
        raise ValueError("label_space must be 'coco_sparse' or 'coco80_contiguous'")
    if label_space == "coco80_contiguous":
        labels = yolo_to_coco_category_ids(labels)
    else:
        try:
            numeric = labels.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("detector labels must contain positive integers") from exc
        if (
            not np.all(np.isfinite(numeric))
            or np.any(numeric != np.floor(numeric))
            or np.any(numeric <= 0.0)
        ):
            raise ValueError("detector labels must contain positive integers")
        labels = numeric.astype(np.int64)
    result = {
        "image_id": resolved_image_id,
        "boxes": _immutable_array(np.asarray(native, dtype=np.float64)),
        "labels": _immutable_array(np.asarray(labels, dtype=np.int64)),
        "scores": _immutable_array(np.asarray(scores, dtype=np.float64)),
    }
    if realization_id is not None:
        result["realization_id"] = _nonnegative_integer(realization_id, name="realization_id")
    return result


__all__ = [
    "LazyNativeCOCOSubset",
    "NativeCOCODataset",
    "NativeCOCOSubset",
    "decode_native_srgb",
    "detector_output_to_native_prediction",
    "load_native_coco_subset",
]
