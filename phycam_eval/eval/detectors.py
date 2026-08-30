"""Optional detector adapters that honor the explicit detector-input boundary."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType, MappingProxyType, SimpleNamespace
from typing import Any, Iterator

import numpy as np

from .._canonical import freeze_json_value
from .model_provenance import (
    describe_artifact_bytes,
    model_identity,
    read_artifact_bytes,
    verify_artifact_bytes,
)
from .preprocess import DetectorInput
from .protocol import validate_torch_device, verify_actual_torch_device
from .torchvision_coco_postprocess import (
    bind_fasterrcnn_coco_sparse_postprocessor,
    bind_retinanet_coco_sparse_postprocessor,
)

_DETR_MODEL_ID = "facebook/detr-resnet-50"
_DETR_REVISION = "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"
_COCO_EVALUATION_MAXIMUM_DETECTIONS = 100
_YOLO_NMS_TIME_LIMIT_POLICY = "isolated_upstream_postprocess_nms_proxy_max_time_img_inf_attested.v1"
_DETR_ARTIFACTS: Mapping[str, Mapping[str, int | str]] = {
    "config.json": {
        "bytes": 4_592,
        "sha256": "e7bcf3992363f27717a863f14b193140ad2e41d4338ee012730e58a92cae17e6",
    },
    "model.safetensors": {
        "bytes": 166_587_896,
        "sha256": "830f5e2eeaada8c8c8281779dcc8ab12833972eb8514ed0a35be6c1d4420ad81",
    },
    "preprocessor_config.json": {
        "bytes": 290,
        "sha256": "0673fea2a6d3cf92cdbab3c7426c0ecdf8a4729a2a4d5199033dcd66a2b8759b",
    },
}
_DETR_ALLOWED_IGNORED_STATE_KEYS = (
    "model.backbone.model.layer1.0.downsample.1.num_batches_tracked",
    "model.backbone.model.layer2.0.downsample.1.num_batches_tracked",
    "model.backbone.model.layer3.0.downsample.1.num_batches_tracked",
    "model.backbone.model.layer4.0.downsample.1.num_batches_tracked",
)
_COCO_SPARSE_CATEGORY_IDS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _shape2(value: Sequence[int], *, name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError(f"{name} must contain height and width")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise TypeError(f"{name}[{index}] must be an integer")
        if int(item) <= 0:
            raise ValueError(f"{name}[{index}] must be positive")
        result.append(int(item))
    return result[0], result[1]


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _strict_json_object(payload: bytes, *, artifact_name: str) -> dict[str, Any]:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-finite JSON number {token!r}")

    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=reject_nonfinite)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"failed to parse pinned {artifact_name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"pinned {artifact_name} must contain a JSON object")
    return value


@contextmanager
def _private_artifact_snapshot(
    artifacts: Mapping[str, bytes],
    *,
    prefix: str,
) -> Iterator[Path]:
    """Materialize immutable payloads for loaders that require local paths."""

    if not artifacts:
        raise ValueError("a private artifact snapshot must not be empty")
    with TemporaryDirectory(prefix=prefix) as temporary_directory:
        root = Path(temporary_directory)
        for name, payload in artifacts.items():
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or name in {".", ".."}
            ):
                raise ValueError("private snapshot artifact names must be plain filenames")
            if not isinstance(payload, bytes):
                raise TypeError("private snapshot artifacts must contain immutable bytes")
            destination = root / name
            with destination.open("xb") as stream:
                written = stream.write(payload)
                if written != len(payload):
                    raise OSError(f"short write while materializing {name}")
                stream.flush()
                os.fsync(stream.fileno())
            destination.chmod(0o400)
        yield root


def _rgb_triplet(value: object, *, name: str, positive: bool = False) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise RuntimeError(f"pinned DETR {name} must contain three values")
    result: list[float] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)):
            raise RuntimeError(f"pinned DETR {name} must contain real values")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"pinned DETR {name} must contain real values") from exc
        if not math.isfinite(number) or (positive and number <= 0.0):
            qualifier = "finite positive" if positive else "finite"
            raise RuntimeError(f"pinned DETR {name} must contain {qualifier} values")
        result.append(number)
    return tuple(result)


def _detr_id2label(value: object, *, source: str) -> dict[int, str]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{source} DETR id2label must be a mapping")
    result: dict[int, str] = {}
    for raw_label, raw_name in value.items():
        if isinstance(raw_label, bool):
            raise RuntimeError(f"{source} DETR id2label contains an invalid label")
        if isinstance(raw_label, int):
            label = raw_label
        elif isinstance(raw_label, str) and raw_label.isdecimal():
            label = int(raw_label)
            if str(label) != raw_label:
                raise RuntimeError(f"{source} DETR id2label labels are not canonical")
        else:
            raise RuntimeError(f"{source} DETR id2label contains an invalid label")
        if not isinstance(raw_name, str) or not raw_name or label in result:
            raise RuntimeError(f"{source} DETR id2label contains an invalid class name")
        result[label] = raw_name
    if set(result) != set(range(91)):
        raise RuntimeError(f"{source} DETR id2label is not the pinned COCO-91 index space")
    return result


def _validate_detr_config(value: Mapping[str, Any]) -> dict[int, str]:
    if value.get("model_type") != "detr":
        raise RuntimeError("pinned DETR config has an unsupported model_type")
    architectures = value.get("architectures")
    if (
        isinstance(architectures, (str, bytes))
        or not isinstance(architectures, Sequence)
        or tuple(architectures) != ("DetrForObjectDetection",)
    ):
        raise RuntimeError("pinned DETR config has an unsupported architecture")
    num_queries = value.get("num_queries")
    if isinstance(num_queries, bool) or num_queries != 100:
        raise RuntimeError("pinned DETR config must expose exactly 100 object queries")
    return _detr_id2label(value.get("id2label"), source="artifact")


def _validate_loaded_detr_config(value: object, artifact_id2label: Mapping[int, str]) -> None:
    if getattr(value, "model_type", None) != "detr":
        raise RuntimeError("loaded DETR model_type drifted from the verified artifact")
    if tuple(getattr(value, "architectures", ())) != ("DetrForObjectDetection",):
        raise RuntimeError("loaded DETR architecture drifted from the verified artifact")
    if getattr(value, "num_queries", None) != 100:
        raise RuntimeError("loaded DETR query count drifted from the verified artifact")
    loaded_id2label = _detr_id2label(getattr(value, "id2label", None), source="loaded model")
    if loaded_id2label != artifact_id2label or getattr(value, "num_labels", None) != 91:
        raise RuntimeError("loaded DETR label space drifted from the verified artifact")


def _validate_detr_preprocessor(value: Mapping[str, Any]) -> tuple[tuple[float, ...], ...]:
    if value.get("image_processor_type") != "DetrImageProcessor":
        raise RuntimeError("pinned DETR preprocessor type is unsupported")
    if value.get("format") != "coco_detection":
        raise RuntimeError("pinned DETR preprocessor does not declare COCO detection format")
    if value.get("do_normalize") is not True:
        raise RuntimeError("pinned DETR preprocessor does not declare RGB normalization")
    if value.get("do_resize") is not True:
        raise RuntimeError("pinned DETR preprocessor resize contract drifted")
    if value.get("do_rescale") not in (None, False):
        raise RuntimeError("pinned DETR preprocessor unexpectedly requires another rescale")
    if value.get("size") != {"shortest_edge": 800, "longest_edge": 1333}:
        raise RuntimeError("pinned DETR preprocessor resize declaration drifted")
    image_mean = _rgb_triplet(value.get("image_mean"), name="image_mean")
    image_std = _rgb_triplet(value.get("image_std"), name="image_std", positive=True)
    if image_mean != (0.485, 0.456, 0.406) or image_std != (0.229, 0.224, 0.225):
        raise RuntimeError("pinned DETR RGB normalization constants drifted")
    return image_mean, image_std


def _detr_model_directory(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser()
    if path.is_file():
        if path.name != "model.safetensors":
            raise ValueError("the pinned DETR checkpoint must be named model.safetensors")
        directory = path.parent
    elif path.is_dir():
        directory = path
    elif path.exists():
        raise ValueError(f"DETR model path is neither a file nor a directory: {path}")
    else:
        raise FileNotFoundError(f"local DETR model path is missing: {path}")
    return directory.resolve(strict=True)


def _complete_yolo_predictor_type() -> type[Any]:
    """Build the pinned predictor variant that cannot abandon a batch tail."""

    from ultralytics.models.yolo.detect import DetectionPredictor

    upstream_postprocess = DetectionPredictor.postprocess
    if not isinstance(upstream_postprocess, FunctionType):
        raise RuntimeError("Ultralytics detection postprocessor is not a Python function")
    if not {"nms", "non_max_suppression"}.issubset(upstream_postprocess.__code__.co_names):
        raise RuntimeError("Ultralytics detection postprocessor no longer exposes its NMS call")
    upstream_nms_module = upstream_postprocess.__globals__.get("nms")
    original_nms = getattr(upstream_nms_module, "non_max_suppression", None)
    if not isinstance(original_nms, FunctionType):
        raise RuntimeError("Ultralytics non_max_suppression is not a Python function")
    time_parameter = inspect.signature(original_nms).parameters.get("max_time_img")
    if time_parameter is None or time_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
        raise RuntimeError("Ultralytics NMS does not expose a keyword max_time_img contract")
    try:
        upstream_postprocess_source = inspect.getsource(upstream_postprocess).encode("utf-8")
        original_nms_source = inspect.getsource(original_nms).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise RuntimeError("Ultralytics detector source is unavailable for provenance") from exc
    upstream_code = upstream_postprocess.__code__
    upstream_defaults = upstream_postprocess.__defaults__
    upstream_kwdefaults = (
        None
        if upstream_postprocess.__kwdefaults__ is None
        else dict(upstream_postprocess.__kwdefaults__)
    )
    upstream_closure = upstream_postprocess.__closure__
    upstream_globals = dict(upstream_postprocess.__globals__)
    original_nms_code = original_nms.__code__
    isolated_nms = FunctionType(
        original_nms_code,
        dict(original_nms.__globals__),
        name=original_nms.__name__,
        argdefs=original_nms.__defaults__,
        closure=original_nms.__closure__,
    )
    isolated_nms.__kwdefaults__ = (
        None if original_nms.__kwdefaults__ is None else dict(original_nms.__kwdefaults__)
    )

    class _CompleteNMSDetectionPredictor(DetectionPredictor):
        _phycam_isolated_nms = staticmethod(isolated_nms)
        _phycam_nms_time_limit_policy = _YOLO_NMS_TIME_LIMIT_POLICY
        _phycam_nms_calls = 0
        _phycam_original_nms_source_sha256 = hashlib.sha256(original_nms_source).hexdigest()
        _phycam_postprocess_calls = 0
        _phycam_upstream_postprocess = staticmethod(upstream_postprocess)
        _phycam_upstream_postprocess_code = upstream_code
        _phycam_upstream_postprocess_source_sha256 = hashlib.sha256(
            upstream_postprocess_source
        ).hexdigest()

        def postprocess(
            self,
            preds: Any,
            img: Any,
            orig_imgs: Any,
            **kwargs: Any,
        ) -> Any:
            """Delegate upstream postprocessing through one isolated NMS binding."""

            nms_calls = 0

            def complete_nms(*args: Any, **nms_kwargs: Any) -> Any:
                nonlocal nms_calls
                if isolated_nms.__code__ is not original_nms_code:
                    raise RuntimeError("Ultralytics NMS implementation drifted during inference")
                nms_calls += 1
                nms_kwargs["max_time_img"] = math.inf
                return isolated_nms(*args, **nms_kwargs)

            isolated_globals = dict(upstream_globals)
            isolated_globals["nms"] = SimpleNamespace(non_max_suppression=complete_nms)
            isolated_postprocess = FunctionType(
                upstream_code,
                isolated_globals,
                name=upstream_postprocess.__name__,
                argdefs=upstream_defaults,
                closure=upstream_closure,
            )
            isolated_postprocess.__kwdefaults__ = (
                None if upstream_kwdefaults is None else dict(upstream_kwdefaults)
            )
            results = isolated_postprocess(self, preds, img, orig_imgs, **kwargs)
            if nms_calls != 1:
                raise RuntimeError("Ultralytics upstream postprocessor did not execute NMS once")
            self._phycam_nms_calls += 1
            self._phycam_postprocess_calls += 1
            return results

    return _CompleteNMSDetectionPredictor


class UltralyticsYOLOAdapter:
    """Run an Ultralytics COCO80 detector on BCHW floating RGB tensors.

    Passing tensors avoids an additional image decode, channel reversal, or
    uint8 conversion inside the adapter. Outputs remain in detector-input
    coordinates and use contiguous COCO80 labels for the COCO runner to map.
    """

    __slots__ = (
        "_execution_device",
        "_model",
        "_predictor_seal",
        "_predictor_type",
        "confidence",
        "device",
        "evaluation_maximum_detections",
        "identity",
        "iou",
        "maximum_detections",
    )

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        confidence: float = 0.001,
        iou: float = 0.7,
        maximum_detections: int = 300,
        evaluation_maximum_detections: int = _COCO_EVALUATION_MAXIMUM_DETECTIONS,
    ) -> None:
        self.confidence = _probability(confidence, name="confidence")
        self.iou = _probability(iou, name="iou")
        self.maximum_detections = _positive_integer(
            maximum_detections,
            name="maximum_detections",
        )
        self.evaluation_maximum_detections = _positive_integer(
            evaluation_maximum_detections,
            name="evaluation_maximum_detections",
        )
        if self.evaluation_maximum_detections != _COCO_EVALUATION_MAXIMUM_DETECTIONS:
            raise ValueError("evaluation_maximum_detections must equal COCO maxDets=100")
        if self.maximum_detections < self.evaluation_maximum_detections:
            raise ValueError("maximum_detections must be at least evaluation_maximum_detections")
        self.device = validate_torch_device(device)
        checkpoint_path = Path(checkpoint)
        try:
            checkpoint_payload = read_artifact_bytes(checkpoint_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"YOLO checkpoint is missing: {checkpoint_path}") from exc
        artifact_record = describe_artifact_bytes(
            checkpoint_payload,
            published_name=checkpoint_path.name,
        )
        try:
            import torch
            import ultralytics
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "UltralyticsYOLOAdapter requires the optional experiments dependencies"
            ) from exc
        with _private_artifact_snapshot(
            {checkpoint_path.name: checkpoint_payload},
            prefix="phycam-yolo-artifact-",
        ) as snapshot:
            self._model = YOLO(str(snapshot / checkpoint_path.name))
        self._predictor_type = _complete_yolo_predictor_type()
        self._predictor_seal = (
            self._predictor_type.__dict__["postprocess"],
            self._predictor_type.__dict__["postprocess"].__code__,
            self._predictor_type._phycam_upstream_postprocess,
            self._predictor_type._phycam_upstream_postprocess_code,
            self._predictor_type._phycam_isolated_nms,
            self._predictor_type._phycam_isolated_nms.__code__,
        )
        identity = model_identity(
            backend="ultralytics-yolo",
            model_id=checkpoint_path.name,
            revision=None,
            artifacts=[artifact_record],
            implementation={
                "adapter": "ultralytics.bchw_float_rgb.v4",
                "artifact_load_semantics": (
                    "single_source_read_then_private_exact_bytes_snapshot.v1"
                ),
                "ultralytics_version": ultralytics.__version__,
                "torch_version": torch.__version__,
                "requested_device": self.device,
                "confidence_threshold": self.confidence,
                "nms_iou_threshold": self.iou,
                "nms_time_limit_policy": _YOLO_NMS_TIME_LIMIT_POLICY,
                "upstream_postprocess_source_sha256": (
                    self._predictor_type._phycam_upstream_postprocess_source_sha256
                ),
                "upstream_nms_source_sha256": (
                    self._predictor_type._phycam_original_nms_source_sha256
                ),
                "maximum_detections_before_coco_limit": self.maximum_detections,
                "maximum_detections": self.evaluation_maximum_detections,
                "global_detection_cap": (
                    "stable_descending_score_mergesort_before_native_mapping.v1"
                ),
                "class_agnostic_nms": False,
                "test_time_augmentation": False,
                "half_precision": False,
                "input_contract": {
                    "layout": "BCHW_RGB",
                    "sample_type": "float32",
                    "range": [0.0, 1.0],
                    "adapter_resize": None,
                    "adapter_uint8_quantization": False,
                },
                "output_label_space": "coco80_contiguous",
            },
        )
        frozen = freeze_json_value(identity)
        assert isinstance(frozen, MappingProxyType)
        self.identity: Mapping[str, Any] = frozen
        self._execution_device: str | None = None

    @property
    def execution_device(self) -> str:
        if self._execution_device is None:
            raise RuntimeError("the adapter has not completed detector inference")
        return self._execution_device

    def detect_batch(self, inputs: Sequence[DetectorInput]) -> tuple[dict[str, Any], ...]:
        """Infer on one ordered batch and return JSON-compatible raw outputs."""

        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
            raise TypeError("inputs must be an ordered sequence of DetectorInput values")
        batch_inputs = tuple(inputs)
        if not batch_inputs:
            raise ValueError("detector batch must not be empty")
        if not all(isinstance(item, DetectorInput) for item in batch_inputs):
            raise TypeError("detector batch must contain DetectorInput values")
        shapes = {item.frame.shape for item in batch_inputs}
        if len(shapes) != 1:
            raise ValueError("detector inputs in one batch must share a shape")
        import torch

        arrays = np.stack(
            [np.array(item.array, dtype=np.float32, copy=True) for item in batch_inputs],
            axis=0,
        )
        tensor = torch.from_numpy(arrays).permute(0, 3, 1, 2).contiguous()
        # Ultralytics 8.4.37 otherwise abandons the unprocessed tail of a batch
        # after ``2 + 0.05 * batch_size`` seconds and merely logs a warning.
        # At the near-threshold-free COCO confidence floor this was observed on
        # real val2017 batches.  A model-local custom predictor makes that time
        # limit unreachable without mutating Ultralytics process-global state.
        previous_predictor = getattr(self._model, "predictor", None)
        observed_predictor_seal = (
            self._predictor_type.__dict__.get("postprocess"),
            getattr(self._predictor_type.__dict__.get("postprocess"), "__code__", None),
            self._predictor_type._phycam_upstream_postprocess,
            self._predictor_type._phycam_upstream_postprocess.__code__,
            self._predictor_type._phycam_isolated_nms,
            self._predictor_type._phycam_isolated_nms.__code__,
        )
        if any(
            observed is not expected
            for observed, expected in zip(observed_predictor_seal, self._predictor_seal)
        ):
            raise RuntimeError("Ultralytics complete-NMS predictor implementation drifted")
        if previous_predictor is None:
            previous_nms_calls = 0
            previous_postprocess_calls = 0
        else:
            if type(previous_predictor) is not self._predictor_type:
                raise RuntimeError("Ultralytics predictor binding drifted before inference")
            previous_nms_calls = previous_predictor._phycam_nms_calls
            previous_postprocess_calls = previous_predictor._phycam_postprocess_calls
        results = self._model.predict(
            source=tensor,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.maximum_detections,
            agnostic_nms=False,
            augment=False,
            half=False,
            device=self.device,
            verbose=False,
            predictor=self._predictor_type,
        )
        predictor = getattr(self._model, "predictor", None)
        if type(predictor) is not self._predictor_type:
            raise RuntimeError("Ultralytics did not retain the complete-NMS predictor")
        if predictor._phycam_nms_time_limit_policy != _YOLO_NMS_TIME_LIMIT_POLICY:
            raise RuntimeError("Ultralytics complete-NMS predictor policy drifted")
        if predictor._phycam_nms_calls != previous_nms_calls + 1:
            raise RuntimeError("Ultralytics did not execute complete NMS exactly once")
        if predictor._phycam_postprocess_calls != previous_postprocess_calls + 1:
            raise RuntimeError("Ultralytics did not execute postprocessing exactly once")
        if len(results) != len(batch_inputs):
            raise RuntimeError("Ultralytics output cardinality does not match the input batch")
        actual_device = getattr(predictor, "device", None)
        if actual_device is None:
            raise RuntimeError("Ultralytics predictor did not expose its execution device")
        verified = verify_actual_torch_device(self.device, actual_device)
        if self._execution_device is None:
            self._execution_device = verified
        elif self._execution_device != verified:
            raise RuntimeError("detector execution device drifted between batches")
        output: list[dict[str, Any]] = []
        for detector_input, result in zip(batch_inputs, results):
            if tuple(result.orig_shape) != detector_input.geometry.output_shape:
                raise RuntimeError("Ultralytics changed the declared detector-input geometry")
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                output.append({"boxes": [], "labels": [], "scores": []})
                continue
            box_values = boxes.xyxy.detach().cpu().numpy()
            label_values = boxes.cls.detach().cpu().numpy().astype(np.int64)
            score_values = boxes.conf.detach().cpu().numpy()
            if not (len(box_values) == len(label_values) == len(score_values) == len(boxes)):
                raise RuntimeError("Ultralytics output fields have inconsistent cardinality")
            # COCOeval's useCats=True implementation applies maxDets within each
            # image/category slice.  Apply the declared detector-level cap here
            # so every model contributes at most 100 predictions per image.
            order = np.argsort(-score_values, kind="mergesort")[
                : self.evaluation_maximum_detections
            ]
            output.append(
                {
                    "boxes": box_values[order].tolist(),
                    "labels": label_values[order].tolist(),
                    "scores": score_values[order].tolist(),
                }
            )
        return tuple(output)


class TorchvisionFasterRCNNAdapter:
    """Pinned Faster R-CNN adapter with detector resizing disabled.

    Torchvision's channel normalization remains part of the model contract,
    while ``min_size``/``max_size`` are fixed to the already-letterboxed input
    shape. This preserves the repository's single explicit geometric
    preprocessing boundary.
    """

    __slots__ = (
        "_execution_device",
        "_model",
        "confidence",
        "device",
        "identity",
        "input_shape",
        "nms_iou",
    )

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        input_shape: Sequence[int] = (640, 640),
        device: str = "cpu",
        confidence: float = 0.001,
        nms_iou: float = 0.5,
    ) -> None:
        self.input_shape = _shape2(input_shape, name="input_shape")
        self.confidence = _probability(confidence, name="confidence")
        self.nms_iou = _probability(nms_iou, name="nms_iou")
        self.device = validate_torch_device(device)
        checkpoint_path = Path(checkpoint)
        try:
            checkpoint_payload = read_artifact_bytes(checkpoint_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Faster R-CNN checkpoint is missing: {checkpoint_path}"
            ) from exc
        artifact_record = describe_artifact_bytes(
            checkpoint_payload,
            published_name=checkpoint_path.name,
        )
        try:
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
        except ImportError as exc:
            raise ImportError(
                "TorchvisionFasterRCNNAdapter requires the optional eval dependencies"
            ) from exc

        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)
        self._model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=min(self.input_shape),
            max_size=max(self.input_shape),
            image_mean=image_mean,
            image_std=image_std,
            box_score_thresh=0.0,
            box_nms_thresh=self.nms_iou,
            box_detections_per_img=100,
        )
        state = torch.load(
            io.BytesIO(checkpoint_payload),
            map_location="cpu",
            weights_only=True,
        )
        self._model.load_state_dict(state, strict=True)
        postprocessor = bind_fasterrcnn_coco_sparse_postprocessor(self._model)
        self._model.eval().to(self.device)
        identity = model_identity(
            backend="torchvision-fasterrcnn",
            model_id="torchvision/fasterrcnn_resnet50_fpn@COCO_V1",
            revision=None,
            artifacts=[artifact_record],
            implementation={
                "adapter": "torchvision.fasterrcnn.fixed_input_float_rgb.v2",
                "artifact_load_semantics": "single_source_read_then_bytes_io.v1",
                "torchvision_version": postprocessor["torchvision_version"],
                "torch_version": torch.__version__,
                "requested_device": self.device,
                "confidence_threshold": self.confidence,
                "nms_iou_threshold": self.nms_iou,
                "maximum_detections": 100,
                "input_contract": {
                    "shape_hw": list(self.input_shape),
                    "layout": "CHW_RGB_list",
                    "sample_type": "float32",
                    "range": [0.0, 1.0],
                    "adapter_resize": None,
                    "normalization_mean_rgb": list(image_mean),
                    "normalization_std_rgb": list(image_std),
                },
                "output_label_space": "coco_sparse",
                "postprocessor_implementation": postprocessor["implementation_id"],
                "postprocessor_binding": "roi_heads.postprocess_detections",
                "allowed_category_ids": list(postprocessor["valid_category_ids"]),
                "coco_sparse_filter_policy": postprocessor["filter_semantics"]["fasterrcnn"],
                "coco_sparse_logit_masking": postprocessor["filter_semantics"]["logit_masking"],
                "coco_sparse_internal_cap_inflation": postprocessor["filter_semantics"][
                    "internal_cap_inflation"
                ],
                "upstream_postprocessor_source_sha256": postprocessor["upstream_source_sha256"][
                    "fasterrcnn"
                ],
                "repository_postprocessor_source_sha256": postprocessor["repository_source_sha256"][
                    "fasterrcnn"
                ],
                "repository_postprocessor_callable": postprocessor["repository_callable"][
                    "fasterrcnn"
                ],
            },
        )
        frozen = freeze_json_value(identity)
        assert isinstance(frozen, MappingProxyType)
        self.identity: Mapping[str, Any] = frozen
        self._execution_device: str | None = None

    @property
    def execution_device(self) -> str:
        if self._execution_device is None:
            raise RuntimeError("the adapter has not completed detector inference")
        return self._execution_device

    def detect_batch(self, inputs: Sequence[DetectorInput]) -> tuple[dict[str, Any], ...]:
        """Infer on one ordered batch and return sparse-COCO predictions."""

        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
            raise TypeError("inputs must be an ordered sequence of DetectorInput values")
        batch_inputs = tuple(inputs)
        if not batch_inputs:
            raise ValueError("detector batch must not be empty")
        if not all(isinstance(item, DetectorInput) for item in batch_inputs):
            raise TypeError("detector batch must contain DetectorInput values")
        if any(item.geometry.output_shape != self.input_shape for item in batch_inputs):
            raise ValueError("detector input shape drifted from the fixed model contract")
        import torch

        tensors = [
            torch.from_numpy(np.array(item.array, dtype=np.float32, copy=True))
            .permute(2, 0, 1)
            .contiguous()
            .to(self.device)
            for item in batch_inputs
        ]
        with torch.inference_mode():
            results = self._model(tensors)
        if len(results) != len(batch_inputs):
            raise RuntimeError("Faster R-CNN output cardinality does not match the input batch")
        parameter_device = next(self._model.parameters()).device
        verified = verify_actual_torch_device(self.device, parameter_device)
        if self._execution_device is None:
            self._execution_device = verified
        elif self._execution_device != verified:
            raise RuntimeError("detector execution device drifted between batches")
        output: list[dict[str, Any]] = []
        for result in results:
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]
            if boxes.device != parameter_device:
                raise RuntimeError("Faster R-CNN outputs drifted from the model device")
            keep = scores >= self.confidence
            output.append(
                {
                    "boxes": boxes[keep].detach().cpu().numpy().tolist(),
                    "labels": labels[keep].detach().cpu().numpy().astype(np.int64).tolist(),
                    "scores": scores[keep].detach().cpu().numpy().tolist(),
                }
            )
        return tuple(output)


class TorchvisionRetinaNetAdapter:
    """Pinned RetinaNet ResNet-50 FPN V2 with detector resizing disabled.

    Torchvision's channel normalization remains part of the model contract,
    while ``min_size``/``max_size`` are fixed to the already-letterboxed input
    shape. The explicit local checkpoint is loaded strictly, and outputs use
    Torchvision's sparse COCO category IDs in detector-input coordinates.
    """

    __slots__ = (
        "_execution_device",
        "_model",
        "confidence",
        "device",
        "identity",
        "input_shape",
        "nms_iou",
    )

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        input_shape: Sequence[int] = (640, 640),
        device: str = "cpu",
        confidence: float = 0.001,
        nms_iou: float = 0.5,
    ) -> None:
        self.input_shape = _shape2(input_shape, name="input_shape")
        self.confidence = _probability(confidence, name="confidence")
        self.nms_iou = _probability(nms_iou, name="nms_iou")
        self.device = validate_torch_device(device)
        checkpoint_path = Path(checkpoint)
        try:
            checkpoint_payload = read_artifact_bytes(checkpoint_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"RetinaNet checkpoint is missing: {checkpoint_path}") from exc
        artifact_record = describe_artifact_bytes(
            checkpoint_payload,
            published_name=checkpoint_path.name,
        )
        try:
            import torch
            from torchvision.models.detection import retinanet_resnet50_fpn_v2
        except ImportError as exc:
            raise ImportError(
                "TorchvisionRetinaNetAdapter requires the optional eval dependencies"
            ) from exc

        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)
        self._model = retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            min_size=min(self.input_shape),
            max_size=max(self.input_shape),
            image_mean=image_mean,
            image_std=image_std,
            score_thresh=self.confidence,
            nms_thresh=self.nms_iou,
            detections_per_img=100,
        )
        state = torch.load(
            io.BytesIO(checkpoint_payload),
            map_location="cpu",
            weights_only=True,
        )
        self._model.load_state_dict(state, strict=True)
        postprocessor = bind_retinanet_coco_sparse_postprocessor(self._model)
        self._model.eval().to(self.device)
        identity = model_identity(
            backend="torchvision-retinanet",
            model_id="torchvision/retinanet_resnet50_fpn_v2@COCO_V1",
            revision=None,
            artifacts=[artifact_record],
            implementation={
                "adapter": "torchvision.retinanet_resnet50_fpn_v2.fixed_input_float_rgb.v2",
                "artifact_load_semantics": "single_source_read_then_bytes_io.v1",
                "torchvision_version": postprocessor["torchvision_version"],
                "torch_version": torch.__version__,
                "requested_device": self.device,
                "confidence_threshold": self.confidence,
                "nms_iou_threshold": self.nms_iou,
                "maximum_detections": 100,
                "input_contract": {
                    "shape_hw": list(self.input_shape),
                    "layout": "CHW_RGB_list",
                    "sample_type": "float32",
                    "range": [0.0, 1.0],
                    "adapter_resize": None,
                    "normalization_mean_rgb": list(image_mean),
                    "normalization_std_rgb": list(image_std),
                },
                "output_label_space": "coco_sparse",
                "postprocessor_implementation": postprocessor["implementation_id"],
                "postprocessor_binding": "postprocess_detections",
                "allowed_category_ids": list(postprocessor["valid_category_ids"]),
                "coco_sparse_filter_policy": postprocessor["filter_semantics"]["retinanet"],
                "coco_sparse_logit_masking": postprocessor["filter_semantics"]["logit_masking"],
                "coco_sparse_internal_cap_inflation": postprocessor["filter_semantics"][
                    "internal_cap_inflation"
                ],
                "upstream_postprocessor_source_sha256": postprocessor["upstream_source_sha256"][
                    "retinanet"
                ],
                "repository_postprocessor_source_sha256": postprocessor["repository_source_sha256"][
                    "retinanet"
                ],
                "repository_postprocessor_callable": postprocessor["repository_callable"][
                    "retinanet"
                ],
            },
        )
        frozen = freeze_json_value(identity)
        assert isinstance(frozen, MappingProxyType)
        self.identity: Mapping[str, Any] = frozen
        self._execution_device: str | None = None

    @property
    def execution_device(self) -> str:
        if self._execution_device is None:
            raise RuntimeError("the adapter has not completed detector inference")
        return self._execution_device

    def detect_batch(self, inputs: Sequence[DetectorInput]) -> tuple[dict[str, Any], ...]:
        """Infer on one ordered batch and return sparse-COCO predictions."""

        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
            raise TypeError("inputs must be an ordered sequence of DetectorInput values")
        batch_inputs = tuple(inputs)
        if not batch_inputs:
            raise ValueError("detector batch must not be empty")
        if not all(isinstance(item, DetectorInput) for item in batch_inputs):
            raise TypeError("detector batch must contain DetectorInput values")
        if any(item.geometry.output_shape != self.input_shape for item in batch_inputs):
            raise ValueError("detector input shape drifted from the fixed model contract")
        import torch

        tensors = [
            torch.from_numpy(np.array(item.array, dtype=np.float32, copy=True))
            .permute(2, 0, 1)
            .contiguous()
            .to(self.device)
            for item in batch_inputs
        ]
        with torch.inference_mode():
            results = self._model(tensors)
        if len(results) != len(batch_inputs):
            raise RuntimeError("RetinaNet output cardinality does not match the input batch")
        parameter_device = next(self._model.parameters()).device
        verified = verify_actual_torch_device(self.device, parameter_device)
        if self._execution_device is None:
            self._execution_device = verified
        elif self._execution_device != verified:
            raise RuntimeError("detector execution device drifted between batches")
        output: list[dict[str, Any]] = []
        for result in results:
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]
            if boxes.device != parameter_device:
                raise RuntimeError("RetinaNet outputs drifted from the model device")
            keep = scores >= self.confidence
            output.append(
                {
                    "boxes": boxes[keep].detach().cpu().numpy().tolist(),
                    "labels": labels[keep].detach().cpu().numpy().astype(np.int64).tolist(),
                    "scores": scores[keep].detach().cpu().numpy().tolist(),
                }
            )
        return tuple(output)


class HuggingFaceDETRAdapter:
    """Pinned DETR ResNet-50 using the repository's detector geometry verbatim.

    The adapter accepts only the exact local ``facebook/detr-resnet-50``
    safetensors snapshot. It verifies the weights, model config, and processor
    config before loading with ``local_files_only=True``. The Hugging Face image
    processor is deliberately not invoked: :class:`DetectorInput` already owns
    the single geometric resize, so this adapter only applies the pinned RGB
    normalization to the existing floating-point tensor.
    """

    __slots__ = (
        "_execution_device",
        "_image_mean",
        "_image_std",
        "_model",
        "confidence",
        "device",
        "identity",
        "input_shape",
    )

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_shape: Sequence[int] = (640, 640),
        device: str = "cpu",
        confidence: float = 0.001,
    ) -> None:
        self.input_shape = _shape2(input_shape, name="input_shape")
        self.confidence = _probability(confidence, name="confidence")
        self.device = validate_torch_device(device)
        model_directory = _detr_model_directory(model_path)

        artifact_records: list[dict[str, Any]] = []
        artifact_payloads: dict[str, bytes] = {}
        for name, expected in _DETR_ARTIFACTS.items():
            artifact_path = model_directory / name
            payload = read_artifact_bytes(artifact_path)
            artifact_payloads[name] = payload
            artifact_records.append(verify_artifact_bytes(payload, expected, published_name=name))
        config = _strict_json_object(
            artifact_payloads["config.json"],
            artifact_name="config.json",
        )
        preprocessor = _strict_json_object(
            artifact_payloads["preprocessor_config.json"],
            artifact_name="preprocessor_config.json",
        )
        artifact_id2label = _validate_detr_config(config)
        self._image_mean, self._image_std = _validate_detr_preprocessor(preprocessor)

        try:
            import huggingface_hub
            import safetensors
            import timm
            import torch
            import torchvision
            import transformers
            from transformers import DetrConfig, DetrForObjectDetection
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceDETRAdapter requires the optional experiments dependencies"
            ) from exc

        with _private_artifact_snapshot(
            artifact_payloads,
            prefix="phycam-detr-artifacts-",
        ) as snapshot:
            runtime_config = DetrConfig.from_pretrained(
                str(snapshot),
                local_files_only=True,
            )
            _validate_loaded_detr_config(runtime_config, artifact_id2label)
            backbone_guards: list[str] = []
            backbone_config = getattr(runtime_config, "backbone_config", None)
            if backbone_config is not None and hasattr(
                backbone_config,
                "use_pretrained_backbone",
            ):
                setattr(backbone_config, "use_pretrained_backbone", False)
                backbone_guards.append("backbone_config.use_pretrained_backbone=false")
            if hasattr(runtime_config, "use_pretrained_backbone"):
                setattr(runtime_config, "use_pretrained_backbone", False)
                backbone_guards.append("config.use_pretrained_backbone=false")
            if not backbone_guards:
                raise RuntimeError(
                    "loaded DETR config cannot disable external pretrained-backbone initialization"
                )

            loaded = DetrForObjectDetection.from_pretrained(
                str(snapshot),
                config=runtime_config,
                local_files_only=True,
                use_safetensors=True,
                output_loading_info=True,
            )
        if not isinstance(loaded, tuple) or len(loaded) != 2 or not isinstance(loaded[1], Mapping):
            raise RuntimeError("DETR loader did not return state-dict loading attestation")
        self._model, loading_info = loaded
        missing_keys = tuple(sorted(loading_info.get("missing_keys", ())))
        unexpected_keys = tuple(sorted(loading_info.get("unexpected_keys", ())))
        mismatched_keys = tuple(sorted(loading_info.get("mismatched_keys", ())))
        error_messages = tuple(str(message) for message in loading_info.get("error_msgs", ()))
        unsupported_unexpected = set(unexpected_keys).difference(_DETR_ALLOWED_IGNORED_STATE_KEYS)
        if missing_keys or mismatched_keys or error_messages or unsupported_unexpected:
            raise RuntimeError(
                "pinned DETR state dict did not load under the supported strict contract"
            )
        loaded_config = getattr(self._model, "config", None)
        if loaded_config is None:
            raise RuntimeError("loaded DETR model did not expose its config")
        _validate_loaded_detr_config(loaded_config, artifact_id2label)
        self._model.eval().to(self.device)

        versions = {
            "huggingface_hub_version": getattr(huggingface_hub, "__version__", None),
            "safetensors_version": getattr(safetensors, "__version__", None),
            "timm_version": getattr(timm, "__version__", None),
            "torch_version": getattr(torch, "__version__", None),
            "torchvision_version": getattr(torchvision, "__version__", None),
            "transformers_version": getattr(transformers, "__version__", None),
        }
        if any(not isinstance(version, str) or not version for version in versions.values()):
            raise RuntimeError("DETR runtime libraries did not expose exact version strings")
        identity = model_identity(
            backend="huggingface-transformers-detr",
            model_id=_DETR_MODEL_ID,
            revision=_DETR_REVISION,
            artifacts=artifact_records,
            implementation={
                "adapter": "transformers.detr_resnet50.detector_input_manual_normalize.v2",
                "artifact_load_semantics": (
                    "single_source_reads_then_private_exact_bytes_snapshot.v1"
                ),
                **versions,
                "requested_device": self.device,
                "local_files_only": True,
                "external_backbone_initialization_disabled": backbone_guards,
                "weights_format": "safetensors",
                "state_dict_loading_attestation": {
                    "missing_keys": list(missing_keys),
                    "unexpected_keys": list(unexpected_keys),
                    "allowed_ignored_state_keys": list(_DETR_ALLOWED_IGNORED_STATE_KEYS),
                    "mismatched_keys": list(mismatched_keys),
                    "error_messages": list(error_messages),
                },
                "confidence_threshold": self.confidence,
                "maximum_detections": 100,
                "object_queries": 100,
                "source_preprocessor_contract": {
                    "image_processor_type": preprocessor["image_processor_type"],
                    "do_resize": preprocessor["do_resize"],
                    "declared_resize": preprocessor["size"],
                    "do_normalize": preprocessor["do_normalize"],
                    "image_mean_rgb": list(self._image_mean),
                    "image_std_rgb": list(self._image_std),
                },
                "input_contract": {
                    "shape_hw": list(self.input_shape),
                    "layout": "BCHW_RGB",
                    "sample_type": "float32",
                    "range_before_normalization": [0.0, 1.0],
                    "repository_detector_input_required": True,
                    "huggingface_image_processor_invoked": False,
                    "adapter_resize": None,
                    "adapter_rescale": None,
                    "adapter_uint8_quantization": False,
                    "normalization_mean_rgb": list(self._image_mean),
                    "normalization_std_rgb": list(self._image_std),
                    "normalized_padding": "forced_zero_outside_letterbox_resized_region",
                    "pixel_mask": "letterbox_resized_region_valid_int64",
                },
                "box_contract": {
                    "format": "xyxy",
                    "coordinates": "detector_input_pixels",
                    "normalized_reference": "pixel_mask_valid_resized_region",
                    "conversion": "scale_by_resized_shape_then_add_letterbox_offset",
                    "adapter_clipping": False,
                },
                "output_label_space": "coco_sparse",
                "allowed_category_ids": list(_COCO_SPARSE_CATEGORY_IDS),
                "postprocessing": {
                    "class_probability": "softmax_excluding_final_no_object_class",
                    "nms": None,
                },
            },
        )
        frozen = freeze_json_value(identity)
        assert isinstance(frozen, MappingProxyType)
        self.identity: Mapping[str, Any] = frozen
        self._execution_device: str | None = None

    @property
    def execution_device(self) -> str:
        if self._execution_device is None:
            raise RuntimeError("the adapter has not completed detector inference")
        return self._execution_device

    def detect_batch(self, inputs: Sequence[DetectorInput]) -> tuple[dict[str, Any], ...]:
        """Infer without another resize and return sparse-COCO DETR outputs."""

        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
            raise TypeError("inputs must be an ordered sequence of DetectorInput values")
        batch_inputs = tuple(inputs)
        if not batch_inputs:
            raise ValueError("detector batch must not be empty")
        if not all(isinstance(item, DetectorInput) for item in batch_inputs):
            raise TypeError("detector batch must contain DetectorInput values")
        if any(item.geometry.output_shape != self.input_shape for item in batch_inputs):
            raise ValueError("detector input shape drifted from the fixed model contract")

        import torch

        arrays = np.stack(
            [np.array(item.array, dtype=np.float32, copy=True) for item in batch_inputs],
            axis=0,
        )
        pixel_values = torch.from_numpy(arrays).permute(0, 3, 1, 2).contiguous().to(self.device)
        mean = torch.tensor(
            self._image_mean, dtype=pixel_values.dtype, device=pixel_values.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self._image_std, dtype=pixel_values.dtype, device=pixel_values.device
        ).view(1, 3, 1, 1)
        pixel_values = ((pixel_values - mean) / std).contiguous()
        pixel_mask = torch.zeros(
            (len(batch_inputs), *self.input_shape),
            dtype=torch.int64,
            device=pixel_values.device,
        )
        for batch_index, detector_input in enumerate(batch_inputs):
            geometry = detector_input.geometry
            top = geometry.pad_top
            left = geometry.pad_left
            resized_height, resized_width = geometry.resized_shape
            pixel_mask[
                batch_index,
                top : top + resized_height,
                left : left + resized_width,
            ] = 1
        # Hugging Face pads normalized tensors with zero and excludes that
        # region from transformer attention. Reproduce that contract after the
        # repository-owned geometric letterbox, independent of its RGB pad.
        pixel_values = pixel_values * pixel_mask[:, None].to(dtype=pixel_values.dtype)
        with torch.inference_mode():
            result = self._model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        try:
            parameter_device = next(self._model.parameters()).device
        except (AttributeError, StopIteration) as exc:
            raise RuntimeError("DETR model does not expose a parameter device") from exc
        verified = verify_actual_torch_device(self.device, parameter_device)
        if self._execution_device is None:
            self._execution_device = verified
        elif self._execution_device != verified:
            raise RuntimeError("detector execution device drifted between batches")

        logits = getattr(result, "logits", None)
        pred_boxes = getattr(result, "pred_boxes", None)
        if not isinstance(logits, torch.Tensor) or not isinstance(pred_boxes, torch.Tensor):
            raise RuntimeError("DETR output did not expose logits and pred_boxes tensors")
        expected_logits_shape = (len(batch_inputs), 100, 92)
        expected_boxes_shape = (len(batch_inputs), 100, 4)
        if tuple(logits.shape) != expected_logits_shape:
            raise RuntimeError(
                f"DETR logits shape drifted from {expected_logits_shape}: {tuple(logits.shape)}"
            )
        if tuple(pred_boxes.shape) != expected_boxes_shape:
            raise RuntimeError(
                f"DETR box shape drifted from {expected_boxes_shape}: {tuple(pred_boxes.shape)}"
            )
        if logits.device != parameter_device or pred_boxes.device != parameter_device:
            raise RuntimeError("DETR outputs drifted from the model device")
        if not torch.is_floating_point(logits) or not torch.is_floating_point(pred_boxes):
            raise RuntimeError("DETR outputs must be floating-point tensors")
        if not bool(torch.isfinite(logits).all().item()) or not bool(
            torch.isfinite(pred_boxes).all().item()
        ):
            raise RuntimeError("DETR outputs contain non-finite values")
        if bool(((pred_boxes < 0.0) | (pred_boxes > 1.0)).any().item()):
            raise RuntimeError("DETR normalized boxes left the supported [0, 1] range")

        probabilities = torch.softmax(logits, dim=-1)
        scores, labels = probabilities[..., :-1].max(dim=-1)
        cx, cy, width, height = pred_boxes.unbind(dim=-1)
        boxes = torch.stack(
            (cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height),
            dim=-1,
        )
        allowed_labels = torch.tensor(
            _COCO_SPARSE_CATEGORY_IDS,
            dtype=labels.dtype,
            device=labels.device,
        )
        output: list[dict[str, Any]] = []
        for batch_index in range(len(batch_inputs)):
            geometry = batch_inputs[batch_index].geometry
            resized_height, resized_width = geometry.resized_shape
            scale = torch.tensor(
                (resized_width, resized_height, resized_width, resized_height),
                dtype=boxes.dtype,
                device=boxes.device,
            )
            offset = torch.tensor(
                (
                    geometry.pad_left,
                    geometry.pad_top,
                    geometry.pad_left,
                    geometry.pad_top,
                ),
                dtype=boxes.dtype,
                device=boxes.device,
            )
            keep = (scores[batch_index] >= self.confidence) & torch.isin(
                labels[batch_index], allowed_labels
            )
            kept_boxes = boxes[batch_index][keep] * scale + offset
            kept_labels = labels[batch_index][keep]
            kept_scores = scores[batch_index][keep]
            if len(kept_scores) > 100:
                raise RuntimeError("DETR emitted more than the pinned 100-query maximum")
            output.append(
                {
                    "boxes": kept_boxes.detach().cpu().numpy().tolist(),
                    "labels": kept_labels.detach().cpu().numpy().astype(np.int64).tolist(),
                    "scores": kept_scores.detach().cpu().numpy().tolist(),
                }
            )
        return tuple(output)


__all__ = [
    "HuggingFaceDETRAdapter",
    "TorchvisionFasterRCNNAdapter",
    "TorchvisionRetinaNetAdapter",
    "UltralyticsYOLOAdapter",
]
