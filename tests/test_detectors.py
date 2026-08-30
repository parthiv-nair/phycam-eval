"""One focused preprocessing/postprocessing contract per reported detector."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import phycam_eval.eval.detectors as detector_module
from phycam_eval.capture import make_ldr_input_frame
from phycam_eval.eval.detectors import HuggingFaceDETRAdapter, UltralyticsYOLOAdapter
from phycam_eval.eval.preprocess import LetterboxConfig, letterbox
from phycam_eval.eval.torchvision_coco_postprocess import (
    fasterrcnn_coco_sparse_postprocess_detections,
    retinanet_coco_sparse_postprocess_detections,
    verify_pinned_torchvision_postprocessors,
)

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def _detector_input(source_shape=(4, 6)):
    frame = make_ldr_input_frame(
        np.full((*source_shape, 3), 0.25, dtype=np.float32),
        image_id="detector-test",
    )
    return letterbox(frame, LetterboxConfig((4, 6), 0.75))


def test_torchvision_contract_accepts_the_pinned_local_build_suffix(monkeypatch) -> None:
    monkeypatch.setattr(torchvision, "__version__", "0.26.0+cu130")
    assert verify_pinned_torchvision_postprocessors()["torchvision_version"] == "0.26.0"


def test_fasterrcnn_filters_sparse_coco_labels_before_the_image_budget() -> None:
    class BoxCoder:
        @staticmethod
        def decode(box_regression, proposals):
            return box_regression

    model = SimpleNamespace(
        box_coder=BoxCoder(),
        score_thresh=0.001,
        nms_thresh=0.5,
        detections_per_img=1,
    )
    logits = torch.full((1, 91), -20.0)
    logits[0, 12] = 20.0  # unused COCO index, higher than the valid candidate
    logits[0, 1] = 19.0
    decoded_boxes = torch.tensor([[[1.0, 1.0, 4.0, 4.0]] * 91])
    boxes, scores, labels = fasterrcnn_coco_sparse_postprocess_detections(
        model,
        logits,
        decoded_boxes,
        [torch.zeros((1, 4))],
        [(8, 8)],
    )
    assert labels[0].tolist() == [1]
    assert len(scores[0]) == len(boxes[0]) == 1


def test_retinanet_filters_sparse_coco_labels_before_feature_topk() -> None:
    class BoxCoder:
        @staticmethod
        def decode_single(box_regression, anchors):
            return anchors + box_regression

    model = SimpleNamespace(
        box_coder=BoxCoder(),
        score_thresh=0.001,
        topk_candidates=1,
        nms_thresh=0.5,
        detections_per_img=1,
    )
    logits = torch.full((1, 1, 91), -20.0)
    logits[0, 0, 12] = 20.0  # unused index must not consume top-k
    logits[0, 0, 1] = 19.0
    result = retinanet_coco_sparse_postprocess_detections(
        model,
        {"cls_logits": [logits], "bbox_regression": [torch.zeros((1, 1, 4))]},
        [[torch.tensor([[1.0, 1.0, 4.0, 4.0]])]],
        [(8, 8)],
    )[0]
    assert result["labels"].tolist() == [1]
    assert len(result["scores"]) == len(result["boxes"]) == 1


def test_yolo_uses_bchw_float_input_and_caps_predictions_before_coco_mapping() -> None:
    count = 101

    def upstream_postprocess():
        return None

    def isolated_nms():
        return None

    class Predictor:
        _phycam_isolated_nms = staticmethod(isolated_nms)
        _phycam_nms_calls = 0
        _phycam_nms_time_limit_policy = (
            "isolated_upstream_postprocess_nms_proxy_max_time_img_inf_attested.v1"
        )
        _phycam_postprocess_calls = 0
        _phycam_upstream_postprocess = staticmethod(upstream_postprocess)
        _phycam_upstream_postprocess_code = upstream_postprocess.__code__

        def __init__(self):
            self._phycam_nms_calls = 0
            self._phycam_postprocess_calls = 0
            self.device = torch.device("cpu")

        def postprocess(self):
            return None

    class Boxes:
        xyxy = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4)
        cls = torch.arange(count, dtype=torch.int64).remainder(2)
        conf = torch.linspace(0.001, 0.999, count)

        def __len__(self):
            return count

    class Model:
        predictor = None
        source = None

        def predict(self, **kwargs):
            self.source = kwargs["source"]
            self.predictor = Predictor()
            self.predictor._phycam_nms_calls = 1
            self.predictor._phycam_postprocess_calls = 1
            return [SimpleNamespace(orig_shape=(4, 6), boxes=Boxes())]

    adapter = object.__new__(UltralyticsYOLOAdapter)
    adapter._execution_device = None
    adapter._model = Model()
    adapter._predictor_type = Predictor
    adapter._predictor_seal = (
        Predictor.__dict__["postprocess"],
        Predictor.__dict__["postprocess"].__code__,
        Predictor._phycam_upstream_postprocess,
        Predictor._phycam_upstream_postprocess_code,
        Predictor._phycam_isolated_nms,
        Predictor._phycam_isolated_nms.__code__,
    )
    adapter.confidence = 0.001
    adapter.device = "cpu"
    adapter.evaluation_maximum_detections = 100
    adapter.identity = {}
    adapter.iou = 0.7
    adapter.maximum_detections = 300

    output = adapter.detect_batch((_detector_input(),))[0]
    assert tuple(adapter._model.source.shape) == (1, 3, 4, 6)
    assert adapter._model.source.dtype is torch.float32
    assert len(output["scores"]) == 100
    assert output["scores"] == pytest.approx(torch.flip(Boxes.conf[-100:], dims=(0,)).tolist())


def test_detr_masks_letterbox_padding_and_scales_boxes_to_the_valid_region(monkeypatch) -> None:
    class Model:
        def __init__(self):
            self.parameter = torch.zeros(1)
            self.call = None

        def parameters(self):
            return iter((self.parameter,))

        def __call__(self, *, pixel_values, pixel_mask):
            self.call = (pixel_values.detach().clone(), pixel_mask.detach().clone())
            logits = torch.full((1, 100, 92), -20.0)
            logits[..., 91] = 20.0
            logits[0, 0, 91] = -20.0
            logits[0, 0, 1] = 20.0
            boxes = torch.full((1, 100, 4), 0.1)
            boxes[0, 0] = torch.tensor([0.5, 0.5, 0.5, 0.5])
            return SimpleNamespace(logits=logits, pred_boxes=boxes)

    adapter = object.__new__(HuggingFaceDETRAdapter)
    adapter._execution_device = None
    adapter._image_mean = (0.485, 0.456, 0.406)
    adapter._image_std = (0.229, 0.224, 0.225)
    adapter._model = Model()
    adapter.confidence = 0.001
    adapter.device = "cpu"
    adapter.identity = {}
    adapter.input_shape = (4, 6)
    monkeypatch.setattr(detector_module, "verify_actual_torch_device", lambda *_: "cpu")

    output = adapter.detect_batch((_detector_input((2, 6)),))[0]
    pixel_values, pixel_mask = adapter._model.call
    assert pixel_mask[0].tolist() == [
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
    ]
    assert torch.count_nonzero(pixel_values[:, :, (0, 3), :]) == 0
    assert output["labels"] == [1]
    np.testing.assert_allclose(output["boxes"], [[1.5, 1.5, 4.5, 2.5]])
