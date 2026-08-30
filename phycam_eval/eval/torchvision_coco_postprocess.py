"""Pinned torchvision postprocessors for the sparse COCO category space.

The COCO-trained torchvision detection heads expose 91 output indices, but only
80 of those indices are categories in the official COCO annotation schema.  A
filter applied after torchvision's detection budgets cannot recover a valid
candidate displaced by one of the unused indices.  This module therefore owns
minimal copies of the torchvision 0.26.0 postprocessors and removes unused
category candidates before each applicable budget.

The model logits are never modified, and the configured upstream budgets are
never enlarged.  Consequently, every retained valid candidate keeps its
upstream box, score, and label.  Relative ordering is unchanged except where
TorchVision leaves exact-score top-k tie ordering unspecified.
"""

from __future__ import annotations

import hashlib
import inspect
from types import MappingProxyType, MethodType
from typing import Any, Mapping

PINNED_TORCHVISION_VERSION = "0.26.0"

COCO_SPARSE_CATEGORY_IDS = (
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

_UPSTREAM_SOURCE_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "fasterrcnn": "ad194bab11211e5ee3ec88297abe9708f69ad49b10bbd1b7dec81eb039b8f8a9",
        "retinanet": "3a9ab809a3822c8c86175e7268059fd0e4048b3d034fab1c18fe4e1bc0ab299d",
    }
)


def _source_sha256(function: object) -> str:
    function = getattr(function, "__func__", function)
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise RuntimeError("cannot inspect the torchvision postprocessor source") from exc
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def fasterrcnn_coco_sparse_postprocess_detections(
    self: object,
    class_logits: object,
    box_regression: object,
    proposals: list[object],
    image_shapes: list[tuple[int, int]],
    _valid_category_ids: tuple[int, ...] = COCO_SPARSE_CATEGORY_IDS,
):
    """Torchvision 0.26.0 RoI postprocessing with pre-budget COCO filtering."""

    import torch
    import torch.nn.functional as F
    from torchvision.ops import boxes as box_ops

    device = class_logits.device
    num_classes = class_logits.shape[-1]
    if num_classes != 91:
        raise RuntimeError("the pinned sparse-COCO postprocessor requires exactly 91 head classes")
    valid_category = torch.zeros(num_classes, dtype=torch.bool, device=device)
    valid_ids = torch.tensor(_valid_category_ids, dtype=torch.int64, device=device)
    valid_category[valid_ids] = True

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
        # Remove unused COCO categories from the upstream-ordered NMS result
        # immediately before the image-wide detections_per_img budget.  Keeping
        # invalid candidates in NMS preserves upstream numerical branch and tie
        # behavior for every valid candidate.
        keep = keep[valid_category[labels[keep]]]
        # keep only topk scoring predictions; the configured budget is unchanged
        keep = keep[: self.detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels


def retinanet_coco_sparse_postprocess_detections(
    self: object,
    head_outputs: Mapping[str, list[object]],
    anchors: list[list[object]],
    image_shapes: list[tuple[int, int]],
    _valid_category_ids: tuple[int, ...] = COCO_SPARSE_CATEGORY_IDS,
) -> list[dict[str, object]]:
    """Torchvision 0.26.0 RetinaNet postprocessing with pre-budget COCO filtering."""

    import torch
    from torchvision.models.detection import _utils as det_utils
    from torchvision.ops import boxes as box_ops

    class_logits = head_outputs["cls_logits"]
    box_regression = head_outputs["bbox_regression"]

    num_images = len(image_shapes)

    detections: list[dict[str, object]] = []

    for index in range(num_images):
        box_regression_per_image = [br[index] for br in box_regression]
        logits_per_image = [cl[index] for cl in class_logits]
        anchors_per_image, image_shape = anchors[index], image_shapes[index]

        image_boxes = []
        image_scores = []
        image_labels = []

        for box_regression_per_level, logits_per_level, anchors_per_level in zip(
            box_regression_per_image, logits_per_image, anchors_per_image
        ):
            num_classes = logits_per_level.shape[-1]
            if num_classes != 91:
                raise RuntimeError(
                    "the pinned sparse-COCO postprocessor requires exactly 91 head classes"
                )
            valid_category = torch.zeros(
                num_classes,
                dtype=torch.bool,
                device=logits_per_level.device,
            )
            valid_ids = torch.tensor(
                _valid_category_ids,
                dtype=torch.int64,
                device=logits_per_level.device,
            )
            valid_category[valid_ids] = True

            # remove low scoring boxes and unused COCO category candidates before
            # the per-feature-level topk_candidates budget
            scores_per_level = torch.sigmoid(logits_per_level).flatten()
            candidate_labels = (
                torch.arange(scores_per_level.numel(), device=scores_per_level.device) % num_classes
            )
            keep_idxs = (scores_per_level > self.score_thresh) & valid_category[candidate_labels]
            scores_per_level = scores_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            # keep only topk scoring predictions; the configured budget is unchanged
            num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = topk_idxs % num_classes

            boxes_per_level = self.box_coder.decode_single(
                box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
            )
            boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        # non-maximum suppression
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
        # Reassert the category boundary on the upstream-ordered NMS result
        # immediately before the image-wide detections_per_img budget.  This is
        # intentionally a candidate filter, not logit masking or cap inflation.
        keep = keep[valid_category[image_labels[keep]]]
        keep = keep[: self.detections_per_img]

        detections.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
            }
        )

    return detections


def verify_pinned_torchvision_postprocessors() -> Mapping[str, Any]:
    """Fail closed unless the installed upstream postprocessors are the pinned sources."""

    import torchvision
    from torchvision.models.detection.retinanet import RetinaNet
    from torchvision.models.detection.roi_heads import RoIHeads

    installed_version = torchvision.__version__
    normalized_version = installed_version.partition("+")[0]
    if normalized_version != PINNED_TORCHVISION_VERSION:
        raise RuntimeError(
            "torchvision version drifted from the pinned postprocessor contract: "
            f"expected {PINNED_TORCHVISION_VERSION!r}, got {installed_version!r}"
        )
    upstream = {
        "fasterrcnn": _source_sha256(RoIHeads.postprocess_detections),
        "retinanet": _source_sha256(RetinaNet.postprocess_detections),
    }
    for name, expected in _UPSTREAM_SOURCE_SHA256.items():
        if upstream[name] != expected:
            raise RuntimeError(f"torchvision {name} postprocessor source drifted")
    owned_defaults = {
        "fasterrcnn": fasterrcnn_coco_sparse_postprocess_detections.__defaults__,
        "retinanet": retinanet_coco_sparse_postprocess_detections.__defaults__,
    }
    expected_defaults = (COCO_SPARSE_CATEGORY_IDS,)
    if any(defaults != expected_defaults for defaults in owned_defaults.values()):
        raise RuntimeError("repository sparse-COCO postprocessor defaults drifted")
    return MappingProxyType(
        {
            "implementation_id": "torchvision.coco_sparse_prebudget_postprocess.v1",
            "torchvision_version": normalized_version,
            "valid_category_ids": COCO_SPARSE_CATEGORY_IDS,
            "upstream_source_sha256": MappingProxyType(upstream),
            "repository_source_sha256": MappingProxyType(
                {
                    "fasterrcnn": _source_sha256(fasterrcnn_coco_sparse_postprocess_detections),
                    "retinanet": _source_sha256(retinanet_coco_sparse_postprocess_detections),
                }
            ),
            "repository_callable": MappingProxyType(
                {
                    "fasterrcnn": (
                        "phycam_eval.eval.torchvision_coco_postprocess."
                        "fasterrcnn_coco_sparse_postprocess_detections"
                    ),
                    "retinanet": (
                        "phycam_eval.eval.torchvision_coco_postprocess."
                        "retinanet_coco_sparse_postprocess_detections"
                    ),
                }
            ),
            "filter_semantics": MappingProxyType(
                {
                    "fasterrcnn": "valid_sparse_coco_before_detections_per_img",
                    "retinanet": (
                        "valid_sparse_coco_before_each_level_topk_and_detections_per_img"
                    ),
                    "logit_masking": False,
                    "internal_cap_inflation": False,
                }
            ),
        }
    )


def _verify_current_binding(
    bound: object,
    *,
    detector: str,
    owned: object,
    upstream: object,
    expected_owner: object,
) -> bool:
    if not callable(bound) or getattr(bound, "__self__", None) is not expected_owner:
        raise RuntimeError(f"the {detector} instance postprocessor has an invalid binding")
    function = getattr(bound, "__func__", bound)
    if function is owned:
        return True
    if function is not upstream:
        raise RuntimeError(f"the {detector} instance postprocessor drifted from pinned upstream")
    return False


def bind_fasterrcnn_coco_sparse_postprocessor(model: object) -> Mapping[str, Any]:
    """Verify and bind the owned postprocessor to a torchvision Faster R-CNN."""

    from torchvision.models.detection.roi_heads import RoIHeads

    provenance = verify_pinned_torchvision_postprocessors()
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is None:
        raise TypeError("model must expose torchvision Faster R-CNN roi_heads")
    bound = getattr(roi_heads, "postprocess_detections", None)
    if bound is None:
        raise TypeError("model roi_heads must expose postprocess_detections")
    if not _verify_current_binding(
        bound,
        detector="fasterrcnn",
        owned=fasterrcnn_coco_sparse_postprocess_detections,
        upstream=RoIHeads.postprocess_detections,
        expected_owner=roi_heads,
    ):
        roi_heads.postprocess_detections = MethodType(
            fasterrcnn_coco_sparse_postprocess_detections, roi_heads
        )
    if (
        getattr(roi_heads.postprocess_detections, "__func__", None)
        is not fasterrcnn_coco_sparse_postprocess_detections
        or getattr(roi_heads.postprocess_detections, "__self__", None) is not roi_heads
    ):
        raise RuntimeError("failed to seal the owned Faster R-CNN postprocessor")
    return provenance


def bind_retinanet_coco_sparse_postprocessor(model: object) -> Mapping[str, Any]:
    """Verify and bind the owned postprocessor to a torchvision RetinaNet."""

    from torchvision.models.detection.retinanet import RetinaNet

    provenance = verify_pinned_torchvision_postprocessors()
    bound = getattr(model, "postprocess_detections", None)
    if bound is None:
        raise TypeError("model must expose torchvision RetinaNet postprocess_detections")
    if not _verify_current_binding(
        bound,
        detector="retinanet",
        owned=retinanet_coco_sparse_postprocess_detections,
        upstream=RetinaNet.postprocess_detections,
        expected_owner=model,
    ):
        model.postprocess_detections = MethodType(
            retinanet_coco_sparse_postprocess_detections, model
        )
    if (
        getattr(model.postprocess_detections, "__func__", None)
        is not retinanet_coco_sparse_postprocess_detections
        or getattr(model.postprocess_detections, "__self__", None) is not model
    ):
        raise RuntimeError("failed to seal the owned RetinaNet postprocessor")
    return provenance


__all__ = [
    "COCO_SPARSE_CATEGORY_IDS",
    "PINNED_TORCHVISION_VERSION",
    "bind_fasterrcnn_coco_sparse_postprocessor",
    "bind_retinanet_coco_sparse_postprocessor",
    "fasterrcnn_coco_sparse_postprocess_detections",
    "retinanet_coco_sparse_postprocess_detections",
    "verify_pinned_torchvision_postprocessors",
]
