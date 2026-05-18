"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""

from __future__ import annotations

# Utilities for filtering animal detections before PRIMA demo inference.
#
# Detectron2 may return both a full-animal box and a local/partial box for the
# same animal. These helpers keep the demo pipeline from rendering the same
# animal multiple times.

from typing import Iterable

import numpy as np

ANIMAL_COCO_IDS = (15, 16, 17, 18, 19, 21, 22)


def _box_areas(boxes: np.ndarray) -> np.ndarray:
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return widths * heights


def _intersection_areas(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    return np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)


def _suppress_duplicate_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float,
    containment_threshold: float,
) -> np.ndarray:
    if len(boxes) <= 1:
        return np.arange(len(boxes), dtype=np.int64)

    boxes = boxes.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)
    areas = _box_areas(boxes)

    contained = np.zeros(len(boxes), dtype=bool)
    for idx, area in enumerate(areas):
        if area <= 0:
            contained[idx] = True
            continue
        larger = np.where(areas > area)[0]
        if len(larger) == 0:
            continue
        covered = _intersection_areas(boxes[idx], boxes[larger]) / area
        if np.any(covered >= containment_threshold):
            contained[idx] = True

    candidates = np.where(~contained)[0]
    if len(candidates) <= 1:
        return candidates

    order = candidates[np.argsort(scores[candidates])[::-1]]
    keep = []
    while len(order) > 0:
        current = order[0]
        keep.append(current)
        rest = order[1:]
        if len(rest) == 0:
            break

        inter = _intersection_areas(boxes[current], boxes[rest])
        union = areas[current] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]

    return np.array(sorted(keep), dtype=np.int64)


def select_animal_boxes(
    det_instances,
    *,
    animal_class_ids: Iterable[int] = ANIMAL_COCO_IDS,
    score_threshold: float = 0.7,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.9,
) -> tuple[np.ndarray, int]:
    """Return filtered animal boxes and the number of duplicate boxes removed."""
    class_ids = set(int(class_id) for class_id in animal_class_ids)
    classes = det_instances.pred_classes.detach().cpu().numpy()
    scores = det_instances.scores.detach().cpu().numpy()

    valid_idx = np.array(
        [
            i
            for i, (class_id, score) in enumerate(zip(classes, scores))
            if int(class_id) in class_ids and float(score) > float(score_threshold)
        ],
        dtype=np.int64,
    )
    if len(valid_idx) == 0:
        return np.zeros((0, 4), dtype=np.float32), 0

    boxes = det_instances.pred_boxes.tensor[valid_idx].detach().cpu().numpy()
    scores = scores[valid_idx]
    keep = _suppress_duplicate_boxes(
        boxes,
        scores,
        iou_threshold=iou_threshold,
        containment_threshold=containment_threshold,
    )
    return boxes[keep], int(len(boxes) - len(keep))
