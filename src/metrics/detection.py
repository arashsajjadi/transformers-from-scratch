"""
Object detection metrics and box utilities.

Implements:
  - box_cxcywh_to_xyxy
  - box_xyxy_to_cxcywh
  - box_iou
  - nms
  - simple_detection_precision_recall
"""

from typing import List, Tuple

import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x_min, y_min, x_max, y_max).

    Args:
        boxes: [..., 4] in (cx, cy, w, h) format.

    Returns:
        [..., 4] in (x_min, y_min, x_max, y_max) format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (x_min, y_min, x_max, y_max) to (cx, cy, w, h).

    Args:
        boxes: [..., 4] in (x_min, y_min, x_max, y_max) format.

    Returns:
        [..., 4] in (cx, cy, w, h) format.
    """
    x_min, y_min, x_max, y_max = boxes.unbind(-1)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes.

    Both inputs must be in (x_min, y_min, x_max, y_max) format.

    Args:
        boxes1: [N, 4]
        boxes2: [M, 4]

    Returns:
        [N, M] IoU matrix.
    """
    # Expand to [N, M, 4] for broadcasting
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    # Intersection
    inter_x_min = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])  # [N, M]
    inter_y_min = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x_max = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y_max = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x_max - inter_x_min).clamp(min=0)
    inter_h = (inter_y_max - inter_y_min).clamp(min=0)
    inter_area = inter_w * inter_h  # [N, M]

    # Union
    union_area = area1[:, None] + area2[None, :] - inter_area  # [N, M]

    iou = inter_area / (union_area + 1e-8)  # [N, M]
    return iou


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Non-Maximum Suppression.

    Removes duplicate detections by keeping only the highest-scoring box
    when two boxes overlap more than iou_threshold.

    Args:
        boxes: [N, 4] in (x_min, y_min, x_max, y_max) format.
        scores: [N] confidence scores.
        iou_threshold: Maximum allowed overlap between kept boxes.

    Returns:
        [K] indices of kept boxes.
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)

    # Sort by score (highest first)
    order = scores.argsort(descending=True)
    keep = []

    while len(order) > 0:
        idx = order[0].item()
        keep.append(idx)

        if len(order) == 1:
            break

        rest = order[1:]
        iou = box_iou(boxes[idx].unsqueeze(0), boxes[rest])  # [1, len(rest)]
        iou = iou.squeeze(0)  # [len(rest)]

        # Keep only boxes with low enough overlap
        order = rest[iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


def simple_detection_precision_recall(
    pred_boxes_list: List[torch.Tensor],
    pred_scores_list: List[torch.Tensor],
    gt_boxes_list: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> Tuple[float, float]:
    """Compute simple precision and recall for a batch of detections.

    Matches each prediction to a ground-truth box by IoU.
    A prediction is a true positive if its IoU with any unmatched GT box >= iou_threshold.

    Args:
        pred_boxes_list: List of [N, 4] tensors in xyxy format, one per image.
        pred_scores_list: List of [N] score tensors.
        gt_boxes_list: List of [M, 4] GT box tensors in xyxy format.
        iou_threshold: IoU threshold for a match.

    Returns:
        (precision, recall) as floats.
    """
    total_tp = 0
    total_pred = 0
    total_gt = 0

    for pred_boxes, pred_scores, gt_boxes in zip(
        pred_boxes_list, pred_scores_list, gt_boxes_list
    ):
        total_pred += len(pred_boxes)
        total_gt += len(gt_boxes)

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue

        iou_matrix = box_iou(pred_boxes, gt_boxes)  # [N, M]
        matched_gt = set()

        # Sort predictions by score
        order = pred_scores.argsort(descending=True)
        for pred_idx in order.tolist():
            best_iou = -1.0
            best_gt = -1
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                iou_val = iou_matrix[pred_idx, gt_idx].item()
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt = gt_idx
            if best_iou >= iou_threshold and best_gt >= 0:
                total_tp += 1
                matched_gt.add(best_gt)

    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_gt, 1)
    return precision, recall
