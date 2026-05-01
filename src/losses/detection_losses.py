"""
Detection loss functions.

Implements:
  - single_object_detection_loss (for TinySingleObjectDetector)
  - grid_detection_loss (for TinyGridDetector)
  - tiny_detr_loss (for TinyDETR with simple greedy matcher)
  - simple_iou_matcher (greedy matcher for DETR)

Box format: normalized (cx, cy, w, h) in [0, 1].
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from src.metrics.detection import box_iou, box_cxcywh_to_xyxy


def single_object_detection_loss(
    class_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    target_class_ids: torch.Tensor,
    target_boxes: torch.Tensor,
    lambda_box: float = 5.0,
) -> torch.Tensor:
    """Combined classification + box regression loss for single-object detection.

    total_loss = cross_entropy(class_logits, target_class_ids)
               + lambda_box * smooth_l1(pred_boxes, target_boxes)

    Args:
        class_logits: [B, num_classes]
        pred_boxes: [B, 4] in normalized cxcywh
        target_class_ids: [B] integer class indices
        target_boxes: [B, 4] in normalized cxcywh
        lambda_box: Weight for box regression loss.

    Returns:
        Scalar total loss.
    """
    cls_loss = F.cross_entropy(class_logits, target_class_ids)
    box_loss = F.smooth_l1_loss(pred_boxes, target_boxes)
    return cls_loss + lambda_box * box_loss


def grid_detection_loss(
    raw: torch.Tensor,
    class_ids_list: List,
    boxes_list: List[torch.Tensor],
    grid_size: int,
    num_classes: int,
    lambda_obj: float = 1.0,
    lambda_cls: float = 1.0,
    lambda_box: float = 5.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Grid-based multi-object detection loss (YOLO-style).

    For each image:
    - Assign each ground-truth box to the grid cell containing its center.
    - Compute objectness loss for all cells.
    - Compute class and box loss only for cells with objects.

    Args:
        raw: [B, G, G, n_pred] where n_pred = 1 + num_classes + 4
        class_ids_list: List of B lists with integer class indices per image.
        boxes_list: List of B tensors [N, 4] in normalized cxcywh.
        grid_size: G (number of grid cells per dimension).
        num_classes: C.
        lambda_obj, lambda_cls, lambda_box: Loss weights.
        device: Target device.

    Returns:
        Scalar total loss.
    """
    if device is None:
        device = raw.device

    B, G, _, n_pred = raw.shape
    # raw[b, r, c, :] = [objectness_logit, cls_logit_0, ..., cls_logit_C-1, cx, cy, w, h]

    # Build targets tensor
    # obj_target: [B, G, G] in {0.0, 1.0}
    # cls_target: [B, G, G] integer class index (only valid where obj=1)
    # box_target: [B, G, G, 4]
    obj_target = torch.zeros(B, G, G, device=device)
    cls_target = torch.zeros(B, G, G, dtype=torch.long, device=device)
    box_target = torch.zeros(B, G, G, 4, device=device)
    has_obj = torch.zeros(B, G, G, dtype=torch.bool, device=device)

    for b in range(B):
        gt_boxes = boxes_list[b]  # [N, 4] cxcywh normalized
        gt_cls = class_ids_list[b]  # list of ints

        if isinstance(gt_cls, torch.Tensor):
            gt_cls = gt_cls.tolist()

        for i, (box, cls_id) in enumerate(zip(gt_boxes, gt_cls)):
            cx, cy = box[0].item(), box[1].item()
            # Map center to grid cell
            col = min(int(cx * G), G - 1)
            row = min(int(cy * G), G - 1)
            obj_target[b, row, col] = 1.0
            cls_target[b, row, col] = int(cls_id)
            box_target[b, row, col] = box
            has_obj[b, row, col] = True

    # ─── Objectness loss (all cells) ──────────────────────────────────────────
    obj_logits = raw[:, :, :, 0]  # [B, G, G]
    obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target)

    # ─── Class loss (only cells with objects) ─────────────────────────────────
    if has_obj.any():
        cls_logits = raw[:, :, :, 1 : 1 + num_classes]  # [B, G, G, C]
        cls_logits_pos = cls_logits[has_obj]  # [N_pos, C]
        cls_target_pos = cls_target[has_obj]  # [N_pos]
        cls_loss = F.cross_entropy(cls_logits_pos, cls_target_pos)
    else:
        cls_loss = torch.tensor(0.0, device=device)

    # ─── Box loss (only cells with objects) ───────────────────────────────────
    if has_obj.any():
        box_logits = raw[:, :, :, 1 + num_classes :]  # [B, G, G, 4]
        # Apply sigmoid to constrain boxes to [0, 1]
        box_preds = torch.sigmoid(box_logits)
        box_preds_pos = box_preds[has_obj]   # [N_pos, 4]
        box_target_pos = box_target[has_obj]  # [N_pos, 4]
        box_loss = F.smooth_l1_loss(box_preds_pos, box_target_pos)
    else:
        box_loss = torch.tensor(0.0, device=device)

    total = lambda_obj * obj_loss + lambda_cls * cls_loss + lambda_box * box_loss
    return total


def simple_iou_matcher(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedily match predicted boxes to ground-truth boxes by IoU.

    This is a simplified version of Hungarian matching used for educational purposes.
    Real DETR uses the Hungarian algorithm for optimal bipartite matching.

    Algorithm:
    1. For each ground-truth box, find the prediction with the highest IoU.
    2. Do not reuse a prediction for two ground-truth boxes.
    3. Unmatched predictions are assigned as "no-object".

    Args:
        pred_boxes: [num_queries, 4] in normalized cxcywh.
        gt_boxes: [num_gt, 4] in normalized cxcywh.

    Returns:
        matched_pred_idx: [num_gt] indices into pred_boxes.
        matched_gt_idx: [num_gt] indices into gt_boxes (just 0..num_gt-1).
    """
    if len(gt_boxes) == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)  # [Q, 4]
    gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)      # [G, 4]

    iou_matrix = box_iou(pred_xyxy, gt_xyxy)    # [Q, G]

    used_pred = set()
    matched_pred_idx = []
    matched_gt_idx = []

    # For each GT box, find the best unmatched prediction
    for gt_idx in range(len(gt_boxes)):
        best_iou = -1.0
        best_pred = -1
        for pred_idx in range(len(pred_boxes)):
            if pred_idx in used_pred:
                continue
            iou_val = iou_matrix[pred_idx, gt_idx].item()
            if iou_val > best_iou:
                best_iou = iou_val
                best_pred = pred_idx
        if best_pred >= 0:
            used_pred.add(best_pred)
            matched_pred_idx.append(best_pred)
            matched_gt_idx.append(gt_idx)

    if not matched_pred_idx:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    return (
        torch.tensor(matched_pred_idx, dtype=torch.long),
        torch.tensor(matched_gt_idx, dtype=torch.long),
    )


def tiny_detr_loss(
    class_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    class_ids_list: List,
    boxes_list: List[torch.Tensor],
    no_object_class: int,
    lambda_class: float = 1.0,
    lambda_box: float = 5.0,
    lambda_no_object: float = 0.2,
    device: torch.device = None,
) -> torch.Tensor:
    """Loss for TinyDETR.

    For each image:
    1. Match predicted queries to ground-truth boxes using simple_iou_matcher.
    2. Assign matched queries a class label and box target.
    3. Assign unmatched queries the no-object class.
    4. Compute cross-entropy loss on all queries.
    5. Compute box loss only on matched queries.

    Args:
        class_logits: [B, num_queries, num_classes+1]
        pred_boxes: [B, num_queries, 4] in normalized cxcywh
        class_ids_list: List of B lists of GT class ids
        boxes_list: List of B tensors [N, 4] in normalized cxcywh
        no_object_class: Index of the no-object class
        lambda_class, lambda_box, lambda_no_object: Loss weights
        device: Target device

    Returns:
        Scalar total loss.
    """
    if device is None:
        device = class_logits.device

    B, Q, C = class_logits.shape
    total_loss = torch.tensor(0.0, device=device)

    cls_losses = []
    box_losses = []

    for b in range(B):
        gt_boxes = boxes_list[b]  # [N, 4]
        gt_cls = class_ids_list[b]
        if isinstance(gt_cls, torch.Tensor):
            gt_cls = gt_cls.tolist()
        N = len(gt_cls)

        pred_cls_b = class_logits[b]   # [Q, C]
        pred_box_b = pred_boxes[b]     # [Q, 4]

        # Default: all queries predict no-object
        cls_target = torch.full((Q,), no_object_class, dtype=torch.long, device=device)
        box_target = torch.zeros(Q, 4, device=device)
        has_match = torch.zeros(Q, dtype=torch.bool, device=device)

        if N > 0:
            gt_boxes_t = gt_boxes.to(device)
            matched_pred, matched_gt = simple_iou_matcher(pred_box_b.detach(), gt_boxes_t)

            if len(matched_pred) > 0:
                for mp, mg in zip(matched_pred.tolist(), matched_gt.tolist()):
                    cls_target[mp] = int(gt_cls[mg])
                    box_target[mp] = gt_boxes_t[mg]
                    has_match[mp] = True

        # Class loss: apply lower weight to no-object class
        weights = torch.ones(C, device=device)
        weights[no_object_class] = lambda_no_object
        cls_loss = F.cross_entropy(pred_cls_b, cls_target, weight=weights)
        cls_losses.append(cls_loss)

        # Box loss: only for matched queries
        if has_match.any():
            box_loss = F.smooth_l1_loss(pred_box_b[has_match], box_target[has_match])
            box_losses.append(box_loss)

    total_loss = lambda_class * sum(cls_losses) / B
    if box_losses:
        total_loss = total_loss + lambda_box * sum(box_losses) / B

    return total_loss
