"""
Loss and metric functions for object detection.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.detection_losses import (
    single_object_detection_loss,
    grid_detection_loss,
    tiny_detr_loss,
)
from src.metrics.detection import box_iou, box_cxcywh_to_xyxy


def single_detector_loss_fn(
    model: nn.Module,
    batch,
    device: torch.device,
    lambda_box: float = 5.0,
) -> torch.Tensor:
    """Loss for TinySingleObjectDetector.

    Args:
        model: TinySingleObjectDetector.
        batch: (images, class_ids, boxes).
        device: Target device.
        lambda_box: Weight for box regression loss.

    Returns:
        scalar total loss.
    """
    images, class_ids, boxes = batch
    images = images.to(device)
    class_ids = torch.tensor(class_ids, dtype=torch.long, device=device) if not isinstance(
        class_ids, torch.Tensor
    ) else class_ids.to(device)
    boxes = boxes.to(device)

    class_logits, pred_boxes = model(images)
    return single_object_detection_loss(class_logits, pred_boxes, class_ids, boxes, lambda_box)


def single_detector_metric_fn(
    model: nn.Module, batch, device: torch.device
) -> Dict[str, float]:
    """Metrics for single-object detection: accuracy and mean IoU."""
    images, class_ids, boxes = batch
    images = images.to(device)
    class_ids = torch.tensor(class_ids, dtype=torch.long, device=device) if not isinstance(
        class_ids, torch.Tensor
    ) else class_ids.to(device)
    boxes = boxes.to(device)

    class_logits, pred_boxes = model(images)
    preds = class_logits.argmax(dim=-1)
    acc = (preds == class_ids).float().mean().item()

    # IoU between predicted and ground-truth boxes
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    gt_xyxy = box_cxcywh_to_xyxy(boxes)
    ious = torch.diagonal(box_iou(pred_xyxy, gt_xyxy))
    mean_iou_val = ious.mean().item()

    return {"accuracy": acc, "mean_iou": mean_iou_val}


def grid_detector_loss_fn(
    model: nn.Module,
    batch,
    device: torch.device,
    lambda_obj: float = 1.0,
    lambda_cls: float = 1.0,
    lambda_box: float = 5.0,
) -> torch.Tensor:
    """Loss for TinyGridDetector.

    Args:
        model: TinyGridDetector.
        batch: (images, class_ids_list, boxes_list) from multi_detection_collate_fn.
        device: Target device.

    Returns:
        scalar loss.
    """
    images, class_ids_list, boxes_list = batch
    images = images.to(device)
    boxes_list = [b.to(device) for b in boxes_list]

    raw = model(images)  # [B, G, G, n_pred]
    return grid_detection_loss(
        raw, class_ids_list, boxes_list,
        model.grid_size, model.num_classes,
        lambda_obj, lambda_cls, lambda_box, device,
    )


def detr_loss_fn(
    model: nn.Module,
    batch,
    device: torch.device,
    lambda_class: float = 1.0,
    lambda_box: float = 5.0,
    lambda_no_object: float = 0.2,
) -> torch.Tensor:
    """Loss for TinyDETR.

    Args:
        model: TinyDETR.
        batch: (images, class_ids_list, boxes_list).
        device: Target device.

    Returns:
        scalar loss.
    """
    images, class_ids_list, boxes_list = batch
    images = images.to(device)
    boxes_list = [b.to(device) for b in boxes_list]

    class_logits, pred_boxes = model(images)
    return tiny_detr_loss(
        class_logits, pred_boxes,
        class_ids_list, boxes_list,
        model.no_object_class,
        lambda_class, lambda_box, lambda_no_object,
        device,
    )
