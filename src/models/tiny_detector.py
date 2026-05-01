"""
Single-object and multi-object (grid) detectors.

TinySingleObjectDetector:
    CNN backbone -> global pool -> class head + box regression head

TinyGridDetector:
    CNN backbone -> grid prediction head
    Each grid cell predicts: objectness, class logits, box coordinates

Box format: normalized (cx, cy, w, h) where all values are in [0, 1].
"""

from typing import Tuple

import torch
import torch.nn as nn


class _CNNBackbone(nn.Module):
    """Small CNN backbone for detection.

    Args:
        in_channels: Input image channels.
        out_channels: Output feature channels.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            # Block 1: [B, C, H, W] -> [B, 32, H/2, W/2]
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: -> [B, 64, H/4, W/4]
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: -> [B, out_channels, H/8, W/8]
            nn.Conv2d(64, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TinySingleObjectDetector(nn.Module):
    """CNN-based single-object detector.

    Given an image, predicts:
    - class_logits: [B, num_classes]
    - box: [B, 4] in normalized (cx, cy, w, h) format

    Args:
        in_channels: Input image channels.
        num_classes: Number of object classes (no background).
        image_size: Input image size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        backbone_out = 128

        self.backbone = _CNNBackbone(in_channels, backbone_out)

        # After backbone: spatial size = image_size // 8 (three poolings above: only 2 here)
        # Block 1 and 2 each have MaxPool2d(2,2) -> image_size // 4
        # Block 3 has no pooling -> still image_size // 4
        feat_size = image_size // 4

        # Global average pooling to get a fixed-size vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.class_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

        # Box regression head: predicts (cx, cy, w, h) normalized to [0, 1]
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid(),  # constrain to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run single-object detection.

        Args:
            x: [B, in_channels, H, W]

        Returns:
            class_logits: [B, num_classes]
            boxes: [B, 4] in normalized (cx, cy, w, h)
        """
        features = self.backbone(x)       # [B, 128, H/4, W/4]
        pooled = self.global_pool(features)  # [B, 128, 1, 1]

        class_logits = self.class_head(pooled)  # [B, num_classes]
        boxes = self.box_head(pooled)             # [B, 4]

        return class_logits, boxes

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TinyGridDetector(nn.Module):
    """YOLO-inspired grid-based multi-object detector.

    Divides the image into a grid_size x grid_size grid.
    Each cell predicts:
        - objectness score (is there an object here?)
        - class logits (which class is it?)
        - box coordinates (cx, cy, w, h) relative to the whole image

    Args:
        in_channels: Input image channels.
        num_classes: Number of object classes (no background).
        grid_size: Number of grid cells per dimension.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        grid_size: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size

        # Number of values predicted per grid cell:
        #   1 (objectness) + num_classes + 4 (box: cx, cy, w, h)
        self.n_pred = 1 + num_classes + 4

        backbone_out = 128

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, backbone_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(backbone_out),
            nn.ReLU(inplace=True),
        )

        # Adaptive pool to get exactly (grid_size x grid_size) feature map
        self.adaptive_pool = nn.AdaptiveAvgPool2d(grid_size)

        # Prediction head: 1x1 conv to produce predictions for each cell
        self.head = nn.Conv2d(backbone_out, self.n_pred, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run grid-based detection.

        Args:
            x: [B, in_channels, H, W]

        Returns:
            predictions: [B, grid_size, grid_size, n_pred]
            Where n_pred = 1 + num_classes + 4
            [0]     = objectness logit
            [1:1+C] = class logits
            [1+C:]  = box (cx, cy, w, h) raw (apply sigmoid for [0,1] range)
        """
        features = self.backbone(x)               # [B, 128, H/4, W/4]
        features = self.adaptive_pool(features)   # [B, 128, grid_size, grid_size]
        raw = self.head(features)                 # [B, n_pred, grid_size, grid_size]

        # Reshape to [B, grid_size, grid_size, n_pred]
        raw = raw.permute(0, 2, 3, 1)

        return raw

    def decode_predictions(
        self,
        raw: torch.Tensor,
        conf_threshold: float = 0.4,
        nms_iou_threshold: float = 0.5,
    ):
        """Post-process raw predictions into boxes and classes.

        Args:
            raw: [B, grid_size, grid_size, n_pred] from forward().
            conf_threshold: Objectness confidence threshold.
            nms_iou_threshold: IoU threshold for NMS.

        Returns:
            List of dicts (one per image):
                'boxes': [N, 4] normalized (cx, cy, w, h)
                'class_ids': [N] integer class predictions
                'scores': [N] objectness scores
        """
        from src.metrics.detection import nms, box_cxcywh_to_xyxy

        batch_results = []
        B = raw.size(0)

        for b in range(B):
            pred = raw[b]  # [G, G, n_pred]
            G = self.grid_size

            objectness = torch.sigmoid(pred[:, :, 0])  # [G, G]
            class_logits = pred[:, :, 1 : 1 + self.num_classes]  # [G, G, C]
            boxes_raw = torch.sigmoid(pred[:, :, 1 + self.num_classes :])  # [G, G, 4]

            # Flatten grid
            scores = objectness.view(-1)  # [G*G]
            boxes = boxes_raw.view(-1, 4)  # [G*G, 4]
            class_ids = class_logits.view(-1, self.num_classes).argmax(dim=-1)

            # Apply confidence threshold
            keep = scores > conf_threshold
            scores = scores[keep]
            boxes = boxes[keep]
            class_ids = class_ids[keep]

            # Apply NMS per class
            if len(boxes) > 0:
                boxes_xyxy = box_cxcywh_to_xyxy(boxes)
                keep_nms = nms(boxes_xyxy, scores, iou_threshold=nms_iou_threshold)
                boxes = boxes[keep_nms]
                scores = scores[keep_nms]
                class_ids = class_ids[keep_nms]

            batch_results.append({
                "boxes": boxes,
                "class_ids": class_ids,
                "scores": scores,
            })

        return batch_results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
