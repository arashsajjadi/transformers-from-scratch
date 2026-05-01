# Object Detection

## What Is Object Detection?

Object detection finds **where** objects are and **what** they are.

Input: an image [H, W, C]
Output: a list of (class, bounding_box) pairs

Example output:
```
[(circle, [x_min=20, y_min=30, x_max=50, y_max=60]),
 (square, [x_min=80, y_min=40, x_max=120, y_max=80])]
```

---

## Bounding Box Formats

Two common formats:

**xyxy** (absolute pixel coordinates):
```
[x_min, y_min, x_max, y_max]
```

**cxcywh** (normalized center format):
```
[cx/W, cy/H, w/W, h/H]  where all values are in [0, 1]
```

This course uses normalized cxcywh internally and converts to xyxy for visualization.

---

## Single-Object Detection

The simplest form: one object per image.

Architecture:
```
image -> CNN backbone -> global average pool -> [class head, box head]
```

Class head: linear -> class logits -> softmax
Box head: linear -> 4 values -> sigmoid -> (cx, cy, w, h)

Loss:
```
total = CrossEntropyLoss(class) + lambda_box * SmoothL1Loss(box)
```

SmoothL1Loss (Huber loss):
- Behaves like MSE for small errors (smooth, no exploding gradients)
- Behaves like L1 for large errors (less sensitive to outliers)

---

## IoU: Intersection over Union

IoU measures how well a predicted box matches the ground-truth box.

```
IoU = area(pred ∩ gt) / area(pred ∪ gt)
```

- IoU = 0: no overlap at all
- IoU = 1: perfect overlap
- IoU >= 0.5: typically considered a correct detection

---

## Multi-Object Detection: Grid Approach

What if there are multiple objects?

The grid approach (YOLO-style):
1. Divide the image into a G×G grid.
2. Each cell is responsible for predicting objects whose center falls in that cell.
3. Each cell predicts: objectness score, class logits, box coordinates.

For G=4, grid_size=4: 4×4=16 cells.

Loss:
```
obj_loss = BCEWithLogitsLoss(objectness for all cells)
cls_loss = CrossEntropyLoss(class for cells WITH objects)
box_loss = SmoothL1Loss(box for cells WITH objects)
total = lambda_obj * obj_loss + lambda_cls * cls_loss + lambda_box * box_loss
```

Post-processing:
- Apply confidence threshold: keep only predictions where objectness > threshold
- Apply NMS: remove duplicate detections

---

## NMS: Non-Maximum Suppression

When multiple predictions overlap the same object, we keep only the best one.

Algorithm:
1. Sort predictions by confidence score (highest first).
2. Take the highest-confidence prediction (a true positive).
3. Remove all other predictions that overlap it by more than IoU threshold.
4. Repeat with the remaining predictions.

---

## Precision and Recall

**Precision**: of all the boxes we predicted, what fraction was correct?
```
Precision = TP / (TP + FP)
```

**Recall**: of all the real objects, what fraction did we find?
```
Recall = TP / (TP + FN)
```

A prediction is a TP if it overlaps a ground-truth box with IoU >= threshold.

High threshold (e.g., 0.75): strict matching required.
Low threshold (e.g., 0.5): more lenient.

---

## Summary

```
Single-object detection:
  CNN -> pool -> class head + box head

Multi-object grid detection:
  CNN -> grid head -> [objectness, class, box] per cell
  Post-process: threshold + NMS

Metrics: IoU, accuracy, precision, recall
```
