# DETR Explained

## The Problem with Grid Detectors

Grid-based detectors (YOLO-style) have a fundamental limitation: one object per cell.

If two objects have their centers in the same grid cell, only one can be detected.

DETR takes a completely different approach.

---

## DETR Key Ideas

**"End-to-End Object Detection with Transformers"** (Carion et al., 2020)

Key ideas:
1. Use a Transformer decoder with **learnable object queries**
2. Each query produces exactly one prediction (class + box)
3. Use **Hungarian matching** to assign predictions to ground truth
4. No NMS required

---

## DETR Architecture

```
Image [B, C, H, W]
  |
  v  CNN Backbone (e.g., ResNet)
  |
  v  Feature map [B, d_model, h, w]
  |
  v  Flatten to sequence: [B, h*w, d_model]
  |
  v  Add 2D positional encoding
  |
  v  Transformer Encoder
  |
  v  Memory [B, h*w, d_model]
  |
  +--- N learnable object queries [B, N, d_model]
  |
  v  Transformer Decoder (queries attend to memory)
  |
  v  Decoder output [B, N, d_model]
  |
  v  Class head + Box head (per query)
  |
  v  N predictions: [(class, box), (class, box), ...]
     + one "no-object" class for unmatched queries
```

---

## Object Queries

Object queries are learnable vectors. There are N of them (e.g., N=100 in original DETR).

At the start of training, they contain random values.
During training, each query learns to focus on a different region or type of object.

In our Tiny DETR with N=5 queries:
- If there are 3 objects in the image, 3 queries should activate with object classes
- 2 queries should predict "no object"

---

## Hungarian Matching (original DETR)

The original DETR uses the Hungarian algorithm for optimal bipartite matching.

Given N predictions and M ground-truth objects (M <= N):
- Find the assignment of predictions to GT objects that minimizes a combined cost
- Cost includes: class probability cost + box L1 cost + box GIoU cost
- Unmatched predictions are assigned the "no object" class

The Hungarian algorithm guarantees the globally optimal assignment.

---

## Simple Greedy Matcher (This Course)

For educational purposes, this course uses a simple greedy matcher:

1. For each ground-truth box, find the prediction with the highest IoU.
2. Do not reuse a prediction.
3. Assign unmatched predictions to "no-object".

This is easier to understand and implement, but not as good as Hungarian matching.

---

## Why DETR Needs No NMS

In grid-based detectors, multiple cells can predict the same object.
NMS removes duplicates after prediction.

In DETR, each query is trained to be responsible for exactly one object (or no object).
The set prediction nature means there are no duplicates by design.

This is a major advantage of the set-based formulation.

---

## Loss Function

```
class_loss = CrossEntropyLoss(predicted class, assigned class)
             (with down-weighted loss for no-object queries)

box_loss = SmoothL1Loss(predicted box, assigned GT box)
           (only for queries matched to real objects)

total = lambda_class * class_loss + lambda_box * box_loss
```

The no-object weight (lambda_no_object = 0.2 in this course) reduces the influence of the
no-object class, since there are usually more no-object queries than real-object queries.

---

## Tiny DETR vs Original DETR

| Feature | Tiny DETR (this course) | Original DETR |
|---------|------------------------|---------------|
| Backbone | Small CNN | ResNet-50 |
| Queries | 5 | 100 |
| Matching | Greedy IoU | Hungarian |
| Dataset | Synthetic shapes | COCO |
| Training | Minutes | Days |
| Purpose | Education | Production |

---

## What You Should See After Training

When you visualize Tiny DETR predictions:
- Some queries should produce reasonable bounding boxes
- Some queries should correctly predict the class
- Some queries should predict "no object" for background

---

## Summary

```
DETR = CNN backbone + Transformer encoder + Transformer decoder + N object queries

Each query → one prediction (class + box or "no object")
Training: match predictions to GT using greedy IoU matcher (or Hungarian in real DETR)
Inference: keep predictions with high object confidence (not no-object)
No NMS needed: the set prediction design prevents duplicates
```
