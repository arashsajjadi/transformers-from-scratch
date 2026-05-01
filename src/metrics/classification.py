"""
Classification metrics.

Implements:
  - accuracy_score_torch
  - macro_f1_from_predictions
  - confusion_matrix_numpy
"""

from typing import List, Optional

import numpy as np
import torch


def accuracy_score_torch(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy from prediction and label tensors.

    Args:
        preds: [N] integer predicted class indices.
        labels: [N] integer true class indices.

    Returns:
        Accuracy as a float in [0, 1].
    """
    return (preds == labels).float().mean().item()


def macro_f1_from_predictions(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Compute macro-averaged F1 score.

    Macro F1 = average F1 across all classes (each class weighted equally).
    This is appropriate when classes are balanced.

    Args:
        preds: [N] predicted class indices.
        labels: [N] true class indices.
        num_classes: Total number of classes.

    Returns:
        Macro F1 score as a float in [0, 1].
    """
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    f1_scores = []
    for c in range(num_classes):
        tp = ((preds_np == c) & (labels_np == c)).sum()
        fp = ((preds_np == c) & (labels_np != c)).sum()
        fn = ((preds_np != c) & (labels_np == c)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def confusion_matrix_numpy(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """Compute a confusion matrix.

    Entry [i, j] = number of examples with true label i predicted as j.

    Args:
        preds: [N] predicted class indices.
        labels: [N] true class indices.
        num_classes: Total number of classes.

    Returns:
        [num_classes, num_classes] integer numpy array.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    for true, pred in zip(labels_np, preds_np):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            cm[true, pred] += 1

    return cm
