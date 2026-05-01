"""
Loss and metric functions for sequence-to-sequence translation.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.synthetic_translation import PAD_IDX


def translation_loss_fn(model: nn.Module, batch, device: torch.device) -> torch.Tensor:
    """Cross-entropy loss for translation, ignoring padding tokens.

    Batch format: (src_ids, tgt_input_ids, tgt_output_ids)

    Args:
        model: Seq2SeqTransformer model.
        batch: Tuple of (src_ids, tgt_input_ids, tgt_output_ids).
        device: Target device.

    Returns:
        scalar loss.
    """
    src, tgt_in, tgt_out = batch
    src = src.to(device)
    tgt_in = tgt_in.to(device)
    tgt_out = tgt_out.to(device)

    logits, _ = model(src, tgt_in)  # [B, tgt_len, vocab_size]

    # Reshape for cross-entropy: [B*tgt_len, vocab_size] and [B*tgt_len]
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        tgt_out.reshape(B * T),
        ignore_index=PAD_IDX,
    )
    return loss


def translation_token_accuracy(
    model: nn.Module, batch, device: torch.device
) -> Dict[str, float]:
    """Compute token-level accuracy (excluding padding).

    Args:
        model: Seq2SeqTransformer model.
        batch: (src_ids, tgt_input_ids, tgt_output_ids).
        device: Target device.

    Returns:
        dict with 'token_accuracy'.
    """
    src, tgt_in, tgt_out = batch
    src = src.to(device)
    tgt_in = tgt_in.to(device)
    tgt_out = tgt_out.to(device)

    logits, _ = model(src, tgt_in)
    preds = logits.argmax(dim=-1)  # [B, tgt_len]

    # Only count non-padding positions
    non_pad = (tgt_out != PAD_IDX)
    correct = ((preds == tgt_out) & non_pad).sum().item()
    total = non_pad.sum().item()

    acc = correct / max(total, 1)
    return {"token_accuracy": acc}
