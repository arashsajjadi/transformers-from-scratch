"""
Custom collate functions for variable-length data.
"""

from typing import List, Tuple

import torch


def pad_sequence_batch(
    sequences: List[torch.Tensor],
    pad_value: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of variable-length sequences to the same length.

    Args:
        sequences: List of 1D tensors of varying lengths.
        pad_value: Value to use for padding.

    Returns:
        (padded_batch, lengths)
        padded_batch: [batch_size, max_len]
        lengths: [batch_size] with original lengths
    """
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(sequences)

    padded = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq

    return padded, lengths
