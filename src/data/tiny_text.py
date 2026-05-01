"""
Text dataset utilities for the tiny sentiment dataset.

The tiny_sentiment.csv file has 90 short sentences labeled as:
  positive, negative, neutral

This module builds a simple character-free tokenizer (word-level),
creates a vocabulary, and returns integer-encoded sequences.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.utils.paths import TINY_SENTIMENT_CSV


# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

LABEL_TO_IDX = {"positive": 0, "negative": 1, "neutral": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


def tokenize(text: str) -> List[str]:
    """Split a sentence into lowercase word tokens.

    Args:
        text: A raw sentence string.

    Returns:
        List of lowercase word strings.
    """
    return text.lower().split()


def build_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    """Build a word-to-index vocabulary from a list of sentences.

    Args:
        texts: List of raw sentence strings.
        min_freq: Minimum token frequency to include. Default 1.

    Returns:
        Dict mapping token string to integer index.
        Index 0 is reserved for PAD, index 1 for UNK.
    """
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, freq in counter.most_common():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode_sentence(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    """Encode a sentence to a fixed-length integer tensor.

    Tokens not in vocab are mapped to UNK_IDX.
    Sequences shorter than max_len are padded with PAD_IDX.
    Sequences longer than max_len are truncated.

    Args:
        text: A raw sentence string.
        vocab: Word-to-index mapping.
        max_len: Fixed output length.

    Returns:
        torch.Tensor of shape [max_len] with dtype torch.long.
    """
    tokens = tokenize(text)
    ids = [vocab.get(t, UNK_IDX) for t in tokens[:max_len]]
    # Pad to max_len
    ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class SentimentDataset(Dataset):
    """Dataset for the tiny sentiment classification task.

    Reads data/tiny/tiny_sentiment.csv and returns (input_ids, label) pairs.

    Args:
        csv_path: Path to the CSV file.
        vocab: Word-to-index mapping (built from training data).
        max_len: Maximum sequence length.
    """

    def __init__(
        self,
        csv_path: Path,
        vocab: Dict[str, int],
        max_len: int = 32,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.texts = df["text"].tolist()
        self.labels = [LABEL_TO_IDX[lbl.strip()] for lbl in df["label"]]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        input_ids = encode_sentence(self.texts[idx], self.vocab, self.max_len)
        label = self.labels[idx]
        return input_ids, label


def load_sentiment_data(
    csv_path: Optional[Path] = None,
    max_len: int = 32,
    vocab_min_freq: int = 1,
    train_ratio: float = 0.8,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load the tiny sentiment dataset and return train/val DataLoaders.

    Args:
        csv_path: Path to the CSV file. Uses default if None.
        max_len: Maximum token sequence length.
        vocab_min_freq: Minimum word frequency to include in vocab.
        train_ratio: Fraction of data used for training.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of DataLoader worker processes.
        seed: Random seed for the train/val split.

    Returns:
        (train_loader, val_loader, vocab)
    """
    if csv_path is None:
        csv_path = TINY_SENTIMENT_CSV

    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()

    # Build vocabulary from all data (small dataset, so we use all)
    vocab = build_vocab(texts, min_freq=vocab_min_freq)

    dataset = SentimentDataset(csv_path, vocab, max_len)

    # Split into train and val
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, vocab
