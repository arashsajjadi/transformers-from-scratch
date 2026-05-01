"""
Translation dataset utilities for the tiny English-to-French dataset.

The tiny_translation_en_fr.csv has 100 simple sentence pairs.
This module builds source and target vocabularies and returns
token-ID sequences padded to a fixed length.

Special tokens:
  <pad>  = 0  padding
  <unk>  = 1  unknown word
  <bos>  = 2  beginning of sequence (used in decoder input)
  <eos>  = 3  end of sequence (used in target output)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.utils.paths import TINY_TRANSLATION_CSV

# Special token strings and their fixed indices
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX, BOS_TOKEN: BOS_IDX, EOS_TOKEN: EOS_IDX}


def tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens.

    Args:
        text: A sentence string.

    Returns:
        List of word strings.
    """
    return text.lower().split()


def build_vocab(sentences: List[str]) -> Dict[str, int]:
    """Build a word vocabulary with special tokens.

    Args:
        sentences: List of sentence strings.

    Returns:
        Dict mapping token to integer index.
    """
    vocab = dict(SPECIAL_TOKENS)
    counter: Counter = Counter()
    for sent in sentences:
        counter.update(tokenize(sent))
    for word, _ in counter.most_common():
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode_source(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    """Encode a source sentence (no BOS/EOS, just tokens + padding).

    Args:
        text: Source sentence string.
        vocab: Source vocabulary.
        max_len: Fixed output length.

    Returns:
        torch.Tensor of shape [max_len].
    """
    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(t, UNK_IDX) for t in tokens]
    ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def encode_target_input(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    """Encode a target sentence for decoder input: BOS + tokens (truncated).

    The decoder input starts with <bos> and ends before <eos>.

    Args:
        text: Target sentence string.
        vocab: Target vocabulary.
        max_len: Fixed output length.

    Returns:
        torch.Tensor of shape [max_len].
    """
    tokens = tokenize(text)[: max_len - 1]  # leave room for BOS
    ids = [BOS_IDX] + [vocab.get(t, UNK_IDX) for t in tokens]
    ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def encode_target_output(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    """Encode a target sentence for decoder output: tokens + EOS (truncated).

    The decoder output ends with <eos>.

    Args:
        text: Target sentence string.
        vocab: Target vocabulary.
        max_len: Fixed output length.

    Returns:
        torch.Tensor of shape [max_len].
    """
    tokens = tokenize(text)[: max_len - 1]  # leave room for EOS
    ids = [vocab.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
    ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class TranslationDataset(Dataset):
    """Dataset for the tiny EN→FR translation task.

    Returns (src_ids, tgt_input_ids, tgt_output_ids) for each pair.

    Args:
        csv_path: Path to tiny_translation_en_fr.csv.
        src_vocab: Source vocabulary.
        tgt_vocab: Target vocabulary.
        max_src_len: Max source sequence length.
        max_tgt_len: Max target sequence length.
    """

    def __init__(
        self,
        csv_path: Path,
        src_vocab: Dict[str, int],
        tgt_vocab: Dict[str, int],
        max_src_len: int = 12,
        max_tgt_len: int = 12,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.sources = df["source"].tolist()
        self.targets = df["target"].tolist()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_ids = encode_source(self.sources[idx], self.src_vocab, self.max_src_len)
        tgt_in = encode_target_input(self.targets[idx], self.tgt_vocab, self.max_tgt_len)
        tgt_out = encode_target_output(self.targets[idx], self.tgt_vocab, self.max_tgt_len)
        return src_ids, tgt_in, tgt_out


def load_translation_data(
    csv_path: Optional[Path] = None,
    max_src_len: int = 12,
    max_tgt_len: int = 12,
    train_ratio: float = 0.85,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[str, int]]:
    """Load the tiny translation dataset and return DataLoaders.

    Args:
        csv_path: Path to CSV. Uses default if None.
        max_src_len: Max source sequence length.
        max_tgt_len: Max target sequence length.
        train_ratio: Fraction of data for training.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        seed: Seed for the split.

    Returns:
        (train_loader, val_loader, src_vocab, tgt_vocab)
    """
    if csv_path is None:
        csv_path = TINY_TRANSLATION_CSV

    df = pd.read_csv(csv_path)
    src_vocab = build_vocab(df["source"].tolist())
    tgt_vocab = build_vocab(df["target"].tolist())

    dataset = TranslationDataset(csv_path, src_vocab, tgt_vocab, max_src_len, max_tgt_len)

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

    return train_loader, val_loader, src_vocab, tgt_vocab
