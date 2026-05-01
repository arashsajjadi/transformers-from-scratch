"""
Vision dataset loaders for Fashion-MNIST and CIFAR-10.

These use torchvision and require an internet connection for the first download.
All data is saved to data/raw/ which is gitignored.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from src.utils.paths import DATA_RAW_DIR


def load_fashion_mnist(
    image_size: int = 28,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load Fashion-MNIST with a train/val/test split.

    Fashion-MNIST has 60,000 training and 10,000 test images.
    We further split the training set into train/val.

    Args:
        image_size: Resize target. Keep at 28 to avoid distortion.
        batch_size: Batch size for all loaders.
        num_workers: DataLoader workers.
        pin_memory: Use pinned memory (True for CUDA only).
        seed: Seed for the train/val split.
        val_split: Fraction of training data to use for validation.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.2860], std=[0.3530]),  # Fashion-MNIST stats
    ])

    data_dir = str(DATA_RAW_DIR)

    train_full = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    n_val = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def load_cifar10(
    image_size: int = 32,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10.

    Args:
        image_size: Resize target.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        pin_memory: Pinned memory (CUDA only).
        seed: Split seed.
        val_split: Fraction for validation.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    data_dir = str(DATA_RAW_DIR)

    train_full = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    n_val = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=generator)

    # Override transform for val subset
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
