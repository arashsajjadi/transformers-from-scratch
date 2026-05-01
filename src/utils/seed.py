"""
Reproducibility utilities.

Call seed_everything(42) at the start of every notebook or script to make
all random operations deterministic.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    This makes experiments reproducible across runs on the same machine.
    Note: results may still differ across different hardware or PyTorch versions.

    Args:
        seed: Integer seed value. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # These two settings make CUDA operations deterministic but may be slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[seed] Random seed set to {seed}.")
