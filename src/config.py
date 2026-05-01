"""
Global default hyperparameters for the course.

Each notebook and script may override these for its specific task.
"""

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED: int = 42

# ── Optimization ─────────────────────────────────────────────────────────────
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 1e-4
GRADIENT_CLIP_NORM: float = 1.0

# ── Batch & epochs ───────────────────────────────────────────────────────────
BATCH_SIZE: int = 32
EPOCHS_QUICK: int = 3   # for fast demo runs
EPOCHS_BETTER: int = 10  # for better results

# ── Regularization ───────────────────────────────────────────────────────────
DROPOUT: float = 0.1

# ── Transformer architecture ─────────────────────────────────────────────────
D_MODEL: int = 64
NUM_HEADS: int = 4
NUM_LAYERS: int = 2
DIM_FEEDFORWARD: int = 128

# ── Optimizer / scheduler ────────────────────────────────────────────────────
OPTIMIZER: str = "adamw"
ACTIVATION: str = "gelu"
NORMALIZATION: str = "layernorm"

# ── DataLoader ───────────────────────────────────────────────────────────────
NUM_WORKERS_NOTEBOOK: int = 0   # safe for Jupyter on all platforms
NUM_WORKERS_SCRIPT: int = 2     # safe on Linux/macOS; scripts handle Windows
