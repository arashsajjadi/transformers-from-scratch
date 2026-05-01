# Transformers from Scratch: From Attention to Vision, Segmentation, and Detection

**Author:** Arash Sajjadi

---

## What is this repository?

This is a hands-on mini-course that teaches you how Transformers work by building them from scratch using PyTorch.

You will start with a single attention score, build up to multi-head attention, implement a full Transformer encoder and decoder, then move on to text classification, machine translation, time-series forecasting, image classification with Vision Transformers, semantic segmentation, and finally object detection with a tiny DETR-style model.

Every concept is explained step-by-step. Every tensor shape is shown. Every model is trained from scratch using small, fast datasets so you can learn quickly on any machine.

---

## Why this course exists

Most online tutorials either:
- Explain attention with diagrams but skip the code, or
- Jump straight to `from transformers import BertModel` without explaining what is happening inside

This course fills the gap. You will write every layer yourself. You will see exactly how the math becomes code. You will understand what each tensor means before using it.

---

## What you will learn

- Attention intuition: queries, keys, and values
- Scaled dot-product attention
- Multi-head attention
- Sinusoidal and learned positional encoding
- Transformer encoder block (pre-norm style)
- Transformer encoder for text classification
- Transformer encoder-decoder for translation
- Time-series forecasting with a Transformer
- CNN baseline for image classification
- Vision Transformer (ViT) from scratch
- Semantic segmentation with U-Net
- Semantic segmentation with ViT
- Single-object detection basics
- Multi-object grid-based detection
- DETR-style object detection with queries

---

## What this course does not do

- No pretrained models
- No Hugging Face models or tokenizers
- No `torch.nn.Transformer` black box
- No `timm` library
- No chasing state-of-the-art benchmark numbers

The goal is to understand, not to compete.

---

## Repository structure

```
transformers-from-scratch/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── environment.yml
├── COURSE_OVERVIEW.md
├── QUICKSTART.md
├── ROADMAP.md
├── CONTRIBUTING.md
├── data/
│   ├── tiny/          <- tiny CSV datasets for text tasks
│   ├── raw/           <- downloaded datasets (gitignored)
│   └── processed/     <- preprocessed data (gitignored)
├── notebooks/         <- 17 Jupyter notebooks (00 to 16)
├── src/
│   ├── config.py      <- global default hyperparameters
│   ├── utils/         <- device, seed, paths, cleanup, reporting
│   ├── data/          <- dataset classes and data loaders
│   ├── models/        <- all model implementations from scratch
│   ├── training/      <- training loops and trainers
│   ├── losses/        <- segmentation and detection losses
│   ├── metrics/       <- evaluation metrics
│   └── visualization/ <- plotting and saving figures
├── scripts/           <- standalone training scripts
├── docs/              <- written explanations
├── assets/            <- diagrams and sample output images
├── runs/              <- saved outputs from notebooks and scripts
└── tests/             <- pytest tests
```

---

## Installation

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Conda (any OS)

```bash
conda env create -f environment.yml
conda activate transformers-course
```

---

## Device support

The code automatically selects the best available device:

1. **CUDA** — NVIDIA GPU on Linux or Windows
2. **MPS** — Apple Silicon GPU on macOS
3. **CPU** — fallback for any machine

You do not need a GPU. Every notebook is designed to run on CPU with small default settings.

Every script accepts `--device auto` (default), `--device cuda`, `--device mps`, or `--device cpu`.

---

## Quick start

```bash
jupyter notebook
```

Open `notebooks/00_course_setup.ipynb` first.

---

## Recommended learning path

| Notebook | Topic |
|----------|-------|
| 00 | Set up the environment and verify your device |
| 01 | Understand attention intuitively without math |
| 02 | Implement scaled dot-product attention |
| 03 | Implement multi-head attention |
| 04 | Understand positional encoding |
| 05 | Build a full Transformer encoder block |
| 06 | Text classification with a Transformer |
| 07 | Tiny encoder-decoder for translation |
| 08 | Time-series forecasting with a Transformer |
| 09 | CNN baseline for image classification |
| 10 | Vision Transformer (ViT) from scratch |
| 11 | Semantic segmentation with U-Net |
| 12 | Semantic segmentation with ViT |
| 13 | Single-object detection basics |
| 14 | Multi-object grid-based detection |
| 15 | Tiny DETR-style detection |
| 16 | Final project: compare all models |

---

## Datasets

| Dataset | Used in | Download required | Why |
|---------|---------|-------------------|-----|
| Tiny Sentiment CSV | Notebooks 05, 06 | No (included) | Simple, fast, custom |
| Tiny Translation CSV | Notebook 07 | No (included) | Simple EN-FR pairs |
| Synthetic time-series | Notebook 08 | No (generated) | Controllable sine waves |
| Fashion-MNIST | Notebooks 09, 10 | Yes (auto via torchvision) | Standard image benchmark |
| Synthetic shapes | Notebooks 11–15 | No (generated) | Fast, visual, controllable |
| Oxford-IIIT Pet | Notebooks 11, 12 | Optional | Real segmentation data |

---

## Expected outputs

After running notebooks and scripts you will see:

- Attention heatmaps showing which tokens attend to which
- Positional encoding visualizations
- Training loss and accuracy curves
- Confusion matrices for classification
- Translation output tables
- Time-series forecast plots
- Patch grid visualizations from ViT
- Segmentation overlays on images
- Bounding box predictions on synthetic shapes
- Model comparison tables across all tasks

---

## Saving results

- Small outputs (PNG, CSV, JSON, Markdown) are saved under `runs/`
- Old run outputs are cleaned automatically before each run
- Large model checkpoints are ignored by `.gitignore`
- You can commit small run outputs to share results

---

## Example Outputs

<!-- AUTO-GENERATED-RESULTS:START -->
This section is updated automatically after running notebooks or scripts.
<!-- AUTO-GENERATED-RESULTS:END -->

---

## Run scripts

```bash
python scripts/run_text_classification.py
python scripts/run_translation.py
python scripts/run_timeseries.py
python scripts/run_vit_classification.py
python scripts/run_unet_segmentation.py
python scripts/run_vit_segmentation.py
python scripts/run_tiny_detector.py
python scripts/run_tiny_detr.py
```

Each script saves outputs to the corresponding `runs/` subfolder.

---

## Testing

```bash
pytest tests
```

Tests check that all model output shapes are correct.

---

## Course philosophy

> Build first. Visualize often. Keep models small. Understand every tensor shape.

---

## License

MIT — see [LICENSE](LICENSE)
