# Quick Start Guide

This guide gets you running in under 5 minutes.

---

## 1. Install dependencies

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

---

## 2. Run the setup notebook first

```bash
jupyter notebook
```

Open `notebooks/00_course_setup.ipynb`. This will:
- Check your Python and PyTorch versions
- Print which device was selected (CUDA, MPS, or CPU)
- Save a test plot to `runs/setup/`

---

## 3. Follow the notebooks in order

Start with notebook 00 and go through them in order (00 → 16). Each notebook builds on the concepts from the previous one.

---

## 4. Run a script

To run a training script from the command line:

```bash
python scripts/run_text_classification.py
```

With options:
```bash
python scripts/run_text_classification.py --epochs 10 --batch-size 16 --device auto
```

---

## 5. Check results

All outputs are saved under `runs/`. For example:
```
runs/text_classification/training_curve.png
runs/text_classification/confusion_matrix.png
runs/text_classification/metrics.json
runs/text_classification/session_report.md
```

---

## 6. Run tests

```bash
pytest tests
```

This checks that all model output shapes are correct.

---

## Device fallback

The code automatically selects the best available device:

| Device | When used |
|--------|-----------|
| `cuda` | NVIDIA GPU is available |
| `mps` | Apple Silicon GPU (M1/M2/M3) on macOS |
| `cpu` | No GPU, or as a fallback |

To force a specific device:
```bash
python scripts/run_text_classification.py --device cpu
python scripts/run_text_classification.py --device cuda
python scripts/run_text_classification.py --device mps
```

If a requested device is not available, the code will warn you and fall back to CPU.

---

## Common issues

**Q: I get a CUDA out-of-memory error.**  
A: Reduce `--batch-size`, for example `--batch-size 8`.

**Q: Notebooks are slow on CPU.**  
A: Use the default small settings. Each notebook is designed to finish on CPU in a few minutes.

**Q: Fashion-MNIST download fails.**  
A: Make sure you have internet access. The dataset is downloaded automatically the first time.

**Q: Import error for `src`.**  
A: Make sure you are running from the repository root. The `src/` folder is a local package.

**Q: Windows multiprocessing error in DataLoader.**  
A: This is handled automatically. Scripts use `num_workers=2` but notebooks use `num_workers=0`.

---

## Folder structure of outputs

```
runs/
├── SUMMARY.md                  <- updated after every run
├── setup/                      <- notebook 00 outputs
├── attention/                  <- notebooks 01-05 outputs
├── text_classification/        <- notebook 06 + script outputs
├── translation/                <- notebook 07 + script outputs
├── timeseries/                 <- notebook 08 + script outputs
├── vit_classification/         <- notebooks 09-10 + script outputs
├── segmentation/               <- notebooks 11-12 + script outputs
└── detection/                  <- notebooks 13-15 + script outputs
```
