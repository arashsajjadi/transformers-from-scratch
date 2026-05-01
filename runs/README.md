# Runs Directory

This directory contains outputs saved by notebooks and scripts.

## Structure

```
runs/
├── SUMMARY.md              <- auto-updated after every session
├── setup/                  <- notebook 00: environment setup
├── attention/              <- notebooks 01-05: attention and encoder
├── text_classification/    <- notebook 06 + script
├── translation/            <- notebook 07 + script
├── timeseries/             <- notebook 08 + script
├── vit_classification/     <- notebooks 09-10 + script
├── segmentation/           <- notebooks 11-12 + scripts
└── detection/              <- notebooks 13-15 + scripts
```

## Output types

Each session saves:
- `metrics.json` — final metric values
- `metrics.csv` — same metrics as a CSV table
- `training_curve.png` — loss and accuracy over epochs
- `session_report.md` — human-readable summary
- Additional PNG figures, CSV tables, and sample predictions

## Git policy

Small files (PNG, CSV, JSON, MD) can be committed to share results.
Large model checkpoints (.pt, .pth) are gitignored.

## Cleanup

Scripts and notebooks automatically delete old output files before writing new ones.
This prevents stale results from accumulating.

To skip cleanup:
```python
prepare_run_dir(run_dir, clean=False)
```

Or from the command line:
```bash
python scripts/run_text_classification.py --clean-runs false
```
