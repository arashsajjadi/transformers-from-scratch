"""
Script: Time-Series Forecasting with a Transformer.

Trains on synthetic sine waves and predicts future values.

Usage:
    python scripts/run_timeseries.py
    python scripts/run_timeseries.py --epochs 8 --device auto
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.device import resolve_device, print_device_info, get_num_workers
from src.utils.seed import seed_everything
from src.utils.cleanup import prepare_run_dir
from src.utils.paths import RUNS_TIMESERIES_DIR
from src.utils.reporting import (
    save_metrics_json, save_metrics_csv, save_table_csv,
    save_markdown_report, update_runs_summary
)
from src.data.synthetic_timeseries import load_timeseries_data
from src.models.time_series_transformer import TimeSeriesTransformer
from src.training.timeseries import timeseries_loss_fn, timeseries_metric_fn
from src.training.trainer import Trainer
from src.visualization.plots import plot_training_curves, save_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Time-Series Forecasting")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default="noisy", choices=["clean", "noisy", "multi_freq", "trend"])
    parser.add_argument("--amp", default="false", choices=["true", "false"])
    parser.add_argument("--clean-runs", default="true", choices=["true", "false"])
    return parser.parse_args()


def main():
    args = parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    print_device_info(device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    run_dir = RUNS_TIMESERIES_DIR
    prepare_run_dir(run_dir, clean=(args.clean_runs == "true"))
    use_amp = (args.amp == "true") and (device.type == "cuda")
    num_workers = get_num_workers(is_notebook=False)

    input_length = 48
    forecast_length = 12

    print(f"\nLoading {args.mode} time-series dataset...")
    train_loader, val_loader = load_timeseries_data(
        n_train=800,
        n_val=200,
        input_length=input_length,
        forecast_length=forecast_length,
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=num_workers,
        seed=args.seed,
    )

    model = TimeSeriesTransformer(
        input_size=1,
        forecast_length=forecast_length,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=timeseries_loss_fn,
        device=device,
        metric_fn=timeseries_metric_fn,
        clip_grad_norm=1.0,
        use_amp=use_amp,
    )

    print(f"\nTraining for {args.epochs} epochs on {device}...")
    final_metrics = trainer.train(epochs=args.epochs)

    # ── Forecast plot ──────────────────────────────────────────────────────────
    model.eval()
    sample_x, sample_y = next(iter(val_loader))
    with torch.no_grad():
        pred = model(sample_x.to(device)).cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(sample_x):
            break
        past = sample_x[i, :, 0].numpy()
        true_future = sample_y[i, :, 0].numpy()
        pred_future = pred[i, :, 0]

        t_past = np.arange(len(past))
        t_future = np.arange(len(past), len(past) + len(true_future))

        ax.plot(t_past, past, "b-", label="Past", linewidth=1.5)
        ax.plot(t_future, true_future, "g-", label="True future", linewidth=1.5)
        ax.plot(t_future, pred_future, "r--", label="Predicted", linewidth=1.5)
        ax.axvline(len(past) - 0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"Sample {i+1}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Time-Series Forecasting ({args.mode} mode)")
    plt.tight_layout()
    save_figure(fig, run_dir / "forecast_plot.png")

    # Save sample forecasts
    rows = []
    for i in range(min(4, len(sample_x))):
        true_f = sample_y[i, :, 0].numpy()
        pred_f = pred[i, :, 0]
        for t in range(len(true_f)):
            rows.append({
                "sample": i,
                "step": t,
                "true": round(float(true_f[t]), 4),
                "predicted": round(float(pred_f[t]), 4),
            })
    save_table_csv(rows, run_dir / "sample_forecasts.csv")

    metrics = {
        "mse": round(final_metrics.get("mse", 0.0), 6),
        "mae": round(final_metrics.get("mae", 0.0), 6),
        "val_loss": round(final_metrics["val_loss"], 6),
        "train_loss": round(final_metrics["train_loss"], 6),
    }

    print(f"\nFinal metrics: {metrics}")

    save_metrics_json(metrics, run_dir / "metrics.json")
    save_metrics_csv(metrics, run_dir / "metrics.csv")

    history = trainer.get_history()
    plot_training_curves(history, run_dir / "training_curve.png", title="Time-Series Forecasting")

    save_markdown_report(
        title="Time-Series Forecasting",
        summary=(
            f"Trained a Transformer on synthetic {args.mode} sine waves. "
            f"MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}."
        ),
        metrics=metrics,
        figures=["training_curve.png", "forecast_plot.png"],
        tables=["sample_forecasts.csv"],
        output_path=run_dir / "session_report.md",
        device=str(device),
        hyperparams={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "input_length": input_length,
            "forecast_length": forecast_length,
            "mode": args.mode,
        },
        architecture="past values -> linear proj -> pos enc -> encoder -> regression head",
        loss_fn="MSELoss",
    )

    update_runs_summary(
        session_name="timeseries",
        report_path=run_dir / "session_report.md",
        metrics=metrics,
        figures=["forecast_plot.png"],
    )

    print(f"\nOutputs saved to {run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
