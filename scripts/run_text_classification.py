"""
Script: Text Classification with a Transformer Encoder.

Trains a Transformer on the tiny sentiment dataset (positive/negative/neutral).
Saves metrics, curves, confusion matrix, and a session report to runs/text_classification/.

Usage:
    python scripts/run_text_classification.py
    python scripts/run_text_classification.py --epochs 10 --device auto --amp false
"""

import sys
import argparse
from pathlib import Path

# Add repo root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.utils.device import resolve_device, print_device_info, get_num_workers, get_pin_memory
from src.utils.seed import seed_everything
from src.utils.cleanup import prepare_run_dir
from src.utils.paths import RUNS_TEXT_CLS_DIR
from src.utils.reporting import (
    save_metrics_json, save_metrics_csv, save_markdown_report, update_runs_summary
)
from src.data.tiny_text import load_sentiment_data, LABEL_TO_IDX, IDX_TO_LABEL
from src.models.text_transformer import TextTransformerClassifier
from src.training.trainer import Trainer
from src.training.classification import classification_loss_fn, classification_metric_fn, collect_predictions
from src.metrics.classification import accuracy_score_torch, macro_f1_from_predictions, confusion_matrix_numpy
from src.visualization.plots import plot_training_curves, plot_confusion_matrix, save_figure
from src.visualization.tables import render_table_as_figure
from src.utils.reporting import save_table_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Text Classification with Transformer")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--amp", default="false", choices=["true", "false"])
    parser.add_argument("--clean-runs", default="true", choices=["true", "false"])
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print_device_info(device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    run_dir = RUNS_TEXT_CLS_DIR
    prepare_run_dir(run_dir, clean=(args.clean_runs == "true"))
    use_amp = (args.amp == "true") and (device.type == "cuda")

    num_workers = get_num_workers(is_notebook=False)
    pin_memory = get_pin_memory(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading sentiment dataset...")
    train_loader, val_loader, vocab = load_sentiment_data(
        max_len=32,
        vocab_min_freq=1,
        batch_size=args.batch_size,
        num_workers=num_workers,
        seed=args.seed,
    )
    class_names = list(LABEL_TO_IDX.keys())
    print(f"Vocab size: {len(vocab)}  |  Classes: {class_names}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TextTransformerClassifier(
        vocab_size=len(vocab),
        num_classes=len(class_names),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 2,
        max_len=64,
        dropout=0.1,
    )
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── Training ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=classification_loss_fn,
        device=device,
        metric_fn=classification_metric_fn,
        clip_grad_norm=1.0,
        use_amp=use_amp,
    )

    print(f"\nTraining for {args.epochs} epochs on {device}...")
    final_metrics = trainer.train(epochs=args.epochs)

    # ── Evaluation ────────────────────────────────────────────────────────────
    preds, labels = collect_predictions(model, val_loader, device)
    accuracy = accuracy_score_torch(preds, labels)
    macro_f1 = macro_f1_from_predictions(preds, labels, num_classes=len(class_names))
    cm = confusion_matrix_numpy(preds, labels, num_classes=len(class_names))

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "val_loss": round(final_metrics["val_loss"], 4),
        "train_loss": round(final_metrics["train_loss"], 4),
    }

    print(f"\nFinal metrics: {metrics}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_metrics_json(metrics, run_dir / "metrics.json")
    save_metrics_csv(metrics, run_dir / "metrics.csv")

    # Training curves
    history = trainer.get_history()
    plot_training_curves(history, run_dir / "training_curve.png", title="Text Classification")

    # Confusion matrix
    plot_confusion_matrix(cm, class_names, run_dir / "confusion_matrix.png",
                          title="Sentiment Confusion Matrix")

    # Sample predictions
    import pandas as pd
    from src.data.tiny_text import TINY_SENTIMENT_CSV
    df = pd.read_csv(TINY_SENTIMENT_CSV)
    texts = df["text"].tolist()
    true_labels = df["label"].tolist()

    model.eval()
    from src.data.tiny_text import encode_sentence, build_vocab
    inv_vocab = {v: k for k, v in vocab.items()}

    rows = []
    for i in range(min(10, len(texts))):
        ids = encode_sentence(texts[i], vocab, max_len=32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(ids)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = logits.argmax(dim=-1).item()
        rows.append({
            "text": texts[i][:50],
            "true_label": true_labels[i],
            "predicted": IDX_TO_LABEL[pred_idx],
            "confidence": f"{probs[pred_idx]:.3f}",
        })

    save_table_csv(rows, run_dir / "sample_predictions.csv")

    # Session report
    save_markdown_report(
        title="Text Classification",
        summary=(
            f"Trained a Transformer encoder on the tiny sentiment dataset "
            f"({len(class_names)} classes). Achieved {accuracy:.1%} accuracy and "
            f"macro F1 of {macro_f1:.3f}."
        ),
        metrics=metrics,
        figures=["training_curve.png", "confusion_matrix.png"],
        tables=["sample_predictions.csv"],
        output_path=run_dir / "session_report.md",
        device=str(device),
        hyperparams={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
        },
        architecture="text -> tokenizer -> embedding -> positional encoding -> encoder -> mean pool -> classifier",
        loss_fn="CrossEntropyLoss",
    )

    update_runs_summary(
        session_name="text_classification",
        report_path=run_dir / "session_report.md",
        metrics=metrics,
        figures=["training_curve.png", "confusion_matrix.png"],
    )

    print(f"\nOutputs saved to {run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
