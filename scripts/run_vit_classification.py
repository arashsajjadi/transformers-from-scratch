"""
Script: Vision Transformer (ViT) Image Classification.

Trains ViT on Fashion-MNIST.

Usage:
    python scripts/run_vit_classification.py
    python scripts/run_vit_classification.py --epochs 5 --device auto
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.utils.device import resolve_device, print_device_info, get_num_workers, get_pin_memory
from src.utils.seed import seed_everything
from src.utils.cleanup import prepare_run_dir
from src.utils.paths import RUNS_VIT_CLS_DIR
from src.utils.reporting import (
    save_metrics_json, save_metrics_csv, save_markdown_report, update_runs_summary
)
from src.data.vision_datasets import load_fashion_mnist, FASHION_MNIST_CLASSES
from src.models.vit import VisionTransformer
from src.training.classification import (
    classification_loss_fn, classification_metric_fn, collect_predictions
)
from src.training.trainer import Trainer
from src.metrics.classification import accuracy_score_torch, macro_f1_from_predictions, confusion_matrix_numpy
from src.visualization.plots import plot_training_curves, plot_confusion_matrix, show_image_grid


def parse_args():
    parser = argparse.ArgumentParser(description="ViT Classification on Fashion-MNIST")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
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

    run_dir = RUNS_VIT_CLS_DIR
    prepare_run_dir(run_dir, clean=(args.clean_runs == "true"))
    use_amp = (args.amp == "true") and (device.type == "cuda")

    num_workers = get_num_workers(is_notebook=False)
    pin_memory = get_pin_memory(device)

    print("\nLoading Fashion-MNIST...")
    train_loader, val_loader, test_loader = load_fashion_mnist(
        image_size=28,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
    )
    class_names = FASHION_MNIST_CLASSES

    # ── Model ─────────────────────────────────────────────────────────────────
    model = VisionTransformer(
        image_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

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
    macro_f1 = macro_f1_from_predictions(preds, labels, num_classes=10)
    cm = confusion_matrix_numpy(preds, labels, num_classes=10)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "val_loss": round(final_metrics["val_loss"], 4),
        "train_loss": round(final_metrics["train_loss"], 4),
    }

    print(f"\nFinal metrics: {metrics}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_metrics_json(metrics, run_dir / "vit_metrics.json")
    save_metrics_csv(metrics, run_dir / "vit_metrics.csv")

    history = trainer.get_history()
    plot_training_curves(history, run_dir / "vit_training_curve.png", title="ViT on Fashion-MNIST")

    plot_confusion_matrix(cm, class_names, run_dir / "vit_confusion_matrix.png",
                          title="ViT Confusion Matrix")

    # Patch grid visualization (first image from val)
    import numpy as np
    from src.visualization.patches import show_patches
    val_iter = iter(val_loader)
    sample_images, sample_labels = next(val_iter)
    img_np = sample_images[0].cpu().numpy()
    if img_np.shape[0] == 1:
        img_np = img_np[0]
    show_patches(np.clip(img_np, 0, 1), patch_size=7, save_path=run_dir / "patch_grid.png",
                 title=f"Patches — {class_names[sample_labels[0].item()]}")

    save_markdown_report(
        title="Vision Transformer (ViT) Classification",
        summary=(
            f"Trained ViT on Fashion-MNIST. "
            f"Accuracy: {accuracy:.1%}, Macro F1: {macro_f1:.3f}."
        ),
        metrics=metrics,
        figures=["vit_training_curve.png", "vit_confusion_matrix.png", "patch_grid.png"],
        tables=[],
        output_path=run_dir / "session_report.md",
        device=str(device),
        hyperparams={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patch_size": 7,
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 2,
        },
        architecture="image -> patches -> linear embed -> CLS + pos enc -> encoder -> CLS -> classifier",
        loss_fn="CrossEntropyLoss",
    )

    update_runs_summary(
        session_name="vit_classification",
        report_path=run_dir / "session_report.md",
        metrics=metrics,
        figures=["vit_training_curve.png", "patch_grid.png"],
    )

    print(f"\nOutputs saved to {run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
