"""
Script: ViT Semantic Segmentation on Synthetic Shapes.

Usage:
    python scripts/run_vit_segmentation.py
    python scripts/run_vit_segmentation.py --epochs 5 --device auto
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.utils.device import resolve_device, print_device_info, get_num_workers
from src.utils.seed import seed_everything
from src.utils.cleanup import prepare_run_dir
from src.utils.paths import RUNS_SEGMENTATION_DIR
from src.utils.reporting import (
    save_metrics_json, save_metrics_csv, save_table_csv,
    save_markdown_report, update_runs_summary
)
from src.data.synthetic_shapes import load_shapes_data
from src.models.vit_segmenter import ViTSegmenter
from src.training.segmentation import segmentation_loss_fn, segmentation_metric_fn
from src.training.trainer import Trainer
from src.metrics.segmentation import pixel_accuracy, mean_iou, dice_score
from src.visualization.plots import plot_training_curves
from src.visualization.segmentation import overlay_segmentation_mask


def parse_args():
    parser = argparse.ArgumentParser(description="ViT Segmentation on Synthetic Shapes")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
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

    run_dir = RUNS_SEGMENTATION_DIR
    prepare_run_dir(run_dir, clean=(args.clean_runs == "true"))
    use_amp = (args.amp == "true") and (device.type == "cuda")
    num_workers = get_num_workers(is_notebook=False)

    image_size = 128
    num_classes = 4
    class_names = ["background", "circle", "square", "triangle"]

    print("\nGenerating synthetic shapes dataset...")
    train_loader, val_loader = load_shapes_data(
        n_train=800,
        n_val=200,
        image_size=image_size,
        mode="segmentation",
        batch_size=args.batch_size,
        num_workers=num_workers,
        seed=args.seed,
    )

    model = ViTSegmenter(
        image_size=image_size,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
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
        loss_fn=segmentation_loss_fn,
        device=device,
        metric_fn=segmentation_metric_fn,
        clip_grad_norm=1.0,
        use_amp=use_amp,
    )

    print(f"\nTraining for {args.epochs} epochs on {device}...")
    final_metrics = trainer.train(epochs=args.epochs)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.eval()
    all_preds = []
    all_masks = []
    sample_img = None
    sample_pred = None

    with torch.no_grad():
        for images, masks in val_loader:
            images_d = images.to(device)
            logits = model(images_d)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_masks.append(masks)
            if sample_img is None:
                sample_img = images[0].cpu().numpy()
                sample_pred = preds[0].cpu().numpy()

    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)

    pa = pixel_accuracy(all_preds, all_masks)
    miou = mean_iou(all_preds, all_masks, num_classes=num_classes)
    dice = dice_score(all_preds, all_masks, num_classes=num_classes)

    metrics = {
        "pixel_accuracy": round(pa, 4),
        "mean_iou": round(miou, 4),
        "dice_score": round(dice, 4),
        "val_loss": round(final_metrics["val_loss"], 4),
        "train_loss": round(final_metrics["train_loss"], 4),
    }

    print(f"\nFinal metrics: {metrics}")

    save_metrics_json(metrics, run_dir / "vit_segmenter_metrics.json")
    save_metrics_csv(metrics, run_dir / "vit_segmenter_metrics.csv")

    history = trainer.get_history()
    plot_training_curves(
        history, run_dir / "vit_segmenter_training_curve.png",
        title="ViT Segmenter"
    )

    img0 = sample_img.transpose(1, 2, 0)
    overlay_segmentation_mask(
        img0, sample_pred,
        save_path=run_dir / "vit_segmenter_overlay.png",
        title="ViT Segmenter Overlay",
        class_names=class_names,
    )

    # Compare with U-Net if its metrics exist
    import json
    unet_metrics_path = run_dir / "unet_metrics.json"
    comparison_rows = [{"model": "ViT Segmenter", **metrics}]
    if unet_metrics_path.exists():
        unet_m = json.loads(unet_metrics_path.read_text())
        comparison_rows.append({"model": "U-Net", **unet_m})

    save_table_csv(comparison_rows, run_dir / "vit_vs_unet_comparison.csv")

    save_markdown_report(
        title="ViT Semantic Segmentation",
        summary=(
            f"Trained ViT Segmenter on synthetic shapes. "
            f"Pixel accuracy: {pa:.1%}, Mean IoU: {miou:.3f}, Dice: {dice:.3f}."
        ),
        metrics=metrics,
        figures=["vit_segmenter_training_curve.png", "vit_segmenter_overlay.png"],
        tables=["vit_vs_unet_comparison.csv"],
        output_path=run_dir / "vit_segmenter_report.md",
        device=str(device),
        hyperparams={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patch_size": 16,
            "d_model": 64,
        },
        architecture="image -> patch embed -> pos enc -> encoder -> reshape -> conv decoder -> mask",
        loss_fn="CrossEntropyLoss",
    )

    update_runs_summary(
        session_name="vit_segmentation",
        report_path=run_dir / "vit_segmenter_report.md",
        metrics=metrics,
        figures=["vit_segmenter_overlay.png"],
    )

    print(f"\nOutputs saved to {run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
