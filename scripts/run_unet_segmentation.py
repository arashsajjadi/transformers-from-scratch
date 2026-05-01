"""
Script: U-Net Semantic Segmentation on Synthetic Shapes.

Usage:
    python scripts/run_unet_segmentation.py
    python scripts/run_unet_segmentation.py --epochs 5 --device auto
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.utils.device import resolve_device, print_device_info, get_num_workers, get_pin_memory
from src.utils.seed import seed_everything
from src.utils.cleanup import prepare_run_dir
from src.utils.paths import RUNS_SEGMENTATION_DIR
from src.utils.reporting import (
    save_metrics_json, save_metrics_csv, save_table_csv,
    save_markdown_report, update_runs_summary
)
from src.data.synthetic_shapes import load_shapes_data
from src.models.unet import UNet
from src.training.segmentation import segmentation_loss_fn, segmentation_metric_fn
from src.training.trainer import Trainer
from src.metrics.segmentation import pixel_accuracy, mean_iou, dice_score
from src.visualization.plots import plot_training_curves
from src.visualization.segmentation import overlay_segmentation_mask, plot_segmentation_examples


def parse_args():
    parser = argparse.ArgumentParser(description="U-Net Segmentation on Synthetic Shapes")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    num_classes = 4  # background, circle, square, triangle
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

    model = UNet(in_channels=3, num_classes=num_classes, base_channels=32)
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

    # ── Collect full val metrics ──────────────────────────────────────────────
    model.eval()
    all_preds = []
    all_masks = []
    sample_images = []
    sample_gt = []
    sample_pred = []

    with torch.no_grad():
        for images, masks in val_loader:
            images_d = images.to(device)
            logits = model(images_d)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_masks.append(masks)
            if len(sample_images) < 4:
                sample_images.append(images[:4].cpu().numpy())
                sample_gt.append(masks[:4].cpu().numpy())
                sample_pred.append(preds[:4].cpu().numpy())

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

    save_metrics_json(metrics, run_dir / "unet_metrics.json")
    save_metrics_csv(metrics, run_dir / "unet_metrics.csv")

    history = trainer.get_history()
    plot_training_curves(history, run_dir / "unet_training_curve.png", title="U-Net Segmentation")

    # Segmentation examples
    imgs_np = np.concatenate(sample_images, axis=0)[:4]
    gts_np = np.concatenate(sample_gt, axis=0)[:4]
    preds_np = np.concatenate(sample_pred, axis=0)[:4]

    plot_segmentation_examples(
        imgs_np, gts_np, preds_np,
        save_path=run_dir / "unet_examples.png",
        title="U-Net Segmentation Examples",
    )

    # Overlay for first image
    img0 = imgs_np[0].transpose(1, 2, 0)
    overlay_segmentation_mask(
        img0, preds_np[0],
        save_path=run_dir / "unet_overlay.png",
        title="U-Net Overlay",
        class_names=class_names,
    )

    # IoU per class table
    iou_rows = []
    for c_idx, cname in enumerate(class_names):
        pred_c = (all_preds == c_idx)
        tgt_c = (all_masks == c_idx)
        inter = (pred_c & tgt_c).float().sum().item()
        union = (pred_c | tgt_c).float().sum().item()
        iou_c = inter / (union + 1e-8) if union > 0 else float("nan")
        iou_rows.append({"class": cname, "iou": f"{iou_c:.4f}"})
    save_table_csv(iou_rows, run_dir / "unet_iou_per_class.csv")

    save_markdown_report(
        title="U-Net Semantic Segmentation",
        summary=(
            f"Trained U-Net on synthetic shapes ({num_classes} classes). "
            f"Pixel accuracy: {pa:.1%}, Mean IoU: {miou:.3f}, Dice: {dice:.3f}."
        ),
        metrics=metrics,
        figures=["unet_training_curve.png", "unet_examples.png", "unet_overlay.png"],
        tables=["unet_iou_per_class.csv"],
        output_path=run_dir / "unet_report.md",
        device=str(device),
        hyperparams={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "image_size": image_size,
            "num_classes": num_classes,
        },
        architecture="image -> encoder (down blocks) -> bottleneck -> decoder (up + skip) -> mask",
        loss_fn="CrossEntropyLoss",
    )

    update_runs_summary(
        session_name="unet_segmentation",
        report_path=run_dir / "unet_report.md",
        metrics=metrics,
        figures=["unet_overlay.png", "unet_examples.png"],
    )

    print(f"\nOutputs saved to {run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
