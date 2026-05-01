"""
Script: Tiny DETR Object Detection.

Trains a DETR-style detector with learnable object queries on synthetic shapes.

Usage:
    python scripts/run_tiny_detr.py
    python scripts/run_tiny_detr.py --epochs 12 --device auto
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
from src.utils.paths import RUNS_DETECTION_DIR
from src.utils.reporting import (
    save_metrics_json, save_metrics_csv, save_table_csv,
    save_markdown_report, update_runs_summary
)
from src.data.synthetic_shapes import load_shapes_data
from src.models.tiny_detr import TinyDETR
from src.training.detection import detr_loss_fn
from src.training.trainer import Trainer
from src.metrics.detection import box_cxcywh_to_xyxy
from src.visualization.plots import plot_training_curves
from src.visualization.detection import plot_detection_examples


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny DETR Object Detection")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-queries", type=int, default=5)
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

    run_dir = RUNS_DETECTION_DIR
    prepare_run_dir(run_dir, clean=(args.clean_runs == "true"))
    use_amp = (args.amp == "true") and (device.type == "cuda")
    num_workers = get_num_workers(is_notebook=False)

    image_size = 128
    num_classes = 3
    class_names = ["circle", "square", "triangle"]

    print("\nGenerating multi-object synthetic dataset...")
    train_loader, val_loader = load_shapes_data(
        n_train=800,
        n_val=200,
        image_size=image_size,
        mode="multi_detection",
        max_objects=3,
        batch_size=args.batch_size,
        num_workers=num_workers,
        seed=args.seed,
    )

    model = TinyDETR(
        in_channels=3,
        num_classes=num_classes,
        num_queries=args.num_queries,
        image_size=image_size,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def loss_fn(m, batch, dev):
        return detr_loss_fn(m, batch, dev, lambda_class=1.0, lambda_box=5.0, lambda_no_object=0.2)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        metric_fn=None,
        clip_grad_norm=1.0,
        use_amp=use_amp,
    )

    print(f"\nTraining for {args.epochs} epochs on {device}...")
    final_metrics = trainer.train(epochs=args.epochs)

    # ── Evaluate sample predictions ───────────────────────────────────────────
    model.eval()
    sample_data = []

    with torch.no_grad():
        for images, class_ids_list, boxes_list in val_loader:
            images_d = images.to(device)
            cls_logits, pred_boxes = model(images_d)
            # cls_logits: [B, Q, C+1], pred_boxes: [B, Q, 4]

            for b in range(min(4, len(images))):
                logits_b = cls_logits[b]  # [Q, C+1]
                boxes_b = pred_boxes[b]   # [Q, 4]
                scores_b = torch.softmax(logits_b, dim=-1)  # [Q, C+1]
                conf_b = 1.0 - scores_b[:, model.no_object_class]  # object confidence

                # Keep queries with high object confidence
                keep = conf_b > 0.3
                if keep.any():
                    pred_boxes_kept = boxes_b[keep].cpu()
                    pred_cls_kept = logits_b[keep, :model.num_classes].argmax(dim=-1).cpu().tolist()
                    pred_scores_kept = conf_b[keep].cpu().tolist()
                else:
                    pred_boxes_kept = boxes_b[:1].cpu()
                    pred_cls_kept = [0]
                    pred_scores_kept = [0.0]

                pred_xyxy = box_cxcywh_to_xyxy(pred_boxes_kept).numpy() * image_size

                gt_boxes_b = boxes_list[b].numpy()
                gt_xyxy = box_cxcywh_to_xyxy(
                    torch.tensor(gt_boxes_b, dtype=torch.float32)
                ).numpy() * image_size

                sample_data.append({
                    "image": images[b].numpy(),
                    "gt_boxes": gt_xyxy,
                    "gt_cls": class_ids_list[b],
                    "pred_boxes": pred_xyxy,
                    "pred_cls": pred_cls_kept,
                    "pred_scores": pred_scores_kept,
                })

            if len(sample_data) >= 4:
                break

    # Query table
    query_rows = []
    for b in range(min(2, len(sample_data))):
        d = sample_data[b]
        for qi in range(len(d["pred_boxes"])):
            query_rows.append({
                "image": b,
                "query": qi,
                "pred_class": class_names[d["pred_cls"][qi]] if qi < len(d["pred_cls"]) else "?",
                "confidence": f"{d['pred_scores'][qi]:.3f}" if qi < len(d["pred_scores"]) else "?",
            })
    save_table_csv(query_rows, run_dir / "tiny_detr_query_table.csv")

    metrics = {
        "val_loss": round(final_metrics["val_loss"], 4),
        "train_loss": round(final_metrics["train_loss"], 4),
    }

    print(f"\nFinal metrics: {metrics}")

    save_metrics_json(metrics, run_dir / "tiny_detr_metrics.json")
    save_metrics_csv(metrics, run_dir / "tiny_detr_metrics.csv")

    history = trainer.get_history()
    plot_training_curves(history, run_dir / "tiny_detr_training_curve.png", title="Tiny DETR")

    imgs_np = np.array([d["image"] for d in sample_data[:4]])
    gt_boxes_list = [d["gt_boxes"] for d in sample_data[:4]]
    gt_cls_list = [d["gt_cls"] for d in sample_data[:4]]
    pred_boxes_list_vis = [d["pred_boxes"] for d in sample_data[:4]]
    pred_cls_list_vis = [d["pred_cls"] for d in sample_data[:4]]
    pred_scores_list_vis = [d["pred_scores"] for d in sample_data[:4]]

    plot_detection_examples(
        imgs_np, gt_boxes_list, gt_cls_list,
        pred_boxes_list_vis, pred_cls_list_vis, pred_scores_list_vis,
        save_path=run_dir / "tiny_detr_predictions.png",
        title="Tiny DETR Predictions",
        class_names=class_names,
    )

    save_markdown_report(
        title="Tiny DETR Object Detection",
        summary=(
            f"Trained DETR-style detector with {args.num_queries} object queries. "
            f"Val loss: {metrics['val_loss']:.4f}."
        ),
        metrics=metrics,
        figures=["tiny_detr_training_curve.png", "tiny_detr_predictions.png"],
        tables=["tiny_detr_query_table.csv"],
        output_path=run_dir / "tiny_detr_report.md",
        device=str(device),
        hyperparams={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "num_queries": args.num_queries,
        },
        architecture="image -> CNN -> tokens -> encoder | queries -> decoder -> class + box",
        loss_fn="CrossEntropy (with no-object) + 5*SmoothL1",
    )

    update_runs_summary(
        "tiny_detr",
        run_dir / "tiny_detr_report.md",
        metrics,
        ["tiny_detr_predictions.png"],
    )

    print(f"\nOutputs saved to {run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
