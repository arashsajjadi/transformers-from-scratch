"""
Script: Single-Object and Grid-Based Object Detection on Synthetic Shapes.

Usage:
    python scripts/run_tiny_detector.py
    python scripts/run_tiny_detector.py --mode single --epochs 8 --device auto
    python scripts/run_tiny_detector.py --mode grid --epochs 10 --device auto
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
from src.models.tiny_detector import TinySingleObjectDetector, TinyGridDetector
from src.training.detection import (
    single_detector_loss_fn, single_detector_metric_fn,
    grid_detector_loss_fn,
)
from src.training.trainer import Trainer
from src.metrics.detection import box_cxcywh_to_xyxy, box_iou
from src.visualization.plots import plot_training_curves
from src.visualization.detection import plot_detection_examples


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny Object Detector on Synthetic Shapes")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--mode", default="single", choices=["single", "grid"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
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

    run_dir = RUNS_DETECTION_DIR
    prepare_run_dir(run_dir, clean=(args.clean_runs == "true"))
    use_amp = (args.amp == "true") and (device.type == "cuda")
    num_workers = get_num_workers(is_notebook=False)

    image_size = 128
    num_classes = 3
    class_names = ["circle", "square", "triangle"]

    if args.mode == "single":
        print("\nSingle-object detection mode.")
        print("Generating synthetic single-object dataset...")

        train_loader, val_loader = load_shapes_data(
            n_train=800,
            n_val=200,
            image_size=image_size,
            mode="single_detection",
            batch_size=args.batch_size,
            num_workers=num_workers,
            seed=args.seed,
        )

        model = TinySingleObjectDetector(
            in_channels=3,
            num_classes=num_classes,
            image_size=image_size,
        )
        print(f"Parameters: {model.count_parameters():,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        def loss_fn(m, batch, dev):
            return single_detector_loss_fn(m, batch, dev, lambda_box=5.0)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            metric_fn=single_detector_metric_fn,
            clip_grad_norm=1.0,
            use_amp=use_amp,
        )

        print(f"\nTraining for {args.epochs} epochs on {device}...")
        final_metrics = trainer.train(epochs=args.epochs)

        # Evaluate
        model.eval()
        all_cls_correct = []
        all_ious = []
        sample_data = []

        with torch.no_grad():
            for images, class_ids, boxes in val_loader:
                images_d = images.to(device)
                cls_logits, pred_boxes = model(images_d)
                preds = cls_logits.argmax(dim=-1).cpu()
                class_ids_t = torch.tensor(class_ids, dtype=torch.long) if not isinstance(
                    class_ids, torch.Tensor) else class_ids

                all_cls_correct.append((preds == class_ids_t).float())

                pred_xyxy = box_cxcywh_to_xyxy(pred_boxes.cpu())
                gt_xyxy = box_cxcywh_to_xyxy(boxes.cpu())
                for pi, gi in zip(pred_xyxy, gt_xyxy):
                    iou = box_iou(pi.unsqueeze(0), gi.unsqueeze(0))[0, 0].item()
                    all_ious.append(iou)

                if len(sample_data) < 4:
                    for i in range(min(4 - len(sample_data), len(images))):
                        sample_data.append({
                            "image": images[i].numpy(),
                            "gt_box": gt_xyxy[i].numpy() * image_size,
                            "pred_box": pred_xyxy[i].detach().numpy() * image_size,
                            "gt_cls": int(class_ids_t[i]),
                            "pred_cls": int(preds[i]),
                        })

        accuracy = torch.cat(all_cls_correct).mean().item()
        miou = np.mean(all_ious)

        metrics = {
            "accuracy": round(accuracy, 4),
            "mean_iou": round(float(miou), 4),
            "val_loss": round(final_metrics["val_loss"], 4),
            "train_loss": round(final_metrics["train_loss"], 4),
        }

        print(f"\nFinal metrics: {metrics}")

        save_metrics_json(metrics, run_dir / "single_detector_metrics.json")
        save_metrics_csv(metrics, run_dir / "single_detector_metrics.csv")

        history = trainer.get_history()
        plot_training_curves(history, run_dir / "single_detector_training_curve.png",
                             title="Single-Object Detection")

        # Detection examples
        imgs = np.array([d["image"] for d in sample_data])
        gt_boxes = [d["gt_box"].reshape(1, 4) for d in sample_data]
        pred_boxes_vis = [d["pred_box"].reshape(1, 4) for d in sample_data]
        gt_cls = [[d["gt_cls"]] for d in sample_data]
        pred_cls = [[d["pred_cls"]] for d in sample_data]

        plot_detection_examples(
            imgs, gt_boxes, gt_cls, pred_boxes_vis, pred_cls,
            save_path=run_dir / "single_detector_predictions.png",
            title="Single-Object Detection",
            class_names=class_names,
        )

        save_markdown_report(
            title="Single-Object Detection",
            summary=f"Accuracy: {accuracy:.1%}, Mean IoU: {miou:.3f}.",
            metrics=metrics,
            figures=["single_detector_training_curve.png", "single_detector_predictions.png"],
            tables=[],
            output_path=run_dir / "single_detector_report.md",
            device=str(device),
            hyperparams={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
            architecture="image -> CNN -> global pool -> class head + box head",
            loss_fn="CrossEntropyLoss + 5*SmoothL1Loss",
        )

        update_runs_summary(
            "single_object_detection",
            run_dir / "single_detector_report.md",
            metrics,
            ["single_detector_predictions.png"],
        )

    else:
        # Grid detector
        print("\nGrid-based multi-object detection mode.")
        print("Generating synthetic multi-object dataset...")

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

        model = TinyGridDetector(in_channels=3, num_classes=num_classes, grid_size=4)
        print(f"Parameters: {model.count_parameters():,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        def loss_fn(m, batch, dev):
            return grid_detector_loss_fn(m, batch, dev)

        # For grid, we only track loss (metric_fn requires more complex logic)
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

        metrics = {
            "val_loss": round(final_metrics["val_loss"], 4),
            "train_loss": round(final_metrics["train_loss"], 4),
        }

        print(f"\nFinal metrics: {metrics}")

        save_metrics_json(metrics, run_dir / "grid_detector_metrics.json")
        save_metrics_csv(metrics, run_dir / "grid_detector_metrics.csv")

        history = trainer.get_history()
        plot_training_curves(history, run_dir / "grid_detector_training_curve.png",
                             title="Grid-Based Multi-Object Detection")

        save_markdown_report(
            title="Grid-Based Multi-Object Detection",
            summary=f"Trained grid detector. Val loss: {metrics['val_loss']:.4f}.",
            metrics=metrics,
            figures=["grid_detector_training_curve.png"],
            tables=[],
            output_path=run_dir / "grid_detector_report.md",
            device=str(device),
            hyperparams={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
            architecture="image -> CNN -> grid head [objectness, class, box] per cell",
            loss_fn="BCE + CrossEntropy + SmoothL1",
        )

        update_runs_summary(
            "grid_detection",
            run_dir / "grid_detector_report.md",
            metrics,
            ["grid_detector_training_curve.png"],
        )

    print(f"\nOutputs saved to {run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
