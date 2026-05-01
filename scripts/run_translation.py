"""
Script: Encoder-Decoder Translation.

Trains a tiny Seq2Seq Transformer on the tiny EN-FR dataset.
The goal is educational overfitting on 100 sentence pairs.

Usage:
    python scripts/run_translation.py
    python scripts/run_translation.py --epochs 20 --device auto
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.utils.device import resolve_device, print_device_info, get_num_workers, get_pin_memory
from src.utils.seed import seed_everything
from src.utils.cleanup import prepare_run_dir
from src.utils.paths import RUNS_TRANSLATION_DIR
from src.utils.reporting import (
    save_metrics_json, save_metrics_csv, save_table_csv, save_markdown_report, update_runs_summary
)
from src.data.synthetic_translation import (
    load_translation_data, PAD_IDX, BOS_IDX, EOS_IDX
)
from src.models.seq2seq_transformer import Seq2SeqTransformer, greedy_decode
from src.training.translation import translation_loss_fn, translation_token_accuracy
from src.training.trainer import Trainer
from src.visualization.plots import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Encoder-Decoder Translation")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", default="false", choices=["true", "false"])
    parser.add_argument("--clean-runs", default="true", choices=["true", "false"])
    return parser.parse_args()


def decode_ids(ids, vocab_inv):
    """Convert integer IDs back to words."""
    tokens = []
    for idx in ids:
        if idx in (PAD_IDX, BOS_IDX):
            continue
        if idx == EOS_IDX:
            break
        tokens.append(vocab_inv.get(int(idx), "<unk>"))
    return " ".join(tokens)


def main():
    args = parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    print_device_info(device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    run_dir = RUNS_TRANSLATION_DIR
    prepare_run_dir(run_dir, clean=(args.clean_runs == "true"))
    use_amp = (args.amp == "true") and (device.type == "cuda")
    num_workers = get_num_workers(is_notebook=False)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading translation dataset...")
    train_loader, val_loader, src_vocab, tgt_vocab = load_translation_data(
        max_src_len=12,
        max_tgt_len=12,
        batch_size=args.batch_size,
        num_workers=num_workers,
        seed=args.seed,
    )
    src_vocab_inv = {v: k for k, v in src_vocab.items()}
    tgt_vocab_inv = {v: k for k, v in tgt_vocab.items()}
    print(f"Source vocab: {len(src_vocab)}  |  Target vocab: {len(tgt_vocab)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        max_src_len=16,
        max_tgt_len=16,
        dropout=0.1,
        pad_idx=PAD_IDX,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── Training ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=translation_loss_fn,
        device=device,
        metric_fn=translation_token_accuracy,
        clip_grad_norm=1.0,
        use_amp=use_amp,
    )

    print(f"\nTraining for {args.epochs} epochs on {device}...")
    final_metrics = trainer.train(epochs=args.epochs)

    # ── Translation examples ──────────────────────────────────────────────────
    import pandas as pd
    from src.utils.paths import TINY_TRANSLATION_CSV

    df = pd.read_csv(TINY_TRANSLATION_CSV)
    rows = []
    for i in range(min(15, len(df))):
        src_text = df["source"].iloc[i]
        tgt_text = df["target"].iloc[i]

        # Encode source
        from src.data.synthetic_translation import encode_source
        src_ids = encode_source(src_text, src_vocab, max_len=12).unsqueeze(0)

        # Greedy decode
        output_ids = greedy_decode(model, src_ids, BOS_IDX, EOS_IDX, max_len=14, device=device)
        translation = decode_ids(output_ids.tolist(), tgt_vocab_inv)

        rows.append({
            "source": src_text,
            "target": tgt_text,
            "predicted": translation,
            "correct": "✓" if tgt_text.lower() == translation.lower() else "✗",
        })

    save_table_csv(rows, run_dir / "translation_examples.csv")

    exact_match = sum(1 for r in rows if r["correct"] == "✓") / max(len(rows), 1)
    metrics = {
        "token_accuracy": round(final_metrics.get("token_accuracy", 0.0), 4),
        "exact_match_sample": round(exact_match, 4),
        "val_loss": round(final_metrics["val_loss"], 4),
        "train_loss": round(final_metrics["train_loss"], 4),
    }

    print(f"\nFinal metrics: {metrics}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_metrics_json(metrics, run_dir / "metrics.json")
    save_metrics_csv(metrics, run_dir / "metrics.csv")

    history = trainer.get_history()
    plot_training_curves(history, run_dir / "training_curve.png", title="Translation")

    save_markdown_report(
        title="Encoder-Decoder Translation",
        summary=(
            f"Trained a Seq2Seq Transformer on 100 EN-FR pairs for {args.epochs} epochs. "
            f"Achieved {exact_match:.0%} exact match on sampled sentences."
        ),
        metrics=metrics,
        figures=["training_curve.png"],
        tables=["translation_examples.csv"],
        output_path=run_dir / "session_report.md",
        device=str(device),
        hyperparams={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        },
        architecture="source -> encoder | target -> masked decoder -> cross-attention -> vocab",
        loss_fn="CrossEntropyLoss (ignore_index=pad)",
    )

    update_runs_summary(
        session_name="translation",
        report_path=run_dir / "session_report.md",
        metrics=metrics,
        figures=["training_curve.png"],
    )

    print(f"\nOutputs saved to {run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
