# Session Report: ViT-Segmenter — Semantic Segmentation

**Date:** 2026-05-03 18:28:21  
**Device:** cuda  

## Summary

ViTSegmenter (image_size=128, patch_size=8, d_model=128, heads=8, layers=4) trained for 20 epochs. Best epoch: 18. Pixel acc: 0.9983, mean IoU: 0.9877, Dice: 0.9938.

## Architecture

```
PatchEmbedding(8×8×3 → 128) + SinusoidalPE + TransformerEncoder×4 (8 heads, FFN 128→256→128) → reshape [B, 128, 16, 16] → PixelDecoder: (BilinearUpsample×2 + Conv3×3 + BN + ReLU + Conv3×3 + BN + ReLU)×3 → Conv1×1(4 classes)
```

**Loss function:** Weighted CrossEntropyLoss + 0.7 × foreground DiceLoss + 0.1 × boundary loss

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| patch_size | 8 |
| d_model | 128 |
| num_heads | 8 |
| num_layers | 4 |
| dim_feedforward | 256 |
| batch_size | 16 |
| epochs | 20 |
| lr | 0.0006 |

## Metrics

| Metric | Value |
|--------|-------|
| pixel_accuracy | 0.9983 |
| mean_iou | 0.9877 |
| dice_score | 0.9938 |
| best_epoch | 18 |
| best_miou_during_training | 0.9877 |
| final_train_loss | 0.1514 |
| final_val_loss | 0.1492 |
| num_params | 624644 |
| num_epochs | 20 |
| patch_size | 8 |
| d_model | 128 |
| num_heads | 8 |
| num_layers | 4 |
| dim_feedforward | 256 |

## Figures

![vit_segmenter_training_curve.png](vit_segmenter_training_curve.png)
![vit_segmenter_overlay.png](vit_segmenter_overlay.png)
![vit_segmenter_examples.png](vit_segmenter_examples.png)

## Tables

- [vit_vs_unet_comparison.csv](vit_vs_unet_comparison.csv)
