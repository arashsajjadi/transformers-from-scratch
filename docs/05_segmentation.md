# Semantic Segmentation

## What Is Semantic Segmentation?

Semantic segmentation assigns a class label to **every pixel** in an image.

Input: an image [H, W, C]
Output: a class map [H, W] where each pixel has a class index

Example:
```
Input image:
  [a photo with a circle and a square on a gray background]

Output mask:
  0 0 0 0 0 0 0 0 0 0   (0 = background)
  0 0 1 1 1 0 0 0 0 0   (1 = circle)
  0 0 1 1 1 0 2 2 0 0   (2 = square)
  0 0 0 0 0 0 2 2 0 0
```

---

## U-Net Architecture

U-Net is the standard architecture for semantic segmentation.
It has an encoder (downsampling path) and a decoder (upsampling path).

```
Input [B, C, H, W]
  |
  v DownBlock 1: Conv -> BN -> ReLU -> MaxPool
  |   skip1 = features before pooling [B, 32, H, W]
  |   x = pooled [B, 32, H/2, W/2]
  |
  v DownBlock 2
  |   skip2 [B, 64, H/2, W/2]
  |   x [B, 64, H/4, W/4]
  |
  v DownBlock 3
  |   skip3 [B, 128, H/4, W/4]
  |   x [B, 128, H/8, W/8]
  |
  v Bottleneck: DoubleConv
  |   x [B, 256, H/8, W/8]
  |
  v UpBlock 3: Upsample + Concat(skip3) + DoubleConv
  |   x [B, 128, H/4, W/4]
  |
  v UpBlock 2: Upsample + Concat(skip2) + DoubleConv
  |   x [B, 64, H/2, W/2]
  |
  v UpBlock 1: Upsample + Concat(skip1) + DoubleConv
  |   x [B, 32, H, W]
  |
  v Conv1x1
  output [B, num_classes, H, W]
```

---

## Skip Connections

The key innovation in U-Net is skip connections.

When an encoder block downsamples the image, spatial detail is lost.
Skip connections pass the pre-pooling features directly to the decoder.

This gives the decoder both:
- **Semantic context** from the bottleneck (what objects are there?)
- **Spatial detail** from the skip connections (where exactly are they?)

---

## Loss Functions

### CrossEntropyLoss

```
CE = -sum_c y_c * log(p_c)
```

Treats each pixel independently. Works well when classes are balanced.

### Dice Loss

```
Dice = 1 - (2 * |P ∩ G|) / (|P| + |G|)
```

Directly optimizes the overlap between prediction and ground truth.
Better when some classes cover very few pixels.

This course uses CrossEntropyLoss for training and reports Dice as a metric.

---

## Metrics

### Pixel Accuracy

```
pixel_accuracy = correct pixels / total pixels
```

Simple but misleading when one class (background) dominates.

### Mean IoU (Intersection over Union)

For each class c:
```
IoU_c = (predicted c AND true c) / (predicted c OR true c)
```

Mean IoU = average over all classes.

This is the standard metric for segmentation.

### Dice Score

Similar to IoU but with a different formula.

```
Dice_c = 2 * |P_c ∩ G_c| / (|P_c| + |G_c|)
```

---

## ViT for Segmentation

ViT was designed for classification (uses only the CLS token).
To use it for segmentation, we keep the patch tokens.

After the encoder, patch tokens have shape [B, num_patches, d_model].
We reshape them to [B, d_model, h_patches, w_patches] — a feature map.
Then a small convolutional decoder upsamples this to [B, num_classes, H, W].

```
Image -> patch embedding -> Transformer encoder
-> reshape tokens -> convolutional decoder -> upsample -> segmentation map
```

U-Net vs ViT Segmenter:
- U-Net uses skip connections for fine spatial detail
- ViT Segmenter uses self-attention for global context
- For small datasets, U-Net often wins
- For large datasets, ViT Segmenter can be competitive
