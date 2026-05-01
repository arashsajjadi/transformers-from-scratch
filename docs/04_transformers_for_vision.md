# Transformers for Vision

## Why Not Just Use a CNN?

CNNs are excellent at capturing local patterns through convolution.
But they have limitations:

- A convolution kernel only sees a local neighborhood
- To capture global structure, you need many layers
- Long-range dependencies (e.g., left eye relates to right eye) are slow to learn

Vision Transformers capture global relationships in a single attention layer.

---

## The Core Idea: Images as Sequences of Patches

A CNN processes pixels with sliding windows.
A Vision Transformer divides the image into non-overlapping patches and treats each patch as a token.

```
Image: 28 x 28 pixels, 1 channel
Patch size: 7 x 7 pixels
Number of patches: (28/7) * (28/7) = 4 * 4 = 16 patches
Each patch: 7 * 7 * 1 = 49 pixels (flattened)
```

These 49 values are projected to d_model dimensions using a linear layer.
This is the "patch embedding".

---

## ViT Architecture

```
Input image [B, 1, 28, 28]
  |
  | split into patches
  v
[B, 16, 49]  (16 patches, 49 pixels each)
  |
  | linear projection (patch embedding)
  v
[B, 16, d_model]  (16 patch tokens)
  |
  | prepend learnable CLS token
  v
[B, 17, d_model]  (1 CLS + 16 patches)
  |
  | add learned positional embedding
  v
[B, 17, d_model]
  |
  | Transformer encoder (N blocks)
  v
[B, 17, d_model]
  |
  | take CLS token (position 0)
  v
[B, d_model]
  |
  | linear classifier
  v
[B, num_classes]
```

---

## The CLS Token

The CLS token (Classification token) is a learnable vector added at position 0.

It has no corresponding patch in the image. Its role is to aggregate information from all other patches through self-attention.

After the encoder, the CLS token output is used for classification.

---

## Positional Embedding in ViT

Unlike text Transformers that use sinusoidal encoding, ViT typically uses **learned 1D positional embeddings**.

Each patch index (0 to 16, including CLS) has its own learnable embedding vector.

The model learns which patches are near each other from the training data.

---

## Key Difference: CNN vs ViT

| Property | CNN | ViT |
|----------|-----|-----|
| Inductive bias | Local patterns first | Global from layer 1 |
| Positional info | Implicit (convolution is local) | Explicit (positional embedding) |
| Data efficiency | High (less data needed) | Low (needs lots of data) |
| Global context | Grows with depth | Available in every layer |
| Parameters for small data | Better | Worse |

For Fashion-MNIST (60K images), a small ViT performs comparably to a CNN with enough training.

---

## What Each Patch Attends To

Early layers: patches mostly attend to adjacent patches.
Later layers: patches start attending across the whole image.

This is why ViT is good for tasks that need global context (e.g., object detection, where an object can be anywhere).

---

## Memory and Computation

Attention has O(n²) complexity in sequence length.
For a 224×224 image with 16×16 patches:
  - Number of patches: (224/16)² = 196
  - Attention matrix: 196 × 196 = 38,416 values per head

This is manageable for small images.
For large images, techniques like windowed attention (Swin Transformer) are used.

---

## Summary

```
Image → patches → linear embedding → [CLS] token + patches → positional encoding
→ Transformer encoder → CLS token output → classifier → predictions
```

Key point: once you have patch embeddings, it is the same Transformer as for text.
