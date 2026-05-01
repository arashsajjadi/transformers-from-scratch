# Course Overview: Transformers from Scratch

**Author:** Arash Sajjadi

This document is a complete syllabus. Each session maps to one notebook and, for later sessions, to one training script.

---

## Session 00 — Environment Setup

**Goal:** Verify the environment, select the device, and save a test output.

**Dataset:** None.

**Architecture:** None.

**Loss:** None.

**Metrics:** None.

**Main output:** `runs/setup/device_info.csv`, `runs/setup/test_plot.png`

**Exercises:**
1. Change the random seed and re-run. Do the plots change?
2. What device did the code select on your machine?
3. Install the requirements on a second machine (or colab) and compare device info.

---

## Session 01 — Attention Intuition

**Goal:** Understand what attention is without heavy math. Use a single sentence example.

**Dataset:** One hand-crafted sentence: *"The animal did not cross the street because it was tired"*

**Architecture:**
```
tokens -> manual scores -> softmax weights -> weighted sum
```

**Loss:** None.

**Metrics:** None.

**Main output:** `runs/attention/attention_heatmap.png`, `runs/attention/manual_attention_scores.csv`

**Exercises:**
1. Change the sentence. Which word does "it" attend to now?
2. What happens if all scores are equal?
3. What is the difference between hard attention (argmax) and soft attention (softmax)?
4. Draw a diagram of attention as a library lookup.

---

## Session 02 — Scaled Dot-Product Attention

**Goal:** Implement the core attention formula from the Attention Is All You Need paper.

**Dataset:** Random tensors.

**Architecture:**
```
Q, K, V -> QK^T / sqrt(d_k) -> softmax -> weighted sum of V
```

**Loss:** None.

**Metrics:** Shape checks.

**Main output:** `runs/attention/scaled_dot_product_heatmap.png`, `runs/attention/masked_attention_heatmap.png`

**Exercises:**
1. Remove the sqrt(d_k) scaling. What happens to the values?
2. What does the causal mask do? When would you use it?
3. Try d_k = 1 and d_k = 1000. Compare the softmax outputs.
4. Write a function that shows which positions are masked.

---

## Session 03 — Multi-Head Attention

**Goal:** Implement multi-head attention and understand why multiple heads help.

**Dataset:** Random tensors.

**Architecture:**
```
input -> linear QKV projections -> split heads -> attention -> concat -> output projection
```

**Loss:** None.

**Metrics:** Shape checks.

**Main output:** `runs/attention/multi_head_attention_heads.png`

**Exercises:**
1. Set num_heads=1. Is this the same as single-head attention?
2. What must be true about d_model and num_heads?
3. Inspect the attention weights of each head. Do they look different?
4. Add a print statement inside the head split. Confirm the shapes.

---

## Session 04 — Positional Encoding

**Goal:** Understand why Transformers need positional information and compare two encoding methods.

**Dataset:** Example sentences: "dog bites man" and "man bites dog"

**Architecture:**
```
token embeddings -> add positional encoding -> positional embeddings
```

**Loss:** None.

**Metrics:** None.

**Main output:** `runs/attention/positional_encoding_heatmap.png`, `runs/attention/position_similarity.png`

**Exercises:**
1. Why does a Transformer without positional encoding treat all positions the same?
2. What is the difference between sinusoidal and learned encoding?
3. Plot the cosine similarity between position 0 and all other positions.
4. What happens if you set max_len much smaller than the sequence length at inference?

---

## Session 05 — Transformer Encoder Block

**Goal:** Build a complete Transformer encoder block and train it on a synthetic task.

**Dataset:** Synthetic integer sequences (label = 1 if first token equals last token).

**Architecture:**
```
embeddings -> multi-head self-attention (pre-norm) -> residual -> feed-forward (pre-norm) -> residual
```

**Loss:** CrossEntropyLoss

**Metrics:** Accuracy

**Main output:** `runs/attention/encoder_training_curve.png`, `runs/attention/encoder_metrics.json`

**Exercises:**
1. Switch from pre-norm to post-norm. Does training behave differently?
2. Add a third encoder layer. Does accuracy improve?
3. Change the classification task. Try: label=1 if the sequence is sorted.
4. What is the residual connection doing? Try removing it.

---

## Session 06 — Text Classification

**Goal:** Train a Transformer encoder to classify short sentences as positive, negative, or neutral.

**Dataset:** `data/tiny/tiny_sentiment.csv` (90 examples, 3 classes)

**Architecture:**
```
text -> tokenizer -> embeddings -> positional encoding -> encoder -> mean pooling -> classifier
```

**Loss:** CrossEntropyLoss

**Metrics:** Accuracy, macro F1, confusion matrix

**Main output:** `runs/text_classification/training_curve.png`, `runs/text_classification/confusion_matrix.png`

**Exercises:**
1. Replace mean pooling with CLS token pooling. Compare results.
2. Add more training examples. Does accuracy improve?
3. What is the vocabulary size? What happens to OOV words?
4. Visualize the attention weights for a misclassified sentence.
5. Try without positional encoding. Does it make a difference on short sentences?

---

## Session 07 — Encoder-Decoder Translation

**Goal:** Build a tiny encoder-decoder Transformer and achieve near-perfect translation on a tiny dataset (educational overfitting).

**Dataset:** `data/tiny/tiny_translation_en_fr.csv` (100 English-to-French pairs)

**Architecture:**
```
source -> encoder | target -> masked decoder -> cross-attention -> vocabulary projection
```

**Loss:** CrossEntropyLoss (ignoring padding)

**Metrics:** Token accuracy, exact match accuracy

**Main output:** `runs/translation/training_curve.png`, `runs/translation/translation_examples.csv`

**Exercises:**
1. What is teacher forcing? Try training without it.
2. What is the causal mask in the decoder? Remove it and observe what happens.
3. Visualize cross-attention weights. Which source words align to which target words?
4. Add 50 more sentence pairs. Does the model generalize better?
5. Implement beam search instead of greedy decoding.

---

## Session 08 — Time-Series Forecasting

**Goal:** Show that Transformers work on numerical sequences, not just text.

**Dataset:** Synthetic sine waves with noise and trend.

**Architecture:**
```
past values -> linear projection -> positional encoding -> encoder -> regression head -> future values
```

**Loss:** MSELoss

**Metrics:** MSE, MAE

**Main output:** `runs/timeseries/forecast_plot.png`, `runs/timeseries/metrics.json`

**Exercises:**
1. Increase forecast_length. At what point does accuracy degrade?
2. Try a multi-frequency input. Can the model learn to separate them?
3. Add a trend to the signal. Does the model learn to extrapolate it?
4. Compare the Transformer to a simple linear baseline.

---

## Session 09 — CNN Baseline for Images

**Goal:** Build a CNN classifier and understand its strengths before ViT.

**Dataset:** Fashion-MNIST (28x28 grayscale, 10 classes)

**Architecture:**
```
image -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Linear
```

**Loss:** CrossEntropyLoss

**Metrics:** Accuracy, macro F1, confusion matrix

**Main output:** `runs/vit_classification/cnn_training_curve.png`, `runs/vit_classification/cnn_confusion_matrix.png`

**Exercises:**
1. What is the receptive field of two Conv layers with kernel size 3?
2. Add a third convolutional layer. Does accuracy improve?
3. Remove pooling. How does the number of parameters change?
4. Look at wrong predictions. What types of errors does the CNN make?

---

## Session 10 — Vision Transformer (ViT)

**Goal:** Implement ViT from scratch and compare it to the CNN baseline.

**Dataset:** Fashion-MNIST (28x28, 1 channel, 10 classes)

**Architecture:**
```
image -> patchify -> patch embedding -> CLS token -> positional embedding -> encoder -> CLS -> classifier
```

**Loss:** CrossEntropyLoss

**Metrics:** Accuracy, macro F1, confusion matrix

**Main output:** `runs/vit_classification/vit_training_curve.png`, `runs/vit_classification/patch_grid.png`

**Exercises:**
1. Change patch_size from 7 to 4. How many patches do you get? Does accuracy change?
2. Try without the CLS token. Use mean pooling instead.
3. Compare ViT vs CNN: which trains faster? Which is more accurate?
4. Visualize the attention weights over patches for a few images.
5. What happens if you train ViT on very few examples (10 per class)?

---

## Session 11 — U-Net Semantic Segmentation

**Goal:** Implement U-Net and train it to segment synthetic shapes.

**Dataset:** Synthetic shapes (circle, square, triangle, background) at 128x128.

**Architecture:**
```
image -> encoder (down blocks) -> bottleneck -> decoder (up blocks with skip connections) -> segmentation map
```

**Loss:** CrossEntropyLoss (DiceLoss explained but not used for training by default)

**Metrics:** Pixel accuracy, mean IoU, Dice score

**Main output:** `runs/segmentation/unet_overlay.png`, `runs/segmentation/unet_metrics.json`

**Exercises:**
1. What is a skip connection? Remove them and observe what happens.
2. What is the difference between CrossEntropyLoss and DiceLoss for segmentation?
3. Add more shapes or classes. How does mean IoU change?
4. Try Oxford-IIIT Pet (optional section). Compare synthetic vs real data performance.
5. Visualize the feature maps at the bottleneck.

---

## Session 12 — ViT Semantic Segmentation

**Goal:** Use patch tokens from a ViT encoder as a feature map for segmentation.

**Dataset:** Synthetic shapes (same as session 11).

**Architecture:**
```
image -> patch embedding -> encoder -> reshape tokens to feature map -> convolutional decoder -> upsample -> mask
```

**Loss:** CrossEntropyLoss

**Metrics:** Pixel accuracy, mean IoU, Dice score

**Main output:** `runs/segmentation/vit_segmenter_overlay.png`, `runs/segmentation/vit_vs_unet_comparison.csv`

**Exercises:**
1. Compare ViT Segmenter to U-Net on the same dataset.
2. Why does ViT need an upsampling decoder for segmentation?
3. What happens to the number of patch tokens when you change patch_size?
4. Visualize the patch token feature map before the decoder.

---

## Session 13 — Single-Object Detection

**Goal:** Teach the two-output structure of detection: class + box.

**Dataset:** Synthetic shapes, one object per image.

**Architecture:**
```
image -> CNN backbone -> global pooling -> class head + box head
```

**Loss:** CrossEntropyLoss (class) + SmoothL1Loss (box)

**Metrics:** Classification accuracy, mean IoU, box L1 error

**Main output:** `runs/detection/single_detector_predictions.png`, `runs/detection/single_detector_metrics.json`

**Exercises:**
1. What is SmoothL1Loss? Why is it better than MSELoss for boxes?
2. Change lambda_box from 5.0 to 0.1. What happens to box accuracy?
3. What is IoU? When is it 0? When is it 1?
4. Try normalizing box coordinates differently (pixel coords vs 0-1).

---

## Session 14 — Multi-Object Grid Detection

**Goal:** Extend detection to multiple objects using a grid prediction scheme (YOLO-style intuition).

**Dataset:** Synthetic shapes, 1-3 objects per image.

**Architecture:**
```
image -> CNN backbone -> grid head [objectness, class, box per cell]
```

**Loss:** BCE (objectness) + CrossEntropy (class) + SmoothL1 (box)

**Metrics:** Mean IoU, objectness accuracy, precision, recall

**Main output:** `runs/detection/grid_detector_predictions.png`, `runs/detection/grid_detector_metrics.json`

**Exercises:**
1. What is NMS? Remove it. How do predictions change?
2. Change the confidence threshold from 0.4 to 0.7. How does precision vs recall change?
3. What happens when two objects fall in the same grid cell?
4. Draw a diagram of the 4x4 grid on a 128x128 image.

---

## Session 15 — Tiny DETR

**Goal:** Implement a DETR-style detector with a Transformer decoder and learned object queries.

**Dataset:** Synthetic shapes, 1-3 objects per image.

**Architecture:**
```
image -> CNN backbone -> flatten feature tokens -> encoder -> object queries -> decoder -> class + box heads
```

**Loss:** CrossEntropy (with no-object class) + SmoothL1 (for matched objects)

**Metrics:** Mean IoU, class accuracy, no-object accuracy, precision, recall

**Main output:** `runs/detection/tiny_detr_predictions.png`, `runs/detection/tiny_detr_metrics.json`

**Exercises:**
1. What is the difference between grid detection and DETR query detection?
2. Why does DETR not need NMS?
3. What is Hungarian matching? (Described in docs, not implemented here.)
4. Increase num_queries from 5 to 10. What changes?
5. Visualize which object query activated for which detected object.

---

## Session 16 — Final Project: Model Comparison

**Goal:** Review all models, compare architectures, and consolidate what you have learned.

**Dataset:** All saved metrics from previous runs.

**New training:** None.

**Main output:**
- `runs/SUMMARY.md` — complete summary table
- `runs/final_comparison_table.csv`
- `runs/final_model_comparison.png`
- `runs/final_gallery.png`

**Exercises:**
1. Which model achieved the best accuracy per task?
2. Which model has the most parameters?
3. Which model would you choose for a real project, and why?
4. What would you change about the Tiny DETR to improve it?
5. Design a new experiment using the tools from this course.
