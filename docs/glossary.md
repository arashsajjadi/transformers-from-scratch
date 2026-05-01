# Glossary

---

**Attention**
A mechanism that computes a weighted sum of value vectors, where the weights are determined by the similarity between query and key vectors.

**Attention weights**
The softmax output in scaled dot-product attention. Values in [0, 1] that sum to 1 across the key dimension.

**Autoregressive**
Generating one token at a time, where each new token is conditioned on all previously generated tokens.

**Backbone**
A feature extractor (usually a CNN) that converts raw input (image) into a feature representation.

**Batch**
A group of training examples processed together. Using batches is more efficient than processing one example at a time.

**Bounding box**
A rectangle that encloses an object in an image. Usually represented as (x_min, y_min, x_max, y_max) or (cx, cy, w, h).

**BOS (Beginning of Sequence)**
A special token added at the start of decoder input sequences.

**CLS token**
A learnable vector prepended to the sequence. Its final representation is used for classification.

**Cross-attention**
Attention where queries come from one sequence (decoder) and keys/values come from another (encoder output).

**d_k**
Dimension of query and key vectors per head. Equal to d_model / num_heads.

**d_model**
The main embedding dimension used throughout a Transformer model.

**DataLoader**
A PyTorch utility that wraps a Dataset and provides batches with optional shuffling and multi-process loading.

**DETR**
DEtection TRansformer. A Transformer-based object detector that uses object queries and Hungarian matching.

**Dice Loss**
A segmentation loss that directly optimizes the overlap between predicted and ground-truth regions. Good for imbalanced classes.

**dim_feedforward**
The hidden dimension of the feed-forward network inside each Transformer block. Typically 4× d_model.

**Dropout**
A regularization technique that randomly zeros some activations during training. Helps prevent overfitting.

**Embedding**
A learned vector representation of a discrete token (word, class, position).

**EOS (End of Sequence)**
A special token that signals the end of a generated sequence.

**F1 Score**
The harmonic mean of precision and recall. Macro F1 averages F1 across all classes equally.

**Feed-forward network (FFN)**
The position-wise MLP inside each Transformer block: Linear -> GELU -> Linear.

**Feature map**
A 3D tensor [channels, height, width] output by a CNN. Encodes spatial features.

**GELU**
Gaussian Error Linear Unit. An activation function: x * P(X <= x) for X ~ N(0,1). Smoother than ReLU.

**Grid detector**
A detection model that divides the image into a grid and predicts objects per cell (YOLO-style).

**Hungarian matching**
An optimal bipartite matching algorithm used in the original DETR to assign predictions to ground truth.

**IoU (Intersection over Union)**
The ratio of the intersection area to the union area of two bounding boxes. Ranges from 0 (no overlap) to 1 (perfect overlap).

**Key (K)**
In attention: a vector that represents "what this position offers". Compared against queries to compute scores.

**LayerNorm**
Normalizes activations across the feature dimension. Stabilizes training.

**Learnable**
A parameter that is updated during training via gradient descent.

**Logits**
The raw output of a linear layer before applying softmax. Can be any real number.

**Mask**
A tensor that blocks certain positions from being attended to. True = blocked in this course.

**Mean pooling**
Averaging all token representations to produce a single vector.

**MPS (Metal Performance Shaders)**
Apple Silicon GPU backend for PyTorch.

**NMS (Non-Maximum Suppression)**
A post-processing step in detection that removes duplicate boxes by keeping only the highest-scoring box in an overlapping region.

**No-object class**
In DETR-style detectors, the extra class assigned to queries that do not match any ground-truth object.

**num_heads**
Number of parallel attention heads in multi-head attention.

**Object query**
A learnable vector in DETR that is decoded into one object prediction. The model learns what to look for.

**OOV (Out of Vocabulary)**
A word not seen during training. Mapped to the UNK token.

**PAD (Padding)**
A special token added to make sequences the same length in a batch.

**Patch**
A small fixed-size region of an image, used as a token by Vision Transformers.

**Patch embedding**
A linear layer that maps flattened patch pixels to the model's d_model dimension.

**Pixel accuracy**
Fraction of pixels correctly classified in a segmentation map.

**Positional encoding**
A vector added to token embeddings to inject information about their position in the sequence.

**Pre-norm**
Applying LayerNorm before the attention or FFN sub-layer. More stable than post-norm.

**Query (Q)**
In attention: a vector representing "what am I looking for?". Compared against keys.

**Receptive field**
The area of the input that a single unit in a CNN sees.

**Residual connection**
Adding the input of a sub-layer to its output: x = x + SubLayer(x). Helps gradient flow.

**Scaled dot-product attention**
The core attention formula: softmax(QK^T / sqrt(d_k)) V.

**Segmentation mask**
A tensor of the same spatial size as the input image, where each pixel has a class label.

**Self-attention**
Attention where queries, keys, and values all come from the same sequence.

**Seq2Seq**
Sequence-to-sequence. A model that takes a sequence as input and produces a different sequence as output.

**Sigmoid**
A function that maps any real number to [0, 1]: 1 / (1 + e^(-x)).

**Skip connection**
In U-Net: a connection that passes encoder features directly to the corresponding decoder level.

**SmoothL1Loss**
A loss for regression: behaves like MSE for small errors and like L1 for large errors. More robust than MSE.

**Softmax**
Converts a vector of raw scores to a probability distribution (all values in [0,1], sum to 1).

**Teacher forcing**
During training, feeding ground-truth tokens as decoder input instead of the model's own predictions.

**Token**
The basic unit of input: a word, a character, or an image patch.

**Transformer**
An architecture built entirely from attention and feed-forward layers, with no recurrence or convolution.

**UNK**
Unknown token. Used for words not in the vocabulary.

**Value (V)**
In attention: the information retrieved when a position is attended to.

**ViT (Vision Transformer)**
A Transformer that processes images by splitting them into patches and treating each patch as a token.
