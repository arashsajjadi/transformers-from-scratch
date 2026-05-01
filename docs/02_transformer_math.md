# Transformer Math

## Scaled Dot-Product Attention

The full formula for attention is:

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

Where:
- Q: query matrix, shape [seq_q, d_k]
- K: key matrix, shape [seq_k, d_k]
- V: value matrix, shape [seq_k, d_v]
- d_k: dimension of the query/key vectors
- sqrt(d_k): scaling factor

### Why scale by sqrt(d_k)?

When d_k is large, the dot products Q K^T tend to have large magnitudes.
This pushes the softmax into regions with very small gradients.
Dividing by sqrt(d_k) keeps the values in a reasonable range.

Example: if d_k = 64, we divide by 8.

---

## Tensor Shapes

| Tensor | Shape | Meaning |
|--------|-------|---------|
| Q | [batch, heads, seq_q, d_k] | Query vectors per head |
| K | [batch, heads, seq_k, d_k] | Key vectors per head |
| V | [batch, heads, seq_k, d_v] | Value vectors per head |
| QK^T | [batch, heads, seq_q, seq_k] | Raw attention scores |
| softmax(QK^T) | [batch, heads, seq_q, seq_k] | Attention weights (sum to 1) |
| output | [batch, heads, seq_q, d_v] | Attended values |

---

## Multi-Head Attention

Instead of computing one big attention function, we run it h times in parallel.

Each run ("head") uses its own linear projection of Q, K, V:

```
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

Then we concatenate all heads and project back to d_model:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

Where:
- d_model: total model dimension
- h: number of heads
- d_k = d_v = d_model / h (each head sees a slice of the full dimension)

### Why multiple heads?

Each head can specialize in a different type of relationship:
- One head might track syntactic dependencies
- Another might track semantic similarity
- Another might track positional proximity

---

## Feed-Forward Network

After multi-head attention, each position goes through a two-layer MLP:

```
FFN(x) = Linear(GELU(Linear(x, W_1, b_1)), W_2, b_2)
```

The hidden dimension is typically 4× d_model.

This is applied independently to each position (hence "position-wise").

---

## Residual Connections and LayerNorm

Every sub-layer uses:

```
x = x + SubLayer(LayerNorm(x))   (pre-norm style)
```

Or:
```
x = LayerNorm(x + SubLayer(x))   (post-norm style)
```

This course uses pre-norm, which is more stable during training.

**LayerNorm** normalizes across the feature dimension (d_model), not the batch dimension.

```
LayerNorm(x) = (x - mean(x)) / (std(x) + eps) * gamma + beta
```

---

## Full Encoder Block

```
Input x [batch, seq, d_model]
  |
  +--> LayerNorm --> MultiHeadSelfAttention --> + --> LayerNorm --> FFN --> + --> Output
       (norm1)                                  |     (norm2)                |
       <----- residual connection 1 -----------+     <--- residual 2 ------+
```

In code:

```python
# Pre-norm self-attention
x = x + dropout(self_attention(layer_norm(x)))

# Pre-norm feed-forward
x = x + dropout(feed_forward(layer_norm(x)))
```

---

## Positional Encoding

Transformers have no inherent sense of order. We inject position information by adding a positional vector to each token embedding.

Sinusoidal encoding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Properties:
- Each position has a unique encoding
- Similar positions have similar encodings
- No learned parameters
- Can generalize to longer sequences than seen at training

---

## Parameter Count for a Transformer Encoder

For d_model = 64, num_heads = 4, num_layers = 2, dim_feedforward = 128, vocab_size = 1000:

| Component | Parameters |
|-----------|------------|
| Token embedding | vocab_size × d_model = 64,000 |
| W_Q, W_K, W_V (per layer) | 3 × d_model × d_model = 12,288 |
| W_O (per layer) | d_model × d_model = 4,096 |
| FFN (per layer) | 2 × d_model × dim_feedforward = 16,384 |
| LayerNorms (per layer) | 4 × d_model × 2 = 512 |
| Total (2 layers) | ~100,000 |

---

## Summary

```
Scaled Dot-Product Attention:
  score = QK^T / sqrt(d_k)
  weight = softmax(score)
  output = weight * V

Multi-Head Attention:
  Run attention h times with different linear projections
  Concatenate and project back to d_model

Transformer Block:
  x = x + Attention(LayerNorm(x))
  x = x + FFN(LayerNorm(x))
```
