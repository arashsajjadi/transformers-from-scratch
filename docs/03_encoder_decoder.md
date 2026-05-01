# Encoder-Decoder Transformer

## When Do You Need a Decoder?

An encoder-only Transformer is good for:
- Classification
- Understanding tasks (reading comprehension, sentence similarity)

A decoder is needed when you want to **generate** output token by token:
- Machine translation
- Summarization
- Text generation
- Image captioning

---

## Architecture Overview

```
Source sequence (English)
  |
  v
[Token Embedding + Positional Encoding]
  |
  v
[ Encoder Block 1 ]
[ Encoder Block 2 ]
      ...
  |
  v
[Encoder Output]  --------> provides keys and values for cross-attention
                             |
Target sequence (French)     |
  |                          |
  v                          |
[Token Embedding + Positional Encoding]
  |                          |
  v                          |
[ Decoder Block 1 ] <--------+
  masked self-attention
  cross-attention
  feed-forward
  |
[ Decoder Block 2 ] <--------+
      ...
  |
  v
[Linear + Softmax] -> probability over target vocabulary
```

---

## The Three Attention Sub-Layers

### 1. Masked Self-Attention (in decoder)

The decoder generates tokens left to right.
When generating position t, it must not see tokens at positions t+1, t+2, ...

We enforce this with a causal mask:

```
Causal mask for seq_len=4:
position 0 can see: [0]
position 1 can see: [0, 1]
position 2 can see: [0, 1, 2]
position 3 can see: [0, 1, 2, 3]
```

In matrix form (True = blocked):
```
[ F  T  T  T ]
[ F  F  T  T ]
[ F  F  F  T ]
[ F  F  F  F ]
```

### 2. Cross-Attention (in decoder)

After masked self-attention, the decoder queries the encoder output.

- **Query**: comes from the decoder (the current partial output)
- **Key and Value**: come from the encoder output (the full source)

This is how the decoder "reads" the source sentence.

### 3. Self-Attention (in encoder)

Processes all source tokens simultaneously.
Each token can attend to every other source token.
No causal masking — the encoder sees the full input.

---

## Teacher Forcing

During **training**, we feed the ground-truth target tokens as decoder input.

Example (translate "I am happy" to "je suis heureux"):
- Decoder input:  `<BOS>  je    suis`
- Target output:   `je    suis  heureux  <EOS>`

This is called "teacher forcing" because we give the model the correct answer instead of its own prediction.

During **inference**, we generate one token at a time:
- Start with `<BOS>`
- Feed it through the decoder → get next token
- Append next token → feed again → get next token
- Stop when `<EOS>` is generated

---

## Greedy Decoding

At each step, pick the token with the highest probability:

```python
next_token = logits.argmax(dim=-1)
```

This is fast but not optimal. Beam search explores multiple candidates and is better for translation quality.

---

## Why Educational Overfitting?

For this course, we train on only 100 sentence pairs. The goal is NOT generalization.

We want to see:
1. Does the loss go down?
2. Does the cross-attention learn meaningful alignments?
3. Does the model learn to produce `<EOS>` at the right time?

These are the important things to verify when building a sequence-to-sequence model from scratch.

---

## Padding and Masking

Since sequences have different lengths, we pad shorter sequences to match the longest in a batch.

The encoder ignores padding positions (src_key_padding_mask).
The decoder also ignores padding in the target (tgt_key_padding_mask).

---

## Summary

```
Encoder:
  src -> embedding + pos_enc -> encoder_blocks -> memory

Decoder:
  tgt -> embedding + pos_enc
      -> masked_self_attention(tgt, tgt)    (causal)
      -> cross_attention(query=tgt, key/value=memory)
      -> feed_forward
      -> linear + softmax -> token probabilities
```
