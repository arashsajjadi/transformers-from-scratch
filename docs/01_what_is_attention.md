# What Is Attention?

## The Core Idea

Attention is a way for a model to focus on the most relevant parts of its input when producing an output.

Think of how you read a sentence to answer a question.

Question: "What did the cat eat?"
Sentence: "The hungry cat quickly ate the fish."

When you look for the answer, your eyes do not treat all words equally. You focus more on "cat", "ate", and "fish". You mostly skip "the", "hungry", and "quickly".

Attention does the same thing — but with numbers.

---

## The Three Ingredients

Every attention operation has three parts:

| Name | Role | Analogy |
|------|------|---------|
| **Query (Q)** | "What am I looking for?" | A search query |
| **Key (K)** | "What does each position offer?" | An index label |
| **Value (V)** | "What information do I retrieve?" | The actual content |

The model computes a score between the query and each key.
High score → that position is relevant → take more of its value.
Low score → that position is not relevant → take less of its value.

---

## How the Score Is Computed

For a query q and a key k, the score is simply the dot product:

```
score(q, k) = q · k
```

A higher dot product means the query and key are more aligned.

Then we apply softmax to turn raw scores into weights that sum to 1.

Finally, we compute a weighted sum of the values:

```
output = softmax(scores) · V
```

---

## Why "Soft" Attention?

Hard attention would pick the single best key and ignore all others.
Soft attention gives every key some weight.

This means the output is a blend of all values, weighted by how relevant each one is.

Soft attention is differentiable, which means we can train it with backpropagation.

---

## A Concrete Example

Sentence: "The animal did not cross the street because it was tired"

When we ask "what does **it** refer to?", the attention mechanism should assign a high weight to "animal" and low weight to everything else.

Let's say we run attention with the token "it" as the query:

```
Scores (before softmax):
  The     = 0.1
  animal  = 2.8   <-- high score
  did     = 0.2
  not     = 0.1
  cross   = 0.3
  the     = 0.1
  street  = 0.5
  because = 0.2
  it      = 1.1   <-- also high (self-attention)
  was     = 0.4
  tired   = 0.6
```

After softmax, "animal" gets the highest weight. The output for "it" will be mostly influenced by the representation of "animal".

---

## Why Does This Matter?

Before attention, models processed sequences step-by-step (RNNs, LSTMs). Long-range dependencies were hard to learn because information had to travel through many steps.

Attention lets any position directly connect to any other position in a single step.

This is why Transformers handle long sequences so much better.

---

## Key Properties of Attention

1. **Permutation equivariant**: attention itself does not know about order. That is why we add positional encoding separately.

2. **Variable length**: the same attention function works on sequences of any length.

3. **Interpretable**: we can visualize which positions attended to which.

4. **Parallelizable**: unlike RNNs, all attention operations can run simultaneously.

---

## What Attention Is Not

- Attention is not magic. It is just a weighted average of values.
- Attention does not understand meaning on its own. It needs to be trained.
- Attention weights do not always match human intuition perfectly.

---

## Summary

```
Query (Q)       "What am I looking for?"
Key   (K)       "What does each word offer?"
Value (V)       "What information do I get?"

score = Q · K^T
weight = softmax(score)
output = weight · V
```

The next step is to scale the scores before softmax. That is called Scaled Dot-Product Attention, covered in doc 02.
