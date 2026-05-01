"""
Encoder-decoder Transformer for sequence-to-sequence tasks (e.g., translation).

Architecture:
    source tokens → encoder → encoder output
    target tokens → decoder (with encoder output as cross-attention keys/values)
    decoder output → linear projection → vocabulary logits

Decoding:
    During training: teacher forcing (feed ground-truth tokens to decoder).
    During inference: greedy_decode (generate one token at a time).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.transformer_encoder import TransformerEncoder
from src.models.transformer_decoder import TransformerDecoder, generate_causal_mask
from src.models.positional_encoding import SinusoidalPositionalEncoding


class Seq2SeqTransformer(nn.Module):
    """Transformer encoder-decoder for translation.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        d_model: Embedding and model dimension.
        num_heads: Attention heads.
        num_encoder_layers: Number of encoder blocks.
        num_decoder_layers: Number of decoder blocks.
        dim_feedforward: FFN hidden size.
        max_src_len: Max source sequence length.
        max_tgt_len: Max target sequence length.
        dropout: Dropout probability.
        pad_idx: Padding token index (must be same in src and tgt vocabs).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        max_src_len: int = 16,
        max_tgt_len: int = 16,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.tgt_vocab_size = tgt_vocab_size

        # Source embedding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.src_pos_enc = SinusoidalPositionalEncoding(d_model, max_src_len, dropout)

        # Target embedding
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_pos_enc = SinusoidalPositionalEncoding(d_model, max_tgt_len, dropout)

        # Encoder and decoder stacks
        self.encoder = TransformerEncoder(
            d_model, num_heads, num_encoder_layers, dim_feedforward, dropout
        )
        self.decoder = TransformerDecoder(
            d_model, num_heads, num_decoder_layers, dim_feedforward, dropout
        )

        # Final linear projection: decoder output → vocabulary logits
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding weights and output projection."""
        scale = self.d_model ** 0.5
        nn.init.normal_(self.src_embedding.weight, mean=0.0, std=1.0 / scale)
        nn.init.normal_(self.tgt_embedding.weight, mean=0.0, std=1.0 / scale)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def encode(
        self,
        src: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode source tokens.

        Args:
            src: [batch, src_len] source token IDs.

        Returns:
            encoder_output: [batch, src_len, d_model]
            src_padding_mask: [batch, src_len] bool tensor
        """
        src_padding_mask = (src == self.pad_idx)

        # Embeddings + positional encoding
        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        src_emb = self.src_pos_enc(src_emb)

        # Encode
        encoder_output, _ = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        return encoder_output, src_padding_mask

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        return_cross_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode one step given encoder output.

        Args:
            tgt: [batch, tgt_len] target token IDs (decoder input).
            encoder_output: [batch, src_len, d_model]
            src_padding_mask: [batch, src_len] bool mask for source padding.
            return_cross_weights: Whether to return cross-attention weights.

        Returns:
            logits: [batch, tgt_len, tgt_vocab_size]
            cross_weights: [batch, heads, tgt_len, src_len] or None
        """
        tgt_len = tgt.size(1)
        tgt_padding_mask = (tgt == self.pad_idx)

        # Causal mask: prevent attending to future tokens
        causal_mask = generate_causal_mask(tgt_len, device=tgt.device)

        # Target embeddings + positional encoding
        tgt_emb = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        tgt_emb = self.tgt_pos_enc(tgt_emb)

        # Decode
        decoder_output, cross_weights = self.decoder(
            tgt_emb,
            encoder_output,
            causal_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            src_key_padding_mask=src_padding_mask,
            return_last_cross_weights=return_cross_weights,
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)  # [batch, tgt_len, vocab_size]

        return logits, cross_weights

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        return_cross_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Full forward pass (teacher forcing).

        Args:
            src: [batch, src_len] source token IDs.
            tgt: [batch, tgt_len] target token IDs (decoder input: BOS + tokens).
            return_cross_weights: Whether to return cross-attention weights.

        Returns:
            logits: [batch, tgt_len, tgt_vocab_size]
            cross_weights: or None
        """
        encoder_output, src_padding_mask = self.encode(src)
        logits, cross_weights = self.decode(
            tgt, encoder_output, src_padding_mask, return_cross_weights
        )
        return logits, cross_weights


def greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    bos_idx: int,
    eos_idx: int,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a translation using greedy decoding.

    At each step, pick the most likely next token.
    Stop when EOS is generated or max_len is reached.

    Args:
        model: Trained Seq2SeqTransformer.
        src: [1, src_len] source token IDs (batch size 1).
        bos_idx: Beginning-of-sequence token index.
        eos_idx: End-of-sequence token index.
        max_len: Maximum output length.
        device: Target device.

    Returns:
        [output_len] tensor of generated token indices.
    """
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        encoder_output, src_padding_mask = model.encode(src)

        # Start with BOS token
        generated = torch.tensor([[bos_idx]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits, _ = model.decode(generated, encoder_output, src_padding_mask)
            # Take the last token's prediction
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_idx:
                break

    return generated[0]  # [output_len]
