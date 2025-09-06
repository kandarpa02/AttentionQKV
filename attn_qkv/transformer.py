from .encoder_and_decoder import (
    FFNResidualLayerNorm,
    EncoderBlock,
    DecoderBlock
)

import jax.numpy as jnp
from jax import Array
import jax
import flax.linen as nnx
from typing import Any


class PositionalEncoding(nnx.Module):
    max_len: int
    d_model: int

    @nnx.compact
    def __call__(self, x):
        pe = self.param('pos_emb', nnx.zeros, (1, self.max_len, self.d_model))
        return x + pe[:, :x.shape[1], :]


class Transformer(nnx.Module):
    num_layers: int
    num_heads: int
    d_model: int
    d_ff: int
    vocab_size: int
    max_len: int
    dropout: float
    pad_id: int
    use_bias: bool = True
    train: bool = True

    def create_masks(self, src_tokens, tgt_tokens):
        """Create encoder and decoder masks in broadcastable form."""
        
        # Encoder padding mask
        enc_mask = (src_tokens == self.pad_id).astype(jnp.float32) * -1e9
        enc_padding_mask = enc_mask[:, None, None, :]

        # Decoder padding mask
        dec_mask = (tgt_tokens == self.pad_id).astype(jnp.float32) * -1e9
        dec_padding_mask = dec_mask[:, None, None, :]

        # Decoder causal mask
        tgt_len = tgt_tokens.shape[1]
        causal_mask = jnp.tril(jnp.ones((1, 1, tgt_len, tgt_len), dtype=jnp.float32))
        causal_mask = (1.0 - causal_mask) * -1e9

        # Combine causal and padding masks
        combined_mask = jnp.maximum(dec_padding_mask, causal_mask)

        return enc_padding_mask, combined_mask

    @nnx.compact
    def __call__(self, src_tokens, tgt_tokens, rng):
        rng1, rng2 = jax.random.split(rng, 2)

        # Create masks
        enc_mask, tgt_mask = self.create_masks(src_tokens, tgt_tokens)

        # Embeddings
        src_embed = nnx.Embed(self.vocab_size, self.d_model)(src_tokens)
        tgt_embed = nnx.Embed(self.vocab_size, self.d_model)(tgt_tokens)

        # Add positional encodings
        src_embed = PositionalEncoding(self.max_len, self.d_model)(src_embed)
        tgt_embed = PositionalEncoding(self.max_len, self.d_model)(tgt_embed)

        # Apply dropout
        src_embed = nnx.Dropout(self.dropout)(src_embed, rng=rng1, deterministic=not self.train)
        tgt_embed = nnx.Dropout(self.dropout)(tgt_embed, rng=rng2, deterministic=not self.train)

        # Encoder stack
        x = src_embed
        for i in range(self.num_layers):
            x = EncoderBlock(self.num_heads, self.dropout, self.use_bias, self.train)(
                x, mask=enc_mask, rng=jax.random.fold_in(rng1, i)
            )
        enc_output = x

        # Decoder stack
        y = tgt_embed
        for i in range(self.num_layers):
            y = DecoderBlock(self.num_heads, self.dropout, self.use_bias, self.train)(
                y, enc_output, enc_mask, tgt_mask, rng=jax.random.fold_in(rng2, i)
            )

        # Final output projection
        logits = nnx.Dense(self.vocab_size)(y)
        return logits