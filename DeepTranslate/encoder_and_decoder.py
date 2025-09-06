from .attention_layer import MultiheadAttention, create_masks, prepare_mask_for_attention
import jax.numpy as jnp
from jax import Array
import jax
import flax.linen as nnx
from flax.linen.initializers import lecun_normal
from typing import Any


class FFNResidualLayerNorm(nnx.Module):
    dropout:float
    train:bool
    d_model:int

    @nnx.compact
    def __call__(self, x, rng):

        # 1st Dense
        ffn = nnx.Dense(self.d_model*4)(x)
        ffn = nnx.gelu(ffn)
    
        # Dropout
        ffn = nnx.Dropout(self.dropout)(ffn, rng=rng, deterministic=not self.train)
        # 2nd Dense
        ffn = nnx.Dense(self.d_model)(ffn)

        residual = x + ffn
        output = nnx.LayerNorm()(residual)
        return output
    

class EncoderBlock(nnx.Module):
    num_heads: int
    dropout: float
    d_ffn: int
    use_bias: bool = True
    train: bool = True

    @nnx.compact
    def __call__(self, x, mask=None, rng:Array|Any=None):
        rng1, rng2 = jax.random.split(rng, 2)
        attn_out = MultiheadAttention(
            self.num_heads, 
            self.dropout, 
            self.use_bias, 
            self.train
            )(x, x, x, mask, rng1)
        
        x = x + attn_out
        x = nnx.LayerNorm()(x)

        ffn_out = FFNResidualLayerNorm(self.dropout, self.train, self.d_ffn)(x, rng2)
        return ffn_out
    
class DecoderBlock(nnx.Module):
    num_heads: int
    dropout: float
    d_ffn: int
    use_bias: bool = True
    train: bool = True

    @nnx.compact
    def __call__(self, x, enc_output, enc_mask, combined_mask, rng):
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        # Self-attention
        attn_output = MultiheadAttention(
            self.num_heads,
            self.dropout,
            self.use_bias,
            self.train)(x, x, x, mask=combined_mask, rng=rng1)
        x = x + attn_output
        x = nnx.LayerNorm()(x)

        # Cross-attention
        cross_output = MultiheadAttention(
            self.num_heads,
            self.dropout,
            self.use_bias,
            self.train)(x, enc_output, enc_output, mask=enc_mask, rng=rng2)
        x = x + cross_output
        x = nnx.LayerNorm()(x)

        # Feed-forward
        x = FFNResidualLayerNorm(self.dropout, self.train, self.d_ffn)(x, rng3)

        return x
