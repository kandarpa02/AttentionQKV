from .attention_layer import MultiheadAttention, create_masks, prepare_mask_for_attention
import jax.numpy as jnp
from jax import Array
import jax
import flax.linen as nn
from flax.linen.initializers import lecun_normal
from typing import Any


class FFNResidualLayerNorm(nn.Module):
    dropout:float
    train:bool
    d_model:int

    @nn.compact
    def __call__(self, x, rng):

        # 1st Dense
        ffn = nn.Dense(self.d_model*4)(x)
        ffn = nn.gelu(ffn)
    
        # Dropout
        ffn = nn.Dropout(self.dropout)(ffn, rng=rng, deterministic=not self.train)
        # 2nd Dense
        ffn = nn.Dense(self.d_model)(ffn)

        residual = x + ffn
        output = nn.LayerNorm()(residual)
        return output
    

class EncoderBlock(nn.Module):
    num_heads: int
    dropout: float
    d_ffn: int
    use_bias: bool = True
    train: bool = True

    @nn.compact
    def __call__(self, x, mask, rng):
        rng1, rng2 = jax.random.split(rng, 2)
        attn_out = MultiheadAttention(
            self.num_heads, 
            self.dropout, 
            self.use_bias, 
            self.train
            )(x, x, x, mask, rng1)
        
        x = x + attn_out
        x = nn.LayerNorm()(x)

        ffn_out = FFNResidualLayerNorm(self.dropout, self.train, self.d_ffn)(x, rng2)
        return ffn_out
    
class DecoderBlock(nn.Module):
    num_heads: int
    dropout: float
    d_ffn: int
    use_bias: bool = True
    train: bool = True

    @nn.compact
    def __call__(self, x, enc_output, enc_mask, combined_mask, rng):
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        # Self-attention
        attn_output = MultiheadAttention(
            self.num_heads,
            self.dropout,
            self.use_bias,
            self.train)(x, x, x, mask=combined_mask, rng=rng1)
        x = x + attn_output
        x = nn.LayerNorm()(x)

        # Cross-attention
        cross_output = MultiheadAttention(
            self.num_heads,
            self.dropout,
            self.use_bias,
            self.train)(x, enc_output, enc_output, mask=enc_mask, rng=rng2)
        x = x + cross_output
        x = nn.LayerNorm()(x)

        # Feed-forward
        x = FFNResidualLayerNorm(self.dropout, self.train, self.d_ffn)(x, rng3)

        return x
