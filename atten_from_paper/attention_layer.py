import jax.numpy as jnp
from jax import Array
import jax
import flax.linen as nnx
from flax.linen.initializers import lecun_normal
from typing import Any

def split_heads(x, num_heads):
    batch, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    x = x.reshape(batch, seq_len, num_heads, head_dim)
    return jnp.transpose(x, (0, 2, 1, 3)) 

def merge_heads(x):
    batch, num_heads, seq_len, head_dim = x.shape
    x = jnp.transpose(x, (0, 2, 1, 3))
    return x.reshape(batch, seq_len, num_heads * head_dim)


def create_masks(src_tokens, tgt_tokens, pad_id: int):
    """
    Create encoder padding mask and decoder combined mask.

    Returns:
        enc_padding_mask : [batch, 1, 1, src_len] 
        combined_mask    : [batch, 1, tgt_len, tgt_len] 
    """
    # Encoder padding mask: -1e9 where pad, 0 where valid
    enc_mask = (src_tokens == pad_id).astype(jnp.float32)
    enc_padding_mask = enc_mask[:, None, None, :] * -1e9

    # Decoder causal mask: -1e9 above diagonal, 0 on/below
    tgt_len = tgt_tokens.shape[1]
    causal_mask = jnp.tril(jnp.ones((1, 1, tgt_len, tgt_len), dtype=jnp.float32))
    causal_mask = (1.0 - causal_mask) * -1e9

    # Decoder padding mask: -1e9 where pad, 0 where valid
    dec_mask = (tgt_tokens == pad_id).astype(jnp.float32)
    dec_padding_mask = dec_mask[:, None, None, :] * -1e9

    # Combine
    combined_mask = jnp.maximum(dec_padding_mask, causal_mask)
    return enc_padding_mask, combined_mask


def prepare_mask_for_attention(mask, num_heads: int, seq_q: int, seq_k: int):
    """
    Ensure mask is broadcastable to [batch, heads, seq_q, seq_k].
    Mask should contain 0 where valid, -1e9 where invalid.

    Args:
        mask: [batch, 1, 1, seq_k]  OR  [1, 1, seq_q, seq_k]
        num_heads: int
        seq_q, seq_k: int

    Returns:
        mask: [batch, num_heads, seq_q, seq_k]
    """
    if mask.ndim == 4:
        # broadcast along heads if needed
        if mask.shape[1] == 1:
            mask = jnp.repeat(mask, num_heads, axis=1)
        return mask
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")


def multi_head_attention(
        q_vec:Array, k_vec:Array,  v_vec:Array, 
        qw:Array, kw:Array, vw:Array, ow:Array,
        qb:Array|Any=None, kb:Array|Any=None,
        vb:Array|Any=None, ob:Array|Any=None,
        num_heads:int=1, 
        mask:Any=None,
        dropout:float=0.1,
        deterministic:bool=False,
        rng:Array|Any=None
        ):
    
    """
    Multi-Head Attention (Vaswani et al., 2017)

    Implements the key mechanism from *"Attention Is All You Need"*, which 
    replaced recurrent and convolutional layers with attention-based 
    computations for sequence modeling.

    Core Idea:
        - Inputs are linearly projected into **queries (Q)**, **keys (K)**, 
          and **values (V)** using learned matrices.
        - Attention scores are computed by a **scaled dot product** 
          (QK^T / sqrt(d_k)), capturing pairwise token interactions.
        - A **softmax** over these scores produces a distribution that 
          weights values (V) by relevance to each query.
        - Multiple attention heads allow the model to jointly attend to 
          different representation subspaces.
        - Outputs of all heads are concatenated and linearly projected 
          via W^O to form the final representation.

    Benefits:
        - Eliminates recurrence, enabling full parallelization over 
          sequence elements.
        - Captures **long-range dependencies** without vanishing gradients.
        - Provides richer contextual embeddings via multi-head diversity.
        - Forms the foundation of modern Transformer architectures in 
          NLP, vision, and multimodal tasks.

    Shapes:
        X   : [batch, seq_len, hidden_dim]
        Q/K/V: [batch, heads, seq_len, head_dim]
        Out : [batch, seq_len, hidden_dim]

    Reference:
        Vaswani et al., "Attention Is All You Need", NeurIPS 2017
    """
    # splitting rng key
    k1, k2 = jax.random.split(rng)

    # 1. Linear Projections (learnable)
    # Project input X into Q, K, V spaces
    def linear(x, w, b=None):
        ## Optonal bias term 
        if b is not None: return (x @ w) + b
        else: return x @ w
        
    Q = linear(q_vec, qw, qb)
    K = linear(k_vec, kw, kb)
    V = linear(v_vec, vw, vb)

    # 2. Split into multiple heads
    Query = split_heads(Q, num_heads)
    Key   = split_heads(K, num_heads)
    Value = split_heads(V, num_heads)

    Batch, Seq, Dim = q_vec.shape
    assert Dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
    head_dim = Dim // num_heads

    # 3. Scaled Dot-Product Attention
    #   a) Similarity score between Q and K^T
    scores = Query @ jnp.swapaxes(Key, -1, -2) / jnp.sqrt(head_dim)
    scores = scores.astype(jnp.float32)

    #   b) (Optional) Mask for autoregressive decoding
    # 3. Apply mask if provided
    if mask is not None:
        if isinstance(mask, bool):
            if mask:  # causal mask
                mask = jnp.tril(jnp.ones((1, num_heads, Query.shape[2], Key.shape[2]), dtype=jnp.float32))
                mask = jnp.where(mask == 1, 0.0, -1e9)  # valid=0, invalid=-inf
            else:
                mask = None
        else:
            mask = prepare_mask_for_attention(mask, num_heads, Query.shape[2], Key.shape[2])

        if mask is not None:
            scores = scores + mask

    #   c) Softmax -> attention weights
    weights = jax.nn.softmax(scores, axis = -1)
    weights = nnx.Dropout(dropout)(weights, deterministic=deterministic, rng=k1)

    #   d) Weighted sum of Values
    attended = weights @ Value  # [batch, heads, seq_len, head_dim]

    # 4. Merge heads back
    merged = merge_heads(attended)  # [batch, seq_len, hidden_dim]

    # 5. Final linear projection (output projection)
    out = linear(merged, ow, ob)
    out = nnx.Dropout(dropout)(out, deterministic=deterministic, rng=k2)
    return out

class MultiheadAttention(nnx.Module):
    num_heads: int
    dropout: float
    use_bias: bool = True
    train: bool = True

    @nnx.compact
    def __call__(self, q_vec, k_vec, v_vec, mask=None, rng=None):
        d_model = q_vec.shape[-1]
        shape = (d_model, d_model)

        # Learnable params
        qw = self.param("qw", lecun_normal(), shape)

        kw = self.param("kw", lecun_normal(), shape)

        vw = self.param("vw", lecun_normal(), shape)

        ow = self.param("ow", lecun_normal(), shape)

        if self.use_bias:
            qb = self.param("qb", nnx.zeros, (d_model,))
            kb = self.param("kb", nnx.zeros, (d_model,))
            vb = self.param("vb", nnx.zeros, (d_model,))
            ob = self.param("ob", nnx.zeros, (d_model,))
        else:
            qb = kb = vb = ob = None

        # Call functional impl
        return multi_head_attention(
            q_vec, k_vec, v_vec,
            qw, kw, vw, ow,
            qb, kb, vb, ob,
            num_heads=self.num_heads,
            mask=mask,
            dropout=self.dropout,
            deterministic=not self.train,
            rng=rng
        )
    
