import jax.numpy as jnp
from jax import Array
import jax
import flax.linen as nnx
from flax.linen.initializers import lecun_normal

def split_heads(x, num_heads):
    batch, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    x = x.reshape(batch, seq_len, num_heads, head_dim)
    return jnp.transpose(x, (0, 2, 1, 3)) 

def merge_heads(x):
    batch, num_heads, seq_len, head_dim = x.shape
    x = jnp.transpose(x, (0, 2, 1, 3))
    return x.reshape(batch, seq_len, num_heads * head_dim)


def multi_head_attention(
        q_vec:Array, 
        k_vec:Array, 
        v_vec:Array, 
        qw:Array, 
        kw:Array, 
        vw:Array, 
        ow,
        num_heads:int, mask:bool=True
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

    # 1. Linear Projections (learnable)
    # Project input X into Q, K, V spaces
    Q = q_vec @ qw
    K = k_vec @ kw
    V = v_vec @ vw

    # 2. Split into multiple heads
    Query = split_heads(Q, num_heads)
    Key   = split_heads(K, num_heads)
    Value = split_heads(V, num_heads)

    Batch, Seq, Dim = q_vec.shape
    head_dim = Dim // num_heads

    # 3. Scaled Dot-Product Attention
    #   a) Similarity score between Q and K^T
    scores = Query @ jnp.swapaxes(Key, -1, -2) / jnp.sqrt(head_dim)
    scores = scores.astype(jnp.float32)

    #   b) (Optional) Mask for autoregressive decoding
    scores = jnp.where(
            mask == True,
            scores - 1e10 * (1 - jnp.tril(jnp.ones((1, 1, Seq, Seq), dtype=jnp.float32))),
            scores
        )

    #   c) Softmax -> attention weights
    weights = jax.nn.softmax(scores, axis = -1)

    #   d) Weighted sum of Values
    attended = weights @ Value  # [batch, heads, seq_len, head_dim]

    # 4. Merge heads back
    merged = merge_heads(attended)  # [batch, seq_len, hidden_dim]

    # 5. Final linear projection (output projection)
    out = merged @ ow
    return out

class MultiheadAttention(nnx.Module):
    num_heads:int

    @nnx.compact
    def __call__(self, q_vec, k_vec, v_vec, mask:bool):
        shape = [q_vec.shape[-1], q_vec.shape[-1]]
        qw = self.param(
            'qw', 
            lecun_normal(), 
            shape
            )
        
        kw = self.param(
            'kw',
            lecun_normal(),
            shape
        )

        vw = self.param(
            'vw',
            lecun_normal(),
            shape
        )

        ow = self.param(
            'ow',
            lecun_normal(),
            shape
        )

        return multi_head_attention(
            q_vec, 
            k_vec,
            v_vec,
            qw,
            kw,
            vw,
            ow,
            num_heads=self.num_heads
            mask=mask
        )