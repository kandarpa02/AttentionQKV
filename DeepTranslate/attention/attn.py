import jax 
import jax.numpy as jnp
import haiku as hk

__all__ = ["MhAttention"]

def split_heads(x:jnp.ndarray, n_heads:int):
    batch, seq, dim = x.shape
    n_dim = dim // n_heads
    x = x.reshape(batch, seq, n_heads, n_dim)
    x = jnp.transpose(x, (0, 2, 1, 3))
    return x

def merge_heads(x):
    batch, n_heads, seq, n_dim = x.shape
    x = jnp.transpose(x, (0, 2, 1, 3))
    x = x.reshape(batch, seq, n_heads * n_dim)
    return x


class MhAttention(hk.Module):
    def __init__(self, d_model, n_heads, mask = False, name="attention"):
        super().__init__(name = name)
        self.d_model = d_model
        self.n_heads = n_heads
        self.mask = mask
    
    def __call__(self, x:jnp.ndarray):
        qw = hk.get_parameter('qw', shape=[self.d_model, self.d_model], init=hk.initializers.RandomNormal())
        kw = hk.get_parameter('kw', shape=[self.d_model, self.d_model], init=hk.initializers.RandomNormal())
        vw = hk.get_parameter('vw', shape=[self.d_model, self.d_model], init=hk.initializers.RandomNormal())
        ow = hk.get_parameter('ow', shape=[self.d_model, self.d_model], init=hk.initializers.RandomNormal())

        Q = x @ qw
        K = x @ kw
        V = x @ vw

        Q = split_heads(Q, self.n_heads)
        K = split_heads(K, self.n_heads)
        V = split_heads(V, self.n_heads)

        score = Q @ jnp.swapaxes(K, -1, -2) / jnp.sqrt(self.d_model)
        score = score.astype(jnp.float32)

        if self.mask:
            _, _, seq_l, _ = x.shape
            mask = jnp.tril(jnp.ones((1, 1, seq_l, seq_l), dtype=jnp.float32))
            score = score - 1e10 * (1.0 - mask)

        weights = jax.nn.softmax(score, axis=-1)
        attn = weights @ V

        merged = merge_heads(attn)
        out = merged @ ow
        return out
    




