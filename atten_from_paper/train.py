import jax
import jax.numpy as jnp
import optax
from typing import List, Callable, Dict, Iterable
from jax.random import PRNGKey
from functools import partial


def optim_wrapper(optimizer, grads, opt_state, params=None):
    try:
        return optimizer.update(grads, opt_state, params)
    except TypeError:
        return optimizer.update(grads, opt_state)

@jax.jit
def train_step(model, optimizer, params, opt_state, x, y):
    def loss_fn_wrapped(params):
        logits = model(params, x)
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, logits.shape[-1])).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn_wrapped)(params)
    updates, opt_state = optim_wrapper(optimizer, grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def eval_step(model, params, x, y):
    logits = model(params, x)
    loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, logits.shape[-1])).mean()
    return loss

import time
def train_session(epochs, 
                  data:Dict[str, Iterable], 
                  model:Dict[str, Callable], 
                  optimizer, opt_state):
    
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = 0.0
        train_batches = 0
        for xb, yb in data['train_loader']:
            params, opt_state, loss = train_step(model['train'], optimizer, params, opt_state, xb, yb)
            train_loss += loss
            train_batches += 1
        train_loss /= train_batches

        val_loss = 0.0
        val_batches = 0
        for xb, yb in data['test_loader']:
            loss = eval_step(model['val'], params, xb, yb)
            val_loss += loss
            val_batches += 1
        val_loss /= val_batches

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch {epoch+1}: "
            f"Train Loss {train_loss:.4f} "
            f"Val Loss {val_loss:.4f} "
            f"({epoch_time:.2f}s)")
        
        return params, opt_state