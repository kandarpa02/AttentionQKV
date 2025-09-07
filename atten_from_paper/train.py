import jax
import jax.numpy as jnp
import optax
from typing import List, Callable, Dict, Iterable, Any
from jax.random import PRNGKey
from functools import partial
import os
from flax.training import checkpoints
import time


def optim_wrapper(optimizer, grads, opt_state, params=None):
    try:
        return optimizer.update(grads, opt_state, params)
    except TypeError:
        return optimizer.update(grads, opt_state)

def _train_step(model, optimizer, params, opt_state, x, y, z, rng):
    def loss_fn_wrapped(params):
        logits = model(params, x, y, rng)
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(z, logits.shape[-1])).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn_wrapped)(params)
    updates, opt_state = optim_wrapper(optimizer, grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def _eval_step(model, params, x, y, z, rng):
    logits = model(params, x, y, rng)
    loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(z, logits.shape[-1])).mean()
    return loss

train_step = jax.jit(_train_step, static_argnums=(0,))
eval_step = jax.jit(_eval_step, static_argnums=(0,))


def train_session(
    epochs, 
    data: dict, 
    model: dict, 
    optimizer, 
    params: dict, 
    rng: jax.Array,
    opt_state: dict,
    ckpt_path: str | None = None, 
    save_per_epoch: int = 1,
):
    # If checkpoint exists, restore
    start_epoch = 0
    if ckpt_path is not None and os.path.exists(ckpt_path):
        state = checkpoints.restore_checkpoint(ckpt_path, target=None)
        if state is not None:
            params = state["params"]
            opt_state = state["opt_state"]
            start_epoch = state.get("epoch", 0)
            print(f"Resuming from checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        start_time = time.time()

        train_loss = 0.0
        train_batches = 0
        for xb, yb, zb in data['train_loader']:
            params, opt_state, loss = train_step(model['train'], optimizer, params, opt_state, xb, yb, zb, rng)
            train_loss += loss
            train_batches += 1
        train_loss /= train_batches

        val_loss = 0.0
        val_batches = 0
        for xb, yb, zb in data['test_loader']:
            loss = eval_step(model['val'], params, xb, yb, zb, rng)
            val_loss += loss
            val_batches += 1
        val_loss /= val_batches

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch {epoch+1}: "
              f"Train Loss {train_loss:.4f} "
              f"Val Loss {val_loss:.4f} "
              f"({epoch_time:.2f}s)")

        if ckpt_path is not None and ((epoch + 1) % save_per_epoch == 0):
            state = {
                "epoch": epoch + 1,
                "params": params,
                "opt_state": opt_state
            }
            checkpoints.save_checkpoint(ckpt_path, target=state, step=epoch + 1, overwrite=True)
            print(f"Checkpoint saved at epoch {epoch+1}")

    return params, opt_state
