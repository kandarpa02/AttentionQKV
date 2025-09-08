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

# Pre-compile the training and evaluation steps with a dummy batch
def precompile_functions(model, optimizer, params, opt_state, rng, sample_batch):
    xb, yb, zb = sample_batch
    print("Pre-compiling training step...")
    train_step = jax.jit(partial(_train_step, model['train'], optimizer), static_argnums=(0, 1))
    train_step(params, opt_state, xb, yb, zb, rng)  # First call compiles
    print("Pre-compiling evaluation step...")
    eval_step = jax.jit(partial(_eval_step, model['val']), static_argnums=(0,))
    eval_step(params, xb, yb, zb, rng)  # First call compiles
    return train_step, eval_step

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
    # Pre-compile with a sample batch
    sample_batch = next(iter(data['train_loader']))
    train_step, eval_step = precompile_functions(model, optimizer, params, opt_state, rng, sample_batch)

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

        # Training loop with progress tracking
        for i, (xb, yb, zb) in enumerate(data['train_loader']):
            params, opt_state, loss = train_step(params, opt_state, xb, yb, zb, rng)
            train_loss += loss
            train_batches += 1
            if i % 100 == 0:  # Print progress every 100 batches
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss:.4f}")

        train_loss /= train_batches

        # Validation loop
        val_loss = 0.0
        val_batches = 0
        for xb, yb, zb in data['test_loader']:
            loss = eval_step(params, xb, yb, zb, rng)
            val_loss += loss
            val_batches += 1
        val_loss /= val_batches

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Time {epoch_time:.2f}s")

        if ckpt_path is not None and ((epoch + 1) % save_per_epoch == 0):
            state = {"epoch": epoch + 1, "params": params, "opt_state": opt_state}
            checkpoints.save_checkpoint(ckpt_path, state, epoch + 1, overwrite=True)
            print(f"Checkpoint saved at epoch {epoch+1}")

    return params, opt_state