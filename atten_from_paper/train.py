import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
from flax import struct
import optax
import os
import time
from typing import Any

class TrainState(train_state.TrainState):
    pass

def _train_step(state: TrainState, batch: dict, rng: jax.Array) -> tuple[TrainState, float]:
    def loss_fn(params):
        logits = state.apply_fn(params, batch[0], batch[1], rng)
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(batch[2], logits.shape[-1])).mean()
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss.item()

def _eval_step(state: TrainState, batch: dict, rng: jax.Array) -> float:
    logits = state.apply_fn(state.params, batch[0], batch[1], rng)
    loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(batch[2], logits.shape[-1])).mean()
    return loss.item()

def precompile_functions(train_state: TrainState, val_state: TrainState, rng: jax.Array, sample_batch: dict):
    print("Pre-compiling training step...")
    train_step = jax.jit(_train_step)
    train_step(train_state, sample_batch, rng)

    print("Pre-compiling evaluation step...")
    eval_step = jax.jit(_eval_step)
    eval_step(val_state, sample_batch, rng)

    return train_step, eval_step

def train_session(
    epochs: int,
    data: dict,
    train_state: TrainState,
    val_state: TrainState,
    rng: jax.Array,
    ckpt_path: str | None = None,
    save_per_epoch: int = 1,
) -> tuple[TrainState, TrainState]:
    sample_batch = next(iter(data['train_loader']))
    train_step, eval_step = precompile_functions(train_state, val_state, rng, sample_batch)

    start_epoch = 0
    if ckpt_path and os.path.exists(ckpt_path):
        restored = checkpoints.restore_checkpoint(ckpt_path, target={'train': train_state, 'val': val_state})
        if restored:
            train_state = restored['train']
            val_state = restored['val']
            start_epoch = train_state.step
            print(f"Resumed from checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train_loss = 0.0
        train_batches = 0

        for i, (xb, yb, zb) in enumerate(data['train_loader']):
            batch = (xb, yb, zb)
            train_state, loss = train_step(train_state, batch, rng)
            train_loss += loss
            train_batches += 1
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Train Loss: {loss:.4f}")

        train_loss /= train_batches

        val_loss = 0.0
        val_batches = 0
        for xb, yb, zb in data['test_loader']:
            batch = (xb, yb, zb)
            loss = eval_step(val_state, batch, rng)
            val_loss += loss
            val_batches += 1
        val_loss /= val_batches

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Time {epoch_time:.2f}s")

        if ckpt_path and ((epoch + 1) % save_per_epoch == 0):
            state = {'train': train_state, 'val': val_state}
            checkpoints.save_checkpoint(ckpt_path, state, epoch + 1, overwrite=True)
            print(f"Checkpoint saved at epoch {epoch+1}")

    return train_state, val_state
