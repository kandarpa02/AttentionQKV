import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from flax import struct
import os
import time
from typing import Any

# Define the training state
class TrainState(train_state.TrainState):
    pass

def _train_step(state: TrainState, batch: dict, rng: jax.Array) -> tuple[TrainState, float]:
    """Single training step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['x'], batch['y'], rng)
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(batch['z'], logits.shape[-1])).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def _eval_step(state: TrainState, batch: dict, rng: jax.Array) -> float:
    """Single evaluation step."""
    logits = state.apply_fn({'params': state.params}, batch['x'], batch['y'], rng)
    loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(batch['z'], logits.shape[-1])).mean()
    return loss

def precompile_functions(state: TrainState, rng: jax.Array, sample_batch: dict):
    """JIT compile train and eval steps."""
    print("Pre-compiling training step...")
    train_step = jax.jit(_train_step)
    train_step(state, sample_batch, rng)  # First call compiles
    print("Pre-compiling evaluation step...")
    eval_step = jax.jit(_eval_step)
    eval_step(state, sample_batch, rng)  # First call compiles
    return train_step, eval_step

def train_session(
    epochs: int,
    data: dict,
    state: TrainState,
    rng: jax.Array,
    ckpt_path: str | None = None,
    save_per_epoch: int = 1,
) -> TrainState:
    """Main training loop."""
    sample_batch = next(iter(data['train_loader']))
    train_step, eval_step = precompile_functions(state, rng, sample_batch)

    start_epoch = 0
    if ckpt_path and os.path.exists(ckpt_path):
        restored_state = checkpoints.restore_checkpoint(ckpt_path, target=state)
        if restored_state:
            state = restored_state
            start_epoch = restored_state.step
            print(f"Resumed from checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train_loss = 0.0
        train_batches = 0

        # Training loop
        for i, (xb, yb, zb) in enumerate(data['train_loader']):
            batch = {'x': xb, 'y': yb, 'z': zb}
            state, loss = train_step(state, batch, rng)
            train_loss += loss
            train_batches += 1
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss:.4f}")

        train_loss /= train_batches

        # Validation loop
        val_loss = 0.0
        val_batches = 0
        for xb, yb, zb in data['test_loader']:
            batch = {'x': xb, 'y': yb, 'z': zb}
            loss = eval_step(state, batch, rng)
            val_loss += loss
            val_batches += 1
        val_loss /= val_batches

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Time {epoch_time:.2f}s")

        # Save checkpoint
        if ckpt_path and ((epoch + 1) % save_per_epoch == 0):
            checkpoints.save_checkpoint(ckpt_path, state, epoch + 1, overwrite=True)
            print(f"Checkpoint saved at epoch {epoch+1}")

    return state
