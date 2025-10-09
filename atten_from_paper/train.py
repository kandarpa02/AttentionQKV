import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import os
import time
from typing import Any

class Trainer:
    def __init__(
        self,
        model_t: Any,
        model_v: Any,
        train_state: train_state.TrainState,
        val_state: train_state.TrainState,
        train_loader: Any,
        val_loader: Any,
        rng: jax.Array,
        ckpt_load_path: str | None = None,
        ckpt_save_path: str | None = None,
        save_per_epoch: int = 1,
    ):
        self.model_t = model_t
        self.model_v = model_v
        self.train_state = train_state
        self.val_state = val_state
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.rng = rng
        self.ckpt_load_path = ckpt_load_path
        self.ckpt_save_path = ckpt_save_path
        self.save_per_epoch = save_per_epoch
        
        # For tracking
        self.train_losses = []
        self.val_losses = []
        self.start_epoch = 0

        self._precompile()
        self._restore_checkpoint()

    def _precompile(self):
        sample_batch = next(iter(self.train_loader))
        print("Pre-compiling training step...")
        self.train_step = jax.jit(self._train_step)
        self.train_step(self.train_state, sample_batch, self.rng)

        print("Pre-compiling evaluation step...")
        self.eval_step = jax.jit(self._eval_step)
        self.eval_step(self.val_state, sample_batch, self.rng)

    def _restore_checkpoint(self):
        if self.ckpt_load_path and os.path.exists(self.ckpt_load_path):
            restored = checkpoints.restore_checkpoint(
                self.ckpt_load_path,
                target={'train': self.train_state, 'val': self.val_state}
            )
            if restored:
                self.train_state = restored['train']
                self.val_state = restored['val']
                self.start_epoch = self.train_state.step
                print(f"Resumed from checkpoint at epoch {self.start_epoch}")

    def _train_step(self, state: train_state.TrainState, batch: tuple, rng: jax.Array):
        pad_id = getattr(self.model_t, "pad_id", 0)  # use pad_id from model if present

        def loss_fn(params):
            src, dec_inp, target = batch
            logits = self.model_t.apply(params, src, dec_inp, rng, deterministic=False)
            
            # Mask out PAD tokens from loss
            pad_mask = (target != pad_id)
            loss = optax.softmax_cross_entropy(
                logits,
                jax.nn.one_hot(target, logits.shape[-1])
            )
            loss = (loss * pad_mask).sum() / pad_mask.sum()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def _eval_step(self, state: train_state.TrainState, batch: tuple, rng: jax.Array):
        pad_id = getattr(self.model_v, "pad_id", 0)
        src, dec_inp, target = batch

        logits = self.model_v.apply(state.params, src, dec_inp, rng, deterministic=True)

        pad_mask = (target != pad_id)
        loss = optax.softmax_cross_entropy(
            logits,
            jax.nn.one_hot(target, logits.shape[-1])
        )
        loss = (loss * pad_mask).sum() / pad_mask.sum()
        return loss

    def train_epoch(self):
        train_loss = 0.0
        train_batches = 0
        for xb, yb, zb in self.train_loader:
            batch = (xb, yb, zb)
            
            # Split RNG for each batch to ensure randomness
            self.rng, subkey = jax.random.split(self.rng)
            self.train_state, loss = self.train_step(self.train_state, batch, subkey)
            
            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches
        self.train_losses.append(train_loss)
        return train_loss

    def evaluate(self):
        self.val_state = self.val_state.replace(params=self.train_state.params)
        val_loss = 0.0
        val_batches = 0

        for xb, yb, zb in self.val_loader:
            batch = (xb, yb, zb)
            self.rng, subkey = jax.random.split(self.rng)
            loss = self.eval_step(self.val_state, batch, subkey)
            val_loss += loss.item()
            val_batches += 1

        val_loss /= val_batches
        self.val_losses.append(val_loss)
        return val_loss

    def save_checkpoint(self, epoch):
        if self.ckpt_save_path and ((epoch + 1) % self.save_per_epoch == 0):
            state = {'train': self.train_state, 'val': self.val_state}
            checkpoints.save_checkpoint(
                self.ckpt_save_path,
                target=state,
                step=epoch + 1,
                overwrite=False,
                keep=epoch
            )
            print(f"Checkpoint saved at epoch {epoch + 1}")

    def train(self, epochs: int):
        for epoch in range(self.start_epoch, epochs):
            start_time = time.time()
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            epoch_time = time.time() - start_time

            print(f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Time {epoch_time:.2f}s")
            self.save_checkpoint(epoch)
