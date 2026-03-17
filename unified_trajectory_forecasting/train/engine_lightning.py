"""
train/engine_lightning.py
=========================
PyTorch Lightning dispatch used by the diffusion-model family
(TSDiff-Q, TSDiff-MS, TSDiff-Cond, CSDI, RATD, NsDiff).

Because each upstream model already ships as a ``LightningModule``,
this engine simply instantiates the appropriate PL Trainer, hands it the
GluonTS data loaders produced by the adapter, and calls ``trainer.fit()``.

If the upstream model is not a ``LightningModule`` (e.g. CSDI/RATD which
use a plain PyTorch loop internally), the engine falls back to a minimal
custom loop that calls ``model.training_step`` per batch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from models.base import BaseModel


# ---------------------------------------------------------------------------
# GluonTS → PyTorch DataLoader helper
# ---------------------------------------------------------------------------

def _gluonts_to_loader(
    dataset,
    context_length: int,
    prediction_length: int,
    batch_size: int,
    mode: str = "train",
):
    """
    Wrap a GluonTS ListDataset in a PyTorch DataLoader that yields
    ``(context, target)`` float32 tensors.

    Falls back gracefully if GluonTS is not available.
    """
    try:
        from gluonts.dataset.loader import TrainDataLoader, InferenceDataLoader
        from gluonts.torch.batchify import batchify
        from uncond_ts_diff.utils import create_splitter
    except ImportError:
        return None  # caller must handle

    from gluonts.itertools import Cached
    splitter = create_splitter(
        past_length=context_length,
        future_length=prediction_length,
        mode=mode,
    )

    if mode == "train":
        loader = TrainDataLoader(
            Cached(dataset),
            batch_size=batch_size,
            stack_fn=batchify,
            transform=splitter,
            num_batches_per_epoch=128,
        )
    else:
        loader = InferenceDataLoader(
            dataset,
            batch_size=batch_size,
            stack_fn=batchify,
            transform=splitter,
        )
    return loader


# ---------------------------------------------------------------------------
# Simple custom loop fallback (for CSDI / RATD / NsDiff)
# ---------------------------------------------------------------------------

def _custom_loop(
    model: BaseModel,
    train_dataset,
    val_dataset,
    config: dict,
    output_dir: Path,
) -> dict:
    """Minimal custom training loop for non-Lightning diffusion models."""
    from data.adapters.gluonts_adapter import _trajectories_to_entries

    tc = config["training"]
    n_past   = int(config["data"]["n_past"])
    n_future = int(config["data"]["n_future"])
    epochs   = int(tc.get("epochs", 1000))
    bs       = int(tc.get("batch_size", 64))
    lr       = float(tc.get("learning_rate", 1e-3))
    grad_clip= float(tc.get("grad_clip", 1.0))
    es_pat   = int(tc.get("early_stopping_patience", 50))
    save_int = int(tc.get("save_interval", 100))

    device_str = tc.get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    inner = getattr(model, "model", None)
    params = inner.parameters() if inner else []
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=float(tc.get("weight_decay", 1e-5)))

    # Convert GluonTS entries to simple numpy arrays.
    def _ds_to_arrays(ds):
        targets = [np.array(entry["target"], dtype=np.float32) for entry in ds]
        T = min(len(t) for t in targets)
        # context = first n_past steps, target = last n_future steps.
        ctx = np.stack([t[:n_past]   for t in targets if len(t) >= n_past + n_future])
        tgt = np.stack([t[n_past:n_past + n_future] for t in targets if len(t) >= n_past + n_future])
        return torch.tensor(ctx), torch.tensor(tgt)

    train_ctx, train_tgt = _ds_to_arrays(train_dataset)
    val_ctx,   val_tgt   = _ds_to_arrays(val_dataset)

    N = train_ctx.shape[0]
    best_val_loss = float("inf")
    best_epoch    = 0
    no_improve    = 0
    loss_hist: list[float] = []
    val_hist:  list[float] = []

    from tqdm import tqdm
    for epoch in tqdm(range(epochs), desc=f"Training {model.name}"):
        if inner: inner.train()
        perm = torch.randperm(N)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, N, bs):
            idx = perm[start:start + bs]
            batch = (train_ctx[idx], train_tgt[idx])
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            if grad_clip > 0 and inner:
                torch.nn.utils.clip_grad_norm_(inner.parameters(), grad_clip)
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        loss_hist.append(ep_loss / max(n_batches, 1))

        # Validation.
        if inner: inner.eval()
        with torch.no_grad():
            vl = model.validation_step((val_ctx, val_tgt)).item()
        if inner: inner.train()
        val_hist.append(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_epoch    = epoch
            no_improve    = 0
            model.save(output_dir / "model_best.pth")
        else:
            no_improve += 1

        if es_pat > 0 and no_improve >= es_pat:
            break

        if save_int > 0 and (epoch + 1) % save_int == 0:
            model.save(output_dir / f"checkpoint_epoch_{epoch+1}.pth")

    model.save(output_dir / "model_final.pth")
    results = {"loss_history": loss_hist, "val_loss_history": val_hist,
               "best_epoch": best_epoch, "best_val_loss": best_val_loss}
    with open(output_dir / "train_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# Lightning dispatch
# ---------------------------------------------------------------------------

def run(
    model: BaseModel,
    train_dataset,
    val_dataset,
    config: dict,
    output_dir: Path,
) -> dict:
    """
    Train a diffusion model using PyTorch Lightning (preferred) or a custom
    loop fallback.

    Parameters
    ----------
    model         : A BaseModel wrapping a GluonTS / Lightning module.
    train_dataset : GluonTS ListDataset (train trajectories).
    val_dataset   : GluonTS ListDataset (truncated for validation).
    config        : Merged config dict.
    output_dir    : Directory for checkpoints and results.

    Returns
    -------
    results dict (see ``engine_pytorch.run``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tc = config["training"]
    n_past   = int(config["data"]["n_past"])
    n_future = int(config["data"]["n_future"])
    epochs   = int(tc.get("epochs", 1000))
    bs       = int(tc.get("batch_size", 64))

    inner = getattr(model, "model", None)

    # Check if the inner model is a LightningModule.
    try:
        import pytorch_lightning as pl
        is_lightning = inner is not None and isinstance(inner, pl.LightningModule)
    except ImportError:
        is_lightning = False

    if is_lightning:
        # ---- PyTorch Lightning path ----
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

        train_loader = _gluonts_to_loader(
            train_dataset, n_past, n_future, bs, mode="train"
        )
        val_loader = _gluonts_to_loader(
            val_dataset, n_past, n_future, bs, mode="val"
        )

        callbacks = [
            ModelCheckpoint(
                dirpath=str(output_dir),
                filename="model_best",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            ),
        ]
        es_patience = int(tc.get("early_stopping_patience", 50))
        if es_patience > 0:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=es_patience, mode="min"))

        trainer = Trainer(
            max_epochs=epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=callbacks,
            default_root_dir=str(output_dir),
            gradient_clip_val=float(tc.get("grad_clip", 1.0)),
            enable_progress_bar=True,
        )
        trainer.fit(inner, train_dataloaders=train_loader,
                    val_dataloaders=val_loader)

        best_ckpt = output_dir / "model_best.ckpt"
        if not best_ckpt.exists():
            # PL may append version suffix; find it.
            ckpts = list(output_dir.rglob("model_best*.ckpt"))
            if ckpts:
                best_ckpt = ckpts[0]

        results = {
            "loss_history":     [],
            "val_loss_history": [],
            "best_epoch":       trainer.current_epoch,
            "best_val_loss":    float("nan"),
            "checkpoint":       str(best_ckpt),
        }
        with open(output_dir / "train_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    else:
        # ---- Custom loop fallback ----
        return _custom_loop(model, train_dataset, val_dataset, config, output_dir)
