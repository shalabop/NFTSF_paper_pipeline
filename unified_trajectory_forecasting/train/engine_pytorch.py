"""
train/engine_pytorch.py
=======================
Custom PyTorch training loop used by the NFTSF normalizing-flow model.

Mirrors the training logic from NFTSF_ssh/train_model.py::train_model()
(lines 286-549) and adapts it to the unified BaseModel interface.

Key features carried over from NFTSF_ssh
-----------------------------------------
- Adam optimiser with configurable weight decay
- ReduceLROnPlateau LR scheduler
- Gradient norm clipping
- Mini-batch or full-batch training
- Train / validation loss tracking with early stopping
- Periodic checkpoint saving
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from models.base import BaseModel


def run(
    model: BaseModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: dict,
    output_dir: Path,
) -> dict:
    """
    Train *model* using a custom PyTorch loop.

    Parameters
    ----------
    model       : A BaseModel instance (typically NFTSFModel).
    train_loader: DataLoader yielding ``(context, target)`` mini-batches.
    val_loader  : DataLoader yielding ``(context, target)`` mini-batches.
    config      : Merged config dict.
    output_dir  : Directory where checkpoints and loss histories are saved.

    Returns
    -------
    results : dict with keys ``loss_history``, ``val_loss_history``,
              ``best_epoch``, ``best_val_loss``.
    """
    tc = config["training"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = tc.get("device", "auto")
    if device_str == "auto":
        import torch as _t
        device_str = "cuda" if _t.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    epochs          = int(tc.get("epochs", 1000))
    lr              = float(tc.get("learning_rate", 1e-3))
    weight_decay    = float(tc.get("weight_decay", 1e-5))
    grad_clip       = float(tc.get("grad_clip", 1.0))
    use_scheduler   = bool(tc.get("use_scheduler", True))
    sched_patience  = int(tc.get("scheduler_patience", 15))
    sched_factor    = float(tc.get("scheduler_factor", 0.5))
    es_patience     = int(tc.get("early_stopping_patience", 50))
    save_interval   = int(tc.get("save_interval", 100))

    # Move model to device.
    inner = getattr(model, "model", None)
    if inner is not None:
        inner.to(device)
        params = inner.parameters()
    else:
        params = model.parameters()  # type: ignore[attr-defined]

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=sched_factor, patience=sched_patience
        )
        if use_scheduler
        else None
    )

    loss_history: list[float]     = []
    val_loss_history: list[float] = []
    best_val_loss   = float("inf")
    best_epoch      = 0
    no_improve      = 0

    for epoch in tqdm(range(epochs), desc=f"Training {model.name}"):

        # ---- Training pass ----
        if inner is not None:
            inner.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.training_step(batch)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[engine_pytorch] Loss is {loss.item()} at epoch {epoch}. Stopping.")
                return {
                    "loss_history":     loss_history,
                    "val_loss_history": val_loss_history,
                    "best_epoch":       best_epoch,
                    "best_val_loss":    best_val_loss,
                }
            loss.backward()
            if grad_clip > 0 and inner is not None:
                torch.nn.utils.clip_grad_norm_(inner.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        loss_history.append(avg_loss)

        # ---- Validation pass ----
        avg_val_loss = 0.0
        for batch in val_loader:
            avg_val_loss += model.validation_step(batch).item()
        avg_val_loss /= max(len(val_loader), 1)
        val_loss_history.append(avg_val_loss)

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # ---- Early stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch    = epoch
            no_improve    = 0
            model.save(output_dir / "model_best.pth")
        else:
            no_improve += 1

        if es_patience > 0 and no_improve >= es_patience:
            print(f"\n[engine_pytorch] Early stop at epoch {epoch+1} "
                  f"(no improvement for {no_improve} epochs).")
            break

        # ---- Periodic checkpoint ----
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            model.save(output_dir / f"checkpoint_epoch_{epoch+1}.pth")

    # Final checkpoint.
    model.save(output_dir / "model_final.pth")

    results = {
        "loss_history":     loss_history,
        "val_loss_history": val_loss_history,
        "best_epoch":       best_epoch,
        "best_val_loss":    best_val_loss,
    }
    with open(output_dir / "train_results.json", "w") as f:
        json.dump(
            {k: (v if not isinstance(v, list) else v) for k, v in results.items()},
            f, indent=2,
        )
    return results
