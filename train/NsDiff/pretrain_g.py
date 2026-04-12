#!/usr/bin/env python3
"""
pretrain_g.py
=============
Stage 2 of NsDiff pretraining: pretrain g_psi (g_backbone / SigmaEstimation).

Paper: "Non-stationary Diffusion For Probabilistic Time Series Forecasting"
       Section 4 — "We follow previous works (Kim et al., 2021; Liu et al.,
       2024b) to train the prior scale of uncertainty g_psi(X). We use the
       input variance to predict the output variance."

Specifically:
    - g_psi(X) predicts sigma_{Y_0} (the actual future variance)
    - Loss: MSE between sqrt(g_psi(X)) and sqrt(sigma_{Y_0})
      i.e. matching predicted std to actual future std
    - sigma_{Y_0} is computed via wv_sigma_trailing on the concatenated
      [context, future] window, taking the last pred_len steps
    - g_psi is INDEPENDENT of f_phi — it only takes context X as input
      so Stage 2 can run in parallel with or before Stage 1

Data setup (matches your pipeline):
    Train windows : positions_train[:, :800]  sliding stride windows
    Val window    : positions_train[:, 800:]  single window per trajectory
                    context=800:900, target=900:1000

Usage:
    python pretrain_g.py \
        --config configs/nsdiff/double_well.yaml \
        --input  DATA/double_well.npz \
        --out    checkpoints/nsdiff/double_well/pretrain
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

ROOT         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "NsDiff"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))
sys.path.insert(0, PROJECT_ROOT)

import NsDiff.src.layer.g_backbone as G
from NsDiff.src.utils.sigma import wv_sigma_trailing
from NsDiff.dataset_md import get_dataloader_md

EPS = 1e-8


# ---------------------------------------------------------------------------
# Compute actual future variance sigma_{Y_0}
# ---------------------------------------------------------------------------

def compute_y_sigma(batch_x, batch_y, rolling_length, pred_len, device):
    """
    Compute sigma_{Y_0}: the actual trailing variance of the forecast window.

    Paper: the forward process uses sigma_{Y_0} as the variance at t=0.
           We compute it as the trailing wavelet variance over the
           concatenated [context, future] sequence, then take the last
           pred_len steps.

    Parameters
    ----------
    batch_x        : (B, ctx_len, 1) — context
    batch_y        : (B, pred_len, 1) — ground truth future
    rolling_length : int — window size for wv_sigma_trailing
    pred_len       : int — forecast horizon H

    Returns
    -------
    y_sigma : (B, pred_len, 1) — actual future variance, clipped to EPS
    """
    # Concatenate context + future: (B, ctx_len + pred_len, 1)
    xy = torch.cat([batch_x, batch_y], dim=1)

    # Trailing variance: uses history up to each point
    # wv_sigma_trailing with discard_rep=False pads beginning by rolling_length
    # and returns T+1 windows — we take the last pred_len
    y_sigma = wv_sigma_trailing(xy, rolling_length)   # (B, ctx_len+pred_len+1, 1)
    y_sigma = y_sigma[:, -pred_len:, :] + EPS          # (B, pred_len, 1)
    return y_sigma


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_g(model, val_loader, rolling_length, pred_len, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_x, batch_y, _, _, _ = batch
            batch_x = batch_x.to(device).float()  # (B, ctx_len, 1)
            batch_y = batch_y.to(device).float()  # (B, pred_len, 1)

            # Predict variance from context only
            gx      = torch.clamp(model(batch_x), min=EPS)  # (B, pred_len, 1)

            # Actual future variance
            y_sigma = compute_y_sigma(
                batch_x, batch_y, rolling_length, pred_len, device)

            # Loss: MSE between predicted std and actual std
            # Paper: "use the input variance to predict the output variance"
            # Match sqrt(gx) vs sqrt(y_sigma) = matching standard deviations
            loss = (torch.sqrt(gx) - torch.sqrt(y_sigma)).square().mean()

            total_loss += loss.item()
            n_batches  += 1
    model.eval()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: Pretrain g_backbone (g_psi) — variance estimator. "
            "Independent of Stage 1."
        )
    )
    parser.add_argument("--config", "-c", required=True,
                        help="Path to NsDiff yaml config")
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to combined DATA/ds.npz file")
    parser.add_argument("--out",    "-o", required=True,
                        help="Output directory for pretrained weights")
    parser.add_argument("--device", "-d", default="cuda:0")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pretrain_g_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    device         = torch.device(args.device)
    ctx_len        = int(config["context_length"])
    pred_len       = int(config["prediction_length"])
    rolling_length = int(config["rolling_length"])

    # Pretraining-specific hyperparameters
    lr           = float(config.get("pretrain_g_lr",      config["lr"]))
    epochs       = int(config.get("pretrain_g_epochs",    config.get("pretrain_epochs", 100)))
    patience     = int(config.get("pretrain_g_patience",  config.get("patience", 20)))
    batch_size   = int(config["batch_size"])
    stride       = int(config["stride"])
    val_size     = int(config["val_size"])
    test_size    = int(config["test_size"])
    val_interval = int(config.get("valid_epoch_interval", 1))

    print(f"=== Stage 2: Pretrain g_backbone (g_psi) ===")
    print(f"context_length   : {ctx_len}")
    print(f"prediction_length: {pred_len}")
    print(f"rolling_length   : {rolling_length}")
    print(f"lr               : {lr}")
    print(f"epochs           : {epochs}")
    print(f"patience         : {patience}")

    # ------------------------------------------------------------------
    # Data — same split as train_nsdiff.py and pretrain_mu.py
    # ------------------------------------------------------------------
    train_loader, val_loader, _ = get_dataloader_md(
        npz_path          = args.input,
        context_length    = ctx_len,
        prediction_length = pred_len,
        batch_size        = batch_size,
        stride            = stride,
        val_size          = val_size,
        test_size         = test_size,
    )

    print(f"train batches : {len(train_loader)}")
    print(f"val   batches : {len(val_loader)}")

    # ------------------------------------------------------------------
    # Model — exactly same instantiation as train_nsdiff.py
    # kernel_size=1 means minimal smoothing; hidden_size=32 is compact
    # ------------------------------------------------------------------
    model = G.SigmaEstimation(
        ctx_len, pred_len, 1,
        kernel_size=1,
        hidden_size=32,
    ).float().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"g_backbone parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimizer
    # Paper: g_psi trained independently from f_phi
    # Using a slightly lower LR than f_phi since variance is smoother
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2,
        verbose=True,
    )

    best_val_loss  = float("inf")
    patience_count = 0
    train_losses   = []
    val_losses     = []
    val_epochs     = []

    print(f"\n{'='*55}")
    print(f"Training g_backbone for up to {epochs} epochs")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch_x, batch_y, _, _, _ = batch
            batch_x = batch_x.to(device).float()  # (B, ctx_len, 1)
            batch_y = batch_y.to(device).float()  # (B, pred_len, 1)

            # Compute actual future variance sigma_{Y_0}
            # This is the supervision signal for g_psi
            y_sigma = compute_y_sigma(
                batch_x, batch_y, rolling_length, pred_len, device)

            # Forward: predict variance from context g_psi(X)
            optimizer.zero_grad()
            gx = torch.clamp(model(batch_x), min=EPS)  # (B, pred_len, 1)

            # Loss: match predicted std to actual future std
            # Paper: "use the input variance to predict the output variance"
            # This is equivalent to matching the scale of the LSNM endpoint
            # N(f_phi(X), g_psi(X)) to the actual data distribution
            loss = (torch.sqrt(gx) - torch.sqrt(y_sigma)).square().mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train = float(np.mean(epoch_losses))
        train_losses.append(avg_train)

        if epoch % val_interval == 0 or epoch == 1:
            val_loss = validate_g(
                model, val_loader, rolling_length, pred_len, device)
            val_losses.append(val_loss)
            val_epochs.append(epoch)

            scheduler.step(val_loss)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss  = val_loss
                patience_count = 0
                torch.save(
                    model.state_dict(),
                    os.path.join(args.out, "cond_pred_model_g.pth"),
                )
            else:
                patience_count += 1

            print(
                f"[{epoch:4d}/{epochs}]  "
                f"train={avg_train:.6f}  val={val_loss:.6f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
                + ("  ← best" if is_best else f"  (patience {patience_count}/{patience})")
            )

            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            print(f"[{epoch:4d}/{epochs}]  train={avg_train:.6f}")

    # ------------------------------------------------------------------
    # Save losses
    # ------------------------------------------------------------------
    np.savez(
        os.path.join(args.out, "pretrain_g_losses.npz"),
        train_losses = np.array(train_losses),
        val_losses   = np.array(val_losses),
        val_epochs   = np.array(val_epochs),
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train Std-MSE")
    if val_losses:
        ax.plot(val_epochs, val_losses, "o-", label="Val Std-MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (std space)")
    ax.set_title("g_backbone (g_psi) Pretraining")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "pretrain_g_curves.png"), dpi=150)
    plt.close()

    print(f"\nBest val loss : {best_val_loss:.6f}")
    print(f"Saved         : {args.out}/cond_pred_model_g.pth")


if __name__ == "__main__":
    main()