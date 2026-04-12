#!/usr/bin/env python3
"""
train_nsdiff_pretrained.py
==========================
Stage 3 of NsDiff pretraining pipeline: train the diffusion model xi_theta
with FROZEN pretrained f_phi (mu_backbone) and g_psi (g_backbone).

Paper: "Non-stationary Diffusion For Probabilistic Time Series Forecasting"
       The paper's recommended workflow:
         1. Pretrain f_phi via MSE (pretrain_mu.py)
         2. Pretrain g_psi via variance matching (pretrain_g.py)
         3. Freeze f_phi and g_psi, train only xi_theta (this script)

       With frozen f_phi and g_psi, xi_theta receives stable, high-quality
       conditional mean y_0_hat = f_phi(X) and conditional variance
       gx = g_psi(X) at every training step. This is what makes the
       diffusion model converge faster and more stably than joint training.

The KL loss trained here is:
    L = E_t [ ||e - xi_theta(y_t, y_0_hat, gx, t)||^2
              + sigma_tilde/sigma_theta
              - log(sigma_tilde/sigma_theta) ]

where:
    y_t        = noisy future at timestep t (via q_sample)
    y_0_hat    = f_phi(X) — frozen mean prediction
    gx         = g_psi(X) — frozen variance prediction
    sigma_tilde = posterior variance (analytically derived from schedule)
    sigma_theta = xi_theta's predicted variance

Data setup (matches your pipeline):
    Train windows : positions_train[:, :800]  sliding stride windows
    Val window    : positions_train[:, 800:]  single window per trajectory
                    context=800:900, target=900:1000

Usage:
    python train_nsdiff_pretrained.py \
        --config  configs/nsdiff/double_well.yaml \
        --input   DATA/double_well.npz \
        --pretrain checkpoints/nsdiff/double_well/pretrain \
        --out     checkpoints/nsdiff/double_well/diffusion
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

from NsDiff.src.models.NsDiff import NsDiff
import NsDiff.src.layer.mu_backbone as ns_Transformer
import NsDiff.src.layer.g_backbone as G
from NsDiff.src.layer.nsdiff_utils import q_sample, cal_sigma_tilde, cal_forward_noise
from NsDiff.src.utils.sigma import wv_sigma_trailing
from NsDiff.dataset_md import get_dataloader_md

EPS = 1e-8


# ---------------------------------------------------------------------------
# Build args — same as train_nsdiff.py / pretrain_mu.py
# ---------------------------------------------------------------------------

def build_args(config, device):
    c         = config
    label_len = c["context_length"] // 2
    return SimpleNamespace(
        seq_len              = c["context_length"],
        pred_len             = c["prediction_length"],
        label_len            = label_len,
        device               = device,
        features             = None,
        enc_in               = 1,
        dec_in               = 1,
        c_out                = 1,
        d_model              = c["d_model"],
        n_heads              = c["n_heads"],
        e_layers             = c["e_layers"],
        d_layers             = c["d_layers"],
        d_ff                 = c["d_ff"],
        moving_avg           = c["moving_avg"],
        timesteps            = c["diffusion_steps"],
        factor               = c.get("factor", 3),
        distil               = c.get("distil", True),
        beta_schedule        = c.get("beta_schedule", "linear"),
        beta_start           = c["beta_start"],
        beta_end             = c["beta_end"],
        embed                = "fixed",
        freq                 = "h",
        dropout              = c.get("dropout", 0.05),
        activation           = c.get("activation", "gelu"),
        output_attention     = False,
        do_predict           = True,
        k_z                  = c.get("k_z", 1e-2),
        k_cond               = c.get("k_cond", 1),
        p_hidden_dims        = [64, 64],
        p_hidden_layers      = c.get("p_hidden_layers", 2),
        d_z                  = c.get("d_z", 8),
        CART_input_x_embed_dim = c.get("CART_input_x_embed_dim", 32),
    )


# ---------------------------------------------------------------------------
# One training step — only xi_theta is optimized
# f_phi and g_psi are frozen (torch.no_grad() context)
# ---------------------------------------------------------------------------

def process_diffusion_batch(
    model,                  # NsDiff — xi_theta
    cond_pred_model,        # mu_backbone — FROZEN f_phi
    cond_pred_model_g,      # g_backbone  — FROZEN g_psi
    batch_x, batch_y,
    pred_len, label_len, rolling_length, device
):
    """
    Forward pass for diffusion training with frozen f_phi and g_psi.

    Key difference from train_nsdiff.py's process_train_batch:
    - f_phi and g_psi are called under torch.no_grad() — no gradients flow
      back into them. Only xi_theta receives gradients.
    - This is the paper's recommended training procedure after pretraining.

    Returns
    -------
    loss : scalar — KL diffusion loss
    """
    n = batch_x.size(0)

    # ------------------------------------------------------------------
    # Compute sigma_{Y_0}: actual future variance (supervision signal)
    # This is used in the forward process noise schedule
    # ------------------------------------------------------------------
    xy      = torch.cat([batch_x, batch_y], dim=1)
    y_sigma = wv_sigma_trailing(xy, rolling_length)
    y_sigma = y_sigma[:, -pred_len:, :] + EPS   # (B, pred_len, 1)

    # ------------------------------------------------------------------
    # Get f_phi(X) and g_psi(X) — NO gradients (frozen)
    # ------------------------------------------------------------------
    with torch.no_grad():
        # f_phi(X): conditional mean prediction
        dec_inp = torch.cat([
            batch_x[:, -label_len:, :],
            torch.zeros([n, pred_len, 1], device=device),
        ], dim=1)
        y_0_hat, _ = cond_pred_model(batch_x, None, dec_inp, None)
        # y_0_hat: (B, pred_len, 1)

        # g_psi(X): conditional variance prediction
        gx = torch.clamp(cond_pred_model_g(batch_x), min=EPS)
        # gx: (B, pred_len, 1)

    # ------------------------------------------------------------------
    # Antithetic timestep sampling — reduces variance of gradient estimate
    # Paper: "Draw t ~ Uniform({1,...,T})"
    # Using antithetic pairs t and (T-1-t) for variance reduction
    # ------------------------------------------------------------------
    t = torch.randint(0, model.num_timesteps, (n // 2 + 1,), device=device)
    t = torch.cat([t, model.num_timesteps - 1 - t], dim=0)[:n]

    # ------------------------------------------------------------------
    # Forward diffusion process: q(y_t | Y_0, f_phi(X), g_psi(X), sigma_{Y_0})
    # Paper Eq. — uses uncertainty-aware noise schedule
    # ------------------------------------------------------------------
    y_T_mean = y_0_hat   # endpoint mean = f_phi(X)
    e        = torch.randn_like(batch_y)

    # Compute forward noise variance at timestep t
    forward_noise = cal_forward_noise(
        model.betas_tilde, model.betas_bar, gx, y_sigma, t)
    noise = e * torch.sqrt(forward_noise)

    # Compute posterior variance sigma_tilde (target for xi_theta's sigma)
    sigma_tilde = cal_sigma_tilde(
        model.alphas, model.alphas_cumprod,
        model.alphas_cumprod_sum, model.alphas_cumprod_prev,
        model.alphas_cumprod_sum_prev,
        model.betas_tilde_m_1, model.betas_bar_m_1,
        gx, y_sigma, t,
    )

    # Sample noisy y_t via closed-form q(y_t | Y_0)
    y_t = q_sample(
        batch_y, y_T_mean,
        model.alphas_bar_sqrt,
        model.one_minus_alphas_bar_sqrt,
        t, noise=noise,
    )

    # ------------------------------------------------------------------
    # Reverse process: xi_theta predicts noise e and variance sigma_theta
    # from (y_t, y_0_hat, gx, t)
    # ------------------------------------------------------------------
    output, sigma_theta = model(batch_x, None, y_t, y_0_hat, gx, t)
    sigma_theta = sigma_theta + EPS

    # ------------------------------------------------------------------
    # KL loss: minimize KL(q(y_{t-1}|y_t, Y_0) || p_theta(y_{t-1}|y_t))
    # Simplified form from paper:
    # L = ||e - output||^2 + sigma_tilde/sigma_theta - log(sigma_tilde/sigma_theta)
    # The constant -1 does not affect gradients and is omitted
    # ------------------------------------------------------------------
    kl_loss = (
        ((e - output)).square().mean()
        + (sigma_tilde / sigma_theta).mean()
        - torch.log(sigma_tilde / sigma_theta).mean()
    )

    return kl_loss


# ---------------------------------------------------------------------------
# Validation — same loss as training
# ---------------------------------------------------------------------------

def validate_diffusion(
    model, cond_pred_model, cond_pred_model_g,
    val_loader, pred_len, label_len, rolling_length, device,
):
    model.eval()
    # Note: cond_pred_model and cond_pred_model_g stay frozen (already eval)
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_x, batch_y, _, _, _ = batch
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()

            loss = process_diffusion_batch(
                model, cond_pred_model, cond_pred_model_g,
                batch_x, batch_y,
                pred_len, label_len, rolling_length, device,
            )
            total_loss += loss.item()
            n_batches  += 1
    model.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: Train NsDiff diffusion model with frozen pretrained "
            "f_phi (mu_backbone) and g_psi (g_backbone)"
        )
    )
    parser.add_argument("--config",   "-c", required=True)
    parser.add_argument("--input",    "-i", required=True,
                        help="Path to combined DATA/ds.npz file")
    parser.add_argument("--pretrain", "-p", required=True,
                        help="Directory containing cond_pred_model.pth and "
                             "cond_pred_model_g.pth from pretraining")
    parser.add_argument("--out",      "-o", required=True,
                        help="Output directory for diffusion model weights")
    parser.add_argument("--device",   "-d", default="cuda:0")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    device         = torch.device(args.device)
    ctx_len        = int(config["context_length"])
    pred_len       = int(config["prediction_length"])
    rolling_length = int(config["rolling_length"])
    label_len      = ctx_len // 2

    lr           = float(config["lr"])
    epochs       = int(config["epochs"])
    patience     = int(config["patience"])
    batch_size   = int(config["batch_size"])
    stride       = int(config["stride"])
    val_size     = int(config["val_size"])
    test_size    = int(config["test_size"])
    val_interval = int(config.get("valid_epoch_interval", 1))

    print(f"=== Stage 3: Train diffusion model xi_theta ===")
    print(f"context_length   : {ctx_len}")
    print(f"prediction_length: {pred_len}")
    print(f"rolling_length   : {rolling_length}")
    print(f"lr               : {lr}")
    print(f"epochs           : {epochs}")
    print(f"patience         : {patience}")
    print(f"pretrain dir     : {args.pretrain}")

    # ------------------------------------------------------------------
    # Data
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

    model_args = build_args(config, device)

    # ------------------------------------------------------------------
    # Load FROZEN pretrained f_phi (mu_backbone)
    # ------------------------------------------------------------------
    cond_pred_model = ns_Transformer.Model(model_args).float().to(device)
    mu_ckpt = os.path.join(args.pretrain, "cond_pred_model.pth")
    if not os.path.exists(mu_ckpt):
        raise FileNotFoundError(
            f"Pretrained mu_backbone not found at {mu_ckpt}. "
            f"Run pretrain_mu.py first."
        )
    cond_pred_model.load_state_dict(
        torch.load(mu_ckpt, map_location=device, weights_only=False))
    cond_pred_model.eval()
    # Freeze ALL parameters — no gradients will flow back into f_phi
    for p in cond_pred_model.parameters():
        p.requires_grad_(False)
    print(f"Loaded and FROZEN f_phi from {mu_ckpt}")

    # ------------------------------------------------------------------
    # Load FROZEN pretrained g_psi (g_backbone)
    # ------------------------------------------------------------------
    cond_pred_model_g = G.SigmaEstimation(
        ctx_len, pred_len, 1,
        kernel_size=1, hidden_size=32,
    ).float().to(device)
    g_ckpt = os.path.join(args.pretrain, "cond_pred_model_g.pth")
    if not os.path.exists(g_ckpt):
        raise FileNotFoundError(
            f"Pretrained g_backbone not found at {g_ckpt}. "
            f"Run pretrain_g.py first."
        )
    cond_pred_model_g.load_state_dict(
        torch.load(g_ckpt, map_location=device, weights_only=False))
    cond_pred_model_g.eval()
    # Freeze ALL parameters
    for p in cond_pred_model_g.parameters():
        p.requires_grad_(False)
    print(f"Loaded and FROZEN g_psi from {g_ckpt}")

    # ------------------------------------------------------------------
    # Diffusion model xi_theta — only this gets trained
    # ------------------------------------------------------------------
    model = NsDiff(model_args, device).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"xi_theta trainable parameters: {n_params:,}")

    # Optimizer only over xi_theta parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2,
        verbose=True,
    )

    best_val_loss  = float("inf")
    patience_count = 0
    train_losses   = []
    val_losses     = []
    val_epochs     = []

    print(f"\n{'='*60}")
    print(f"Training xi_theta (diffusion) for up to {epochs} epochs")
    print(f"f_phi and g_psi are FROZEN")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        # Keep frozen models in eval mode throughout
        cond_pred_model.eval()
        cond_pred_model_g.eval()

        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch_x, batch_y, _, _, _ = batch
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()

            # zero_grad BEFORE forward pass (fix from earlier analysis)
            optimizer.zero_grad()

            loss = process_diffusion_batch(
                model, cond_pred_model, cond_pred_model_g,
                batch_x, batch_y,
                pred_len, label_len, rolling_length, device,
            )

            if torch.isnan(loss):
                print(f"WARNING: NaN loss at epoch {epoch} — skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        if not epoch_losses:
            print(f"WARNING: No valid batches in epoch {epoch}")
            continue

        avg_train = float(np.mean(epoch_losses))
        train_losses.append(avg_train)

        if epoch % val_interval == 0 or epoch == 1:
            val_loss = validate_diffusion(
                model, cond_pred_model, cond_pred_model_g,
                val_loader, pred_len, label_len, rolling_length, device,
            )
            val_losses.append(val_loss)
            val_epochs.append(epoch)

            scheduler.step(val_loss)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss  = val_loss
                patience_count = 0
                torch.save(
                    model.state_dict(),
                    os.path.join(args.out, "model.pth"),
                )
                # Also copy frozen weights to out dir for forecasting
                torch.save(
                    cond_pred_model.state_dict(),
                    os.path.join(args.out, "cond_pred_model.pth"),
                )
                torch.save(
                    cond_pred_model_g.state_dict(),
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
    # Save training curves
    # ------------------------------------------------------------------
    np.savez(
        os.path.join(args.out, "diffusion_losses.npz"),
        train_losses = np.array(train_losses),
        val_losses   = np.array(val_losses),
        val_epochs   = np.array(val_epochs),
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train KL")
    if val_losses:
        ax.plot(val_epochs, val_losses, "o-", label="Val KL")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Loss")
    ax.set_title("xi_theta (NsDiff) Diffusion Training — Pretrained f_phi, g_psi frozen")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "diffusion_curves.png"), dpi=150)
    plt.close()

    print(f"\nBest val KL loss : {best_val_loss:.6f}")
    print(f"Saved model.pth  : {args.out}/model.pth")
    print(f"Saved f_phi      : {args.out}/cond_pred_model.pth")
    print(f"Saved g_psi      : {args.out}/cond_pred_model_g.pth")
    print(
        "\nAll three files are ready for forecast_nsdiff.py "
        "(pass --ckpt pointing to this directory)."
    )


if __name__ == "__main__":
    main()