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

import NsDiff.src.layer.mu_backbone as ns_Transformer
from NsDiff.dataset_md import get_dataloader_md


def build_mu_args(config, device):
    c = config
    label_len = c["context_length"] // 2
    return SimpleNamespace(
        seq_len = c["context_length"],
        pred_len = c["prediction_length"],
        label_len = label_len,
        device = device,
        features = None,
        enc_in = 1,
        dec_in = 1,
        c_out = 1,
        d_model = c["d_model"],
        n_heads = c["n_heads"],
        e_layers = c["e_layers"],
        d_layers = c["d_layers"],
        d_ff = c["d_ff"],
        moving_avg = c["moving_avg"],
        factor = c.get("factor", 3),
        distil = c.get("distil", True),
        embed = "fixed",
        freq = "h",
        dropout = c.get("dropout", 0.05),
        activation = c.get("activation", "gelu"),
        output_attention = False,
        do_predict = True,
        p_hidden_dims = [64, 64],
        p_hidden_layers = c.get("p_hidden_layers", 2),
        timesteps            = c["diffusion_steps"],
        beta_schedule        = c.get("beta_schedule", "linear"),
        beta_start           = c["beta_start"],
        beta_end             = c["beta_end"],
        k_z                  = c.get("k_z", 1e-2),
        k_cond               = c.get("k_cond", 1),
        d_z                  = c.get("d_z", 8),
        CART_input_x_embed_dim = c.get("CART_input_x_embed_dim", 32),
    )


def forward_mu(model, batch_x, label_len, pred_len, device):
    """
    Run one forward pass of f_phi (Non-stationary Transformer).

    Parameters
    ----------
    batch_x  : (B, ctx_len, 1) — context window
    label_len: int — overlap between encoder and decoder (ctx_len // 2)
    pred_len : int — forecast horizon H

    Returns
    -------
    y_0_hat : (B, pred_len, 1) — predicted conditional mean
    """
    B = batch_x.size(0)

    # Decoder input: last label_len of context + zeros for forecast region
    # This matches exactly what process_train_batch does in train_nsdiff.py
    dec_inp = torch.cat([
        batch_x[:, -label_len:, :],
        torch.zeros([B, pred_len, 1], device=device),
    ], dim=1)  # (B, label_len + pred_len, 1)

    # x_mark passed as None (embed="fixed" so temporal marks unused)
    y_0_hat, _ = model(batch_x, None, dec_inp, None)
    # y_0_hat: (B, pred_len, 1) 
    return y_0_hat



def validate_mu(model, val_loader, label_len, pred_len, device, criterion):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_x, batch_y, _, _, _ = batch
            batch_x = batch_x.to(device).float()  # (B, ctx_len, 1)
            batch_y = batch_y.to(device).float()  # (B, pred_len, 1)

            y_0_hat = forward_mu(model, batch_x, label_len, pred_len, device)
            loss = criterion(y_0_hat, batch_y)

            total_loss += loss.item()
            n_batches  += 1
    model.train()
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Pretrain mu_backbone (f_phi) via MSE regression"
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
    with open(os.path.join(args.out, "pretrain_mu_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    device   = torch.device(args.device)
    ctx_len  = int(config["context_length"])
    pred_len = int(config["prediction_length"])
    label_len = ctx_len // 2

    # Pretraining-specific hyperparameters — separate from diffusion training
    lr         = float(config.get("pretrain_mu_lr",      config["lr"]))
    epochs     = int(config.get("pretrain_mu_epochs",    config.get("pretrain_epochs", 100)))
    patience   = int(config.get("pretrain_mu_patience",  config.get("patience", 20)))
    batch_size = int(config["batch_size"])
    stride     = int(config["stride"])
    val_size   = int(config["val_size"])
    test_size  = int(config["test_size"])
    val_interval = int(config.get("valid_epoch_interval", 1))

    print(f"=== Stage 1: Pretrain mu_backbone (f_phi) ===")
    print(f"context_length  : {ctx_len}")
    print(f"prediction_length: {pred_len}")
    print(f"label_len       : {label_len}")
    print(f"lr              : {lr}")
    print(f"epochs          : {epochs}")
    print(f"patience        : {patience}")

    train_loader, val_loader, _ = get_dataloader_md(
        npz_path = args.input,
        context_length = ctx_len,
        prediction_length = pred_len,
        batch_size = batch_size,
        stride = stride,
        val_size = val_size,
        test_size         = test_size,
    )

    print(f"train batches : {len(train_loader)}")
    print(f"val   batches : {len(val_loader)}")

    model_args = build_mu_args(config, device)
    model = ns_Transformer.Model(model_args).float().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"mu_backbone parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2)

    best_val_loss  = float("inf")
    patience_count = 0
    train_losses = []
    val_losses = []
    val_epochs = []

    print(f"\n{'='*55}")
    print(f"Training mu_backbone for up to {epochs} epochs")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch_x, batch_y, _, _, _ = batch
            batch_x = batch_x.to(device).float()  # (B, ctx_len, 1)
            batch_y = batch_y.to(device).float()  # (B, pred_len, 1)

            optimizer.zero_grad()
            y_0_hat = forward_mu(model, batch_x, label_len, pred_len, device)

            loss = criterion(y_0_hat, batch_y)

            loss.backward()
            # Gradient clipping for stability (same as train_nsdiff.py)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train = float(np.mean(epoch_losses))
        train_losses.append(avg_train)

        if epoch % val_interval == 0 or epoch == 1:
            val_loss = validate_mu(
                model, val_loader, label_len, pred_len, device, criterion)
            val_losses.append(val_loss)
            val_epochs.append(epoch)

            scheduler.step(val_loss)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss  = val_loss
                patience_count = 0
                torch.save(
                    model.state_dict(),
                    os.path.join(args.out, "cond_pred_model.pth"),
                )
            else:
                patience_count += 1

            print(
                f"[{epoch:4d}/{epochs}]  "
                f"train={avg_train:.6f}  val={val_loss:.6f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
                + (" <-- best" if is_best else f"  (patience {patience_count}/{patience})")
            )

            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            print(f"[{epoch:4d}/{epochs}]  train={avg_train:.6f}")

    np.savez(
        os.path.join(args.out, "pretrain_mu_losses.npz"),
        train_losses = np.array(train_losses),
        val_losses   = np.array(val_losses),
        val_epochs   = np.array(val_epochs),
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train MSE")
    if val_losses:
        ax.plot(val_epochs, val_losses, "o-", label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("mu_backbone (f_phi) Pretraining")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "pretrain_mu_curves.png"), dpi=150)
    plt.close()

    print(f"\nBest val MSE : {best_val_loss:.6f}")
    print(f"Saved        : {args.out}/cond_pred_model.pth")


if __name__ == "__main__":
    main()