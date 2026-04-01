# train_ccdm.py
import argparse
import json
import os
import sys
import yaml
import random
import numpy as np
import torch

ROOT     = os.path.dirname(os.path.abspath(__file__))
CCDM_DIR = os.path.join(ROOT, "architectures", "CCDM")
sys.path.insert(0, CCDM_DIR)
sys.path.insert(0, ROOT)

from dataset_md   import get_dataloader_md   
from custom_model import DiffMTS           

def build_configs(cfg, num_feat, device):

    class Configs: pass
    c = Configs()

    c.data_name        = cfg["data"]["data_name"]
    c.cont_len         = cfg["data"]["context_length"]
    c.pred_len         = cfg["data"]["prediction_length"]
    c.use_window_norm  = cfg["data"]["use_window_norm"]
    c.num_feat         = num_feat

    c.n_emb            = cfg["model"]["n_emb"]
    c.cont_hidden_dim  = cfg["model"]["cont_hidden_dim"]
    c.pred_hidden_dim  = cfg["model"]["pred_hidden_dim"]
    c.step_hidden_dim  = cfg["model"]["step_hidden_dim"]
    c.time_hidden_dim  = cfg["model"]["time_hidden_dim"]
    c.n_depth          = cfg["model"]["n_depth"]
    c.n_heads          = cfg["model"]["n_heads"]
    c.attn_dropout     = cfg["model"]["attn_dropout"]
    c.mlp_ratio        = cfg["model"]["mlp_ratio"]
    c.non_attn         = cfg["model"]["non_attn"]

    c.n_steps          = cfg["diffusion"]["n_steps"]
    c.beta_start       = cfg["diffusion"]["beta_start"]
    c.beta_end         = cfg["diffusion"]["beta_end"]
    c.beta_schedule    = cfg["diffusion"]["beta_schedule"]
    c.parameterization = cfg["diffusion"]["parameterization"]
    c.step_dist        = cfg["diffusion"]["step_dist"]

    c.use_contrast     = cfg["contrastive"]["use_contrast"]
    c.contrast_weight  = cfg["contrastive"]["contrast_weight"]
    c.n_negatives      = cfg["contrastive"]["n_negatives"]
    c.temperature      = cfg["contrastive"]["temperature"]

    c.n_epochs         = cfg["train"]["epochs"]
    c.init_lr          = cfg["train"]["lr"]

    c.device = device
    return c


def main():
    parser = argparse.ArgumentParser(description="Train CCDM on MD data")
    parser.add_argument("--config", "-c", required=True, help="Path to config_double_well.yaml")
    parser.add_argument("--input",  "-i", required=True, help="Path to data/double_well.npz")
    parser.add_argument("--out",    "-o", required=True, help="Output folder, e.g. checkpoints/ccdm/double_well")
    parser.add_argument("--device", "-d", default="cuda:0")
    parser.add_argument("--two_stage",      action="store_true", default=False)
    parser.add_argument("--pretrain_epoch", type=int, default=None)
    parser.add_argument("--refine_epochs",  type=int, default=30)
    args = parser.parse_args()

    seed = 2024
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(json.dumps(cfg, indent=4))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader, mean, std = get_dataloader_md(
        npz_path = args.input,
        context_length = cfg["data"]["context_length"],
        prediction_length = cfg["data"]["prediction_length"],
        batch_size = cfg["train"]["batch_size"],
        normalization = cfg["train"]["normalization"],
        stride=cfg["train"]["stride"],
        val_size = cfg["train"]["val_size"],
        test_size = cfg["train"]["test_size"],
    )

    print(f"mean={mean:.4f}  std={std:.4f}  "
          f"train={len(train_loader)}  "
          f"val={len(val_loader)}  "
          f"test={len(test_loader)}")

    np.save(os.path.join(args.out, "mean.npy"), np.array([mean]))
    np.save(os.path.join(args.out, "std.npy"), np.array([std]))
    np.save(os.path.join(args.out, "context_length.npy"), np.array([cfg["data"]["context_length"]]))
    np.save(os.path.join(args.out, "prediction_length.npy"),np.array([cfg["data"]["prediction_length"]]))

    batch_x, _, _, x_mark, y_mark  = next(iter(train_loader))
    D = batch_x.shape[-1]
    print(f"num_feat (D) = {D}")
    configs  = build_configs(cfg, D, device)
    configs.valid_epoch_interval = cfg["train"].get("valid_epoch_interval", 50)
    configs.save_epoch_interval  = cfg["train"].get("save_epoch_interval", 100)

    from network import Denoiser
    n_params = sum(p.numel() for p in Denoiser(configs).parameters())
    print(f"Parameters: {n_params:,}")

    model = DiffMTS(configs, train_loader, val_loader, test_loader, out_dir=args.out)

    model.train(
        is_refine = args.two_stage,
        model_path = cfg["train"]["model_path"],
        refine_epochs  = args.refine_epochs,
        loss_path = cfg["train"]["training_curv"]
    )

    print(f"Saved to: {args.out}/")


if __name__ == "__main__":
    main()