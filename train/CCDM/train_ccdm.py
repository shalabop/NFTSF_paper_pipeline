# train_ccdm.py
import argparse
import json
import os
import sys
import yaml
import random
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))                     # train/CSDI
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))        # project root

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "CCDM"))   # for diff_models
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))           # for CSDI package
sys.path.insert(0, PROJECT_ROOT)  

from CCDM.dataset_md   import get_dataloader_md   
from CCDM.custom_model import DiffMTS           

def build_configs(cfg, num_feat, device):

    class Configs: pass
    c = Configs()

    c.data_name = cfg["data"]["data_name"]
    c.cont_len = cfg["data"]["context_length"]
    c.pred_len = cfg["data"]["prediction_length"]
    c.use_window_norm  = cfg["data"]["use_window_norm"]
    c.num_feat = num_feat

    c.n_emb = cfg["model"]["n_emb"]
    c.cont_hidden_dim = cfg["model"]["cont_hidden_dim"]
    c.pred_hidden_dim = cfg["model"]["pred_hidden_dim"]
    c.step_hidden_dim = cfg["model"]["step_hidden_dim"]
    c.time_hidden_dim = cfg["model"]["time_hidden_dim"]
    c.n_depth = cfg["model"]["n_depth"]
    c.n_heads = cfg["model"]["n_heads"]
    c.attn_dropout = cfg["model"]["attn_dropout"]
    c.mlp_ratio = cfg["model"]["mlp_ratio"]
    c.non_attn = cfg["model"]["non_attn"]

    c.n_steps = cfg["diffusion"]["n_steps"]
    c.beta_start = cfg["diffusion"]["beta_start"]
    c.beta_end = cfg["diffusion"]["beta_end"]
    c.beta_schedule = cfg["diffusion"]["beta_schedule"]
    c.parameterization = cfg["diffusion"]["parameterization"]
    c.step_dist = cfg["diffusion"]["step_dist"]

    c.use_contrast = cfg["contrastive"]["use_contrast"]
    c.contrast_weight  = cfg["contrastive"]["contrast_weight"]
    c.n_negatives = cfg["contrastive"]["n_negatives"]
    c.temperature = cfg["contrastive"]["temperature"]

    c.n_epochs = cfg["train"]["epochs"]
    c.init_lr = cfg["train"]["lr"]

    c.device = device
    c.prediction_length = cfg["data"]["prediction_length"]
    c.context_length = cfg["data"]["context_length"]
    return c


def main():
    parser = argparse.ArgumentParser(description="Train CCDM on MD data")
    parser.add_argument("--config", "-c", required=True, help="Path to config_double_well.yaml")
    parser.add_argument("--input",  "-i", required=True, help="Path to data/data_train.npz")
    parser.add_argument("--out",    "-o", required=True, help="Output folder, e.g. checkpoints/ccdm/double_well")
    parser.add_argument("--device", "-d", default="cuda:0")
    parser.add_argument("--two_stage", "-ts",     action="store_true", default=False)
    parser.add_argument("--refine_epochs", "-re" , type=int, default=0)
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

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    context_length = config["data"]["context_length"]
    prediction_length = config["data"]["prediction_length"]
    batch_size = config["train"]["batch_size"]
    val_size = config["train"]["val_size"]
    test_size = config["train"]["test_size"]
    stride = config["train"]["stride"]

    # Data
    train_loader, val_loader = get_dataloader_md(
        flag="train",
        npz_path = args.input,
        context_length = context_length,
        prediction_length = prediction_length,
        batch_size = batch_size,
        val_size = val_size,
        test_size = test_size,
        stride = stride,
    )

    np.save(os.path.join(args.out, "context_length.npy"), np.array([cfg["data"]["context_length"]]))
    np.save(os.path.join(args.out, "prediction_length.npy"),np.array([cfg["data"]["prediction_length"]]))

    batch_x, _, _, x_mark, y_mark  = next(iter(train_loader))
    D = batch_x.shape[-1]
    print(f"num_feat (D) = {D}")
    configs  = build_configs(cfg, D, device)
    configs.valid_epoch_interval = cfg["train"]["valid_epoch_interval"]
    configs.save_epoch_interval  = cfg["train"]["save_epoch_interval"]

    from network import Denoiser
    n_params = sum(p.numel() for p in Denoiser(configs).parameters())
    print(f"Parameters: {n_params:,}")

    model = DiffMTS(configs, train_loader, val_loader, None, out_dir=args.out)

    model.train(
        is_refine = args.two_stage,
        model_path = args.out, #path to checkp
        refine_epochs  = args.refine_epochs,
        loss_path = cfg["train"]["training_curv"]
    )

    print(f"Saved to: {args.out}/")


if __name__ == "__main__":
    main()