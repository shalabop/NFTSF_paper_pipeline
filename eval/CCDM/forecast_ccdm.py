# forecast_ccdm.py
import argparse
import os
import sys
import yaml
import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))                     # train/CSDI
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))        # project root

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "CCDM"))   # for diff_models
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))           # for CSDI package
sys.path.insert(0, PROJECT_ROOT)  

from CCDM.network     import Denoiser          # noqa
from CCDM.diffusion   import DDPM              # noqa
from CCDM.dataset_md  import get_dataloader_md # noqa

def forecast_ccdm(denoiser, diffusion, test_loader, n_samples,
                  device, pred_len, num_feat):
    """
    Run reverse-diffusion sampling over the test set.

    Returns
    -------
    samples      : (N_windows, n_samples, H, D)
    ground_truth : (N_windows, H, D)
    contexts     : (N_windows, L, D)
    """
    denoiser.eval()
    all_samples, all_gt, all_ctx = [], [], []

    with torch.no_grad():
        #for batch_x, batch_y0, _ in tqdm(test_loader, desc="Forecasting"):
        for batch_x, batch_y0, local_scaler, x_mark, y0_mark in tqdm(test_loader, desc="Forecasting"):
            batch_x  = batch_x.float().to(device)   # (B, L, D)
            batch_y0 = batch_y0.float().to(device)  # (B, H, D)
            x_mark, y0_mark = x_mark.float().to(device), y0_mark.float().to(device)
        
            batch_x_n = batch_x


            # (B*n_samples, H, D)
            samples = diffusion.sampling(n_samples, batch_x_n,
                                         x_mark=None, y0_mark=None)

            B = batch_x.shape[0]
            samples = samples.view(B, n_samples, pred_len, num_feat)  # (B, S, H, D)

            all_samples.append(samples.cpu().numpy())
            all_gt.append(batch_y0.cpu().numpy())
            all_ctx.append(batch_x.cpu().numpy())

    return (np.concatenate(all_samples, axis=0),   # (N, S, H, D)
            np.concatenate(all_gt,      axis=0),   # (N, H, D)
            np.concatenate(all_ctx,     axis=0))   # (N, L, D)


def build_configs(cfg, num_feat, device):

    class Configs: pass
    c = Configs()

    c.cont_len         = cfg["data"]["context_length"]
    c.pred_len         = cfg["data"]["prediction_length"]
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

    c.device = device
    return c


def main():
    parser = argparse.ArgumentParser(description="Forecast with CCDM")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--ckpt",   "-k", required=True)
    parser.add_argument("--out",    "-o", required=True)
    parser.add_argument("--device", "-d", default="cuda:0")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prediction_length = cfg["data"]["prediction_length"]
    context_length = cfg["data"]["context_length"]
    n_samples = cfg["train"]["n_samples"]
    batch_size = cfg["train"]["batch_size"]
    test_size = cfg["train"]["test_size"]
    
    val_size = cfg["train"]["val_size"]
    stride = cfg["train"]["stride"]

    print(f"prediction_length : {prediction_length}")
    print(f"context_length    : {context_length}")
    print(f"n_samples         : {n_samples}")
    print(f"checkpoint        : {args.ckpt}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Data
    _, _, test_loader, mean, std = get_dataloader_md(
        npz_path = args.input,
        context_length = context_length,
        prediction_length = prediction_length,
        batch_size = batch_size,
        test_size = test_size,
        val_size = val_size,
        stride = stride,
    )

    # Infer D
    batch_x, _, _, _,_ = next(iter(test_loader))
    D = batch_x.shape[-1]
    print(f"num_feat (D) : {D}")

    # Build model
    configs  = build_configs(cfg, D, device)
    denoiser = Denoiser(configs).to(device)
    diffusion = DDPM(denoiser, configs).to(device)

    denoiser.load_state_dict(
        torch.load(args.ckpt, map_location=device, weights_only=False))
    print(f"Loaded checkpoint : {args.ckpt}")


    # Forecast
    samples, gt, contexts = forecast_ccdm(
        denoiser, diffusion, test_loader,
        n_samples, device, prediction_length, D,
    )

    ci90_lower = np.percentile(samples,  5, axis=1)   # (N, H, D)
    ci90_upper = np.percentile(samples, 95, axis=1)
    ci50_lower = np.percentile(samples, 25, axis=1)
    ci50_upper = np.percentile(samples, 75, axis=1)
    median     = np.percentile(samples, 50, axis=1)

    if samples.shape[-1] == 1:
        samples    = samples[..., 0]    # (N, S, H)
        gt         = gt[..., 0]         # (N, H)
        contexts   = contexts[..., 0]   # (N, L)
        median     = median[..., 0]     # (N, H)
        ci90_lower = ci90_lower[..., 0] # (N, H)
        ci90_upper = ci90_upper[..., 0]
        ci50_lower = ci50_lower[..., 0]
        ci50_upper = ci50_upper[..., 0]

    raw              = np.load(args.input)
    positions        = raw["positions"]
    time             = raw["time"]
    train_test_split = int(raw["train_test_split"])

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ####fixes shape
    samples=samples.transpose(0,2,1) # (N, H, S, D)
    #print(samples.shape)
    #samples shape is (N, prediction_length, samples)
        
    np.savez(
        args.out,
        samples           = samples,
        ground_truth      = gt,
        contexts          = contexts,
        median            = median,
        ci90_lower        = ci90_lower,
        ci90_upper        = ci90_upper,
        ci50_lower        = ci50_lower,
        ci50_upper        = ci50_upper,
        full_trajectories = positions,
        time              = time,
        time_train        = time[:train_test_split],
        time_test         = time[train_test_split:train_test_split + prediction_length],
        train_test_split  = train_test_split,
        prediction_length = prediction_length,
        context_length    = context_length,
        n_samples         = n_samples,
    )
    print(f"Saved: {args.out}")

    assert not np.isnan(samples).any(),      "NaN in samples"
    assert not np.isinf(samples).any(),      "Inf in samples"
    assert np.all(ci90_lower <= ci90_upper), "CI90 ordering violated"
    print("Sanity checks passed TRUE")


if __name__ == "__main__":
    main()