#!/usr/bin/env python3
"""
forecast_nf_dataloader.py – NF-TSF forecast using the same DataLoader
as CSDI, making timing comparisons fair.  The model is called exactly once
per batch with tiled contexts, matching the batching pattern of CSDI.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
import time as timelib

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "NF"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))
sys.path.insert(0, PROJECT_ROOT)

from architecture import create_nfm
from CSDI.dataset_md import get_dataloader_md   # reuse the same dataset loader


def parse_args():
    p = argparse.ArgumentParser(description="NF-TSF forecast (DataLoader version)")
    p.add_argument("--config", "-c", required=True, help="Path to YAML config")
    p.add_argument("--input", "-i", required=True, help="Path to test .npz file")
    p.add_argument("--ckpt", "-k", required=True, help="Path to model .pth")
    p.add_argument("--out", "-o", required=True, help="Output .npz path")
    p.add_argument("--device", "-d", default="cuda:0")
    p.add_argument("--train_test_split", "-tts", type=int, default=None,help="Override train_test_split")
    p.add_argument("--prediction_length", type=int)
    p.add_argument("--context_length", type=int)
    p.add_argument("--num_of_samples", type=int)
    p.add_argument("--test_size", type=int)
    return p.parse_args()


def load_model(config_path, model_path, context_length, prediction_length, device):
    """Load NF model from checkpoint, reading architecture params from config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Defaults (same as original NF script)
    flow_blocks   = cfg.get("flow_blocks", 6)
    hidden_units  = cfg.get("hidden_units", 64)
    hidden_layers = cfg.get("hidden_layers", "1,2")

    hidden_layers_list = tuple(int(x) for x in str(hidden_layers).split(","))

    model = create_nfm(
        device,
        latent_size=prediction_length,
        context_size=context_length,
        K=flow_blocks,
        hidden_units=hidden_units,
        hidden_layers_list=hidden_layers_list,
    )
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"Loaded NF model from {model_path}")
    print(f"  flow_blocks={flow_blocks}, hidden_units={hidden_units}, "
          f"hidden_layers={hidden_layers_list}")
    return model


def evaluate_nf(model, test_loader, n_samples, device, prediction_length, context_length):
    """
    Batched inference for NF, exactly analogous to evaluate_csdi.

    For each batch:
      - observed_data: (B, L+H, 1) – the full window (context+forecast)
    We extract context (first L steps) and ground truth (last H steps),
    tile the context n_samples times, call model.sample once, and reshape.
    """
    model.eval()
    all_samples = []
    all_gt = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Forecasting"):
            observed = batch["observed_data"]          # (B, L+H, 1)
            B = observed.shape[0]

            # Extract context and ground truth
            ctx = observed[:, :context_length, 0]      # (B, L)
            gt = observed[:, -prediction_length:, 0]   # (B, H)
            ctx = ctx.float().to(device)
            gt = gt.float().to(device)

            # Tile context: (B, L) -> (B * n_samples, L)
            ctx_tiled = ctx.repeat_interleave(n_samples, dim=0)

            # Sample from the flow
            samp, _ = model.sample(B * n_samples, ctx_tiled)   # (B*S, H)

            # Reshape to (B, S, H) then transpose to (B, H, S)
            samp = samp.view(B, n_samples, prediction_length)   # (B, S, H)
            samp = samp.permute(0, 2, 1)                        # (B, H, S)

            all_samples.append(samp.cpu().numpy())
            all_gt.append(gt.cpu().numpy())

    samples_out = np.concatenate(all_samples, axis=0)   # (N, H, S)
    gt_out = np.concatenate(all_gt, axis=0)             # (N, H)
    return samples_out, gt_out


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract hyperparameters (same keys as CSDI config)
    prediction_length = args.prediction_length
    context_length    = args.context_length
    num_of_samples    = args.num_of_samples
    batch_size        = config["batch_size"]
    test_size         = args.test_size
    stride            = -1 #config["stride"] dont matter for testinf
    val_size          = -1 #config["val_size"]

    # Load test data to get full trajectories and time array
    data = np.load(args.input)
    positions = data["positions"]
    time = data["time"] if "time" in data else None

    if args.train_test_split is not None:
        train_test_split = args.train_test_split
        print(f"Using train_test_split = {train_test_split} (CLI override)")
    else:
        train_test_split = int(data["train_test_split"])
        print(f"Using train_test_split = {train_test_split} (from .npz)")

    print(f"prediction_length : {prediction_length}")
    print(f"context_length    : {context_length}")
    print(f"num_of_samples    : {num_of_samples}")
    print(f"batch_size        : {batch_size}")

    # Create test loader exactly as CSDI does
    test_loader = get_dataloader_md(
        flag="test",
        npz_path=args.input,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=64 ,#batch_size,
        test_size=test_size,
        stride=stride,
        val_size=val_size,
        train_test_split=train_test_split,
    )

    # Load model
    model = load_model(args.config, args.ckpt, context_length,
                       prediction_length, device)

    # Timing: only the evaluation loop
    time_start = timelib.time()
    samples, gt = evaluate_nf(model, test_loader, num_of_samples,
                              device, prediction_length, context_length)
    time_elapsed = timelib.time() - time_start
    print(f"time_elapsed {time_elapsed}")

    N = len(samples)

    # Confidence intervals (same as CSDI)
    ci90_lower = np.percentile(samples,  5, axis=2)
    ci90_upper = np.percentile(samples, 95, axis=2)
    ci50_lower = np.percentile(samples, 25, axis=2)
    ci50_upper = np.percentile(samples, 75, axis=2)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        args.out,
        samples=samples,
        ground_truth=gt,
        ci90_lower=ci90_lower,
        ci90_upper=ci90_upper,
        ci50_lower=ci50_lower,
        ci50_upper=ci50_upper,
        full_trajectories=positions[:N],
        time=time,
        time_train=time[:train_test_split] if time is not None else None,
        time_test=time[train_test_split:train_test_split+prediction_length] if time is not None else None,
        train_test_split=train_test_split,
        prediction_length=prediction_length,
        num_of_samples=num_of_samples,
        context_length=context_length,
        time_elapsed=time_elapsed,
    )
    print(f"Saved: {args.out}")
    print(f"  samples shape: {samples.shape} (N, H, S)")
    print(f"  ground_truth shape: {gt.shape} (N, H)")
    print(f"  time_elapsed: {time_elapsed:.2f} s")

    # Sanity checks
    assert not np.isnan(samples).any(), "NaN in samples"
    assert not np.isinf(samples).any(), "Inf in samples"
    assert np.all(ci90_lower <= ci90_upper), "CI90 ordering violated"
    print("Sanity checks passed ✓")


if __name__ == "__main__":
    main()
    
    
'''

python eval/NF/forecast_nf_ex.py \
  --config configs/nf/alanine_phi_100_100_forecast.yaml \
  --input DATA/alanine_phi_test.npz \
  --ckpt checkpoints_100_100/nf/alanine_phi.pth \
  --out results/nf/alanine_phi_100_100_test.npz \
  --device cuda:0 --test_size 3000 \
  --prediction_length 100 --context_length 100 \
  --num_of_samples 1000 --train_test_split 900
  
python eval/NF/forecast_nf_ex.py \
  --config configs/nf/alanine_psi_100_100_forecast.yaml \
  --input DATA/alanine_psi_test.npz \
  --ckpt checkpoints_100_100/nf/alanine_psi.pth \
  --out results/nf/alanine_psi_100_100_test.npz \
  --device cuda:0 --test_size 3000 \
  --prediction_length 100 --context_length 100 \
  --num_of_samples 1000 --train_test_split 900
  
  '''
