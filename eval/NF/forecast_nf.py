import os
import sys
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm
import time as timelib 

ROOT = os.path.dirname(os.path.abspath(__file__))                     # train/CSDI
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))        # project root

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "NF"))   # for diff_models
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))           # for CSDI package
sys.path.insert(0, PROJECT_ROOT)                                          # for top-level modules if any

from architecture import create_nfm


def parse_args():
    p = argparse.ArgumentParser(description="NF-TSF forecast script")
    p.add_argument("--model_path",      required=True,
                   help="Path to trained model .pth file")
    p.add_argument("--config",     default=None,
                   help="Path to training config_*.json (for architecture params)")
  
    p.add_argument("--data_path",       required=True,
                   help="Path to test data .npz file")
    p.add_argument("--out",  "-o",      required=True,
                   help="Output .npz path (e.g. results/nf/double_well.npz)")
    p.add_argument("--context_length",   type=int, default=100,
                   help="Context length L (default: 100)")
    p.add_argument("--prediction_length", type=int, default=100,
                   help="Forecast horizon H (default: 100)")
    p.add_argument("--n_samples", type=int, default=500,
                   help="Ensemble size S (default: 500)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_test_split", type=int, default=None,
                   help="Step index where forecast starts (default: T - n_future)")
    p.add_argument("--test_size", type=int, required=False)
    return p.parse_args()


def setup_device(device_arg):
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    print(f"Device: {device}")
    torch.set_default_device(device)
    return device


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_model(model_path, config, context_length, prediction_length, device):
    """Build NF architecture and load weights."""
    # Architecture defaults — overridden by config if provided
    flow_blocks   = 6
    hidden_units  = 64
    hidden_layers = "1,2"

    import yaml
    if config is not None and os.path.exists(config):
        with open(config) as f:
            cfg = yaml.safe_load(f)  
        flow_blocks   = cfg.get("flow_blocks",   flow_blocks)
        hidden_units  = cfg.get("hidden_units",  hidden_units)
        hidden_layers = cfg.get("hidden_layers", hidden_layers)
        print(f"Loaded architecture from config: "
              f"flow_blocks={flow_blocks}, hidden_units={hidden_units}, "
              f"hidden_layers={hidden_layers}")
    else:
        print(f"No config_path provided — using defaults: "
              f"flow_blocks={flow_blocks}, hidden_units={hidden_units}, "
              f"hidden_layers={hidden_layers}")

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
    print(f"Loaded weights: {model_path}")
    return model


def load_test_data(data_path):
    """
    Load test npz.  Expects keys:
        positions  : (N_traj, T)
        time       : (T,)            optional
        train_test_split : scalar    optional
    """
    data = np.load(data_path, allow_pickle=True)
    positions = data["positions"].astype(np.float32)   # (N, T)
    time = data["time"] if "time" in data else None
    tts  = int(data["train_test_split"]) if "train_test_split" in data else None
    print(f"Test data: {positions.shape}  (N_traj, T)")
    return positions, time, tts

def run_forecast(model, positions, context_length, prediction_length, n_samples,
                 mean, std, device, train_test_split, test_size):
    
    test_size = min(test_size, positions.shape[0]) 
    positions = positions[:test_size]
    
    # Limit to test_size trajectories
    positions = positions[:test_size]
    N = positions.shape[0]

    # Context and ground truth windows
    ctx_raw = positions[:, train_test_split - context_length : train_test_split]   # (test_size, L)
    gt_raw  = positions[:, train_test_split : train_test_split + prediction_length] # (test_size, H)

    ctx_norm = ctx_raw

    samples_out = np.zeros((N, prediction_length, n_samples), dtype=np.float32)

    for i in tqdm(range(N), desc="Forecasting"):
        past_norm = torch.tensor(ctx_norm[i], dtype=torch.float32, device=device).unsqueeze(0)  # (1, L)
        past_repeat = past_norm.repeat(n_samples, 1)  # (S, L)

        samp_norm = None
        for attempt in range(3):
            try:
                with torch.no_grad():
                    samp_norm, _ = model.sample(n_samples, past_repeat)  # (S, H)
                break
            except AssertionError as e:
                if attempt == 2:
                    print(f"  WARNING: traj {i} failed after 3 attempts: {e}")
                    samp_norm = torch.zeros(n_samples, prediction_length, device=device)

        samp_real = samp_norm.cpu().numpy()  # (S, H)
        samples_out[i] = samp_real.T         # (H, S)

    return samples_out, gt_raw, ctx_raw

def main():
    args = parse_args()
    set_seed(args.seed)
    device = setup_device(args.device)

    positions, time, tts_npz = load_test_data(args.data_path)
    N, T = positions.shape

    if args.train_test_split is not None:
        train_test_split = args.train_test_split
        print(f"train_test_split: {train_test_split} (CLI override)")
    '''elif tts_npz is not None:
        tts = tts_npz
        print(f"train_test_split: {tts} (from npz)")
    else:
        tts = T - args.prediction_length
        print(f"train_test_split: {tts} (inferred as T - n_future)")'''
    

    assert train_test_split >= args.context_length, \
        f"train_test_split ({train_test_split}) must be >= n_past ({args.context_length})"
    assert train_test_split + args.prediction_length <= T, \
        f"train_test_split + n_future ({train_test_split + args.prediction_length}) exceeds T ({T})"

    context_length=args.context_length
    prediction_length = args.prediction_length
    test_size=args.test_size

    model = load_model(
        args.model_path, args.config,
        context_length, prediction_length, device,
    )

    #start time measurment
    time_start = timelib.time()
    
    samples, ground_truth, contexts = run_forecast(
        model, positions,
        context_length=context_length, prediction_length=prediction_length,
        test_size=test_size,
        n_samples=args.n_samples,
        mean=0, std=0,
        device=device,
        train_test_split=train_test_split,
    )
    
    time_elapsed = time_start - timelib.time()
    
    ci90_lower = np.percentile(samples, 5,  axis=2)   # (N, H)
    ci90_upper = np.percentile(samples, 95, axis=2)
    ci50_lower = np.percentile(samples, 25, axis=2)
    ci50_upper = np.percentile(samples, 75, axis=2)

    #full_trajectories = np.concatenate([contexts, ground_truth], axis=1)  # (N, L+H)
    full_trajectories = positions # (N, T)

    out_path = args.out
    if not out_path.endswith(".npz"):
        out_path += ".npz"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    save_dict = dict(
        samples = samples,           # (N, H, S)
        ground_truth = full_trajectories[:,train_test_split:train_test_split+prediction_length],      # (N, H)
        contexts = contexts,          # (N, L)
        ci90_lower = ci90_lower,        # (N, H)
        ci90_upper = ci90_upper,
        ci50_lower = ci50_lower,
        ci50_upper = ci50_upper,
        full_trajectories = full_trajectories, # (N, L+H)
        train_test_split = train_test_split,
        prediction_length = prediction_length,
        num_of_samples = args.n_samples,
        context_length = context_length,
        time_elapsed = time_elapsed
    )
    if time is not None:
        save_dict["time"]       = time
        save_dict["time_train"] = time[:train_test_split]
        save_dict["time_test"]  = time[train_test_split:]

    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved: {out_path}")
    print(f"  samples      : {samples.shape}")
    print(f"  ground_truth : {ground_truth.shape}")
    print(f"  contexts     : {contexts.shape}")

    assert not np.isnan(samples).any(),      "NaN in samples"
    assert not np.isinf(samples).any(),      "Inf in samples"
    assert np.all(ci90_lower <= ci90_upper), "CI90 ordering violated"
    print("Sanity checks passed")


if __name__ == "__main__":
    main()