import os
import sys
import json
import argparse

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


def parse_args():
    p = argparse.ArgumentParser(description="NF-TSF forecast script")
    p.add_argument("--model_path",        required=True)
    p.add_argument("--config",            default=None)
    p.add_argument("--data_path",         required=True)
    p.add_argument("--out", "-o",         required=True)
    p.add_argument("--context_length",    type=int, default=100)
    p.add_argument("--prediction_length", type=int, default=100)
    p.add_argument("--n_samples",         type=int, default=500)
    p.add_argument("--batch_size",        type=int, default=256,
                   help="Number of (traj × sample) pairs per forward pass. "
                        "Increase for speed, decrease if OOM.")
    p.add_argument("--device",            default="cuda",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--train_test_split",  type=int, default=None)
    p.add_argument("--test_size",         type=int, required=False)
    return p.parse_args()


def setup_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    print(f"Device: {device}")
    return device


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_model(model_path, config_path, context_length, prediction_length, device):
    flow_blocks = 6
    hidden_units = 64
    hidden_layers = "1,2"

    import yaml
    if config_path is not None and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        flow_blocks   = cfg.get("flow_blocks",   flow_blocks)
        hidden_units  = cfg.get("hidden_units",  hidden_units)
        hidden_layers = cfg.get("hidden_layers", hidden_layers)
        print(f"Architecture from config: flow_blocks={flow_blocks}, "
              f"hidden_units={hidden_units}, hidden_layers={hidden_layers}")
    else:
        print(f"Using default architecture: flow_blocks={flow_blocks}, "
              f"hidden_units={hidden_units}, hidden_layers={hidden_layers}")

    hidden_layers_list = tuple(int(x) for x in str(hidden_layers).split(","))

    model = create_nfm(
        device,
        latent_size = prediction_length,
        context_size = context_length,
        K = flow_blocks,
        hidden_units = hidden_units,
        hidden_layers_list = hidden_layers_list,
    )
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"Loaded weights: {model_path}")
    return model


def load_test_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    positions = data["positions"].astype(np.float32)   # (N, T)
    time = data["time"] if "time" in data else None
    tts = int(data["train_test_split"]) if "train_test_split" in data else None
    print(f"Test data: {positions.shape}  (N_traj, T)")
    return positions, time, tts


def run_forecast_batched(
    model,
    positions : np.ndarray,  
    context_length: int,
    prediction_length: int,
    n_samples : int,
    train_test_split: int,
    test_size : int,
    batch_size : int,
    device : torch.device,
):
    test_size = min(test_size, positions.shape[0])
    positions = positions[:test_size]
    N = positions.shape[0]

    tts = train_test_split
    ctx_np = positions[:, tts - context_length : tts]          
    gt_np = positions[:, tts : tts + prediction_length]       

    ctx_tensor = torch.tensor(ctx_np, dtype=torch.float32, device=device)  # (N, L)

    samples_out = np.zeros((N, prediction_length, n_samples), dtype=np.float32)

    time_elapsed= 0

    n_batches = (N + batch_size - 1) // batch_size
    for b in tqdm(range(n_batches), desc="Forecasting (batched)"):
        b_start = b * batch_size
        b_end = min(b_start + batch_size, N)
        B  = b_end - b_start

        ctx_b = ctx_tensor[b_start:b_end]         
        ctx_b_tiled = ctx_b.repeat_interleave(n_samples, dim=0)  

        with torch.no_grad():
            try:
                time_start = timelib.time()
                samp, _ = model.sample(B * n_samples, ctx_b_tiled)  # (B*S, H)
                time_elapsed_inst = timelib.time() - time_start
                print(f"time_elapsed_inst {time_elapsed_inst}")
                
                time_elapsed += time_elapsed_inst
            except AssertionError as e:
                raise RuntimeWarning()
        samp_np = samp.cpu().numpy().reshape(B, n_samples, prediction_length)
        samples_out[b_start:b_end] = samp_np.transpose(0, 2, 1)  # (B, H, S)

    print(f"Inference time : {time_elapsed:.2f}s  "
          f"({time_elapsed / N * 1000:.1f} ms/trajectory, "
          f"{time_elapsed / (N * n_samples) * 1000:.3f} ms/sample)")

    return samples_out, gt_np, ctx_np, time_elapsed

def main():
    args   = parse_args()
    set_seed(args.seed)

    device = setup_device(args.device)

    positions, time, tts_npz = load_test_data(args.data_path)
    N, T = positions.shape

    if args.train_test_split is not None:
        train_test_split = args.train_test_split
        print(f"train_test_split : {train_test_split} (CLI override)")
    elif tts_npz is not None:
        train_test_split = tts_npz
        print(f"train_test_split : {train_test_split} (from npz)")
    else:
        train_test_split = T - args.prediction_length
        print(f"train_test_split : {train_test_split} (inferred as T - H)")

    assert train_test_split >= args.context_length, (
        f"train_test_split ({train_test_split}) < context_length ({args.context_length})")
    assert train_test_split + args.prediction_length <= T, (
        f"train_test_split + prediction_length "
        f"({train_test_split + args.prediction_length}) > T ({T})")

    context_length    = args.context_length
    prediction_length = args.prediction_length
    test_size         = args.test_size if args.test_size is not None else N

    model = load_model(
        args.model_path, args.config,
        context_length, prediction_length, device,
    )

    samples, ground_truth, contexts, time_elapsed = run_forecast_batched(
        model            = model,
        positions        = positions,
        context_length   = context_length,
        prediction_length= prediction_length,
        n_samples        = args.n_samples,
        train_test_split = train_test_split,
        test_size        = test_size,
        batch_size       = args.batch_size,
        #batch_size = 64,
        device           = device,
    )

    N_out = len(samples)

    ci90_lower = np.percentile(samples,  5, axis=2)  
    ci90_upper = np.percentile(samples, 95, axis=2)
    ci50_lower = np.percentile(samples, 25, axis=2)
    ci50_upper = np.percentile(samples, 75, axis=2)

    out_path = args.out
    if not out_path.endswith(".npz"):
        out_path += ".npz"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    save_dict = dict(
        samples           = samples,                      
        ground_truth      = ground_truth,                
        contexts          = contexts,                    
        ci90_lower        = ci90_lower,
        ci90_upper        = ci90_upper,
        ci50_lower        = ci50_lower,
        ci50_upper        = ci50_upper,
        full_trajectories = positions[:N_out],         
        train_test_split  = train_test_split,
        prediction_length = prediction_length,
        context_length    = context_length,
        num_of_samples    = args.n_samples,
        time_elapsed      = time_elapsed,
    )
    if time is not None:
        save_dict["time"]       = time
        save_dict["time_train"] = time[:train_test_split]
        save_dict["time_test"]  = time[train_test_split : train_test_split + prediction_length]

    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved: {out_path}")
    print(f"samples : {samples.shape}  (N, H, S)")
    print(f"ground_truth : {ground_truth.shape}  (N, H)")
    print(f"contexts  : {contexts.shape}  (N, L)")
    print(f"time_elapsed : {time_elapsed:.2f}s")

    assert not np.isnan(samples).any(), "NaN in samples"
    assert not np.isinf(samples).any(), "Inf in samples"
    assert np.all(ci90_lower <= ci90_upper), "CI90 ordering violated"
    print("sanity checks passed")


if __name__ == "__main__":
    main()