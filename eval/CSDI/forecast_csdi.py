# forecast_csdi.py
import argparse
import os
import sys
import yaml
import numpy as np
import torch
from tqdm import tqdm

ROOT     = os.path.dirname(os.path.abspath(__file__))
CSDI_DIR = os.path.join(ROOT, "architectures", "CSDI")
sys.path.insert(0, CSDI_DIR)
sys.path.insert(0, ROOT)

from models.CSDI.main_model import CSDI_Forecasting
from models.CSDI.dataset_md import get_dataloader_md


def denormalize(x, mean, std, normalization):
    """Reverse global normalization. Not used for 'local'."""
    if normalization == "zscore":
        return x * std + mean
    elif normalization == "std":
        return x * std
    return x  


def evaluate_csdi(model, test_loader, n_samples, device,
                  prediction_length, normalization, mean, std):
    """
    Returns samples and gt in original (denormalized) units.

    For normalization="local": each sample is denormalized by its own
    per-sample scaler stored in batch["local_scaler"].

    For all other modes: global denormalization applied after collecting
    all batches.
    """
    model.eval()
    all_samples = []
    all_gt      = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Forecasting"):  
            
            (observed_data, observed_mask, observed_tp,
             gt_mask, _, _, _) = model.process_data(batch)
            
            ##debug##################clear
            #ctx_len = observed_data.shape[-1] - prediction_length
            #observed_data_zero = observed_data.clone()
            #observed_data_zero[:, :, :ctx_len] = 0.0  

            cond_mask = gt_mask
            side_info = model.get_side_info(observed_tp, cond_mask)
            samples = model.impute(observed_data, cond_mask, side_info, n_samples)

            samples = samples[:, :, 0, -prediction_length:]  
            samples = samples.permute(0, 2, 1)               
            gt      = observed_data[:, 0, -prediction_length:].clone()  

            if normalization == "local":
                scaler = batch["local_scaler"].to(device)    
                samples = samples * scaler.unsqueeze(-1)     
                gt      = gt * scaler                        
                all_samples.append(samples.cpu().numpy())
                all_gt.append(gt.cpu().numpy())
            else:
                all_samples.append(samples.cpu().numpy())
                all_gt.append(gt.cpu().numpy())

    samples_out = np.concatenate(all_samples, axis=0)  
    gt_out      = np.concatenate(all_gt,      axis=0)  

    if normalization != "local":
        samples_out = denormalize(samples_out, mean, std, normalization)
        gt_out      = denormalize(gt_out,      mean, std, normalization)

    return samples_out, gt_out


def main():
    parser = argparse.ArgumentParser(description="Forecast with CSDI")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--ckpt",   "-k", required=True)
    parser.add_argument("--out",    "-o", required=True)
    parser.add_argument("--device", "-d", default="cuda:0")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["model"]["is_unconditional"] = 0

    prediction_length = config["train"]["prediction_length"]
    context_length    = config["train"]["context_length"]
    num_of_samples    = config["train"]["num_of_samples"]
    batch_size        = config["train"]["batch_size"]
    test_size         = config["train"]["test_size"]
    stride            = config["train"]["stride"]
    normalization     = config["train"]["normalization"]
    val_size         = config["train"]["val_size"]

    print(f"prediction_length : {prediction_length}")
    print(f"context_length    : {context_length}")
    print(f"num_of_samples    : {num_of_samples}")
    print(f"normalization     : {normalization}")

    ckpt_dir = os.path.dirname(args.ckpt)

    _, _, test_loader, mean, std = get_dataloader_md(
        npz_path          = args.input,
        context_length    = context_length,
        prediction_length = prediction_length,
        batch_size        = batch_size,
        test_size         = test_size,
        stride            = stride,
        normalization     = normalization,
        val_size=val_size
    )

    batch = next(iter(test_loader))
    print("gt_mask sample:", batch["gt_mask"][0, :, 0])

    for fname, varname in [("mean.npy", "mean"),("std.npy", "std"),("normalization.npy", "normalization")]:
        path = os.path.join(ckpt_dir, fname)
        if os.path.exists(path):
            val = np.load(path)[0]
            if varname == "mean":
                mean = float(val)
            elif varname == "std":
                std = float(val)
            else:
                normalization = str(val)
            print(f"Loaded {varname}={val} from checkpoint")

    model = CSDI_Forecasting(config, args.device, target_dim=1).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device, weights_only=False))
    print(f"Loaded: {args.ckpt}")

    samples, gt = evaluate_csdi(
        model, test_loader, num_of_samples,
        args.device, prediction_length,
        normalization, mean, std,
    )

    data             = np.load(args.input)
    positions        = data["positions"]
    time             = data["time"]
    train_test_split = int(data["train_test_split"])
    N                = len(samples)

    ci90_lower = np.percentile(samples,  5, axis=2)
    ci90_upper = np.percentile(samples, 95, axis=2)
    ci50_lower = np.percentile(samples, 25, axis=2)
    ci50_upper = np.percentile(samples, 75, axis=2)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        args.out,
        samples           = samples,
        ground_truth      = gt,
        ci90_lower        = ci90_lower,
        ci90_upper        = ci90_upper,
        ci50_lower        = ci50_lower,
        ci50_upper        = ci50_upper,
        full_trajectories = positions[:N],
        time              = time,
        time_train        = time[:train_test_split],
        time_test         = time[train_test_split: train_test_split + prediction_length],
        train_test_split  = train_test_split,
        prediction_length = prediction_length,
        num_of_samples    = num_of_samples,
    )
    print(f"Saved: {args.out}")

    assert not np.isnan(samples).any(),      "NaN in samples"
    assert not np.isinf(samples).any(),      "Inf in samples"
    assert np.all(ci90_lower <= ci90_upper), "CI90 ordering violated"
    print("Sanity checks passed TRUE")


if __name__ == "__main__":
    main()