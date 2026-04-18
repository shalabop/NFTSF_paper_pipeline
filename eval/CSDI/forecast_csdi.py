# forecast_csdi.py
import argparse
import os
import sys
import yaml
import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))                     # train/CSDI
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))        # project root

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "CSDI"))   # for diff_models
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))           # for CSDI package
sys.path.insert(0, PROJECT_ROOT)                                          # for top-level modules if any

from CSDI.main_model import CSDI_Forecasting
from CSDI.dataset_md import get_dataloader_md
from CSDI.utils import train

def evaluate_csdi(model, test_loader, n_samples, device,
                  prediction_length):
    model.eval()
    all_samples = []
    all_gt      = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Forecasting"):  
            
            (observed_data, observed_mask, observed_tp,
             gt_mask, _, _, _) = model.process_data(batch)


            cond_mask = gt_mask
            side_info = model.get_side_info(observed_tp, cond_mask)
            samples = model.impute(observed_data, cond_mask, side_info, n_samples)

            samples = samples[:, :, 0, -prediction_length:]  
            samples = samples.permute(0, 2, 1)               
            gt = observed_data[:, 0, -prediction_length:].clone()  

            all_samples.append(samples.cpu().numpy())
            all_gt.append(gt.cpu().numpy())

    samples_out = np.concatenate(all_samples, axis=0)  
    gt_out      = np.concatenate(all_gt,      axis=0)  

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
    val_size         = config["train"]["val_size"]

    print(f"prediction_length : {prediction_length}")
    print(f"context_length    : {context_length}")
    print(f"num_of_samples    : {num_of_samples}")

    ckpt_dir = os.path.dirname(args.ckpt)

    test_loader = get_dataloader_md(
        flag="test",
        npz_path          = args.input,
        context_length    = context_length,
        prediction_length = prediction_length,
        batch_size        = batch_size,
        test_size         = test_size,
        stride            = stride,
        val_size=val_size
    )

    batch = next(iter(test_loader))
    print("gt_mask sample:", batch["gt_mask"][0, :, 0])

    model = CSDI_Forecasting(config, args.device, target_dim=1).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device, weights_only=False))
    print(f"Loaded: {args.ckpt}")

    samples, gt = evaluate_csdi(
        model, test_loader, num_of_samples,
        args.device, prediction_length,
    )

    data = np.load(args.input)
    positions = data["positions"]
    time = data["time"]
    train_test_split = int(data["train_test_split"])
    N = len(samples)

    ci90_lower = np.percentile(samples,  5, axis=2)
    ci90_upper = np.percentile(samples, 95, axis=2)
    ci50_lower = np.percentile(samples, 25, axis=2)
    ci50_upper = np.percentile(samples, 75, axis=2)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    #assert samples.shape == (N, prediction_length, 100)
    print(samples.shape)

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
        context_length    = context_length,
    )
    print(f"Saved: {args.out}")

    assert not np.isnan(samples).any(),      "NaN in samples"
    assert not np.isinf(samples).any(),      "Inf in samples"
    assert np.all(ci90_lower <= ci90_upper), "CI90 ordering violated"
    print("Sanity checks passed TRUE")


if __name__ == "__main__":
    main()