#!/usr/bin/env python
"""
RATD Forecast Script
Test : positions_test[:, 800:]  (steps 800-999) — 9000 traj × 1 window = 9000
       obs=800-899 (context), pred=900-999 (target)
"""

import torch
import numpy as np
import os
import argparse
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys

ROOT         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "RATD"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))
sys.path.insert(0, PROJECT_ROOT)

from RATD.custom_model import RATD_Forecasting


class RATDTestDataset(Dataset):
    """
    One window per test trajectory:
        observed  : steps 800-899  (L=100)
        target    : steps 900-999  (H=100)
    positions shape passed in: (N_traj, 200) — already sliced to last L+H steps
    """
    def __init__(self, positions, references, indices, L, H, k):
        self.positions  = positions                              # (N_traj, L+H)
        self.references = torch.from_numpy(references).float()  # (N_refs,  H)
        self.indices    = indices                                # (N_traj,  k)
        self.L = L
        self.H = H
        self.k = k

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx):
        full = torch.from_numpy(self.positions[idx].copy()).float()  # (L+H,)

        observed_mask = torch.ones(self.L + self.H)
        gt_mask       = torch.cat([torch.ones(self.L), torch.zeros(self.H)])
        timepoints    = torch.arange(self.L + self.H).float()

        ref_futures = self.references[self.indices[idx]]   # (k, H)
        reference   = ref_futures.flatten().unsqueeze(-1)  # (k*H, 1)

        return {
            "observed_data": full.unsqueeze(0),            # (1, L+H)
            "observed_mask": observed_mask.unsqueeze(0),   # (1, L+H)
            "gt_mask"      : gt_mask.unsqueeze(0),         # (1, L+H)
            "timepoints"   : timepoints,                   # (L+H,)
            "reference"    : reference,                    # (k*H, 1)
        }



def forecast_ratd(config):
    device = torch.device(
        config['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    L = config['retrieval']['L']               # 100
    H = config['retrieval']['H']               # 100
    k = config['retrieval']['k']               # 3
    n_samples = config['forecast'].get('n_samples', 100)
    test_size = config['forecast'].get('test_size', None)  # None = all 9000

    data_test      = np.load(config['path']['test_data'])
    positions_test = data_test['positions']                  # (9000, 1000)

    test_positions = positions_test[:, -(L + H):]            # (9000, 200)

    if test_size is not None:
        test_positions = test_positions[:test_size]          # (test_size, 200)

    N_test = test_positions.shape[0]
    print(f"test trajectories : {N_test}")
    print(f"test window       : obs steps 800-899, pred steps 900-999")

    data_train = np.load(config['path']['train_data'])
    time       = data_train['time']                          # (1000,)

    ref_futures  = np.load(config['path']['futures_path'])   # (39000, 100)
    test_indices = torch.load(
        config['path']['ref_path'].replace('.pt', '_test.pt'))  # (9000, 3)

    if test_size is not None:
        test_indices = test_indices[:test_size]

    print(f"futures      : {ref_futures.shape}")
    print(f"test_indices : {test_indices.shape}")

    assert len(test_indices) == N_test, \
        f"index/window mismatch: {len(test_indices)} vs {N_test}"
    assert test_indices.max() < len(ref_futures), \
        f"index out of bounds: max={test_indices.max()} >= {len(ref_futures)}"

    test_ds = RATDTestDataset(
        test_positions, ref_futures, test_indices, L, H, k)
    test_loader = DataLoader(
        test_ds,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train'].get('num_workers', 2),
    )
    print(f"test samples : {len(test_ds)}")

    model = RATD_Forecasting(config, device, target_dim=1).to(device)
    ckpt  = os.path.join(config['train']['path'], 'best_model.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {ckpt}")

    all_samples = []
    all_gt      = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Forecasting"):
            batch = {bk: bv.to(device) for bk, bv in batch.items()}

            # samples      : (B, n_samples, 1, L+H)
            # observed_data: (B, 1, L+H)
            samples, observed_data, _, _, _ = model.evaluate(
                batch, n_samples=n_samples)

            all_samples.append(samples.cpu().numpy())
            all_gt.append(observed_data.cpu().numpy())

    samples_out = np.concatenate(all_samples, axis=0)    # (N, n_samples, 1, L+H)
    gt_out      = np.concatenate(all_gt,      axis=0)    # (N, 1, L+H)

    # squeeze feature dim, take only forecast horizon
    samples_out  = samples_out[:, :, 0, -H:]             # (N, n_samples, H)
    gt_out       = gt_out[:, 0, :]                       # (N, L+H)

    contexts     = gt_out[:, :L]                          # (N, L)
    gt_forecast  = gt_out[:, L:]                          # (N, H)

    samples_out  = samples_out.transpose(0, 2, 1)         # (N, H, n_samples)

    ci90_lower = np.percentile(samples_out, 5,  axis=2)   # (N, H)
    ci90_upper = np.percentile(samples_out, 95, axis=2)
    ci50_lower = np.percentile(samples_out, 25, axis=2)
    ci50_upper = np.percentile(samples_out, 75, axis=2)

    out_dir = os.path.dirname(config['forecast']['out_path'])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        config['forecast']['out_path'],
        samples           = samples_out,              # (N, H, n_samples)
        ground_truth      = gt_forecast,               # (N, H)
        contexts          = contexts,                  # (N, L)
        ci90_lower        = ci90_lower,                # (N, H)
        ci90_upper        = ci90_upper,
        ci50_lower        = ci50_lower,
        ci50_upper        = ci50_upper,
        full_trajectories = positions_test[:N_test],   # (N, 1000)
        time              = time,                      # (1000,)
        train_test_split  = 900,   # forecast starts at step 900 of time array
        prediction_length = H,
        num_of_samples    = n_samples,
        L                 = L,
        H                 = H,
    )
    print(f"Saved: {config['forecast']['out_path']}")

    assert not np.isnan(samples_out).any(), "NaN in samples"
    assert not np.isinf(samples_out).any(), "Inf in samples"
    assert np.all(ci90_lower <= ci90_upper), "CI90 ordering violated"
    print("Sanity checks passed ✓")

    return samples_out, gt_forecast


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    forecast_ratd(config)