#!/usr/bin/env python
"""
RATD Forecasting Script
Usage: conda run -n unified_tsf python forecast_ratd.py --config ratd_config.yaml
"""

import torch
import numpy as np
import os
import argparse
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from custom_model import RATD_Forecasting


class RATDTestDataset(Dataset):
    def __init__(self, positions, windows, references, indices, L, H, k):
        self.positions = positions
        self.windows = windows
        self.references = torch.from_numpy(references).float()
        self.indices = indices
        self.L = L
        self.H = H
        self.k = k

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        traj, start = self.windows[idx]
        
        full = self.positions[traj, start:start+self.L+self.H]
        full = torch.from_numpy(full).float()
        
        observed_mask = torch.ones(self.L+self.H)
        gt_mask = torch.cat([torch.ones(self.L), torch.zeros(self.H)])
        timepoints = torch.arange(self.L+self.H).float()
        
        ref_indices = self.indices[idx]
        ref_futures = self.references[ref_indices]
        ref_flat = ref_futures.flatten()
        reference = ref_flat.unsqueeze(-1)
        
        return {
            "observed_data": full.unsqueeze(0),
            "observed_mask": observed_mask.unsqueeze(0),
            "gt_mask": gt_mask.unsqueeze(0),
            "timepoints": timepoints,
            "reference": reference
        }


def get_windows(positions, L, H, stride):
    windows = []
    for traj_idx in range(positions.shape[0]):
        seq = positions[traj_idx]
        max_start = len(seq) - L - H
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride):
            windows.append((traj_idx, start))
    return windows

def forecast_ratd(config):
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = np.load(config['path']['dataset_path'])
    positions = data['positions']
    time = data.get('time', np.arange(positions.shape[1]))  # if time exists
    train_test_split = int(data['train_test_split'])
    val_start = int(config['train']['val_start'])
    
    L = config['retrieval']['L']
    H = config['retrieval']['H']
    stride = config['retrieval']['stride']
    k = config['retrieval']['k']
    n_samples = config['forecast'].get('n_samples', 20)
    
    # Test data: steps 300-500 (for predicting 400-500)
    test_size=config["forecast"]["test_size"]
    test_data = positions[:test_size, train_test_split - L:]  # Steps 300-500
    print(f"Test data shape: {test_data.shape}")
    
    # Load retrieval outputs
    references = np.load(config['path']['futures_path'])
    indices = torch.load(config['path']['ref_path'])
    
    # Create test windows
    test_windows = get_windows(test_data, L, H, stride)
    
    # Create mapping from window to index
    train_data = positions[:test_size, :val_start]
    all_windows = get_windows(train_data, L, H, stride)
    window_to_idx = {win: i for i, win in enumerate(all_windows)}
    
    test_valid_windows = [w for w in test_windows if w in window_to_idx]
    test_indices = torch.stack([indices[window_to_idx[w]] for w in test_valid_windows])
    
    # Create dataset
    test_ds = RATDTestDataset(test_data, test_valid_windows, references, test_indices, L, H, k)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # Load model
    model = RATD_Forecasting(config, device, target_dim=1).to(device)
    checkpoint_path = os.path.join(config['train']['path'], 'best_model.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Generate predictions
    all_samples = []      # full predicted trajectories (context + future)
    all_gt = []           # ground truth full trajectories
    all_contexts = []     # context part
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Forecasting"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            samples, observed_data, _, _, _ = model.evaluate(batch, n_samples=n_samples)
            
            # samples: (B, n_samples, K, L+H)
            # observed_data: (B, K, L+H)
            
            all_samples.append(samples.cpu().numpy())
            all_gt.append(observed_data.cpu().numpy())
            all_contexts.append(observed_data[:, :, :L].cpu().numpy())
    
    samples_out = np.concatenate(all_samples, axis=0)           # (N, n_samples, 1, L+H)
    gt_out = np.concatenate(all_gt, axis=0)                    # (N, 1, L+H)
    contexts = np.concatenate(all_contexts, axis=0)            # (N, 1, L)
    
    samples_out = samples_out[:, :, 0, :]                       # (N, n_samples, L+H)
    samples=samples_out[:,:,-H:]
    
    print(samples.shape)
    #raise RuntimeError("asda")
    gt_out = gt_out[:, 0, :]                                   # (N, L+H)
    contexts = contexts[:, 0, :]                               # (N, L)
    
    N = samples.shape[0]
    
    ci90_lower = np.percentile(samples, 5, axis=1)          # (N, L+H)
    ci90_upper = np.percentile(samples, 95, axis=1)
    ci50_lower = np.percentile(samples, 25, axis=1)
    ci50_upper = np.percentile(samples, 75, axis=1)
    
    # Save results
    out_dir = os.path.dirname(config['forecast']['out_path'])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    np.savez(
        config['forecast']['out_path'],
        samples=samples,
        ground_truth=gt_out,
        ci90_lower=ci90_lower,
        ci90_upper=ci90_upper,
        ci50_lower=ci50_lower,
        ci50_upper=ci50_upper,
        contexts=contexts,
        full_trajectories=positions[:N],
        time=time,
        time_train=time[:train_test_split],
        time_test=time[train_test_split:],
        train_test_split=train_test_split,
        prediction_length=H,
        num_of_samples=n_samples,
        L=L,
        H=H,
    )
    print(f"Saved: {config['forecast']['out_path']}")
    
    assert not np.isnan(samples).any(), "NaN in samples"
    assert not np.isinf(samples).any(), "Inf in samples"
    assert np.all(ci90_lower <= ci90_upper), "CI90 ordering violated"
    print("Sanity checks passed ✓")
    
    return samples, gt_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    forecast_ratd(config)