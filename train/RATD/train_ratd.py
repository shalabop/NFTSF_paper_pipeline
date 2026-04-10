#!/usr/bin/env python
"""
RATD Training Script
Train  : positions_train[:, :800]   (steps 0-799)   — 3000 traj × 13 windows = 39000
Val    : positions_train[:, 800:]   (steps 800-999) — 3000 traj × 1  window  =  3000
         obs=800-899 (context), pred=900-999 (target)
"""

import torch
import torch.optim as optim
import numpy as np
import os
import argparse
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

ROOT         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "RATD"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))
sys.path.insert(0, PROJECT_ROOT)

from RATD.custom_model import RATD_Forecasting

class RATDDataset(Dataset):
    def __init__(self, positions, time, windows, references, indices, L, H, k,
                 time_offset=0):
        self.positions  = positions
        self.time       = time
        self.windows    = windows
        self.references = torch.from_numpy(references).float()
        self.indices    = indices
        self.L          = L
        self.H          = H
        self.k          = k
        self.time_offset = time_offset       
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        traj, start = self.windows[idx]

        full = torch.from_numpy(
            self.positions[traj, start:start + self.L + self.H].copy()
        ).float()

        observed_mask = torch.ones(self.L + self.H)
        gt_mask       = torch.cat([torch.ones(self.L), torch.zeros(self.H)])

        timepoints = torch.from_numpy(
            self.time[self.time_offset + start :
                      self.time_offset + start + self.L + self.H].copy()
        ).float()

        ref_futures = self.references[self.indices[idx]]   # (k, H)
        reference   = ref_futures.flatten().unsqueeze(-1)  # (k*H, 1)

        return {
            "observed_data": full.unsqueeze(0),
            "observed_mask": observed_mask.unsqueeze(0),
            "gt_mask"      : gt_mask.unsqueeze(0),
            "timepoints"   : timepoints,
            "reference"    : reference,
        }


def get_windows(positions, L, H, stride):
    """Sliding windows over (N_traj, T) array."""
    windows = []
    for traj_idx in range(positions.shape[0]):
        max_start = positions.shape[1] - L - H
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride):
            windows.append((traj_idx, start))
    return windows



def train_ratd(config):
    device = torch.device(
        config['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_train = np.load(config['path']['train_data'], allow_pickle=True)
    positions  = data_train['positions']   # (3000, 1000)
    time       = data_train['time']        # (1000,)

    L = config['retrieval']['L']        # 100
    H = config['retrieval']['H']        # 100
    stride = config['train']['stride']       # 50
    #stride = config['data']['retrieval_stride'] #since they have to be the same
    k = config['retrieval']['k']        # 3
    val_start = int(config['train']['val_start'])  # 800

    train_positions = positions[:, :val_start]   # (3000, 800)
    val_positions   = positions[:, val_start:]   # (3000, 200)

    print(f"train_positions : {train_positions.shape}  (steps 0-{val_start-1})")
    print(f"val_positions   : {val_positions.shape}  (steps {val_start}-999)")

    ref_futures   = np.load(config['path']['futures_path'])    # (39000, 100)
    train_indices = torch.load(
        config['path']['ref_path'].replace('.pt', '_train.pt'))  # (39000, 3)
    val_indices   = torch.load(
        config['path']['ref_path'].replace('.pt', '_val.pt'))    # (3000,  3)

    print(f"futures       : {ref_futures.shape}")
    print(f"train_indices : {train_indices.shape}")
    print(f"val_indices   : {val_indices.shape}")

    train_windows = get_windows(train_positions, L, H, stride)
    val_windows   = get_windows(val_positions,   L, H, stride)

    print(f"stride used : {stride}")
    print(f"train_windows : {len(train_windows)}")
    print(f"val_windows : {len(val_windows)}")
    print(f"train_indices : {len(train_indices)}")
    print(f"val_indices : {len(val_indices)}")

    assert len(train_windows) == len(train_indices), \
        f"train window/index mismatch: {len(train_windows)} vs {len(train_indices)}. " \
        f"Did build_retrieval.py use the same stride ({stride})?"
    assert len(val_windows) == len(val_indices), \
        f"val window/index mismatch: {len(val_windows)} vs {len(val_indices)}. " \
        f"Did build_retrieval.py use the same stride ({stride})?"

    train_ds = RATDDataset(train_positions, time, train_windows, ref_futures, train_indices, L, H, k, time_offset=0)
    val_ds = RATDDataset(val_positions, time, val_windows, ref_futures, val_indices, L, H, k, time_offset=val_start)

    train_loader = DataLoader(
        train_ds, batch_size=config['train']['batch_size'],
        shuffle=True,  num_workers=config['train'].get('num_workers', 2))
    val_loader   = DataLoader(
        val_ds,   batch_size=config['train']['batch_size'],
        shuffle=False, num_workers=config['train'].get('num_workers', 2))

    print(f"train samples : {len(train_ds)}")
    print(f"val   samples : {len(val_ds)}")

    model     = RATD_Forecasting(config, device, target_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=config['train']['learning_rate'])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['train']['milestones'],
        gamma=config['train']['gamma'])

    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    os.makedirs(config['train']['path'], exist_ok=True)

    best_val_loss = float('inf')
    train_losses  = []
    val_epochs    = []
    val_losses    = []

    print(f"\n{'='*60}")
    print(f"Training RATD for {config['train']['epochs']} epochs")
    print(f"{'='*60}")

    for epoch in range(config['train']['epochs']):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            loss = model(batch, is_train=1)

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at epoch {epoch+1}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['train']['grad_clip'])
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_train = total_loss / max(n_batches, 1)
        train_losses.append(avg_train)
        scheduler.step()

        if (epoch + 1) % config['train']['val_every'] == 0:
            model.eval()
            val_loss = 0.0
            n_val    = 0

            with torch.no_grad():
                for batch in tqdm(val_loader,
                                  desc=f"Epoch {epoch+1} Val", leave=False):
                    batch = {key: val.to(device) for key, val in batch.items()}
                    val_loss += model(batch, is_train=0).item()
                    n_val    += 1

            avg_val = val_loss / max(n_val, 1)
            val_epochs.append(epoch + 1)
            val_losses.append(avg_val)

            print(f"[{epoch+1:3d}/{config['train']['epochs']}]  "
                  f"train={avg_train:.6f}  val={avg_val:.6f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(),
                           os.path.join(config['train']['path'], 'best_model.pth'))
                print(f"  best saved  (val={avg_val:.6f})")
        else:
            print(f"[{epoch+1:3d}/{config['train']['epochs']}]  "
                  f"train={avg_train:.6f}")

    np.savez(
        os.path.join(config['train']['path'], 'training_curves.npz'),
        train_losses = np.array(train_losses),
        val_epochs   = np.array(val_epochs),
        val_losses   = np.array(val_losses),
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label='Train')
    if val_losses:
        ax.plot(val_epochs, val_losses, 'o-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('RATD Training'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['train']['path'], 'training_curves.png'),
                dpi=150)
    plt.close()

    print(f"\nBest val loss : {best_val_loss:.6f}")
    print(f"Saved         : {config['train']['path']}/best_model.pth")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_ratd(config)