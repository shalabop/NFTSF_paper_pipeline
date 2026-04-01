#!/usr/bin/env python
"""
RATD Training Script
Usage: python train_ratd.py --config ratd_config.yaml
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

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "architectures", "RATD"))
sys.path.insert(0, ROOT)

from custom_model import RATD_Forecasting


class RATDDataset(Dataset):
    def __init__(self, positions, time, windows, references, indices, L, H, k):
        self.positions  = positions
        self.time       = time
        self.windows    = windows
        self.references = torch.from_numpy(references).float()  # (N_refs, H)
        self.indices    = indices                                # (N_windows, k)
        self.L = L ##observed past length
        self.H = H #predicted future length
        self.k = k # refernces number

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        traj, start = self.windows[idx]

        full = torch.from_numpy(self.positions[traj, start:start + self.L + self.H].copy()).float()
        observed_mask = torch.ones(self.L + self.H) ##all ones for context window + future target
        gt_mask = torch.cat([torch.ones(self.L), torch.zeros(self.H)]) ## L number of 1s and H number of 0s
        
        ###!!!!!!!!!!!!!!!!!!!!!!!!
        #timepoints = torch.arange(self.L + self.H).float() ## 0,1,2 ... H+L
        timepoints=self.time[start:start + self.L + self.H].copy() ## actual time points for the window

        ref_futures = self.references[self.indices[idx]]  # (top k, H)
        
        #print(ref_futures.shape)
        reference = ref_futures.flatten().unsqueeze(-1) # could be wrong?
        #print(reference.shape)
        #reference=ref_futures
        #raise RuntimeError("asdasd")
        return {
            "observed_data" : full.unsqueeze(0),           # (1, L+H) Full window
            "observed_mask" : observed_mask.unsqueeze(0),  # (1, L+H) all 1s
            "gt_mask"       : gt_mask.unsqueeze(0),        # (1, L+H) 0-L=0s and L+1-H=1s
            "timepoints"    : timepoints,                   # (L+H,) 
            "reference"     : reference,                    # (k*H, 1)
        }


def get_windows(data, L, H, stride):
    """Sliding windows over (N_traj, T) array."""
    windows = []
    for traj_idx in range(data.shape[0]):
        max_start = data.shape[1] - L - H
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride):
            windows.append((traj_idx, start))
    return windows


def train_ratd(config):
    device = torch.device(
        config['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data = np.load(config['path']['dataset_path'])
    time= data["time"]                           # (N_traj, T)
    positions = data['positions']                   # (N_traj, T)
    train_test_split = int(data['train_test_split'])       # 400

    L = config['retrieval']['L']                     # 100
    H = config['retrieval']['H']                     # 100
    stride = config['train']['stride']
    k = config['retrieval']['k']
    val_start = int(config['train']['val_start'])          # 200

    # Matches your other models exactly:
    #   train : t=0..299   (train_test_split - H = 300)
    #   val   : t=200..399 (val_start to train_test_split)
    train_end  = train_test_split - H                      # 300
    train_data = positions[:, :train_end]                  # (N, 300)
    val_data   = positions[:, val_start:train_test_split]  # (N, 200) = L+H

    print(f"train_data : {train_data.shape}  (t=0..{train_end-1})")
    print(f"val_data   : {val_data.shape}  (t={val_start}..{train_test_split-1})")

    # Val references — separate file, one per val window
    '''val_refs_path = config['path'].get('val_futures_path', None)
    val_idx_path  = config['path'].get('val_ref_path',     None)

    if val_refs_path and os.path.exists(val_refs_path):
        val_refs = np.load(val_refs_path)
        val_indices = torch.load(val_idx_path)
        print(f"Loaded val references: {val_refs.shape}")
    else:
        print("WARNING: no val references found — using train references for val")
        val_refs = train_refs
        val_indices = train_indices[:len(positions)]  # one per trajectory
        
        print("=="*30)'''
        
    ref_futures = np.load(config['path']['futures_path'])   # shape (N_refs, H)

    train_indices = torch.load(config['path']['ref_path'].replace('.pt', '_train.pt'))  # (N_train_windows, k)
    val_indices   = torch.load(config['path']['ref_path'].replace('.pt', '_val.pt'))    # (N_val_windows, k)

    print(f"train_indices shape: {train_indices.shape}")
    print(f"val_indices shape: {val_indices.shape}")

    print(f"train_refs    : {ref_futures.shape}")
    print(f"train_indices : {train_indices.shape}")

    train_windows = get_windows(train_data, L, H, stride)
    #val_windows   = [(traj, 0) for traj in range(val_data.shape[0])]
    val_windows = get_windows(val_data, L, H, stride)
    
    print(f"train_windows : {len(train_windows)}")
    print(f"val_windows   : {len(val_windows)}")

    assert len(train_windows) == len(train_indices), f"train window/index mismatch: {len(train_windows)} vs {len(train_indices)}"

    train_ds = RATDDataset(train_data, time, train_windows,ref_futures, train_indices, L, H, k)
    val_ds = RATDDataset(val_data, time,   val_windows,ref_futures,   val_indices,   L, H, k)
    
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'],shuffle=True,  num_workers=config['train'].get('num_workers', 2))
    val_loader = DataLoader(val_ds,   batch_size=config['train']['batch_size'],shuffle=False, num_workers=config['train'].get('num_workers', 2))

    print(f"train samples : {len(train_ds)}")
    print(f"val samples   : {len(val_ds)}")

    model = RATD_Forecasting(config, device, target_dim=1).to(device)
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['train']['milestones'],
        gamma=config['train']['gamma'])

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

        for batch in tqdm(train_loader,desc=f"Epoch {epoch+1} Train", leave=False):
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            loss = model(batch, is_train=1) ##1 for training

            if torch.isnan(loss):
                #dprint(f"NaN loss at epoch {epoch+1}, skipping batch")
                raise RuntimeError(f"NaN loss in barch {epoch+1}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip'])
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            

        avg_train = total_loss / max(n_batches, 1)
        train_losses.append(avg_train)
        print(f'Epoch {epoch+1} loss: {avg_train}')
        

        avg_val = None
        if (epoch + 1) % config['train']['val_every'] == 0:

            model.eval()
            val_loss   = 0.0
            n_val      = 0

            with torch.no_grad():
                for batch in tqdm(val_loader,desc=f"Epoch {epoch+1} Val", leave=False):
                    batch = {key: val.to(device) for key, val in batch.items()}
                    val_loss += model(batch, is_train=0).item()
                    n_val += 1

            avg_val = val_loss / max(n_val, 1)
            val_epochs.append(epoch)
            val_losses.append(avg_val)
            
            val_str = f"val={avg_val:.6f}" if avg_val is not None else "val=--"
            print(f"[{epoch+1:3d}/{config['train']['epochs']}]  "
            f"train={avg_train:.6f}  {val_str}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(),os.path.join(config['train']['path'],'best_model.pth'))
                print(f" best saved  epoch={epoch+1}"f"val={avg_val:.6f}")

        scheduler.step()

        #if (epoch + 1) % config['train']['print_every'] == 0 or epoch == 0:
        

    np.savez(os.path.join(config['train']['path'],'training_curves.npz'),train_losses=train_losses,
            val_epochs=val_epochs,val_losses=val_losses)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label='Train')
    ax.plot(val_epochs, val_losses, 'o-', label='Val')   # plot at correct epochs
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('RATD Training')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['train']['path'],'training_curves.png'), dpi=150)
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