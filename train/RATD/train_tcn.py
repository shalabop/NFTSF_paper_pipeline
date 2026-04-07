"""
TCN Training for Double-Well Data
Matches original training logic with config file support
"""

import sys

import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))                     # train/CSDI
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))        # project root

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "RATD"))   # for diff_models
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))           # for CSDI package
sys.path.insert(0, PROJECT_ROOT)  

# import existing modules
from TCN_master.TCN.ts_cnn.tstcn import TimeSeriesTCN


def train_tcn(config):    
    # Device
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"\nLoading data from {config['data']['train_data']}")
    
    train_data = np.load(config['data']['train_data'])
    #test_data = np.load(config['data']['test_data'])
    positions_train = train_data['positions']  # (N_traj, T)
    #positions_test = test_data['positions']  # (N_traj, T)
    #train_test_split = int(train_data['train_test_split'])
    #train_test_split=int(config['data']['train_test_split'])
    val_start = int(config["train"]["val_start"]) 
    
    lockback_L = config['data']['L']
    pred_steps = config['data']['H']
    
    # Normalize based on training data
    #train_end = train_test_split - pred_steps
    #train_data = positions[:, :train_end]
    #mean = train_data.mean()
    #std = train_data.std()
    
    #print(f"Normalization: mean={mean:.4f}, std={std:.4f}")
    
    #positions_norm = ((positions - mean) / std).astype(np.float32)
    
    train_positions = positions_train[:, :val_start]         # (N, 300)
    val_positions   = positions_train[:, val_start:]
    #test_positions  = positions_test[:, -pred_steps-lockback_L:]
    
    print(f"Train shape: {train_positions.shape}")
    print(f"Val shape: {val_positions.shape}")
    
    class BatchDataset(torch.utils.data.Dataset):
        """Simple dataset that returns full sequences for batch processing"""
        def __init__(self, positions):
            self.positions = positions
            
        def __len__(self):
            return self.positions.shape[0]  
            
        def __getitem__(self, idx):
            return self.positions[idx]  #
    
    train_dataset = BatchDataset(train_positions)
    val_dataset = BatchDataset(val_positions)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train'].get('num_workers', 1),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train'].get('num_workers', 1)
    )
    

    model = TimeSeriesTCN(
        input_size=config['model']['input_size'],
        output_size=config['model']['output_size'],
        num_channels=config['model']['num_channels'],
        kernel_size=config['model']['kernel_size'],
        dropout=config['model']['dropout']
    )
    
    
    
    model=model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    learning_rate = config['train']['learning_rate']
    epochs = config['train']['epochs']
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=[int(0.6*epochs), int(0.85*epochs)], gamma=0.1)
    criterion = nn.MSELoss()
    
    best_val = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*55}")
    print(f"Training TCN encoder for {epochs} epochs on {device}")
    print(f"{'='*55}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        
        for batch in train_loader:
            #batch is shaped (B,T)
            B = batch.shape[0] #batch size == 64
            T = batch.shape[1] #number of trainin steps
            max_t = T - lockback_L - pred_steps
            
            if max_t <= 0:
                continue
            
            starts = np.random.randint(0, max_t, size=B)
            
            x_list, y_list = [], []
            for i, s in enumerate(starts):
                x_list.append(batch[i, s:s + lockback_L])
                y_list.append(batch[i, s + lockback_L:s + lockback_L + pred_steps])
            
            x_tensor = torch.stack(x_list).unsqueeze(1).to(device)  # (B, 1, L) ##past
            y_tensor = torch.stack(y_list).to(device)                # (B, H) ##future
            
            assert x_tensor.shape == (B, 1, lockback_L), f"x shape: {x_tensor.shape}"
            assert y_tensor.shape == (B, pred_steps), f"y shape: {y_tensor.shape}"
            
            optimizer.zero_grad()
            output = model(x_tensor)  # (B, H) #taks the past windoes and returns future window 
            
            assert output.shape == y_tensor.shape, f"output {output.shape} != target {y_tensor.shape}"
            
            loss = criterion(output, y_tensor)
            loss.backward()
            
            if config['train'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip'])
            
            optimizer.step()
            
            train_loss += loss.item()
            n_train_batches += 1
        
        train_loss /= max(n_train_batches, 1)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # batch: (B, val_T) where val_T = tts - val_start = 200
                B = batch.shape[0]
                T = batch.shape[1]

                # Fixed single window: last L steps as context, next H as target
                # val data is already positions[:, val_start:tts]
                # so context = first L steps, target = last H steps
                if T < lockback_L + pred_steps:
                    continue   
                
                x_tensor = batch[:, :lockback_L].unsqueeze(1).to(device)   # (B, 1, L)
                y_tensor = batch[:, lockback_L:lockback_L+pred_steps].to(device)              # (B, H)

                val_loss      += criterion(model(x_tensor), y_tensor).item()
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)
        if val_loss==0:
            raise RuntimeError("Validation loss == 0! check data leakage")
        val_losses.append(val_loss)
        
        scheduler.step()
        
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save(model.state_dict(), config['path']['encoder_path'])
        
        if (epoch + 1) % config['train']['print_every'] == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}]  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}"
                  + ("  ← best" if is_best else ""))
            
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('TCN Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('tcn_training_curves.png', dpi=150)
    print(f"\nBest val loss : {best_val:.5f}")
    print(f"Saved : {config['path']['encoder_path']}")
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TCN for Double-Well')
    parser.add_argument('--config', "-c", type=str, required=True, help='Config file path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    os.makedirs(os.path.dirname(config['path']['encoder_path']), exist_ok=True)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    train_tcn(config)