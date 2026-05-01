import torch
import numpy as np
import os
import argparse
import yaml
from tqdm import tqdm

from TCN_master.TCN.ts_cnn.tstcn import TimeSeriesTCN

##only for the training set
def all_encode(model, config):
    """
    Build reference database by encoding all sliding windows
    from the training time period.

    train_end = train_test_split - H  (matches train_ratd.py exactly)
    """
    print("=" * 60)
    print("Phase 1: Encoding reference windows")
    print("=" * 60)

    L = config["data"]["L"]
    H = config["data"]["H"]
    stride = config["data"]["retrieval_stride"]
    device = config["train"]["device"]

    data = np.load(config["path"]["dataset_path"])
    positions = data["positions"]                    # (N_traj, T)
    train_test_split = int(data["train_test_split"])
    train_end = train_test_split - H                 # 400-100=300

    train_positions = positions[:, :train_end]             # (N, 300)
    N_traj, T = train_positions.shape

    print(f"train_test_split : {train_test_split}")
    print(f"train_end : {train_end}  (= train_test_split - H)")
    print(f"N_traj : {N_traj}")
    print(f"T (train steps)  : {T}")
    print(f"L={L}  H={H}  stride={stride}")

    max_start_global = T - L - H
    if max_start_global < 0:
        raise RuntimeError(
            f"Training data too short: T={T} < L+H={L+H}. "
            f"Reduce L/H or generate more data.")

    n_windows_per_traj = len(range(0, max_start_global + 1, stride))
    print(f"Windows per traj : {n_windows_per_traj}")
    print(f"Total windows : {N_traj * n_windows_per_traj}")

    hisvec_list = [] ##store embeddings of windoes
    future_list = [] ##store futures (in regaulr space) 

    model.eval()
    with torch.no_grad():
        for traj_idx in tqdm(range(N_traj), desc="Encoding"):
            seq = train_positions[traj_idx]
            max_start = len(seq) - L - H

            for start in range(0, max_start + 1, stride):
                ctx = seq[start : start + L]
                fut = seq[start + L : start + L + H]

                x_tensor = torch.from_numpy(ctx.copy()).float().unsqueeze(0).unsqueeze(1).to(device)  # (1,1,L)

                x_vec = model.encode(x_tensor)   
                hisvec_list.append(x_vec.cpu())
                future_list.append(fut)

    if not hisvec_list:
        raise RuntimeError("No windows encoded — check stride and data length.")

    hisvec_tensor = torch.cat(hisvec_list, dim=0)   
    futures_array = np.stack(future_list)           

    os.makedirs(os.path.dirname(config["path"]["vec_path"]), exist_ok=True)
    torch.save(hisvec_tensor.float(), config["path"]["vec_path"])
    np.save(config["path"]["futures_path"], futures_array)

    print(f"\nSaved {len(hisvec_list)} embeddings")
    print(f"  Embeddings : {hisvec_tensor.shape}")
    print(f"  Futures    : {futures_array.shape}")

    return hisvec_tensor


def all_retrieval(model, k, config):
    """
    k: number of references to create per embedding
    """
    print("=" * 60)
    print(f"Phase 2: Building retrieval indices (k={k})")
    print("=" * 60)

    L = config["data"]["L"]
    H = config["data"]["H"]
    stride = config["data"]["retrieval_stride"]
    device = config["train"]["device"]

    data = np.load(config["path"]["dataset_path"])
    positions = data["positions"]
    train_test_split = int(data["train_test_split"])
    train_end = train_test_split - H                 # same as all_encode
    val_start=config["train"]["val_start"]

    train_data = positions[:, :train_end]
    N_traj, T  = train_data.shape
    
    val_data = positions[:, val_start:train_test_split]
    test_data = positions[:, train_end:] 

    print(f"train_test_split : {train_test_split}")
    print(f"train_end : {train_end}")
    print(f"train_data : {train_data.shape}")

    # Load precomputed embeddings
    all_repr = torch.load(config["path"]["vec_path"])       # (N_refs, embed_dim)
    print(f"Reference bank : {all_repr.shape}")

    ##create mapping between embeddings and futures
    ##embedding are all valid L length windows in train/val/test
    ##futures are all from the train set
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    

    for split_name, split_data in splits.items():
        print(f"\nProcessing {split_name} split...")
        N_traj, T = split_data.shape

        windows = []
        for traj_idx in range(N_traj):
            seq = split_data[traj_idx]
            max_start = len(seq) - L - H
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, stride):
                windows.append((traj_idx, start))

        print(f"  Number of windows: {len(windows)}")

        references = []
        model.eval()
        with torch.no_grad():
            for traj_idx, start in tqdm(windows, desc=f"Building {split_name} indices"):
                ctx = split_data[traj_idx, start:start+L]
                x_tensor = torch.from_numpy(ctx.copy()).float().unsqueeze(0).unsqueeze(1).to(device)
                x_vec = model.encode(x_tensor).cpu()
                distances = torch.cdist(x_vec, all_repr, p=2)  # (1, N_refs)
                _, top_k = torch.topk(-distances, k, dim=1)   # (1, k)
                references.append(top_k.squeeze(0).int())             # (k,)

        references_tensor = torch.stack(references)  # (N_windows, k)
        save_path = config["path"]["ref_path"].replace('.pt', f'_{split_name}.pt')
        torch.save(references_tensor, save_path)
        print(f"  Saved {split_name} indices to {save_path}")

        # Sanity check: load back one to verify
        test_path = config["path"]["ref_path"].replace('.pt', '_train.pt')
        if os.path.exists(test_path):
            test_idx = torch.load(test_path)
            print(f"\nSanity: train indices shape = {test_idx.shape}")

        print(f"\nSaved retrieval indices: {references_tensor.shape}")
        print(f"  Expected : ({N_traj * len(range(0, T-L-H+1, stride))}, {k})")

        if split_name == 'train':
            train_indices = references_tensor

    futures_array = np.load(config["path"]["futures_path"])   # (N_refs, H)
    print(f"\n=== Sanity check ===")
    print(f"Train indices rows: {train_indices.shape[0]}")
    print(f"Reference futures rows: {futures_array.shape[0]}")
    assert train_indices.shape[0] == futures_array.shape[0], f"MISMATCH: train indices rows ({train_indices.shape[0]}) != futures rows ({futures_array.shape[0]})"
    print(" Sanity check passed: train indices count matches reference futures.")

    print("\nAll retrieval indices generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RATD Retrieval Builder")
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--type",   "-t", type=str, required=True,
                        choices=["encode", "retrieval"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Loading TCN encoder from {config['path']['encoder_path']}")
    model = TimeSeriesTCN(
        input_size   = config['model']['input_size'],
        output_size  = config['model']['output_size'],
        num_channels = config['model']['num_channels'],
        kernel_size  = config['model']['kernel_size'],
        dropout      = config['model']['dropout'],
    )

    if not os.path.exists(config['path']['encoder_path']):
        raise FileNotFoundError(
            f"Encoder weights not found: {config['path']['encoder_path']}")

    model.load_state_dict(torch.load(config['path']['encoder_path'], map_location='cpu'))
    model.to(config['train']['device'])
    model.eval()
    print(f"Model loaded  params={sum(p.numel() for p in model.parameters()):,}")

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.type == 'encode':
        all_encode(model, config)
    elif args.type == 'retrieval':
        all_retrieval(model, config['model']['k'], config)