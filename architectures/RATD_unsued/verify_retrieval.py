"""
Retrieval Verification Script for Double-Well Data
Plots retrieved contexts and futures to verify retrieval quality
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import yaml
import os
from pathlib import Path

# Import your model
from TCN_master.TCN.ts_cnn.tstcn import TimeSeriesTCN


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_retrieval_database(model, positions_norm, L, H, stride, device):
    """
    Build memory bank from all training windows
    
    Returns:
        memory_contexts: (N_windows, L) - context windows
        memory_futures: (N_windows, H) - future segments
        memory_embeddings: (N_windows, embed_dim) - L2 normalized embeddings
    """
    print("Building memory bank from training data...")
    
    memory_contexts = []
    memory_futures = []
    memory_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for traj_idx in range(positions_norm.shape[0]):
            seq = positions_norm[traj_idx]
            max_start = len(seq) - L - H
            
            for start in range(0, max_start + 1, stride):
                ctx = seq[start:start + L]
                fut = seq[start + L:start + L + H]
                
                # Encode context
                ctx_tensor = torch.tensor(ctx).unsqueeze(0).unsqueeze(1).to(device)
                emb = model.encode(ctx_tensor).cpu()
                
                memory_contexts.append(ctx)
                memory_futures.append(fut)
                memory_embeddings.append(emb.squeeze())
    
    memory_embeddings = torch.stack(memory_embeddings)  # (N_windows, embed_dim)
    memory_contexts = np.stack(memory_contexts)
    memory_futures = np.stack(memory_futures)
    
    print(f"Memory bank shape: {memory_embeddings.shape}")
    print(f"Memory contexts shape: {memory_contexts.shape}")
    print(f"Memory futures shape: {memory_futures.shape}")
    
    return memory_contexts, memory_futures, memory_embeddings


def retrieve_neighbors(query_ctx, memory_contexts, memory_futures, memory_embeddings, 
                       model, device, k=5):
    """
    Retrieve top-k nearest neighbors for a query context
    
    Returns:
        top_idx: indices of top-k neighbors
        top_vals: distances to top-k neighbors
        query_fut: true future of query (if provided, else None)
    """
    with torch.no_grad():
        # Encode query
        q_tensor = torch.tensor(query_ctx).unsqueeze(0).unsqueeze(1).to(device)
        query_vec = model.encode(q_tensor).cpu()
        
        # Compute distances
        distances = torch.norm(memory_embeddings - query_vec, dim=1)
        
        # Get top-k
        top_vals, top_idx = torch.topk(distances, min(k, len(distances)), largest=False)
    
    return top_idx.numpy(), top_vals.numpy()


def verify_retrieval(config):
    """Main verification function"""
    
    # Device
    device = torch.device(config['retrieval']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parameters
    L = config['retrieval']['L']
    H = config['retrieval']['H']
    stride = config['retrieval'].get('database_stride', 50)
    n_neighbors = config['retrieval'].get('k', 5)
    n_queries = config['retrieval'].get('n_queries', 50)
    
    # Load data
    print(f"\nLoading data from {config['data']['dataset_path']}")
    data = np.load(config['data']['dataset_path'])
    positions = data['positions']  # (N_traj, T)
    train_test_split = int(data['train_test_split'])
    
    # Normalize based on training data
    train_end = train_test_split - H
    train_data = positions[:, :train_end]
    mean = train_data.mean()
    std = train_data.std()
    
    positions_norm = ((positions - mean) / std).astype(np.float32)
    
    print(f"Normalization: mean={mean:.4f}, std={std:.4f}")
    print(f"Data shape: {positions.shape}")
    
    # Split data
    train_positions = positions_norm[:, :train_end]
    test_start = train_end
    test_end = min(positions.shape[1], train_test_split + H)
    test_positions = positions_norm[:, test_start:test_end]
    
    print(f"Train shape: {train_positions.shape}")
    print(f"Test shape: {test_positions.shape}")
    
    # Load model
    print(f"\nLoading model from {config['retrieval']['model_path']}")
    model = TimeSeriesTCN(
        input_size=config['model']['input_size'],
        output_size=config['model']['output_size'],
        num_channels=config['model']['num_channels'],
        kernel_size=config['model']['kernel_size'],
        dropout=config['model']['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(config['retrieval']['model_path'], map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # Build retrieval database from training data
    memory_contexts, memory_futures, memory_embeddings = build_retrieval_database(
        model, train_positions, L, H, stride, device
    )
    
    # Find valid test sequences (long enough)
    min_len = L + H
    valid_test_indices = [i for i in range(test_positions.shape[0]) 
                          if len(test_positions[i]) >= min_len]
    
    if not valid_test_indices:
        raise ValueError(f"No test sequence has at least {min_len} steps. "
                         f"Check your test set length or reduce L/H.")
    
    print(f"\nValid test sequences: {len(valid_test_indices)}")
    
    # Random query selection
    np.random.seed(42)  # For reproducibility
    query_idx = np.random.choice(valid_test_indices)
    query_seq = test_positions[query_idx]
    
    # Get query context and future
    query_ctx = query_seq[:L]
    query_fut = query_seq[L:L+H]
    
    print(f"\nQuery #{query_idx}")
    print(f"  context end: {query_ctx[-1]:+.3f}")
    print(f"  true future mean: {query_fut.mean():+.3f}")
    
    # Retrieve neighbors
    top_idx, top_vals = retrieve_neighbors(
        query_ctx, memory_contexts, memory_futures, memory_embeddings,
        model, device, k=n_neighbors
    )
    
    # Print results
    print(f"\nTop-{n_neighbors} retrieved neighbors:")
    print(f"  {'rank':<5} {'dist':<8} {'ctx_end':<10} {'fut_mean':<10} {'match'}")
    print(f"  {'-'*50}")
    
    for rank, (idx, dist) in enumerate(zip(top_idx, top_vals), 1):
        ctx_end = memory_contexts[idx, -1]
        fut_mean = memory_futures[idx].mean()
        match = "✓" if np.sign(query_ctx[-1]) == np.sign(fut_mean) else "✗"
        print(f"  {rank:<5} {dist:<8.4f} {ctx_end:<+10.3f} {fut_mean:<+10.3f} {match}")
    
    # Aggregate statistics over multiple queries
    print(f"\nAggregate check over {n_queries} random queries...")
    agreements = []
    
    for _ in range(min(n_queries, len(valid_test_indices))):
        qi = np.random.choice(valid_test_indices)
        ctx = test_positions[qi, :L]
        
        top_idx_local, _ = retrieve_neighbors(
            ctx, memory_contexts, memory_futures, memory_embeddings,
            model, device, k=n_neighbors
        )
        
        agree = np.mean([
            np.sign(ctx[-1]) == np.sign(memory_futures[idx].mean())
            for idx in top_idx_local
        ])
        agreements.append(agree)
    
    mean_agree = np.mean(agreements)
    print(f"Mean well agreement: {mean_agree:.3f}")
    print(f"  > 0.8 = excellent retrieval quality ✓")
    print(f"  > 0.6 = acceptable")
    print(f"  < 0.5 = encoder not learning — check training")
    
    # ── Plot ───────────────────────────────────────────────────────────────────
    print("\nGenerating plot...")
    
    fig = plt.figure(figsize=(15, 8))
    n_plot = min(len(top_idx), 5)  # Only plot top 5
    gs = gridspec.GridSpec(2, n_plot, figure=fig, hspace=0.45, wspace=0.3)
    
    t_ctx = np.arange(L)
    t_fut = np.arange(L, L + H)
    
    # Top row: context comparison
    ax_top = fig.add_subplot(gs[0, :])
    for rank in range(n_plot):
        idx = top_idx[rank]
        alpha = max(0.3, 0.8 - rank * 0.12)  # Ensure alpha >= 0.3
        ax_top.plot(t_ctx, memory_contexts[idx],
                    color='steelblue', alpha=alpha, linewidth=1.2,
                    label=f"Match {rank+1} (d={top_vals[rank]:.3f})" if rank < 3 else None)
    
    ax_top.plot(t_ctx, query_ctx, color='red', linewidth=2.0, label='Query context')
    ax_top.axhline(0, color='grey', linewidth=0.5, linestyle='--', label='barrier (x=0)')
    ax_top.set_title(f"Query #{query_idx}: Context Windows — "
                     f"query end={query_ctx[-1]:+.2f}  "
                     f"well={'negative' if query_ctx[-1] < 0 else 'positive'}")
    ax_top.set_xlabel("Time steps")
    ax_top.set_ylabel("Position (normalized)")
    ax_top.legend(fontsize=8)
    ax_top.grid(alpha=0.3)
    
    # Bottom row: retrieved futures vs true future
    for rank in range(n_plot):
        idx = top_idx[rank]
        ax = fig.add_subplot(gs[1, rank])
        
        fut_mean = memory_futures[idx].mean()
        match_str = "✓" if np.sign(query_ctx[-1]) == np.sign(fut_mean) else "✗"
        color = 'steelblue' if np.sign(query_ctx[-1]) == np.sign(fut_mean) else 'orange'
        
        ax.plot(t_fut, memory_futures[idx], color=color, linewidth=1.5, 
                label=f"Retrieved future {match_str}")
        ax.plot(t_fut, query_fut, color='red', linewidth=1.5, linestyle='--', 
                label='True future')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.set_title(f"Ref {rank+1}  d={top_vals[rank]:.3f}\n"
                     f"fut_mean={fut_mean:+.2f} {match_str}", fontsize=9)
        ax.set_xlabel("Time steps")
        if rank == 0:
            ax.set_ylabel("Position (normalized)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    # Save plot
    plt.suptitle(f"Retrieval Verification — mean well agreement: {mean_agree:.1%}",
                 fontsize=12, fontweight='bold')
    
    output_path = config['retrieval'].get('plot_path', 'retrieval_verification.pdf')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_path}")
    
    return mean_agree


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify TCN retrieval for Double-Well')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory if needed
    output_dir = os.path.dirname(config['retrieval'].get('plot_path', 'retrieval_verification.pdf'))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run verification
    verify_retrieval(config)