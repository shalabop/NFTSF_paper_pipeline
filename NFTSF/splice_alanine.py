#!/usr/bin/env python3
"""
splice_alanine.py
=================
Converts alanine-dipeptide-3x250ns-backbone-dihedrals.npz into
the multi_sim .npy format expected by NF-TSF:

    shape (num_time_steps, 1 + num_sims)
    col 0  = time index (0 .. num_time_steps-1)
    col 1+ = trajectory chunks (one column per chunk)

Each 250 000-frame trajectory is sliced into non-overlapping chunks of
`chunk_size` frames. With chunk_size=1000 (matching your double_well default)
you get 750 chunks total — comparable to num_sims=750.

You can choose which dihedral column to use:
    col 0 = phi  (the double-well coordinate — recommended)
    col 1 = psi

Usage:
    python splice_alanine.py                         # defaults: phi, 1000-frame chunks
    python splice_alanine.py --dihedral psi          # use psi instead
    python splice_alanine.py --chunk_size 500        # shorter chunks → more of them
    python splice_alanine.py --train_frac 0.8        # 80/20 train/test split
"""

import argparse
import numpy as np
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',       default='alanine-dipeptide-3x250ns-backbone-dihedrals.npz')
    p.add_argument('--output_dir',  default='./data')
    p.add_argument('--dihedral',    default='phi', choices=['phi', 'psi'],
                   help='Which dihedral angle to use as the 1-D coordinate')
    p.add_argument('--chunk_size',  type=int, default=1000,
                   help='Frames per chunk — match your TOTAL_TIME/TIME_STEP (default 1000)')
    p.add_argument('--train_frac',  type=float, default=0.8,
                   help='Fraction of chunks used for training (rest = test)')
    p.add_argument('--seed',        type=int, default=0)
    return p.parse_args()


def splice(npz_path, dihedral='phi', chunk_size=1000):
    """
    Load the .npz, pick the dihedral column, slice every trajectory into
    non-overlapping chunks of `chunk_size`, return as (chunk_size, N) array.
    """
    data = np.load(npz_path)
    col = 0 if dihedral == 'phi' else 1

    chunks = []
    for key in sorted(data.keys()):           # arr_0, arr_1, arr_2
        traj = data[key][:, col]              # (250000,)  — float32 radians
        n_chunks = len(traj) // chunk_size
        # Trim any remainder so all chunks are exactly chunk_size frames
        trimmed = traj[:n_chunks * chunk_size]
        split = trimmed.reshape(n_chunks, chunk_size)  # (250, 1000)
        chunks.append(split)

    all_chunks = np.concatenate(chunks, axis=0)        # (750, 1000)
    return all_chunks                                   # rows = simulations


def to_multi_sim(chunks, chunk_size):
    """
    chunks : (num_sims, chunk_size)
    returns: (chunk_size, 1 + num_sims)  — col 0 = frame index, rest = trajs
    """
    time_col = np.arange(chunk_size, dtype=np.float32).reshape(-1, 1)
    data = np.hstack([time_col, chunks.T.astype(np.float32)])
    return data


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading:     {args.input}")
    print(f"Dihedral:    {args.dihedral}  (col {'0=phi' if args.dihedral=='phi' else '1=psi'})")
    print(f"Chunk size:  {args.chunk_size} frames")

    chunks = splice(args.input, dihedral=args.dihedral, chunk_size=args.chunk_size)
    print(f"Total chunks: {len(chunks)}  (shape {chunks.shape})")

    # Shuffle then split train / test
    idx = rng.permutation(len(chunks))
    n_train = int(len(chunks) * args.train_frac)
    train_chunks = chunks[idx[:n_train]]
    test_chunks  = chunks[idx[n_train:]]

    print(f"Train chunks: {len(train_chunks)}   Test chunks: {len(test_chunks)}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, f'alanine_{args.dihedral}_train.npy')
    test_path  = os.path.join(args.output_dir, f'alanine_{args.dihedral}_test.npy')

    train_data = to_multi_sim(train_chunks, args.chunk_size)
    test_data  = to_multi_sim(test_chunks,  args.chunk_size)

    np.save(train_path, train_data)
    np.save(test_path,  test_data)

    print(f"\nSaved train: {train_path}  shape={train_data.shape}")
    print(f"Saved test:  {test_path}   shape={test_data.shape}")
    print(f"\nDrop-in replacement for double_well — same format:")
    print(f"  python train.py --data {train_path} ...")


if __name__ == '__main__':
    main()
