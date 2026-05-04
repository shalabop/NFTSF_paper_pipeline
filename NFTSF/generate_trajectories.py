#!/usr/bin/env python3
"""
Generate trajectory data for NF-TSF experiments.

Produces .npy files in multi_sim format: shape (num_time_steps, 1 + num_sims)
where column 0 is the time index and columns 1..N are simulation trajectories.

Usage:
    python generate_trajectories.py --landscape linear_gaussian --num_sims 9000 --output_path ./data/train.npy
    python generate_trajectories.py --landscape single_well --num_sims 3000 --output_path ./data/test.npy
"""

import argparse
import os
import random
import torch
import numpy as np
from generators import (
    single_well_generator,
    double_wells_generator,
    linear_gaussian_generator,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate trajectory data for NF-TSF')
    parser.add_argument('--landscape', type=str, required=True,
                        choices=['linear_gaussian', 'single_well', 'double_well'],
                        help='Type of landscape to generate')
    parser.add_argument('--num_sims', type=int, required=True,
                        help='Number of simulation trajectories')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output .npy file path')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for generation (cpu recommended for large batches)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducible data generation (default: 0)')
    return parser.parse_args()


def set_seed(seed):
    """Seed all RNGs used by data generation for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

# Simulation resolution: 10000 internal steps at dt=0.01 over 100 s.
# Recording resolution: every 10th step => 1000 recorded points, spacing 0.1 s.
TOTAL_TIME = 100.0
TIME_STEP = 0.01      # internal integration timestep (do not increase this)
SAMPLE_EVERY = 10     # store one state per this many integration steps

# Linear Gaussian defaults
LINEAR_NOISE_STD = 0.1
LINEAR_SLOPE_RANGE = (0.5, 3.0)
LINEAR_INTERCEPT_RANGE = (-1.0, 1.0)

# Single well defaults (overdamped Langevin, reduced units k_B=1)
SW_X0_MEAN = 0.0
SW_X0_STD = 0.5
SW_ZETA = 1.0
SW_K = 1.0
SW_X_MU = 0.0
SW_T = 1.0
SW_BOLTZMANN = 1.0

# Double well defaults (overdamped Langevin, reduced units k_B=1)
DW_X0_MEAN = 0.0
DW_X0_STD = 0.3
DW_ZETA = 1.0
DW_LEFT_WELL = -1.0
DW_RIGHT_WELL = 1.0
DW_BARRIER_HEIGHT = 2.0
DW_T = 1.5
DW_BOLTZMANN = 1.0
DW_TILT = 0.0


def generate(landscape, num_sims, device):
    """Generate positions and times tensors for the given landscape."""
    if landscape == 'linear_gaussian':
        positions, times = linear_gaussian_generator(
            total_time=TOTAL_TIME, time_step=TIME_STEP,
            noise_std=LINEAR_NOISE_STD,
            slope_range=LINEAR_SLOPE_RANGE,
            intercept_range=LINEAR_INTERCEPT_RANGE,
            num_of_simulations=num_sims,
            device=device,
            sample_every=SAMPLE_EVERY,
        )
    elif landscape == 'single_well':
        positions, times = single_well_generator(
            total_time=TOTAL_TIME, time_step=TIME_STEP,
            x0_mean=SW_X0_MEAN, zeta=SW_ZETA, k=SW_K,
            x_mu=SW_X_MU, T=SW_T, boltzmann=SW_BOLTZMANN,
            num_of_simulations=num_sims, x0_std=SW_X0_STD,
            device=device,
            sample_every=SAMPLE_EVERY,
        )
    elif landscape == 'double_well':
        positions, times = double_wells_generator(
            total_time=TOTAL_TIME, time_step=TIME_STEP,
            x0_mean=DW_X0_MEAN, zeta=DW_ZETA,
            left_well=DW_LEFT_WELL, right_well=DW_RIGHT_WELL,
            barrier_height=DW_BARRIER_HEIGHT,
            T=DW_T, boltzmann=DW_BOLTZMANN, tilt=DW_TILT,
            num_of_simulations=num_sims, x0_std=DW_X0_STD,
            device=device,
            sample_every=SAMPLE_EVERY,
        )
    return positions, times


def to_multi_sim_format(positions, times):
    """
    Convert generator output to multi_sim .npy format.

    Input:
        positions: tensor shape (num_sims, num_time_steps)
        times: tensor shape (num_time_steps,)
    Output:
        numpy array shape (num_time_steps, 1 + num_sims)
        Column 0 = time values, columns 1..N = trajectories
    """
    pos_np = positions.cpu().numpy()   # (num_sims, T)
    time_np = times.cpu().numpy()      # (T,)

    # Transpose to (T, num_sims) and prepend time column
    pos_transposed = pos_np.T          # (T, num_sims)
    time_col = time_np.reshape(-1, 1)  # (T, 1)
    data = np.hstack([time_col, pos_transposed])

    return data


def main():
    args = parse_args()

    # Seed all RNGs before any stochastic operation
    set_seed(args.seed)
    print(f"Random seed: {args.seed} (landscape='{args.landscape}')")

    # Create output directory if needed
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {args.num_sims} trajectories for '{args.landscape}'...")
    positions, times = generate(args.landscape, args.num_sims, args.device)
    print(f"  positions shape: {tuple(positions.shape)}")
    print(f"  times shape: {tuple(times.shape)}")

    data = to_multi_sim_format(positions, times)
    print(f"  multi_sim format shape: {data.shape}")

    np.save(args.output_path, data)
    print(f"  Saved to: {args.output_path}")


if __name__ == '__main__':
    main()
