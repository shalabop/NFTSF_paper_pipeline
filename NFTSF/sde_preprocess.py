"""
sde_preprocess.py
=================
Load a pre-generated SDE dataset, split it into train / val / test partitions
at the trajectory level, and save each partition as a .npy file.

Usage
-----
    python sde_preprocess.py --model_key sw_sle_em --data_dir /path/to/Trajectories --output_dir ./split_data
    python sde_preprocess.py --model_key sw_gle_oe_em --data_dir ./Data/Trajectories --output_dir ./split_data
    python sde_preprocess.py --model_key sw_sle_em --data_dir . --output_dir ./split_data --no_val

    # With downsampling and diagnostics:
    python sde_preprocess.py \\
        --model_key sw_sle_em \\
        --data_dir "/path/to/raw" \\
        --output_dir "./split_data" \\
        --skip 10 \\
        --max_timesteps 1000 \\
        --time_stride 10 \\
        --plot_dataset \\
        --plot_n 8

Output files (written to --output_dir)
---------------------------------------
    <model_key>_x_train.npy   positions — train set  (72%)
    <model_key>_v_train.npy   velocities — train set
    <model_key>_x_val.npy     positions — val set     (8%)   [omitted with --no_val]
    <model_key>_v_val.npy     velocities — val set           [omitted with --no_val]
    <model_key>_x_test.npy    positions — test set   (20%)
    <model_key>_v_test.npy    velocities — test set

Split proportions (applied at the trajectory / row level)
----------------------------------------------------------
    test  : 20%   (outer_test_size)
    val   :  8%   (val_size × remaining 80%)
    train : 72%

Input files expected in --data_dir
------------------------------------
    <model_key>_<test_id>_<skip>_x.npy
    <model_key>_<test_id>_<skip>_v.npy
Default: test_id=1, skip=10  →  sw_sle_em_1_10_x.npy / sw_sle_em_1_10_v.npy

Note on --skip vs --time_stride
---------------------------------
    --skip      : selects the raw filename (e.g. <model_key>_<test_id>_<skip>_x.npy).
                  It does NOT downsample trajectories. It is a filename parameter only.
    --time_stride: performs actual downsampling along the time axis of the loaded data
                  by keeping every TIME_STRIDE-th timestep. Applied after --max_timesteps
                  truncation and before --max_windows row truncation.
"""

from __future__ import annotations

import argparse
import os

from sde_loader import load_sde_dataset, split_sde_dataset, save_sde_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split a pre-generated SDE dataset and save train/val/test .npy files."
    )
    p.add_argument(
        '--model_key', required=True,
        help='SDE model key, e.g. sw_sle_em or sw_gle_oe_em',
    )
    p.add_argument(
        '--data_dir', required=True,
        help='Directory containing the raw <model_key>_<id>_<skip>_x/v.npy files',
    )
    p.add_argument(
        '--output_dir', required=True,
        help='Directory where split .npy files are written',
    )
    p.add_argument(
        '--test_id', type=int, default=1,
        help='Parameter-set ID encoded in the filename (default: 1)',
    )
    p.add_argument(
        '--skip', type=int, default=10,
        help=(
            'Skip value encoded in the filename (default: 10). '
            'Selects file <model_key>_<test_id>_<skip>_x/v.npy. '
            'Does NOT downsample trajectories — use --time_stride for that.'
        ),
    )
    p.add_argument(
        '--outer_test_size', type=float, default=0.2,
        help='Fraction held out as test set (default: 0.2)',
    )
    p.add_argument(
        '--val_size', type=float, default=0.1,
        help='Fraction of train+val pool used for validation (default: 0.1)',
    )
    p.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducible splits (default: 42)',
    )
    p.add_argument(
        '--no_val', action='store_true', default=False,
        help='Skip saving val files — write only train and test',
    )
    p.add_argument(
        '--max_windows', type=int, default=None,
        help=(
            'If provided, keep only the first MAX_WINDOWS rows (windows/trajectories) '
            'from the loaded dataset before splitting. '
            'Useful for quick experiments on large files. '
            'Must be a positive integer. Omit to use the full dataset.'
        ),
    )
    p.add_argument(
        '--max_timesteps', type=int, default=None,
        help=(
            'If provided, keep only the first MAX_TIMESTEPS time steps of each '
            'trajectory (i.e., truncate along axis=1) before splitting. '
            'Must be a positive integer. Omit to use the full trajectory length.'
        ),
    )
    p.add_argument(
        '--time_stride', type=int, default=1,
        help=(
            'If >1, keep every TIME_STRIDE-th timestep after loading and optional '
            'max_timesteps truncation. This performs preprocessing-time downsampling '
            'and is separate from --skip, which only selects the input filename.'
        ),
    )
    p.add_argument(
        '--plot_dataset', action='store_true', default=False,
        help='If set, save quick diagnostic plots of the loaded/preprocessed dataset.',
    )
    p.add_argument(
        '--plot_n', type=int, default=8,
        help='Number of trajectories to plot (default: 8).',
    )
    return p.parse_args()


def _save_trajectory_plot(
    data,
    title: str,
    save_path: str,
    plot_n: int,
    ylabel: str = "value",
) -> None:
    import matplotlib.pyplot as plt
    n_plot = min(plot_n, data.shape[0])
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(n_plot):
        traj = data[i].flatten() if data.ndim > 2 else data[i]
        ax.plot(traj, alpha=0.7, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Timestep index")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  saved : {save_path}")


def _save_distribution_plot(
    data,
    title: str,
    save_path: str,
) -> None:
    import matplotlib.pyplot as plt
    flat = data.flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(flat, bins=100, color="steelblue", edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  saved : {save_path}")


def save_diagnostic_plots(
    dataset: dict,
    model_key: str,
    output_dir: str,
    plot_n: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")

    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    print(f"\nDiagnostic plots → {diag_dir}/")

    x = dataset['x']
    _save_trajectory_plot(
        data=x,
        title=f"{model_key} — position trajectories (first {min(plot_n, x.shape[0])})",
        save_path=os.path.join(diag_dir, f"{model_key}_x_trajectories.png"),
        plot_n=plot_n,
        ylabel="x",
    )
    _save_distribution_plot(
        data=x,
        title=f"{model_key} — position value distribution",
        save_path=os.path.join(diag_dir, f"{model_key}_x_distribution.png"),
    )

    if dataset['v'] is not None:
        v = dataset['v']
        _save_trajectory_plot(
            data=v,
            title=f"{model_key} — velocity trajectories (first {min(plot_n, v.shape[0])})",
            save_path=os.path.join(diag_dir, f"{model_key}_v_trajectories.png"),
            plot_n=plot_n,
            ylabel="v",
        )
        _save_distribution_plot(
            data=v,
            title=f"{model_key} — velocity value distribution",
            save_path=os.path.join(diag_dir, f"{model_key}_v_distribution.png"),
        )


def main() -> None:
    args = parse_args()

    print(f"\nSDE Preprocessing")
    print(f"  model_key  : {args.model_key}")
    print(f"  data_dir   : {args.data_dir}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  test_id={args.test_id}  skip={args.skip}  seed={args.seed}")
    print(f"  test={args.outer_test_size*100:.0f}%  "
          f"val={args.val_size*(1-args.outer_test_size)*100:.0f}%  "
          f"train={(1-args.outer_test_size)*(1-args.val_size)*100:.0f}%")

    # --- Load ---
    dataset = load_sde_dataset(
        model_key=args.model_key,
        data_dir=args.data_dir,
        test_id=args.test_id,
        skip=args.skip,
    )
    v_shape = dataset['v'].shape if dataset['v'] is not None else None
    print(f"\nLoaded : x={dataset['x'].shape}  v={v_shape}  "
          f"has_memory={dataset['has_memory']}")

    # --- Time splice: truncate each trajectory to the first max_timesteps columns ---
    if args.max_timesteps is not None:
        if args.max_timesteps <= 0:
            raise ValueError(
                f"--max_timesteps must be a positive integer, got {args.max_timesteps}"
            )
        T_available = dataset['x'].shape[1]
        if args.max_timesteps >= T_available:
            print(
                f"  time splice  : requested {args.max_timesteps} timesteps but "
                f"trajectories only have {T_available}; using full length"
            )
        else:
            dataset['x'] = dataset['x'][:, :args.max_timesteps]
            if dataset['v'] is not None:
                dataset['v'] = dataset['v'][:, :args.max_timesteps]
            v_shape_ts = dataset['v'].shape if dataset['v'] is not None else None
            print(
                f"  time splice  : kept first {args.max_timesteps} of {T_available} "
                f"timesteps  → x={dataset['x'].shape}  v={v_shape_ts}"
            )
    else:
        print(f"  time splice  : none (--max_timesteps not set)")

    # --- Time-stride downsampling (separate from --skip filename parameter) ---
    if args.time_stride <= 0:
        raise ValueError(
            f"--time_stride must be a positive integer, got {args.time_stride}"
        )
    if args.time_stride == 1:
        print(f"  time stride  : none (--time_stride=1, no downsampling applied)")
    else:
        T_before = dataset['x'].shape[1]
        dataset['x'] = dataset['x'][:, ::args.time_stride]
        if dataset['v'] is not None:
            dataset['v'] = dataset['v'][:, ::args.time_stride]
        v_shape_stride = dataset['v'].shape if dataset['v'] is not None else None
        T_after = dataset['x'].shape[1]
        print(
            f"  time stride  : kept every {args.time_stride}-th step, "
            f"{T_before} → {T_after} timesteps  "
            f"→ x={dataset['x'].shape}  v={v_shape_stride}"
        )

    # --- Window splice: keep only the first max_windows rows before splitting ---
    if args.max_windows is not None:
        if args.max_windows <= 0:
            raise ValueError(
                f"--max_windows must be a positive integer, got {args.max_windows}"
            )
        n_available = dataset['x'].shape[0]
        if args.max_windows >= n_available:
            print(
                f"  window splice: requested {args.max_windows} windows but dataset only "
                f"has {n_available}; using full dataset"
            )
        else:
            dataset['x'] = dataset['x'][:args.max_windows]
            if dataset['v'] is not None:
                dataset['v'] = dataset['v'][:args.max_windows]
            v_shape_ws = dataset['v'].shape if dataset['v'] is not None else None
            print(
                f"  window splice: kept first {args.max_windows} of {n_available} windows  "
                f"→ x={dataset['x'].shape}  v={v_shape_ws}"
            )
    else:
        print(f"  window splice: none (--max_windows not set)")

    # --- Split ---
    split = split_sde_dataset(
        dataset,
        outer_test_size=args.outer_test_size,
        val_size=args.val_size,
        random_seed=args.seed,
    )
    print(f"\nSplit:")
    print(f"  train : {split['x_train'].shape[0]} trajectories")
    if not args.no_val:
        print(f"  val   : {split['x_val'].shape[0]} trajectories")
    print(f"  test  : {split['x_test'].shape[0]} trajectories")

    # --- Save split files ---
    print(f"\nSaving to {args.output_dir}/")
    paths = save_sde_split(split, args.output_dir, include_val=not args.no_val)
    print(f"\nDone. {len(paths)} files written.")

    # --- Optional diagnostic plots (after all preprocessing, before return) ---
    if args.plot_dataset:
        save_diagnostic_plots(
            dataset=dataset,
            model_key=args.model_key,
            output_dir=args.output_dir,
            plot_n=args.plot_n,
        )


if __name__ == '__main__':
    main()
