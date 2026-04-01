
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 10,
    "axes.titlesize" : 11,
    "axes.labelsize" : 10,
    "legend.fontsize": 8,
    "figure.dpi"     : 150,
})


Y_LIMITS = {
    "double_well" : (-2.0,  2.0),
    "single_well" : (-1.5,  5.0),
    "linear"      : (None, None),
    "sine":  (-9, 9),
}

REF_LINES = {
    "double_well" : [-1.0, 1.0],
    "single_well" : [],
    "linear"      : [],
    "sine": [],
}

DATASETS = list(Y_LIMITS.keys())


def load_denorm_stats(data_path):
    d = np.load(data_path, allow_pickle=True)
    if "mean" in d and "std" in d:
        return float(0), float(d["std"])
    raise KeyError(f"'mean' and 'std' not found in {data_path}.")


def denormalize(x, mean, std):
    return x * std + mean


def plot_heatmap_panel(ax, samples, full_traj, time,
                       train_test_split, prediction_length,
                       title, y_limits, ref_lines,
                       num_bins=100, vmax=None):

    forecast_time = time[train_test_split:train_test_split + prediction_length]
    context_time  = time[:train_test_split]
    median        = np.median(samples, axis=1)   # (pred_len,)
    if y_limits[0] is None:
        y_min = min(np.percentile(samples, 1),
                    full_traj[train_test_split:
                              train_test_split + prediction_length].min())
        y_max = max(np.percentile(samples, 99),
                    full_traj[train_test_split:
                              train_test_split + prediction_length].max())
        margin = (y_max - y_min) * 0.2
        y_min -= margin
        y_max += margin
    else:
        y_min, y_max = y_limits

    # build density heatmap over forecast region
    bins    = np.linspace(y_min, y_max, num_bins)
    density = np.zeros((num_bins - 1, prediction_length))
    for t in range(prediction_length):
        hist, _ = np.histogram(samples[t, :], bins=bins, density=True)
        density[:, t] = hist

    # auto vmax if not provided
    if vmax is None:
        vmax = np.percentile(density[density > 0], 80) if density.max() > 0 else 1.0

    # plot heatmap over forecast region only
    ax.imshow(density,
              extent=[forecast_time[0], forecast_time[-1], bins[0], bins[-1]],
              aspect="auto", origin="lower",
              cmap="magma", vmin=0, vmax=vmax)

    # context trajectory
    ax.plot(context_time,
            full_traj[:train_test_split],
            color="white", linewidth=1.0, alpha=0.9, label="Context")

    # ground truth forecast
    ax.plot(forecast_time,
            full_traj[train_test_split:train_test_split + prediction_length],
            color="cyan", linestyle="--", linewidth=1.2,
            label="Ground Truth", alpha=0.85)

    # median forecast
    ax.plot(forecast_time, median,
            color="red", linewidth=1.5, label="Median", zorder=3)

    # reference lines (well positions etc.)
    for rl in ref_lines:
        ax.axhline(rl, linestyle="--", color="white",
                   alpha=0.4, linewidth=0.8)

    # forecast start line
    ax.axvline(time[train_test_split],
               color="white", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_ylim(y_min, y_max)
    ax.set_title(title, pad=3)
    ax.set_xlabel("Time")
    ax.set_ylabel("x")



def main():
    parser = argparse.ArgumentParser(
        description="Heatmap forecast plots for multiple trajectories of one model")
    parser.add_argument("--results",  "-r", required=True,
                        help="Path to forecast .npz file")
    parser.add_argument("--name",     "-n", required=True,
                        help="Model name e.g. CSDI")
    parser.add_argument("--dataset",  "-d", required=True,
                        choices=DATASETS,
                        help="Dataset name for y-limits and ref lines")
    parser.add_argument("--out",      "-o", required=True,
                        help="Output path e.g. figures/heatmap/csdi_double_well.pdf")
    parser.add_argument("--n-trajs",  "-k", type=int, default=6,
                        help="Number of trajectories to plot (default: 6)")
    parser.add_argument("--indices",  "-i", nargs="+", type=int, default=None,
                        help="Specific trajectory indices. Overrides --n-trajs")
    parser.add_argument("--ncols",         type=int, default=3,
                        help="Number of columns in grid (default: 3)")
    parser.add_argument("--bins",          type=int, default=100,
                        help="Number of histogram bins for heatmap (default: 100)")
    parser.add_argument("--vmax",          type=float, default=None,
                        help="Max density value for colormap (auto if not set)")
    parser.add_argument("--denorm",        default=None,
                        help="Path to data .npz to denormalize results. "
                             "e.g. norm_data/double_well.npz")
    args = parser.parse_args()

    r                 = np.load(args.results)
    samples_all       = r["samples"]           # (N, pred_len, n_samples)
    full_trajs        = r["full_trajectories"] # (N, total_T)
    time              = r["time"]              # (total_T,)
    train_test_split  = int(r["train_test_split"])
    prediction_length = int(r["prediction_length"])

    if args.denorm is not None:
        mean, std = load_denorm_stats(args.denorm)
        #print(f"Denormalizing with mean={mean:.4f} std={std:.4f}")
        samples_all = samples_all * std #denormalize(samples_all, mean, std)
        full_trajs  = full_trajs * std #denormalize(full_trajs,  mean, std)

    y_limits  = Y_LIMITS[args.dataset]
    ref_lines = REF_LINES[args.dataset]

    N       = samples_all.shape[0]
    indices = args.indices if args.indices is not None \
              else list(range(min(args.n_trajs, N)))

    n     = len(indices)
    ncols = min(args.ncols, n)
    nrows = int(np.ceil(n / ncols))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, z in zip(axes[:n], indices):
        if z >= N:
            print(f"WARNING: index {z} >= N={N}, skipping")
            ax.set_visible(False)
            continue

        print(samples_all.shape) ##(N, pred_len, n_samples)
        print(samples_all[z].shape)
        #exit()
        
        plot_heatmap_panel(
            ax               = ax,
            samples          = samples_all[z],      # (pred_len, n_samples)
            full_traj        = full_trajs[z],        # (total_T,)
            time             = time,
            train_test_split = train_test_split,
            prediction_length= prediction_length,
            title            = f"Trajectory {z}",
            y_limits         = y_limits,
            ref_lines        = ref_lines,
            num_bins         = args.bins,
            vmax             = args.vmax,
        )

    for ax in axes[n:]:
        ax.set_visible(False)

    # shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="white",  lw=1,          label="Context"),
        Line2D([0], [0], color="cyan",   lw=1, ls="--", label="Ground Truth"),
        Line2D([0], [0], color="red",    lw=1.5,        label="Median"),
    ]
    fig.legend(handles=legend_elements,
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False,
               facecolor="black", labelcolor="white")

    fig.suptitle(f"{args.name} — {args.dataset.replace('_', ' ').title()}",
                 y=1.06, fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("black")
    for ax in axes[:n]:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

    plt.tight_layout()
    fig.savefig(args.out, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {args.out}")
    plt.show()


if __name__ == "__main__":
    main()