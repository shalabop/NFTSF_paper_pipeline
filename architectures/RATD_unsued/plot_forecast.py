# plot_forecast.py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tueplots import bundles
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def load_denorm_stats(data_path):
    d = np.load(data_path)
    if "mean" in d and "std" in d:
        return float(d["mean"][0]), float(d["std"][0])
    raise KeyError(f"'mean' and 'std' not found in {data_path}.")


def denormalize(x, mean, std):
    return x * std + mean


def plot_trajectory(ax, full_traj, time, time_test, median, ci90_lower, ci90_upper,
                    train_test_split, prediction_length, color="red", samples_to_plot=0):
    ax.plot(time[:train_test_split],full_traj[:train_test_split],
            color="black", linewidth=0.8, label="Context")
    ax.plot(time_test,full_traj[train_test_split:train_test_split + prediction_length],
            color="black", linewidth=0.8, linestyle="--", label="Ground Truth")
    if samples_to_plot is not None:
        # samples_to_plot shape: [num_futures, time]
        # Transpose to [time, num_futures] so ax.plot maps time to the x-axis correctly
        ax.plot(time_test, samples_to_plot.T, color="grey", alpha=0.3, linewidth=0.5)

    ax.plot(time_test, median, color=color, linewidth=0.8, label="Median")
    ax.fill_between(time_test, ci90_lower, ci90_upper,
                    alpha=0.2, color=color, label="90% CI")
    ax.axvline(time[train_test_split], color="grey",
               linewidth=0.6, linestyle="--")


def main():
    parser = argparse.ArgumentParser(description="Plot forecast trajectories.")
    parser.add_argument("--results",  "-r", required=True)
    parser.add_argument("--name",     "-n", required=True)
    parser.add_argument("--out",      "-o", required=True)
    parser.add_argument("--n-trajs",  "-k", type=int, default=6)
    parser.add_argument("--indices",  "-i", nargs="+", type=int, default=None)
    parser.add_argument("--color",    "-c", default="red")
    parser.add_argument("--ncols",         type=int, default=3)
    parser.add_argument("--denorm",   "-d", default=None,
                        help="Path to data .npz to denormalize results. "
                             "e.g. data/double_well.npz")
    parser.add_argument("--futures","-s",default=0)
    args = parser.parse_args()

    results           = np.load(args.results)
    samples           = results["samples"]
    ground_truth      = results["ground_truth"]
    ci90_lower        = results["ci90_lower"]
    ci90_upper        = results["ci90_upper"]
    full_trajectories = results["full_trajectories"]
    time              = results["time"]
    time_test         = results["time_test"]
    train_test_split  = int(results["train_test_split"])
    prediction_length = int(results["prediction_length"])

    if args.denorm is not None:
        mean, std = load_denorm_stats(args.denorm)
        print(f"Denormalizing with mean={mean:.4f} std={std:.4f}")
        samples           = denormalize(samples,           mean, std)
        ground_truth      = denormalize(ground_truth,      mean, std)
        ci90_lower        = denormalize(ci90_lower,        mean, std)
        ci90_upper        = denormalize(ci90_upper,        mean, std)
        full_trajectories = denormalize(full_trajectories, mean, std)

    indices = args.indices if args.indices is not None \
              else list(range(args.n_trajs))

    n     = len(indices)
    ncols = min(args.ncols, n)
    nrows = int(np.ceil(n / ncols))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(bundles.neurips2024(ncols=12, nrows=9, rel_width=1)):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(4 * ncols, 3 * nrows))
        axes = np.array(axes).flatten()

        for ax, z in zip(axes[:n], indices):
            traj_samples = samples[z] 
            median = np.median(samples[z], axis=1)
            
            num_futures = int(args.futures)
            futures_to_plot = traj_samples[:num_futures, :] if num_futures > 0 else None

            plot_trajectory(
                ax               = ax,
                full_traj        = full_trajectories[z],
                time             = time,
                time_test        = time_test,
                median           = median,
                ci90_lower       = ci90_lower[z],
                ci90_upper       = ci90_upper[z],
                train_test_split = train_test_split,
                prediction_length= prediction_length,
                color            = args.color,
                samples_to_plot=futures_to_plot
            )
            ax.set_title(f"Trajectory {z}", fontsize=8)
            ax.set_xlabel("Time")
            ax.set_ylabel("Position")

        for ax in axes[n:]:
            ax.set_visible(False)

        legend_elements = [
            Line2D([0], [0], color="black",    lw=1,          label="Context"),
            Line2D([0], [0], color="black",    lw=1, ls="--", label="Ground Truth"),
            Line2D([0], [0], color=args.color, lw=1,          label="Median"),
            Patch (            facecolor=args.color, alpha=0.2, label="90% CI"),
        ]
        fig.legend(handles=legend_elements, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
        fig.suptitle(args.name, y=1.06)
        fig.tight_layout()
        fig.savefig(args.out, bbox_inches="tight")
        print(f"Saved figure to: {args.out}")
        plt.show()


if __name__ == "__main__":
    main()