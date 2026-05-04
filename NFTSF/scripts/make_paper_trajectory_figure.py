#!/usr/bin/env python3
"""
Generate publication-style trajectory summary figures from NFTSF result NPZ files.

Example 1 (select by window index)
----------------------------------
python scripts/make_paper_trajectory_figure.py \
  --config configs/paper_trajectory_figure_example.json \
  --traj_index 12

Example 2 (select exact trajectory ID)
--------------------------------------
python scripts/make_paper_trajectory_figure.py \
  --config configs/paper_trajectory_figure_example.json \
  --traj_id 7 \
  --traj_window_offset 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, MultipleLocator

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    yaml = None
    _HAS_YAML = False


LANDSCAPE_DISPLAY: dict[str, str] = {
    "single_well": "Single Well",
    "double_well": "Double Well",
    "alanine_phi": r"Alanine $\varphi$ (Phi)",
    "alanine_psi": r"Alanine $\psi$ (Psi)",
}

COORDINATE_LABEL: dict[str, str] = {
    "single_well": r"Position $x$",
    "double_well": r"Position $x$",
    "alanine_phi": r"Angle $\varphi$ (rad)",
    "alanine_psi": r"Angle $\psi$ (rad)",
}


def _land_display(landscape: str) -> str:
    return LANDSCAPE_DISPLAY.get(landscape, landscape.replace("_", " ").title())


def _coord_label(landscape: str) -> str:
    return COORDINATE_LABEL.get(landscape, "Value")


def _pi_formatter(x: float, pos: float) -> str:
    val = round(x / (np.pi / 2)) * (np.pi / 2)
    if abs(val) < 1e-8:
        return r"$0$"
    num = val / np.pi
    if np.isclose(num, 1.0):
        return r"$\pi$"
    if np.isclose(num, -1.0):
        return r"$-\pi$"
    if np.isclose(num, 0.5):
        return r"$\frac{\pi}{2}$"
    if np.isclose(num, -0.5):
        return r"$-\frac{\pi}{2}$"
    return rf"${num:.0f}\pi$"


def _apply_pi_ticks(ax: plt.Axes, landscape: str) -> None:
    if landscape in ("alanine_phi", "alanine_psi"):
        ax.yaxis.set_major_locator(MultipleLocator(np.pi / 2))
        ax.yaxis.set_major_formatter(FuncFormatter(_pi_formatter))


def _compute_shared_ylim(arrays: list[np.ndarray], margin_frac: float = 0.05) -> tuple[float, float]:
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        flat = np.asarray(arr, dtype=np.float64).reshape(-1)
        if flat.size:
            vals.append(flat)
    if not vals:
        return -1.0, 1.0
    all_vals = np.concatenate(vals)
    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return -1.0, 1.0
    margin = margin_frac * (y_max - y_min) if y_max > y_min else 0.1
    return y_min - margin, y_max + margin


def _standardize_samples(samples: np.ndarray, n_future: int) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    before = tuple(samples.shape)
    samp = np.asarray(samples)
    if samp.ndim != 2:
        raise ValueError(f"Expected 2D samples shaped (S, H) or (H, S), got {before}")

    if samp.shape[1] == n_future:
        std_samp = samp
    elif samp.shape[0] == n_future:
        std_samp = samp.T
    else:
        raise ValueError(
            f"Could not infer horizon axis for samples shape {before} with n_future={n_future}"
        )

    after = tuple(std_samp.shape)
    return std_samp.astype(np.float32), before, after


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper trajectory summary figure panels.")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON or YAML config file. CLI flags override config keys.")

    parser.add_argument("--dataset_path", type=str, default="./sw_sle_em_x_test.npy")
    parser.add_argument("--result_npz", type=str, default=None)
    parser.add_argument("--norm_stats_path", type=str, default=None)
    parser.add_argument("--landscape", type=str, default="single_well")
    parser.add_argument("--model", type=str, default="nftsf")
    parser.add_argument("--data_format", type=str, default="multi_sim",
                        choices=["single_sim", "multi_sim", "tnf", "sde_presplit"])

    parser.add_argument("--traj_index", type=int, default=0)
    parser.add_argument("--traj_id", type=int, default=None,
                        help="Exact underlying trajectory ID to select from fullset_traj_indices.")
    parser.add_argument("--traj_window_offset", type=int, default=0,
                        help="Window offset among windows belonging to traj_id (supports negative offsets).")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Maximum number of result samples to retain.")
    parser.add_argument("--n_display_samples", type=int, default=10)

    parser.add_argument("--output_dir", type=str, default="./paper_figures")
    parser.add_argument("--output_name", type=str, default="trajectory_summary")
    parser.add_argument("--formats", nargs="+", default=["png"],
                        help="Output format list, e.g., png svg pdf")

    parser.add_argument("--fig_width", type=float, default=5.5)
    parser.add_argument("--fig_height", type=float, default=4.8)
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--hist_bins_time", type=int, default=None)
    parser.add_argument("--hist_bins_value", type=int, default=None)
    parser.add_argument("--hist_cmap", type=str, default="magma")
    parser.add_argument("--hist_log", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hist_dark", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hist_show_colorbar", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hist_show_bin_edges", action=argparse.BooleanOptionalAction, default=None)

    return parser.parse_args()


def load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}

    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        if not _HAS_YAML:
            raise RuntimeError("YAML config requested but PyYAML is not installed. Use JSON instead.")
        return yaml.safe_load(text) or {}

    return json.loads(text)


def load_result_npz(result_npz: str) -> dict:
    npz_path = Path(result_npz)
    data = np.load(npz_path, allow_pickle=True)
    result = {k: data[k] for k in data.files}

    print(f"Result NPZ: {npz_path}")
    print("NPZ fields and shapes:")
    for key in sorted(result.keys()):
        val = result[key]
        if hasattr(val, "shape"):
            print(f"  - {key}: shape={val.shape}")
        else:
            print(f"  - {key}: type={type(val)}")

    return result


def load_norm_stats(norm_stats_path: str | None) -> tuple[float | None, float | None]:
    if not norm_stats_path:
        return None, None

    stats = np.load(norm_stats_path)
    mean = float(np.asarray(stats["mean"]).reshape(-1)[0])
    std = float(np.asarray(stats["std"]).reshape(-1)[0])
    print(f"Normalization stats: {norm_stats_path} (mean={mean:.6f}, std={std:.6f})")
    return mean, std


def denormalize(arr: np.ndarray, mean: float | None, std: float | None) -> np.ndarray:
    if mean is None or std is None:
        return arr
    return arr * std + mean


def _load_dataset_array(dataset_path: str, data_format: str) -> np.ndarray:
    arr = np.load(dataset_path, allow_pickle=True)

    if data_format == "single_sim":
        data = np.asarray(arr, dtype=np.float32)[:, :, 1]
    elif data_format == "multi_sim":
        data = np.asarray(arr, dtype=np.float32)[:, 1:].T
    elif data_format == "sde_presplit":
        data = np.asarray(arr, dtype=np.float32)
    elif data_format == "tnf":
        tnf = np.asarray(arr)
        if tnf.ndim == 1:
            tnf = tnf[:, None, None]
        elif tnf.ndim == 2:
            tnf = tnf[:, None, :]
        feature_slice = slice(1, None) if tnf.shape[2] > 1 else slice(None)
        data_vals = tnf[:, :, feature_slice].astype(np.float32)
        data = data_vals[:, 0, 0].reshape(1, -1)
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")

    return data


def select_window_and_trajectory(
    result: dict,
    requested_window_index: int,
    requested_traj_id: int | None = None,
    traj_window_offset: int = 0,
) -> tuple[int, int]:
    if "fullset_window_ground_truths" not in result:
        if requested_traj_id is not None:
            raise ValueError(
                "--traj_id requires result NPZ with fullset_window_ground_truths and fullset_traj_indices."
            )
        return 0, int(requested_window_index)

    n_windows = result["fullset_window_ground_truths"].shape[0]
    if requested_traj_id is None:
        window_idx = int(np.clip(requested_window_index, 0, n_windows - 1))
        traj_idx = int(result["fullset_traj_indices"][window_idx]) if "fullset_traj_indices" in result else window_idx
        return window_idx, traj_idx

    if "fullset_traj_indices" not in result:
        raise ValueError(
            "--traj_id was provided, but this NPZ does not contain fullset_traj_indices. "
            "Use --traj_index to select a window instead."
        )

    traj_indices = np.asarray(result["fullset_traj_indices"]).astype(int)
    matches = np.where(traj_indices == int(requested_traj_id))[0]
    if matches.size == 0:
        unique_vals = np.unique(traj_indices)
        unique_preview = unique_vals[:20]
        raise ValueError(
            f"No windows found for traj_id={requested_traj_id}. "
            f"Available trajectory IDs preview: {unique_preview.tolist()} "
            f"(total unique={unique_vals.size})"
        )

    offset = matches.size + int(traj_window_offset) if traj_window_offset < 0 else int(traj_window_offset)
    if offset < 0 or offset >= matches.size:
        raise ValueError(
            f"traj_window_offset={traj_window_offset} is out of range for traj_id={requested_traj_id}. "
            f"This trajectory has {matches.size} available windows. "
            f"Valid offsets are 0 to {matches.size - 1}, or negative offsets down to -{matches.size}."
        )

    window_idx = int(matches[offset])
    traj_idx = int(traj_indices[window_idx])
    return window_idx, traj_idx


def make_sine_wave(length: int, amplitude: float, frequency: float, phase: float,
                   noise_mean: float, noise_std: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(length)
    clean = amplitude * np.sin(2.0 * np.pi * frequency * x + phase)
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(loc=noise_mean, scale=noise_std, size=length)
    return clean, noisy


def _style_axis_text(ax: plt.Axes, landscape: str, dark: bool = False) -> None:
    if dark:
        ax.tick_params(labelsize=int(COMPARE_STYLE["tick_fs"]), colors="white")
    else:
        ax.tick_params(labelsize=int(COMPARE_STYLE["tick_fs"]))
    _apply_pi_ticks(ax, landscape)


COMPARE_STYLE: dict[str, object] = {
    "axis_label_fs": 10,
    "title_fs": 10,
    "tick_fs": 8,
    "legend_fs": 8,
    "truth_lw": 1.0,
    "median_lw": 0.8,
    "boundary_lw": 1.2,
    "sample_lw": 0.8,
    "band_alpha": 0.40,
    "boundary_alpha": 0.4,
    "well_ylim": (-2.2, 2.2),
    "well_yticks": [-2, -1, 0, 1, 2],
}


def _apply_compare_axis_style(ax: plt.Axes, landscape: str, n_past: int, n_future: int,
                              dark: bool = False, show_xlabel: bool = True, show_ylabel: bool = True) -> None:
    n_total = n_past + n_future
    ax.set_xlim(0, n_total)
    ax.set_xticks([0, n_past, n_total])
    if show_xlabel:
        ax.set_xlabel("Forecast step", fontsize=int(COMPARE_STYLE["axis_label_fs"]))
    if show_ylabel:
        ax.set_ylabel(_coord_label(landscape), fontsize=int(COMPARE_STYLE["axis_label_fs"]))

    if landscape in ("single_well", "double_well"):
        ax.set_ylim(*tuple(COMPARE_STYLE["well_ylim"]))
        ax.set_yticks(list(COMPARE_STYLE["well_yticks"]))

    if dark:
        ax.tick_params(labelsize=int(COMPARE_STYLE["tick_fs"]), colors="white")
    else:
        ax.tick_params(labelsize=int(COMPARE_STYLE["tick_fs"]))
    _apply_pi_ticks(ax, landscape)


def plot_sine_panel(ax: plt.Axes, cfg: dict, landscape: str,
                    show_titles: bool = False, show_legend: bool = False) -> None:
    clean, noisy = make_sine_wave(
        length=int(cfg["length"]),
        amplitude=float(cfg["amplitude"]),
        frequency=float(cfg["frequency"]),
        phase=float(cfg["phase"]),
        noise_mean=float(cfg["noise_mean"]),
        noise_std=float(cfg["noise_std"]),
        seed=int(cfg["seed"]),
    )
    x = np.arange(clean.size)
    ax.plot(x, clean, color=cfg.get("clean_color", "tab:blue"),
            linewidth=float(COMPARE_STYLE["truth_lw"]), label="Clean sine")
    ax.plot(x, noisy, color=cfg.get("noisy_color", "tab:orange"),
            linewidth=float(COMPARE_STYLE["sample_lw"]),
            alpha=float(cfg.get("noisy_alpha", 0.8)), label="Noisy sine")
    if show_titles:
        ax.set_title("Synthetic Sine + Noise", fontsize=int(COMPARE_STYLE["title_fs"]))
    ax.set_xlabel("Forecast step", fontsize=int(COMPARE_STYLE["axis_label_fs"]))
    ax.set_ylabel("Value", fontsize=int(COMPARE_STYLE["axis_label_fs"]))
    ax.grid(True, alpha=0.25)
    _style_axis_text(ax, landscape)
    if show_legend:
        ax.legend(loc="best", fontsize=int(COMPARE_STYLE["legend_fs"]))


def plot_raw_trajectory_panel(ax: plt.Axes, full_traj: np.ndarray, n_past: int, n_future: int,
                              landscape: str, shared_ylim: tuple[float, float] | None = None,
                              show_titles: bool = False, show_legend: bool = False) -> None:
    n_total = n_past + n_future
    x = np.arange(n_total)
    ax.plot(x, full_traj[:n_total], color="black", linewidth=float(COMPARE_STYLE["truth_lw"]), label="Raw trajectory")
    ax.axvline(n_past, color="black", linestyle="--", alpha=float(COMPARE_STYLE["boundary_alpha"]),
               linewidth=float(COMPARE_STYLE["boundary_lw"]), label="Forecast boundary")
    if show_titles:
        ax.set_title("Raw Dataset Trajectory", fontsize=int(COMPARE_STYLE["title_fs"]))
    _apply_compare_axis_style(ax, landscape, n_past, n_future)
    if shared_ylim is not None and landscape not in ("single_well", "double_well"):
        ax.set_ylim(*shared_ylim)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="best", fontsize=int(COMPARE_STYLE["legend_fs"]))


def plot_sample_paths_panel(ax: plt.Axes, ground_truth: np.ndarray, samples: np.ndarray,
                            n_past: int, n_future: int, n_display_samples: int,
                            style: dict, landscape: str,
                            shared_ylim: tuple[float, float] | None = None,
                            show_titles: bool = False, show_legend: bool = False) -> None:
    n_show = min(n_display_samples, samples.shape[0])
    x_all = np.arange(n_past + n_future)
    x_future = np.arange(n_past, n_past + n_future)

    ax.plot(x_all[:n_past], ground_truth[:n_past], color="black",
            linewidth=float(style.get("line_width", COMPARE_STYLE["truth_lw"])), label="Past")
    ax.plot(x_future, ground_truth[n_past:n_past+n_future], color="gray",
            linestyle="--", linewidth=float(style.get("line_width", COMPARE_STYLE["median_lw"])), label="Ground truth future")

    for i in range(n_show):
        label = f"Pred samples ({n_show})" if i == 0 else None
        ax.plot(
            x_future,
            samples[i, :n_future],
            color=style.get("sample_color", "tab:blue"),
            alpha=float(style.get("sample_alpha", 0.6)),
            linewidth=float(style.get("sample_line_width", COMPARE_STYLE["sample_lw"])),
            label=label,
        )

    ax.axvline(n_past, color="black", linestyle="--", alpha=float(COMPARE_STYLE["boundary_alpha"]),
               linewidth=float(COMPARE_STYLE["boundary_lw"]))
    if show_titles:
        ax.set_title("Predicted Sample Paths", fontsize=int(COMPARE_STYLE["title_fs"]))
    _apply_compare_axis_style(ax, landscape, n_past, n_future)
    if shared_ylim is not None and landscape not in ("single_well", "double_well"):
        ax.set_ylim(*shared_ylim)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="best", fontsize=int(COMPARE_STYLE["legend_fs"]))


def plot_interval_band_panel(ax: plt.Axes, ground_truth: np.ndarray, samples: np.ndarray,
                             n_past: int, n_future: int, style: dict, landscape: str,
                             shared_ylim: tuple[float, float] | None = None,
                             show_titles: bool = False, show_legend: bool = False) -> None:
    x_all = np.arange(0, n_past + n_future)
    future_steps = np.arange(n_past, n_past + n_future)

    lo90 = np.percentile(samples[:, :n_future], 5.0, axis=0)
    lo50 = np.percentile(samples[:, :n_future], 25.0, axis=0)
    median = np.percentile(samples[:, :n_future], 50.0, axis=0)
    hi50 = np.percentile(samples[:, :n_future], 75.0, axis=0)
    hi90 = np.percentile(samples[:, :n_future], 95.0, axis=0)

    ax.plot(x_all, ground_truth[:n_past + n_future], "k-", linewidth=float(COMPARE_STYLE["truth_lw"]), label="Truth")
    ax.plot(future_steps, median, color="#df1111", linewidth=float(COMPARE_STYLE["median_lw"]), label="Median")
    ax.fill_between(future_steps, lo50, hi50, color="tab:blue", alpha=float(COMPARE_STYLE["band_alpha"]), label="50% band")
    ax.fill_between(future_steps, lo90, lo50, color="tab:orange", alpha=float(COMPARE_STYLE["band_alpha"]), label="90% band")
    ax.fill_between(future_steps, hi50, hi90, color="tab:orange", alpha=float(COMPARE_STYLE["band_alpha"]))

    ax.axvline(x=n_past, color="k", linestyle="--", alpha=float(COMPARE_STYLE["boundary_alpha"]),
               linewidth=float(COMPARE_STYLE["boundary_lw"]))
    if show_titles:
        ax.set_title("Forecast Interval Bands", fontsize=int(COMPARE_STYLE["title_fs"]))
    _apply_compare_axis_style(ax, landscape, n_past, n_future)
    if shared_ylim is not None and landscape not in ("single_well", "double_well"):
        ax.set_ylim(*shared_ylim)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="upper left", fontsize=int(COMPARE_STYLE["legend_fs"]), framealpha=0.8)

def plot_hist2d_panel(ax: plt.Axes, ground_truth: np.ndarray, samples: np.ndarray,
                      n_past: int, n_future: int, hist_cfg: dict,
                      landscape: str, shared_ylim: tuple[float, float], dark: bool = False,
                      show_titles: bool = False, show_legend: bool = False, show_ylabel: bool = True) -> None:
    x_future = np.arange(n_past, n_past + n_future)
    time_rep = np.tile(x_future, (samples.shape[0], 1))

    bins_time = hist_cfg.get("bins_time")
    bins_value = hist_cfg.get("bins_value")
    n_x_bins = int(bins_time) if bins_time is not None else max(12, n_future // 5)
    n_y_bins = int(bins_value) if bins_value is not None else 24
    x_edges = np.linspace(n_past, n_past + n_future, n_x_bins + 1)
    y_edges = np.linspace(shared_ylim[0], shared_ylim[1], n_y_bins + 1)

    print(f"Density histogram bins: n_x_bins={n_x_bins}, n_y_bins={n_y_bins}")

    kwargs: dict = {
        "bins": [x_edges, y_edges],
        "cmap": hist_cfg.get("cmap", "magma"),
        "density": True,
    }
    if bool(hist_cfg.get("log", True)):
        kwargs["norm"] = LogNorm(vmin=1e-6)

    show_edges = bool(hist_cfg.get("show_bin_edges", False))
    if dark or not show_edges:
        kwargs["edgecolors"] = "none"

    hist = ax.hist2d(time_rep.flatten(), samples[:, :n_future].flatten(), **kwargs)
    hist[3].set_rasterized(True)

    gt_color = hist_cfg.get("ground_truth_color", "lime")
    gt_linewidth = float(hist_cfg.get("ground_truth_linewidth", 1.5))
    boundary_color = "white" if dark else "gray"
    label_color = "white" if dark else "black"
    face_color = "black" if dark else "white"

    all_steps = np.arange(0, n_past + n_future)
    ax.plot(all_steps, ground_truth[:n_past+n_future], color=gt_color, linewidth=gt_linewidth, label="Ground truth", zorder=10)
    ax.axvline(n_past, color=boundary_color, linestyle="--", alpha=float(COMPARE_STYLE["boundary_alpha"]),
               linewidth=float(COMPARE_STYLE["boundary_lw"]), label="Forecast boundary")

    if show_titles:
        ax.set_title("Predicted Future Density", fontsize=int(COMPARE_STYLE["title_fs"]), color=label_color)
    _apply_compare_axis_style(ax, landscape, n_past, n_future, dark=dark, show_ylabel=show_ylabel)
    ax.xaxis.label.set_color(label_color)
    ax.yaxis.label.set_color(label_color)
    ax.set_facecolor(face_color)
    if landscape not in ("single_well", "double_well"):
        ax.set_ylim(*shared_ylim)

    if bool(hist_cfg.get("show_colorbar", False)):
        cb = plt.colorbar(hist[3], ax=ax)
        cb.set_label("Density", fontsize=int(COMPARE_STYLE["legend_fs"]), color=label_color)
        if dark:
            cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
            cb.outline.set_edgecolor("white")

    if show_legend:
        leg = ax.legend(loc="best", fontsize=int(COMPARE_STYLE["legend_fs"]))
        if dark:
            leg.get_frame().set_facecolor("black")
            leg.get_frame().set_edgecolor("white")
            for txt in leg.get_texts():
                txt.set_color("white")


def save_figure(fig: plt.Figure, output_dir: Path, output_name: str, formats: list[str], dpi: int) -> list[Path]:
    saved = []
    for ext in formats:
        save_path = output_dir / f"{output_name}.{ext}"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved output: {save_path}")
        saved.append(save_path)
    return saved


def build_figure(context: dict, cfg: dict) -> tuple[plt.Figure, dict[str, plt.Figure]]:
    show_titles = bool(cfg["figure"].get("show_titles", False))
    show_legends = bool(cfg["figure"].get("show_legends", False))
    layout = [
        [".", "sine", "sine", "raw", "raw", "."],
        ["samples", "samples", "bands", "bands", "density", "density"],
    ]
    fig = plt.figure(figsize=(cfg["figure"]["width"], cfg["figure"]["height"]), constrained_layout=True)
    axes = fig.subplot_mosaic(layout)

    print(f"Main figure size: {cfg['figure']['width']} x {cfg['figure']['height']} in")
    print("Using subplot mosaic: sine, raw, samples, bands, density")
    print(f"Histogram dark mode: {bool(cfg['histogram'].get('dark', False))}")

    plot_sine_panel(axes["sine"], cfg["sine"], context["landscape"], show_titles=show_titles, show_legend=show_legends)
    plot_raw_trajectory_panel(
        axes["raw"], context["raw_full_traj"], context["n_past"], context["n_future"],
        context["landscape"], context["shared_ylim"], show_titles=show_titles, show_legend=show_legends,
    )

    plot_sample_paths_panel(
        axes["samples"], context["ground_truth"], context["samples"],
        context["n_past"], context["n_future"], cfg["display"]["n_display_samples"],
        cfg["style"], context["landscape"], context["shared_ylim"], show_titles=show_titles, show_legend=show_legends,
    )
    plot_interval_band_panel(
        axes["bands"], context["ground_truth"], context["samples"],
        context["n_past"], context["n_future"], cfg["style"],
        context["landscape"], context["shared_ylim"], show_titles=show_titles, show_legend=show_legends,
    )
    plot_hist2d_panel(
        axes["density"], context["ground_truth"], context["samples"],
        context["n_past"], context["n_future"], cfg["histogram"],
        context["landscape"], context["shared_ylim"],
        dark=bool(cfg["histogram"].get("dark", False)), show_titles=show_titles, show_legend=show_legends,
        show_ylabel=False,
    )

    panel_figs: dict[str, plt.Figure] = {}

    panel_builders = {
        "sine_panel": lambda ax: plot_sine_panel(ax, cfg["sine"], context["landscape"], show_titles=show_titles, show_legend=show_legends),
        "raw_trajectory_panel": lambda ax: plot_raw_trajectory_panel(
            ax, context["raw_full_traj"], context["n_past"], context["n_future"],
            context["landscape"], context["shared_ylim"], show_titles=show_titles, show_legend=show_legends,
        ),
        "sample_paths_panel": lambda ax: plot_sample_paths_panel(
            ax, context["ground_truth"], context["samples"], context["n_past"], context["n_future"],
            cfg["display"]["n_display_samples"], cfg["style"], context["landscape"], context["shared_ylim"],
            show_titles=show_titles, show_legend=show_legends,
        ),
        "interval_band_panel": lambda ax: plot_interval_band_panel(
            ax, context["ground_truth"], context["samples"], context["n_past"], context["n_future"],
            cfg["style"], context["landscape"], context["shared_ylim"], show_titles=show_titles, show_legend=show_legends,
        ),
        "hist2d_panel": lambda ax: plot_hist2d_panel(
            ax, context["ground_truth"], context["samples"], context["n_past"], context["n_future"],
            cfg["histogram"], context["landscape"], context["shared_ylim"],
            dark=bool(cfg["histogram"].get("dark", False)), show_titles=show_titles, show_legend=show_legends,
        ),
    }

    for name, builder in panel_builders.items():
        pfig, pax = plt.subplots(1, 1, figsize=(cfg["figure"]["panel_width"], cfg["figure"]["panel_height"]))
        builder(pax)
        pfig.tight_layout(rect=[0, 0, 1, 0.98])
        panel_figs[name] = pfig

    return fig, panel_figs


def main() -> None:
    args = parse_args()
    cfg_file = load_config(args.config)
    dataset_path = cfg_file.get("dataset_path", args.dataset_path)
    result_npz = cfg_file.get("result_npz", args.result_npz)
    if result_npz is None:
        raise ValueError("result_npz must be provided via --result_npz or config[result_npz].")
    norm_stats_path = cfg_file.get("norm_stats_path", args.norm_stats_path)
    landscape = cfg_file.get("landscape", args.landscape)
    model = cfg_file.get("model", args.model)
    data_format = cfg_file.get("data_format", args.data_format)
    traj_index = int(cfg_file.get("traj_index", args.traj_index))
    traj_id = cfg_file.get("traj_id", args.traj_id)
    traj_window_offset = int(cfg_file.get("traj_window_offset", args.traj_window_offset))
    random_seed = int(cfg_file.get("random_seed", args.random_seed))
    n_samples = cfg_file.get("n_samples", args.n_samples)
    output_dir_val = cfg_file.get("output_dir", args.output_dir)
    output_name = cfg_file.get("output_name", args.output_name)
    formats = cfg_file.get("formats", args.formats)

    hist_cfg = cfg_file.get("histogram", {})
    cfg = {
        "figure": {
            "width": cfg_file.get("figure", {}).get("width", args.fig_width),
            "height": cfg_file.get("figure", {}).get("height", args.fig_height),
            "panel_width": cfg_file.get("figure", {}).get("panel_width", 5.2),
            "panel_height": cfg_file.get("figure", {}).get("panel_height", 3.6),
            "dpi": cfg_file.get("figure", {}).get("dpi", args.dpi),
            "show_titles": cfg_file.get("figure", {}).get("show_titles", False),
            "show_legends": cfg_file.get("figure", {}).get("show_legends", False),
        },
        "display": {
            "n_display_samples": cfg_file.get("display", {}).get("n_display_samples", args.n_display_samples),
        },
        "sine": {
            "length": cfg_file.get("sine", {}).get("length", 200),
            "amplitude": cfg_file.get("sine", {}).get("amplitude", 1.0),
            "frequency": cfg_file.get("sine", {}).get("frequency", 0.02),
            "phase": cfg_file.get("sine", {}).get("phase", 0.0),
            "noise_mean": cfg_file.get("sine", {}).get("noise_mean", 0.0),
            "noise_std": cfg_file.get("sine", {}).get("noise_std", 0.15),
            "seed": cfg_file.get("sine", {}).get("seed", args.random_seed),
            "clean_color": cfg_file.get("sine", {}).get("clean_color", "tab:blue"),
            "noisy_color": cfg_file.get("sine", {}).get("noisy_color", "tab:orange"),
            "noisy_alpha": cfg_file.get("sine", {}).get("noisy_alpha", 0.8),
        },
        "style": cfg_file.get("style", {}),
        "histogram": {
            "bins_time": hist_cfg.get("bins_time", args.hist_bins_time),
            "bins_value": hist_cfg.get("bins_value", args.hist_bins_value),
            "cmap": hist_cfg.get("cmap", args.hist_cmap),
            "log": hist_cfg.get("log", True if args.hist_log is None else args.hist_log),
            "dark": hist_cfg.get("dark", False if args.hist_dark is None else args.hist_dark),
            "show_colorbar": hist_cfg.get("show_colorbar", False if args.hist_show_colorbar is None else args.hist_show_colorbar),
            "show_bin_edges": hist_cfg.get("show_bin_edges", False if args.hist_show_bin_edges is None else args.hist_show_bin_edges),
            "ground_truth_color": hist_cfg.get("ground_truth_color", "lime"),
            "ground_truth_linewidth": hist_cfg.get("ground_truth_linewidth", 1.5),
        },
    }

    cfg["sine"]["seed"] = cfg_file.get("sine", {}).get("seed", random_seed)

    result = load_result_npz(result_npz)
    mean, std = load_norm_stats(norm_stats_path)

    n_past = int(np.asarray(result["n_past"]).reshape(-1)[0])
    n_future = int(np.asarray(result["n_future"]).reshape(-1)[0])
    print(f"Selected n_past={n_past}, n_future={n_future}")

    window_idx, traj_idx = select_window_and_trajectory(
        result,
        requested_window_index=traj_index,
        requested_traj_id=None if traj_id is None else int(traj_id),
        traj_window_offset=traj_window_offset,
    )
    print("Selection request:")
    print(f"  traj_index/window index: {traj_index}")
    print(f"  traj_id: {traj_id}")
    print(f"  traj_window_offset: {traj_window_offset}")
    print("Selected:")
    print(f"  window_idx: {window_idx}")
    print(f"  underlying traj_idx: {traj_idx}")
    if traj_id is not None and "fullset_traj_indices" in result:
        traj_indices = np.asarray(result["fullset_traj_indices"]).astype(int)
        n_matches = int(np.where(traj_indices == int(traj_id))[0].size)
        print(f"  available windows for traj_id={int(traj_id)}: {n_matches}")

    if "fullset_window_ground_truths" in result and "fullset_window_samples" in result:
        ground_truth = np.asarray(result["fullset_window_ground_truths"][window_idx], dtype=np.float32)
        samples_raw = np.asarray(result["fullset_window_samples"][window_idx], dtype=np.float32)
        if "fullset_window_starts" in result:
            start = int(result["fullset_window_starts"][window_idx])
        else:
            start = 0
    else:
        ground_truth = np.asarray(result["ground_truth"], dtype=np.float32)
        samples_raw = np.asarray(result["samples"], dtype=np.float32)
        start = 0

    if "fullset_window_starts" in result:
        print(f"  window_start: {start}")

    print(f"Ground truth shape (pre-denorm): {ground_truth.shape}")
    print(f"Samples shape before standardization: {samples_raw.shape}")

    if n_samples is not None:
        if samples_raw.ndim == 2 and samples_raw.shape[0] != n_future:
            samples_raw = samples_raw[:int(n_samples)]
        elif samples_raw.ndim == 2 and samples_raw.shape[1] != n_future:
            samples_raw = samples_raw[:, :int(n_samples)]

    ground_truth = denormalize(ground_truth, mean, std)
    samples_raw = denormalize(samples_raw, mean, std)
    samples, before_shape, after_shape = _standardize_samples(samples_raw, n_future)
    print(f"Samples shape before/after standardization: {before_shape} -> {after_shape}")

    dataset = _load_dataset_array(dataset_path, data_format)
    dataset = denormalize(dataset, mean, std)

    traj_idx_clip = int(np.clip(traj_idx, 0, dataset.shape[0] - 1))
    raw_traj = np.asarray(dataset[traj_idx_clip], dtype=np.float32)
    raw_start = int(np.clip(start, 0, max(0, raw_traj.shape[0] - (n_past + n_future))))
    raw_full_traj = raw_traj[raw_start:raw_start + n_past + n_future]

    shared_ylim = _compute_shared_ylim([
        raw_full_traj[:n_past + n_future],
        ground_truth[:n_past + n_future],
        samples[:, :n_future],
    ])
    print(f"Shared y-limits used: ({shared_ylim[0]:.6f}, {shared_ylim[1]:.6f})")

    output_dir = Path(output_dir_val)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "landscape": landscape,
        "model": model,
        "n_past": n_past,
        "n_future": n_future,
        "ground_truth": ground_truth,
        "samples": samples,
        "raw_full_traj": raw_full_traj,
        "shared_ylim": shared_ylim,
    }

    fig, panel_figs = build_figure(context, cfg)

    save_figure(fig, output_dir, output_name, formats, int(cfg["figure"]["dpi"]))
    for panel_name, panel_fig in panel_figs.items():
        save_figure(panel_fig, output_dir, f"{output_name}_{panel_name}", formats,
                    int(cfg["figure"]["dpi"]))
        plt.close(panel_fig)

    plt.close(fig)


if __name__ == "__main__":
    main()
