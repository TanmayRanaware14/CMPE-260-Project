"""Plots for baseline / PPO comparisons.

Colors follow the entity (method), never its rank, using fixed slots from a
colorblind-validated categorical palette. Text uses neutral ink, not series
colors. Metric everywhere: ``pct_vs_oz`` (positive = fewer IR instructions
than LLVM -Oz).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# Fixed entity -> color mapping (categorical slots 1-4, in order).
METHOD_COLORS = {
    "ppo": "#2a78d6",
    "greedy": "#eb6834",
    "random_search": "#1baf7a",
    "random": "#eda100",
    "ppo_argmax": "#e87ba4",
}
METHOD_ORDER = ["random", "random_search", "greedy", "ppo_argmax", "ppo"]
METHOD_LABELS = {
    "ppo": "PPO",
    "greedy": "Greedy",
    "random_search": "Random search",
    "random": "Random",
    "ppo_argmax": "PPO (argmax)",
}


def _canonical(method):
    return "random_search" if str(method).startswith("random_search") else str(method)


def _prepare(frame):
    frame = frame.copy()
    frame["method_key"] = frame["method"].map(_canonical)
    return frame


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _finish(fig, ax, title, output_path):
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _reference_lines(ax, frame):
    """-Oz is the zero line; draw the mean -O3 level as a dashed reference.

    Labels sit in the right margin (outside the axes) so they never collide
    with bars, error bars or value labels.
    """
    ax.axhline(0, color=INK_2, linewidth=1.2)
    ax.annotate("LLVM -Oz", (1.0, 0), xycoords=("axes fraction", "data"),
                xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                color=INK_2, fontsize=8, annotation_clip=False)
    o3 = frame[frame["method"] == "llvm_o3"]["pct_vs_oz"]
    if not o3.empty:
        level = float(o3.mean())
        ax.axhline(level, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.annotate("LLVM -O3", (1.0, level), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                    color=INK_2, fontsize=8, annotation_clip=False)


def _legend_below(ax, ncol):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol,
              frameon=False, fontsize=8, labelcolor=INK_2)


def plot_method_comparison(frame, split, output_path):
    """Mean pct_vs_oz per method on one split, with std error bars."""
    frame = _prepare(frame)
    rows = frame[frame["split"] == split]
    present = [m for m in METHOD_ORDER if m in set(rows["method_key"])]
    if not present:
        raise ValueError(f"No budgeted-method rows for split {split!r}")
    stats = [(m, rows[rows["method_key"] == m]["pct_vs_oz"]) for m in present]
    means = [s.mean() for _, s in stats]
    stds = [s.std(ddof=0) if len(s) > 1 else 0.0 for _, s in stats]

    fig, ax = plt.subplots(figsize=(6.4, 4))
    _style_axes(ax)
    x = np.arange(len(present))
    ax.bar(x, means, width=0.55, yerr=stds, color=[METHOD_COLORS[m] for m in present],
           error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    for xi, mean, std in zip(x, means, stds):
        tip = mean + std if mean >= 0 else mean - std
        ax.annotate(f"{mean:+.1f}%", (xi, tip), xytext=(0, 4 if mean >= 0 else -12),
                    textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in present])
    ax.set_ylabel("IR instructions vs -Oz (%, higher is better)")
    _reference_lines(ax, rows)
    _finish(fig, ax, f"Method comparison on the {split} split (mean ± std)", output_path)


def plot_per_benchmark(frame, split, output_path):
    """Grouped bars: pct_vs_oz per benchmark and method (mean over seeds)."""
    frame = _prepare(frame)
    rows = frame[frame["split"] == split]
    present = [m for m in METHOD_ORDER if m in set(rows["method_key"])]
    benchmarks = sorted(rows["benchmark"].unique())
    if not present or not benchmarks:
        raise ValueError(f"No rows for split {split!r}")
    pivot = (rows[rows["method_key"].isin(present)]
             .groupby(["benchmark", "method_key"])["pct_vs_oz"].mean().unstack())

    fig, ax = plt.subplots(figsize=(max(6.4, 1.4 * len(benchmarks) + 2), 4))
    _style_axes(ax)
    width = 0.8 / len(present)
    x = np.arange(len(benchmarks))
    for i, method in enumerate(present):
        values = [pivot.loc[b, method] if method in pivot.columns and b in pivot.index
                  else np.nan for b in benchmarks]
        ax.bar(x + (i - (len(present) - 1) / 2) * width, values, width=width * 0.9,
               color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax.set_xticks(x)
    ax.set_xticklabels([b.split("/")[-1] for b in benchmarks], rotation=30, ha="right")
    ax.set_ylabel("IR instructions vs -Oz (%, higher is better)")
    _reference_lines(ax, rows)
    _legend_below(ax, len(present))
    _finish(fig, ax, f"Per-benchmark results on the {split} split", output_path)


def plot_generalization(frame, output_path, splits=("train", "validation", "test", "shift")):
    """Mean pct_vs_oz per method across splits - the generalization gap."""
    frame = _prepare(frame)
    splits = [s for s in splits if s in set(frame["split"])]
    present = [m for m in METHOD_ORDER if m in set(frame["method_key"])]
    if not splits or not present:
        raise ValueError("Not enough data for a generalization plot")
    fig, ax = plt.subplots(figsize=(6.8, 4))
    _style_axes(ax)
    width = 0.8 / len(present)
    x = np.arange(len(splits))
    for i, method in enumerate(present):
        values = [frame[(frame["method_key"] == method) & (frame["split"] == s)]
                  ["pct_vs_oz"].mean() for s in splits]
        ax.bar(x + (i - (len(present) - 1) / 2) * width, values, width=width * 0.9,
               color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("IR instructions vs -Oz (%, higher is better)")
    ax.axhline(0, color=INK_2, linewidth=1.2)
    _legend_below(ax, len(present))
    _finish(fig, ax, "Generalization across splits", output_path)


def read_monitor_csv(path):
    """Read an SB3 Monitor CSV (first line is a '#' JSON header)."""
    return pd.read_csv(path, skiprows=1)


def plot_learning_curve(monitor_paths, output_path, window=50):
    """Episode return during PPO training, one thin line per seed."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    _style_axes(ax)
    plotted = 0
    for path in monitor_paths:
        data = read_monitor_csv(path)
        if data.empty:
            continue
        smooth = data["r"].rolling(window=min(window, len(data)), min_periods=1).mean()
        ax.plot(np.arange(1, len(data) + 1), smooth, color=METHOD_COLORS["ppo"],
                linewidth=1.5, alpha=0.8)
        plotted += 1
    if not plotted:
        plt.close(fig)
        raise ValueError("No monitor data to plot")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Episode return (rolling mean, {window})")
    _finish(fig, ax, f"PPO training curve ({plotted} seed{'s' if plotted != 1 else ''})",
            output_path)
