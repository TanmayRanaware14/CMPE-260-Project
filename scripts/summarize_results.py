
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import load_config

METHOD_ORDER = ["llvm_O3", "llvm_Oz", "random", "greedy", "ppo"]
METHOD_LABELS = {
    "llvm_O3": "LLVM -O3",
    "llvm_Oz": "LLVM -Oz",
    "random": "Random (20 passes)",
    "greedy": "Greedy (<=20 passes)",
    "ppo": "PPO (20 passes)",
}
SPLIT_ORDER = ["train", "validation", "test", "cross_family_test"]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize results and draw figures.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--baselines", default="results/baselines.csv")
    parser.add_argument("--ppo", default="results/ppo_eval.csv")
    parser.add_argument("--ppo-dir", default="results/ppo")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--output", default="results/summary.csv")
    return parser.parse_args()


def per_program(data):
    """Average seeds within each benchmark and method first."""
    return (
        data.groupby(["split", "benchmark", "method"], as_index=False)
        .agg(
            reduction_pct=("reduction_pct", "mean"),
            vs_o3_pct=("vs_o3_pct", "mean"),
            vs_oz_pct=("vs_oz_pct", "mean"),
            trials=("trials", "mean"),
            elapsed_seconds=("elapsed_seconds", "mean"),
        )
    )


def summarize(programs):
    rows = []
    for split in SPLIT_ORDER:
        for method in METHOD_ORDER:
            group = programs[(programs.split == split) & (programs.method == method)]
            if group.empty:
                continue
            rows.append({
                "split": split,
                "method": method,
                "programs": len(group),
                "mean_reduction_pct": group.reduction_pct.mean(),
                "median_reduction_pct": group.reduction_pct.median(),
                "mean_vs_oz_pct": group.vs_oz_pct.mean(),
                "beats_or_ties_oz": int((group.vs_oz_pct >= 0).sum()),
                "beats_o3": int((group.vs_o3_pct > 0).sum()),
                "mean_trials": group.trials.mean(),
                "mean_seconds": group.elapsed_seconds.mean(),
            })
    return pd.DataFrame(rows)


def plot_split_bars(summary, path):
    splits = [s for s in SPLIT_ORDER if s in set(summary.split)]
    methods = [m for m in METHOD_ORDER if m in set(summary.method)]
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, method in enumerate(methods):
        values = []
        for split in splits:
            match = summary[(summary.split == split) & (summary.method == method)]
            values.append(match.mean_reduction_pct.iloc[0] if not match.empty else float("nan"))
        positions = [j + (i - (len(methods) - 1) / 2) * width for j in range(len(splits))]
        ax.bar(positions, values, width, label=METHOD_LABELS[method])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([s.replace("_", " ") for s in splits])
    ax.set_ylabel("Mean IR instruction-count reduction (%)")
    ax.set_title("Instruction-count reduction from the original program")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_per_program(programs, path):
    methods = ["llvm_Oz", "random", "greedy", "ppo"]
    data = programs[programs.method.isin(methods)]
    benchmarks = []
    for split in SPLIT_ORDER:
        benchmarks += sorted(data[data.split == split].benchmark.unique())
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for i, method in enumerate(methods):
        values = []
        for benchmark in benchmarks:
            match = data[(data.benchmark == benchmark) & (data.method == method)]
            values.append(match.reduction_pct.iloc[0] if not match.empty else float("nan"))
        positions = [j + (i - (len(methods) - 1) / 2) * width for j in range(len(benchmarks))]
        ax.bar(positions, values, width, label=METHOD_LABELS[method])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels([b.split("/")[-1] for b in benchmarks], rotation=60, ha="right")
    ax.set_ylabel("IR instruction-count reduction (%)")
    ax.set_title("Per-program reduction (train | validation | cBench test | CHStone test)")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_learning_curves(ppo_dir, seeds, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for seed in seeds:
        curve_path = ppo_dir / f"learning_curve_seed{seed}.csv"
        if not curve_path.exists():
            continue
        curve = pd.read_csv(curve_path)
        ax.plot(curve.timesteps, curve.validation_mean_reduction_pct, marker="o", markersize=3, label=f"seed {seed}")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Validation mean reduction (%)")
    ax.set_title("PPO validation reduction during training (sha, dijkstra)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(ROOT / args.config)
    frames = [pd.read_csv(ROOT / args.baselines)]
    ppo_path = ROOT / args.ppo
    if ppo_path.exists():
        frames.append(pd.read_csv(ppo_path))
    data = pd.concat(frames, ignore_index=True)
    data = data[(data.status == "ok") & (data.method != "original")]

    programs = per_program(data)
    summary = summarize(programs)
    output = ROOT / args.output
    summary.to_csv(output, index=False, float_format="%.3f")
    programs.to_csv(output.with_name("per_program.csv"), index=False, float_format="%.3f")

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    figures = ROOT / args.figures
    figures.mkdir(parents=True, exist_ok=True)
    plot_split_bars(summary, figures / "reduction_by_split.png")
    plot_per_program(programs, figures / "reduction_per_program.png")
    plot_learning_curves(ROOT / args.ppo_dir, config["ppo_seeds"], figures / "ppo_learning_curves.png")
    print(f"Wrote {output}, per_program.csv, and figures in {figures}")


if __name__ == "__main__":
    main()
