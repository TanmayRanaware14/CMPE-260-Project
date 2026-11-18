"""Create figures and a summary table from results/*.csv.

    python scripts/make_plots.py [--split test]
"""

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compiler_pass_rl.evaluation import load_results, summarize  # noqa: E402
from compiler_pass_rl.plotting import (  # noqa: E402
    plot_generalization,
    plot_learning_curve,
    plot_method_comparison,
    plot_per_benchmark,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", help="split for the main comparison plots")
    args = parser.parse_args()

    results = ROOT / "results"
    figures = ROOT / "figures"
    frame = load_results(results / "baselines.csv", results / "ppo_evaluation.csv")

    summary = summarize(frame)
    summary.to_csv(results / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    plot_method_comparison(frame, args.split, figures / f"method_comparison_{args.split}.png")
    plot_per_benchmark(frame, args.split, figures / f"per_benchmark_{args.split}.png")
    plot_generalization(frame, figures / "generalization.png")
    monitors = sorted(glob.glob(str(results / "monitor/seed*.monitor.csv")))
    if monitors:
        plot_learning_curve(monitors, figures / "learning_curve.png")
    print("Figures written to", figures)


if __name__ == "__main__":
    main()
