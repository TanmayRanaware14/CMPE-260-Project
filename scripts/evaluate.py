"""Evaluate trained PPO models on every split (train, validation, test, shift).

    python scripts/evaluate.py [--mock] [--models results/models/ppo_seed*.zip]

Writes results/ppo_evaluation.csv and prints a comparison against baselines
(if results/baselines.csv exists) plus the PPO generalization gap.
"""

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compiler_pass_rl.config import get_splits, load_config, set_global_seeds  # noqa: E402
from compiler_pass_rl.environment import CompilerPassEnv  # noqa: E402
from compiler_pass_rl.evaluation import (  # noqa: E402
    evaluate_ppo,
    generalization_gap,
    summarize,
)
from compiler_pass_rl.ppo_agent import load_ppo  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "configs/default.json"))
    parser.add_argument("--models", nargs="+", default=[str(ROOT / "results/models/ppo_seed*.zip")])
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=5,
                        help="sampled-policy episodes per program (results are averaged)")
    parser.add_argument("--no-argmax", action="store_true",
                        help="skip the deterministic (argmax) evaluation")
    parser.add_argument("--output", default=str(ROOT / "results/ppo_evaluation.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    backend = "mock" if args.mock else "compilergym"
    splits = get_splits(config, backend)
    split_names = args.splits or list(splits)

    model_paths = sorted({p for pattern in args.models for p in glob.glob(pattern)})
    if not model_paths:
        raise FileNotFoundError(
            "No model files found. Run scripts/train_ppo.py first "
            f"(looked for: {args.models})"
        )

    everything = [b for s in split_names for b in splits[s]]
    env = CompilerPassEnv(
        benchmarks=everything,
        environment_id=config["environment_id"],
        observation_space=config["observation_space"],
        reward_space=config["reward_space"],
        max_episode_steps=config["max_episode_steps"],
        backend=backend,
        use_action_histogram=config["use_action_histogram"],
        reward_clip=config["reward_clip"],
    )
    records = []
    try:
        for path in model_paths:
            match = re.search(r"seed(\d+)", Path(path).name)
            seed = int(match.group(1)) if match else 0
            model = load_ppo(path)
            if model.observation_space.shape != env.observation_space.shape:
                raise ValueError(
                    f"{path} expects observations of shape {model.observation_space.shape} "
                    f"but the environment produces {env.observation_space.shape}. "
                    "Was it trained with a different config (e.g. use_action_histogram)?"
                )
            # "ppo": the stochastic policy PPO was trained as (actions sampled).
            # "ppo_argmax": always take the most likely action. With idempotent
            # passes this can repeat one action and waste the whole budget.
            modes = [("ppo", False, args.episodes)]
            if not args.no_argmax:
                modes.append(("ppo_argmax", True, 1))
            for split in split_names:
                for method, deterministic, episodes in modes:
                    set_global_seeds(seed)
                    results = evaluate_ppo(
                        model, env, splits[split], split=split, seed=seed,
                        deterministic=deterministic, episodes=episodes, method=method)
                    records.extend(results)
                    mean = sum(r["pct_vs_oz"] for r in results) / len(results)
                    print(f"seed {seed} [{split}] {method:<10} mean pct_vs_oz = {mean:+.2f}%")
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print("Saved evaluation results to:", output)

    frames = [pd.DataFrame(records)]
    baselines = ROOT / "results/baselines.csv"
    if baselines.exists():
        frames.append(pd.read_csv(baselines))
    combined = pd.concat(frames, ignore_index=True)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n" + summarize(combined).round(3).to_string(index=False))
    gap = generalization_gap(combined, "ppo", "train", "test")
    print(f"\nPPO generalization gap (train - test mean pct_vs_oz): {gap:+.2f} points")


if __name__ == "__main__":
    main()
