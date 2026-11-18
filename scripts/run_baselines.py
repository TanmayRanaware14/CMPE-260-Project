"""Run random, random-search, greedy and LLVM -Oz/-O3 reference baselines.

    python scripts/run_baselines.py [--mock] [--splits train validation test]

Every budgeted method gets the same step budget (config max_episode_steps) as PPO.
Writes results/baselines.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compiler_pass_rl.baselines import (  # noqa: E402
    run_compiler_references,
    run_greedy,
    run_random_episode,
    run_random_search,
)
from compiler_pass_rl.config import get_splits, load_config, save_config_snapshot  # noqa: E402
from compiler_pass_rl.environment import make_compilergym_env  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "configs/default.json"))
    parser.add_argument("--splits", nargs="+", default=None,
                        help="default: every split in the config")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip-greedy", action="store_true",
                        help="greedy needs env.fork() and is the slowest baseline")
    parser.add_argument("--output", default=str(ROOT / "results/baselines.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    backend = "mock" if args.mock else "compilergym"
    all_splits = get_splits(config, backend)
    split_names = args.splits or list(all_splits)
    seeds = args.seeds if args.seeds is not None else config["baseline_seeds"]
    max_steps = config["max_episode_steps"]
    candidates = config.get("greedy_candidates") or None

    first = all_splits[split_names[0]][0]
    env = make_compilergym_env(
        benchmark=first,
        environment_id=config["environment_id"],
        observation_space=config["observation_space"],
        reward_space=config["reward_space"],
        backend=backend,
    )
    records = []
    try:
        for split in split_names:
            for benchmark in all_splits[split]:
                print(f"[{split}] {benchmark}")
                records.extend(run_compiler_references(env, benchmark, split=split))
                for seed in seeds:
                    records.append(run_random_episode(
                        env, benchmark, max_steps=max_steps, seed=seed, split=split))
                    records.append(run_random_search(
                        env, benchmark, max_steps=max_steps,
                        trials=config["random_search_trials"], seed=seed, split=split))
                    if not args.skip_greedy:
                        records.append(run_greedy(
                            env, benchmark, max_steps=max_steps,
                            candidate_actions=candidates, seed=seed, split=split))
                        if candidates is None:
                            break  # greedy over all actions is deterministic
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    save_config_snapshot(config, output.with_suffix(".config.json"),
                         {"backend": backend, "seeds": seeds, "splits": split_names})
    print(f"Wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
