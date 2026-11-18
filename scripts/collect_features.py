"""Collect initial Autophase features and instruction counts for every benchmark.

    python scripts/collect_features.py [--mock]

Writes results/initial_features.csv (one row per program).
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compiler_pass_rl.config import get_splits, load_config  # noqa: E402
from compiler_pass_rl.environment import (  # noqa: E402
    instruction_counts,
    make_compilergym_env,
    reset_compat,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "configs/default.json"))
    parser.add_argument("--output", default=str(ROOT / "results/initial_features.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    backend = "mock" if args.mock else "compilergym"
    splits = get_splits(config, backend)

    first = next(iter(splits.values()))[0]
    env = make_compilergym_env(
        benchmark=first,
        environment_id=config["environment_id"],
        observation_space=config["observation_space"],
        reward_space=config["reward_space"],
        backend=backend,
    )
    rows, skipped = [], []
    try:
        for split, benchmarks in splits.items():
            for benchmark in benchmarks:
                try:
                    observation, _ = reset_compat(env, benchmark=benchmark)
                    counts = instruction_counts(env)
                except Exception as exc:
                    skipped.append((benchmark, repr(exc)))
                    print(f"SKIPPED {benchmark}: {exc}")
                    continue
                row = {"split": split, "benchmark": benchmark,
                       "init_count": counts["current"], "oz_count": counts["oz"],
                       "o3_count": counts["o3"]}
                for i, value in enumerate(observation.flatten()):
                    row[f"f{i}"] = int(value)
                rows.append(row)
                print(f"[{split}] {benchmark}: {counts}")
    finally:
        env.close()

    if not rows:
        raise RuntimeError("No benchmarks could be loaded.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output} ({len(skipped)} skipped)")


if __name__ == "__main__":
    main()
