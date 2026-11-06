
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from baselines import run_greedy_sequence, run_random_episode
from config import load_config
from environment import ir_instruction_count, make_compilergym_env, reset_compat

SPLITS = [
    ("train", "train_benchmarks"),
    ("validation", "validation_benchmarks"),
    ("test", "test_benchmarks"),
    ("cross_family_test", "cross_family_test_benchmarks"),
]

CSV_COLUMNS = [
    "benchmark",
    "split",
    "method",
    "seed",
    "initial_count",
    "o3_count",
    "oz_count",
    "final_count",
    "reduction_pct",
    "vs_o3_pct",
    "vs_oz_pct",
    "steps",
    "trials",
    "elapsed_seconds",
    "actions",
    "status",
    "error",
]


def outcome_row(benchmark, split, method, seed, refs, final, steps, trials, elapsed, actions):
    initial, o3, oz = refs
    return {
        "benchmark": benchmark,
        "split": split,
        "method": method,
        "seed": seed,
        "initial_count": initial,
        "o3_count": o3,
        "oz_count": oz,
        "final_count": final,
        "reduction_pct": f"{100.0 * (initial - final) / max(initial, 1):.4f}",
        "vs_o3_pct": f"{100.0 * (o3 - final) / max(o3, 1):.4f}",
        "vs_oz_pct": f"{100.0 * (oz - final) / max(oz, 1):.4f}",
        "steps": steps,
        "trials": trials,
        "elapsed_seconds": f"{elapsed:.4f}",
        "actions": json.dumps(actions),
        "status": "ok",
        "error": "",
    }


def error_row(benchmark, split, method, seed, exc):
    row = {column: "" for column in CSV_COLUMNS}
    row.update(
        benchmark=benchmark,
        split=split,
        method=method,
        seed=seed,
        status="error",
        error=f"{type(exc).__name__}: {exc}",
    )
    return row


def reference_counts(env):
    initial = ir_instruction_count(env)
    o3 = int(np.asarray(env.observation["IrInstructionCountO3"]).reshape(-1)[0])
    oz = int(np.asarray(env.observation["IrInstructionCountOz"]).reshape(-1)[0])
    return initial, o3, oz


def run_benchmark(benchmark, split, config, seeds):
    rows = []
    env = make_compilergym_env(
        benchmark=benchmark,
        environment_id=config["environment_id"],
        observation_space=config["observation_space"],
        reward_space=config["reward_space"],
    )
    max_steps = int(config["max_episode_steps"])
    try:
        reset_compat(env)
        refs = reference_counts(env)
        initial, o3, oz = refs

        rows.append(outcome_row(benchmark, split, "original", "", refs, initial, 0, 0, 0.0, []))
        rows.append(outcome_row(benchmark, split, "llvm_O3", "", refs, o3, "", "", 0.0, []))
        rows.append(outcome_row(benchmark, split, "llvm_Oz", "", refs, oz, "", "", 0.0, []))

        for seed in seeds:
            try:
                started = time.perf_counter()
                result = run_random_episode(env, max_steps=max_steps, seed=seed)
                elapsed = time.perf_counter() - started
                if result["error"]:
                    raise RuntimeError(result["error"])
                final = ir_instruction_count(env)
                rows.append(outcome_row(
                    benchmark, split, "random", seed, refs, final,
                    result["steps"], result["steps"], elapsed, result["actions"],
                ))
            except Exception as exc:
                rows.append(error_row(benchmark, split, "random", seed, exc))

        try:
            started = time.perf_counter()
            result = run_greedy_sequence(env, max_steps=max_steps)
            elapsed = time.perf_counter() - started
            if result["error"]:
                raise RuntimeError(result["error"])
            rows.append(outcome_row(
                benchmark, split, "greedy", "", refs, result["final_count"],
                result["steps"], result["trials"], elapsed, result["actions"],
            ))
        except Exception as exc:
            rows.append(error_row(benchmark, split, "greedy", "", exc))
    except Exception as exc:
        rows.append(error_row(benchmark, split, "original", "", exc))
    finally:
        env.close()
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run original, -O3, -Oz, random, and greedy on every split."
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="results/baselines.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(ROOT / args.config)
    seeds = config.get("random_seeds", [7, 17, 27])
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for split, key in SPLITS:
        for benchmark in config.get(key, []):
            benchmark_rows = run_benchmark(benchmark, split, config, seeds)
            rows.extend(benchmark_rows)
            for row in benchmark_rows:
                if row["status"] != "ok":
                    print(f"[error] {split} {benchmark} {row['method']} {row['error']}", flush=True)
                elif row["method"] in ("greedy", "random"):
                    print(
                        f"[ok] {split:<17} {benchmark:<24} {row['method']:<7} "
                        f"seed={row['seed']!s:<3} reduction={row['reduction_pct']}% "
                        f"vs_Oz={row['vs_oz_pct']}% steps={row['steps']} "
                        f"trials={row['trials']} time={row['elapsed_seconds']}s",
                        flush=True,
                    )

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
