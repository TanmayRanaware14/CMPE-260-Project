import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import run_random_episode
from config import load_config
from environment import make_compilergym_env, reset_compat


CSV_COLUMNS = [
    "benchmark",
    "split",
    "seed",
    "max_steps",
    "steps_taken",
    "initial_ir_count",
    "final_ir_count",
    "instruction_delta",
    "instruction_reduction_pct",
    "total_reward",
    "elapsed_seconds",
    "actions",
    "status",
    "error",
]


def parse_args():
    """Parse the pilot runner's command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the random LLVM pass-ordering baseline on the "
            "configured training benchmarks and save one CSV row per episode."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="Path to the experiment JSON configuration.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 17, 27],
        help="Seeds for the baseline's action sampler.",
    )
    parser.add_argument(
        "--output",
        default="results/random_baseline.csv",
        help="CSV path for one row per attempted episode.",
    )
    return parser.parse_args()


def benchmark_identifier(entry):
    """Return a training benchmark id exactly as the environment expects it.

    Configured ids are forwarded unchanged. A dataset prefix is not added
    here, so an id that already includes one is not prefixed a second time.
    """
    identifier = str(entry).strip()
    if not identifier:
        raise ValueError("Configuration contains an empty training benchmark.")
    return identifier


def load_experiment(config):
    """Read the settings this pilot uses from a loaded configuration."""
    required = [
        "environment_id",
        "observation_space",
        "reward_space",
        "max_episode_steps",
        "train_benchmarks",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(
            "Configuration is missing required keys: " + ", ".join(missing)
        )

    max_steps = int(config["max_episode_steps"])
    if max_steps <= 0:
        raise ValueError(
            f"max_episode_steps must be positive, got {max_steps}."
        )

    benchmarks = [benchmark_identifier(item) for item in config["train_benchmarks"]]
    if not benchmarks:
        raise ValueError("Configuration train_benchmarks is empty.")

    return {
        "environment_id": config["environment_id"],
        "observation_space": config["observation_space"],
        "reward_space": config["reward_space"],
        "max_steps": max_steps,
        "benchmarks": benchmarks,
    }


def print_settings(config_path, settings, seeds, output_path):
    """Print the resolved pilot settings before any episode starts."""
    print("Random baseline experiment")
    print(f"Config: {config_path}")
    print(f"Environment: {settings['environment_id']}")
    print(f"Observation space: {settings['observation_space']}")
    print(f"Reward space: {settings['reward_space']}")
    print(f"Max episode steps: {settings['max_steps']}")
    print("Split: train")
    print("Benchmarks: " + ", ".join(settings["benchmarks"]))
    print("Seeds: " + ", ".join(str(seed) for seed in seeds))
    print(f"Output: {output_path}")
    print()


def read_ir_instruction_count(env):
    """Read CompilerGym's raw IrInstructionCount for the current program state."""
    value = env.observation["IrInstructionCount"]
    counts = np.asarray(value).reshape(-1)
    if counts.size != 1:
        raise RuntimeError(
            "IrInstructionCount did not contain a single value: "
            f"shape {np.asarray(value).shape}"
        )
    count = counts[0]
    if not np.isfinite(count):
        raise RuntimeError(f"IrInstructionCount is not finite: {count}")
    return int(count)


def blank_row(benchmark, seed, max_steps):
    """Return a CSV row with measurements left blank."""
    row = {column: "" for column in CSV_COLUMNS}
    row["benchmark"] = benchmark
    row["split"] = "train"
    row["seed"] = seed
    row["max_steps"] = max_steps
    row["status"] = "error"
    return row


def require_consistent_rollout(result, max_steps):
    """Check that the baseline's step, action, and reward lists agree."""
    actions = list(result["actions"])
    rewards = list(result["rewards"])
    steps = int(result["steps"])

    if steps != len(actions) or steps != len(rewards):
        raise RuntimeError(
            "Random baseline returned mismatched steps, actions, and rewards: "
            f"steps={steps}, actions={len(actions)}, rewards={len(rewards)}"
        )
    if steps > max_steps:
        raise RuntimeError(
            f"Random baseline took {steps} steps, above the limit of {max_steps}."
        )

    reward_sum = float(sum(rewards))
    total_reward = float(result["total_reward"])
    if abs(reward_sum - total_reward) > 1e-6:
        raise RuntimeError(
            "Cumulative reward does not match the sum of per-step rewards: "
            f"{total_reward} vs {reward_sum}"
        )

    return actions, rewards, steps, total_reward


def clear_outcome(row):
    """Remove optimization-outcome fields that are not a completed measurement."""
    row["final_ir_count"] = ""
    row["instruction_delta"] = ""
    row["instruction_reduction_pct"] = ""
    row["total_reward"] = ""


def run_episode(benchmark, seed, settings):
    """Run one seeded random episode and return a CSV row.

    elapsed_seconds covers the baseline call only: its internal reset back to
    the original program, then application of the sampled passes. It does not
    include environment construction, the earlier initial-count read, the
    final-count read, or execution of the compiled program.
    """
    max_steps = settings["max_steps"]
    row = blank_row(benchmark, seed, max_steps)
    env = None

    try:
        env = make_compilergym_env(
            benchmark=benchmark,
            environment_id=settings["environment_id"],
            observation_space=settings["observation_space"],
            reward_space=settings["reward_space"],
        )
        reset_compat(env)
        initial_ir_count = read_ir_instruction_count(env)
        row["initial_ir_count"] = initial_ir_count
        if initial_ir_count <= 0:
            raise RuntimeError(
                "invalid initial IR instruction count: "
                f"{initial_ir_count}"
            )

        started = time.perf_counter()
        try:
            result = run_random_episode(
                env,
                max_steps=max_steps,
                seed=seed,
            )
        finally:
            row["elapsed_seconds"] = f"{time.perf_counter() - started:.6f}"

        actions, _, steps, total_reward = require_consistent_rollout(
            result,
            max_steps,
        )
        row["steps_taken"] = steps
        row["actions"] = json.dumps(actions)

        failure = result.get("error")
        if failure:
            row["status"] = "aborted"
            row["error"] = failure
            clear_outcome(row)
            return row

        final_ir_count = read_ir_instruction_count(env)
        instruction_delta = initial_ir_count - final_ir_count
        instruction_reduction_pct = (
            100.0 * instruction_delta / initial_ir_count
        )

        reset_compat(env)
        restored_ir_count = read_ir_instruction_count(env)
        if restored_ir_count != initial_ir_count:
            raise RuntimeError(
                "Initial IR instruction count did not match the original "
                "program state after reset: "
                f"{initial_ir_count} vs {restored_ir_count}"
            )

        row["final_ir_count"] = final_ir_count
        row["instruction_delta"] = instruction_delta
        row["instruction_reduction_pct"] = f"{instruction_reduction_pct:.6f}"
        row["total_reward"] = f"{total_reward:.10f}"
        row["status"] = "ok"
        row["error"] = ""
        return row

    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        clear_outcome(row)
        return row

    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                print(
                    f"[warn] {benchmark} seed={seed} "
                    f"environment close failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )


def print_progress(row):
    """Print one short line for an attempted episode."""
    if row["status"] == "ok":
        print(
            f"[ok] {row['benchmark']} seed={row['seed']} "
            f"steps={row['steps_taken']} "
            f"reduction={row['instruction_reduction_pct']}% "
            f"reward={row['total_reward']} "
            f"time={row['elapsed_seconds']}s",
            flush=True,
        )
        return

    print(
        f"[{row['status']}] {row['benchmark']} seed={row['seed']} "
        f"{row['error']}",
        flush=True,
    )


def write_csv(rows, output_path):
    """Write attempted episodes, creating the output directory if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, output_path):
    """Print episode counts and per-benchmark means from successful rows."""
    attempted = len(rows)
    successful = sum(row["status"] == "ok" for row in rows)
    failed = attempted - successful

    print()
    print(f"Attempted episodes: {attempted}")
    print(f"Successful episodes: {successful}")
    print(f"Failed or aborted episodes: {failed}")
    print(f"Output CSV: {output_path}")
    print()
    print(
        "Mean raw instruction-count reduction by training benchmark "
        "(successful episodes only):"
    )
    header = (
        f"{'benchmark':<28} {'successful_seeds':>16} "
        f"{'failed':>8} {'mean_reduction_pct':>20}"
    )
    print(header)
    print("-" * len(header))

    benchmarks = []
    for row in rows:
        if row["benchmark"] not in benchmarks:
            benchmarks.append(row["benchmark"])

    for benchmark in benchmarks:
        group = [row for row in rows if row["benchmark"] == benchmark]
        values = [
            float(row["instruction_reduction_pct"])
            for row in group
            if row["status"] == "ok"
        ]
        failures = sum(row["status"] != "ok" for row in group)
        if values:
            mean_text = f"{sum(values) / len(values):.4f}"
        else:
            mean_text = "n/a"
        print(
            f"{benchmark:<28} {len(values):>16} "
            f"{failures:>8} {mean_text:>20}"
        )

    print()
    print(
        "Means use successful episodes only. A negative percentage means "
        "those sequences increased IR instruction count."
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    settings = load_experiment(config)
    output_path = Path(args.output)

    print_settings(args.config, settings, args.seeds, output_path)

    rows = []
    for benchmark in settings["benchmarks"]:
        for seed in args.seeds:
            row = run_episode(benchmark, seed, settings)
            rows.append(row)
            print_progress(row)

    write_csv(rows, output_path)
    print_summary(rows, output_path)


if __name__ == "__main__":
    main()
