
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import PPO

from config import load_config
from environment import CompilerGymWrapper
from evaluation import evaluate_ppo_on_benchmark

SPLITS = [
    ("train", "train_benchmarks"),
    ("validation", "validation_benchmarks"),
    ("test", "test_benchmarks"),
    ("cross_family_test", "cross_family_test_benchmarks"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen PPO checkpoints on every split."
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--checkpoint-dir", default="results/ppo")
    parser.add_argument("--baselines", default="results/baselines.csv")
    parser.add_argument("--output", default="results/ppo_eval.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(ROOT / args.config)
    checkpoint_dir = ROOT / args.checkpoint_dir

    references = {}
    with (ROOT / args.baselines).open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["method"] == "original" and row["status"] == "ok":
                references[row["benchmark"]] = (
                    int(row["initial_count"]),
                    int(row["o3_count"]),
                    int(row["oz_count"]),
                )

    all_benchmarks = [b for _, key in SPLITS for b in config.get(key, [])]
    env = CompilerGymWrapper(
        benchmark=all_benchmarks,
        environment_id=config["environment_id"],
        observation_space=config["observation_space"],
        reward_space=config["reward_space"],
        max_steps=config["max_episode_steps"],
        normalized_observation=True,
        normalized_reward=True,
    )

    rows = []
    try:
        for seed in config["ppo_seeds"]:
            model = PPO.load(str(checkpoint_dir / f"ppo_seed{seed}_best.zip"))
            for split, key in SPLITS:
                for benchmark in config.get(key, []):
                    initial, o3, oz = references[benchmark]
                    started = time.perf_counter()
                    result = evaluate_ppo_on_benchmark(model, env, benchmark)
                    elapsed = time.perf_counter() - started
                    if result["initial_count"] != initial:
                        raise RuntimeError(
                            f"{benchmark}: initial count {result['initial_count']} "
                            f"does not match baseline {initial}"
                        )
                    final = result["final_count"]
                    rows.append({
                        "benchmark": benchmark,
                        "split": split,
                        "method": "ppo",
                        "seed": seed,
                        "initial_count": initial,
                        "o3_count": o3,
                        "oz_count": oz,
                        "final_count": final,
                        "reduction_pct": f"{100.0 * (initial - final) / max(initial, 1):.4f}",
                        "vs_o3_pct": f"{100.0 * (o3 - final) / max(o3, 1):.4f}",
                        "vs_oz_pct": f"{100.0 * (oz - final) / max(oz, 1):.4f}",
                        "steps": len(result["actions"]),
                        "trials": len(result["actions"]),
                        "elapsed_seconds": f"{elapsed:.4f}",
                        "actions": json.dumps(result["actions"]),
                        "status": "error" if result["error"] else "ok",
                        "error": result["error"] or "",
                    })
                    print(
                        f"[ppo seed={seed}] {split:<17} {benchmark:<24} "
                        f"reduction={rows[-1]['reduction_pct']}% vs_Oz={rows[-1]['vs_oz_pct']}%",
                        flush=True,
                    )
    finally:
        env.close()

    output = ROOT / args.output
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
