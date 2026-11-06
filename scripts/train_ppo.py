
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3.common.callbacks import BaseCallback

from config import load_config
from environment import CompilerGymWrapper
from evaluation import evaluate_ppo_on_benchmark
from ppo_agent import train_ppo


class ValidationCheckpoint(BaseCallback):
    """Evaluate on validation programs and keep the best checkpoint.

    Selection uses the mean instruction-count reduction on validation
    programs only. Test programs are never evaluated here.
    """

    def __init__(self, eval_env, benchmarks, every, save_path, log_rows):
        super().__init__()
        self.eval_env = eval_env
        self.benchmarks = benchmarks
        self.every = every
        self.save_path = save_path
        self.log_rows = log_rows
        self.best = -float("inf")
        self.best_step = 0
        self.started = time.perf_counter()

    def _evaluate(self):
        reductions = []
        for benchmark in self.benchmarks:
            result = evaluate_ppo_on_benchmark(self.model, self.eval_env, benchmark)
            reductions.append(
                100.0 * (result["initial_count"] - result["final_count"])
                / max(result["initial_count"], 1)
            )
        mean = sum(reductions) / len(reductions)
        self.log_rows.append({
            "timesteps": self.num_timesteps,
            "validation_mean_reduction_pct": f"{mean:.4f}",
            "elapsed_seconds": f"{time.perf_counter() - self.started:.1f}",
        })
        if mean > self.best:
            self.best = mean
            self.best_step = self.num_timesteps
            self.model.save(str(self.save_path))
        print(
            f"[val] step={self.num_timesteps} mean_reduction={mean:.2f}% "
            f"best={self.best:.2f}% at {self.best_step}",
            flush=True,
        )

    def _on_training_start(self):
        self._evaluate()

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            self._evaluate()
        return True


def make_env(config, benchmarks):
    return CompilerGymWrapper(
        benchmark=benchmarks,
        environment_id=config["environment_id"],
        observation_space=config["observation_space"],
        reward_space=config["reward_space"],
        max_steps=config["max_episode_steps"],
        normalized_observation=True,
        normalized_reward=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on the training split.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--output-dir", default="results/ppo")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(ROOT / args.config)
    seeds = args.seeds or config["ppo_seeds"]
    timesteps = args.timesteps or config["ppo_total_timesteps"]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        train_env = make_env(config, config["train_benchmarks"])
        eval_env = make_env(config, config["validation_benchmarks"])
        log_rows = []
        save_path = output_dir / f"ppo_seed{seed}_best"
        callback = ValidationCheckpoint(
            eval_env,
            config["validation_benchmarks"],
            args.eval_every or config["ppo_eval_every"],
            save_path,
            log_rows,
        )
        started = time.perf_counter()
        try:
            train_ppo(
                train_env,
                total_timesteps=timesteps,
                learning_rate=config["ppo_learning_rate"],
                n_steps=config["ppo_n_steps_multi"],
                batch_size=config["ppo_batch_size"],
                gamma=config["ppo_gamma"],
                ent_coef=config["ppo_ent_coef"],
                seed=seed,
                verbose=0,
                callback=callback,
            )
        finally:
            train_env.close()
            eval_env.close()

        with (output_dir / f"learning_curve_seed{seed}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

        summary = {
            "seed": seed,
            "timesteps": timesteps,
            "best_validation_mean_reduction_pct": callback.best,
            "best_checkpoint_timesteps": callback.best_step,
            "training_seconds": round(time.perf_counter() - started, 1),
            "checkpoint": str(save_path.with_suffix(".zip").relative_to(ROOT)),
        }
        (output_dir / f"train_summary_seed{seed}.json").write_text(
            json.dumps(summary, indent=2)
        )
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
