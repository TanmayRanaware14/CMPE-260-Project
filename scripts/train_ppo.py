"""Train PPO on the TRAIN split only (one model per seed).

    python scripts/train_ppo.py [--mock] [--seeds 7 8 9] [--timesteps 30000]

Outputs:
    results/models/ppo_seed<seed>.zip
    results/monitor/seed<seed>.monitor.csv   (training episode returns)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3.common.monitor import Monitor  # noqa: E402

from compiler_pass_rl.config import (  # noqa: E402
    get_splits,
    load_config,
    save_config_snapshot,
    set_global_seeds,
)
from compiler_pass_rl.environment import CompilerPassEnv  # noqa: E402
from compiler_pass_rl.ppo_agent import train_ppo  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "configs/default.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    backend = "mock" if args.mock else "compilergym"
    train_benchmarks = get_splits(config, backend)["train"]
    seeds = args.seeds if args.seeds is not None else config["ppo_seeds"]
    timesteps = args.timesteps or config["total_timesteps"]

    models_dir = ROOT / "results/models"
    monitor_dir = ROOT / "results/monitor"
    models_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training on {len(train_benchmarks)} programs: {train_benchmarks}")

    for seed in seeds:
        set_global_seeds(seed)
        base_env = CompilerPassEnv(
            benchmarks=train_benchmarks,
            environment_id=config["environment_id"],
            observation_space=config["observation_space"],
            reward_space=config["reward_space"],
            max_episode_steps=config["max_episode_steps"],
            backend=backend,
            use_action_histogram=config["use_action_histogram"],
            reward_clip=config["reward_clip"],
            seed=seed,
        )
        env = Monitor(base_env, filename=str(monitor_dir / f"seed{seed}"))
        try:
            model = train_ppo(
                env,
                total_timesteps=timesteps,
                learning_rate=config["ppo_learning_rate"],
                n_steps=config["ppo_n_steps"],
                batch_size=config["ppo_batch_size"],
                ent_coef=config["ppo_ent_coef"],
                gamma=config["ppo_gamma"],
                net_arch=config["ppo_net_arch"],
                seed=seed,
                verbose=0 if args.quiet else 1,
            )
            path = models_dir / f"ppo_seed{seed}.zip"
            model.save(str(path))
            print(f"Saved {path}")
        finally:
            env.close()

    save_config_snapshot(config, models_dir / "train_config.json",
                         {"backend": backend, "seeds": seeds, "timesteps": timesteps,
                          "train_benchmarks": train_benchmarks})


if __name__ == "__main__":
    main()
