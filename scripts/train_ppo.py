
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from environment import CompilerGymWrapper
from ppo_agent import train_ppo


def main():
    config = load_config(ROOT / "configs/default.json")

    # Initial smoke-training run on one benchmark.
    benchmark = config["train_benchmarks"][0]

    env = CompilerGymWrapper(
    benchmark=benchmark,
    observation_space=config["environment"]["observation_space"],
    reward_space=config["environment"]["reward_space"],
    max_steps=config["environment"]["max_steps"],
)

    try:
        model = train_ppo(
            env,
            total_timesteps=config["total_timesteps"],
            learning_rate=config["ppo_learning_rate"],
            n_steps=config["ppo_n_steps"],
            batch_size=config["ppo_batch_size"],
            seed=config["seed"],
        )

        output = ROOT / "results/ppo_model"
        model.save(str(output))
        print("Saved model to:", output.with_suffix(".zip"))

        print(
            "Note: this initial run trains on one benchmark. "
            "It does not establish generalization."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
