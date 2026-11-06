
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from environment import (
    make_compilergym_env,
    reset_compat,
    step_compat,
    action_count,
)


def main():
    benchmark = "cbench-v1/qsort"

    env = make_compilergym_env(
        benchmark=benchmark,
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
    )

    try:
        observation, info = reset_compat(env)

        print("Benchmark:", getattr(env, "benchmark", benchmark))
        print("Observation shape:", observation.shape)
        print("First 10 features:", observation.flatten()[:10])
        print("Action count:", action_count(env))

        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, step_info = step_compat(
            env, action
        )
        done = terminated or truncated

        print("\nSampled action:", action)
        print("Reward:", reward)
        print("Episode ended:", done)
        print("Next observation shape:", next_observation.shape)
        print("Step information:", step_info)

        print("\nSmoke test completed.")

    finally:
        env.close()


if __name__ == "__main__":
    main()