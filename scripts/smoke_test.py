"""Verify the environment works before building anything on top of it.

    python scripts/smoke_test.py                 # real CompilerGym, one benchmark
    python scripts/smoke_test.py --mock          # mock backend (no CompilerGym)
    python scripts/smoke_test.py --check-config  # confirm every config URI loads
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compiler_pass_rl.config import get_splits, load_config  # noqa: E402
from compiler_pass_rl.environment import (  # noqa: E402
    CompilerPassEnv,
    action_count,
    instruction_counts,
    make_compilergym_env,
    reset_compat,
    step_compat,
)


def check_raw_env(backend, benchmark, config):
    env = make_compilergym_env(
        benchmark=benchmark,
        environment_id=config["environment_id"],
        observation_space=config["observation_space"],
        reward_space=config["reward_space"],
        backend=backend,
    )
    try:
        observation, _ = reset_compat(env, benchmark=benchmark)
        print("Benchmark:          ", getattr(env, "benchmark", benchmark))
        print("Observation shape:  ", observation.shape)
        print("First 10 features:  ", observation.flatten()[:10])
        print("Action count:       ", action_count(env))
        print("Counts at reset:    ", instruction_counts(env))
        action = env.action_space.sample()
        _, reward, done, info = step_compat(env, action)
        print(f"Applied action {action}: reward={reward:.4f} done={done}")
        print("Counts after step:  ", instruction_counts(env))
        fork = env.fork()
        fork.close()
        print("env.fork():          OK (needed by the greedy baseline)")
    finally:
        env.close()


def check_wrapper(backend, benchmarks, config):
    env = CompilerPassEnv(
        benchmarks=benchmarks,
        environment_id=config["environment_id"],
        observation_space=config["observation_space"],
        reward_space=config["reward_space"],
        max_episode_steps=config["max_episode_steps"],
        backend=backend,
        use_action_histogram=config["use_action_histogram"],
        reward_clip=config["reward_clip"],
    )
    try:
        observation, info = env.reset(seed=0)
        assert env.observation_space.contains(observation), "observation outside Box"
        print(f"Wrapper observation: shape={observation.shape} benchmark={info['benchmark']}")
        total, steps, truncated, terminated = 0.0, 0, False, False
        while not (truncated or terminated):
            observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            steps += 1
        print(f"Wrapper episode: {steps} steps, return={total:.4f}, truncated={truncated}")
    finally:
        env.close()


def check_config(backend, config):
    splits = get_splits(config, backend)
    failures = []
    for split, uris in splits.items():
        for uri in uris:
            env = None
            try:
                env = make_compilergym_env(
                    benchmark=uri,
                    environment_id=config["environment_id"],
                    observation_space=config["observation_space"],
                    reward_space=config["reward_space"],
                    backend=backend,
                )
                observation, _ = reset_compat(env, benchmark=uri)
                counts = instruction_counts(env)
                print(f"  OK   [{split}] {uri}  insts={counts['current']} oz={counts['oz']}")
            except Exception as exc:  # report every failure, don't stop at the first
                failures.append((split, uri, exc))
                print(f"  FAIL [{split}] {uri}: {type(exc).__name__}: {exc}")
            finally:
                if env is not None:
                    env.close()
    if failures:
        print(f"\n{len(failures)} benchmark(s) failed. Fix the URIs in configs/default.json.")
        sys.exit(1)
    print("\nAll configured benchmarks loaded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="use the mock backend")
    parser.add_argument("--check-config", action="store_true",
                        help="verify every benchmark URI in the config loads")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs/default.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    backend = "mock" if args.mock else "compilergym"
    splits = get_splits(config, backend)

    if args.check_config:
        check_config(backend, config)
        return

    benchmark = args.benchmark or splits["train"][0]
    print(f"== Raw environment ({backend}) ==")
    check_raw_env(backend, benchmark, config)
    print("\n== Gymnasium wrapper ==")
    check_wrapper(backend, splits["train"][:3], config)
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
