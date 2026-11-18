"""Unit tests. They run on the mock backend, so CompilerGym is not required."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compiler_pass_rl.baselines import (  # noqa: E402
    run_compiler_references,
    run_greedy,
    run_random_episode,
    run_random_search,
)
from compiler_pass_rl.config import get_splits, load_config  # noqa: E402
from compiler_pass_rl.environment import (  # noqa: E402
    AUTOPHASE_DIM,
    CompilerPassEnv,
    action_count,
    instruction_counts,
    reset_compat,
    step_compat,
)
from compiler_pass_rl.evaluation import make_record, summarize  # noqa: E402
from compiler_pass_rl.mock_env import MockLLVMEnv  # noqa: E402

URI = "benchmark://mock-v0/3"


@pytest.mark.parametrize("new_api", [False, True])
def test_compat_helpers_handle_both_api_styles(new_api):
    env = MockLLVMEnv(URI, new_api=new_api)
    obs, info = reset_compat(env, benchmark=URI)
    assert obs.shape == (AUTOPHASE_DIM,) and isinstance(info, dict)
    obs, reward, done, info = step_compat(env, 0)
    assert isinstance(reward, float) and done is False
    assert action_count(env) == 124


def test_mock_is_deterministic_and_idempotent():
    a, b = MockLLVMEnv(URI), MockLLVMEnv(URI)
    seq = [5, 9, 20, 5, 5]
    for action in seq:
        a.step(action)
        b.step(action)
    assert a.observation["IrInstructionCount"] == b.observation["IrInstructionCount"]
    before = a.observation["IrInstructionCount"]
    a.step(5)  # re-applying a pass changes nothing
    assert a.observation["IrInstructionCount"] == before


def test_fork_is_independent():
    env = MockLLVMEnv(URI)
    fork = env.fork()
    for action in range(60):
        fork.step(action)
    assert env.observation["IrInstructionCount"] == env.observation["IrInstructionCountO0"]
    assert fork.observation["IrInstructionCount"] != env.observation["IrInstructionCount"]


def test_wrapper_spaces_truncation_and_forced_benchmark():
    env = CompilerPassEnv([URI, "benchmark://mock-v0/9"], max_episode_steps=5, backend="mock")
    obs, info = env.reset(seed=1, options={"benchmark": "benchmark://mock-v0/9"})
    assert info["benchmark"] == "benchmark://mock-v0/9"
    assert env.observation_space.contains(obs)
    steps, truncated, terminated = 0, False, False
    while not (truncated or terminated):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        steps += 1
    assert steps == 5 and truncated and not terminated


def test_wrapper_samples_multiple_programs():
    benchmarks = [f"benchmark://mock-v0/{i}" for i in range(6)]
    env = CompilerPassEnv(benchmarks, backend="mock", seed=0)
    seen = {env.reset()[1]["benchmark"] for _ in range(40)}
    assert len(seen) > 1


def test_wrapper_cycle_sampling_is_ordered():
    benchmarks = [f"benchmark://mock-v0/{i}" for i in range(3)]
    env = CompilerPassEnv(benchmarks, backend="mock", sampling="cycle")
    seen = [env.reset()[1]["benchmark"] for _ in range(6)]
    assert seen == benchmarks + benchmarks


def test_reward_clip_applies_to_training_signal_only():
    env = CompilerPassEnv([URI], backend="mock", reward_clip=0.001)
    env.reset()
    rewards = [env.step(a)[1] for a in range(40)]
    assert max(abs(r) for r in rewards) <= 0.001 + 1e-12


def test_baselines_report_consistent_records():
    env = MockLLVMEnv(URI)
    rows = [
        run_random_episode(env, URI, max_steps=10, seed=0),
        run_random_search(env, URI, max_steps=10, trials=4, seed=0),
        run_greedy(env, URI, max_steps=10, candidate_actions=None),
        *run_compiler_references(env, URI),
    ]
    by_method = {r["method"]: r for r in rows}
    assert by_method["llvm_oz"]["pct_vs_oz"] == pytest.approx(0.0)
    assert by_method["llvm_o3"]["final_count"] == by_method["llvm_o3"]["o3_count"]
    for row in rows:
        assert row["final_count"] <= row["init_count"] or row["method"] == "random"
    assert by_method["greedy"]["steps"] <= 10
    assert by_method["greedy"]["final_count"] <= by_method["random"]["final_count"]


def test_random_search_not_worse_than_its_own_first_trial():
    env = MockLLVMEnv(URI)
    single = run_random_search(env, URI, max_steps=10, trials=1, seed=3)
    best = run_random_search(env, URI, max_steps=10, trials=12, seed=3)
    assert best["final_count"] <= single["final_count"]


def test_greedy_reward_matches_instruction_count_change():
    env = MockLLVMEnv(URI)
    row = run_greedy(env, URI, max_steps=10, candidate_actions=None)
    denominator = max(row["init_count"] - row["oz_count"], 1)
    expected = (row["init_count"] - row["final_count"]) / denominator
    assert row["total_reward"] == pytest.approx(expected)


def test_make_record_and_summarize():
    import pandas as pd

    counts = {"current": 80, "oz": 100, "o3": 90}
    row = make_record(counts, 200, method="ppo", benchmark="b", seed=0, steps=3,
                      total_reward=1.0, split="test")
    assert row["pct_vs_oz"] == pytest.approx(20.0)
    assert row["pct_vs_init"] == pytest.approx(60.0)
    summary = summarize(pd.DataFrame([row]))
    assert summary.loc[0, "beats_oz_rate"] == 1.0
    assert summary.loc[0, "geomean_ratio_vs_oz"] == pytest.approx(1.25)


def test_splits_do_not_overlap():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    for backend in ("compilergym", "mock"):
        splits = get_splits(config, backend)
        train = set(splits["train"])
        assert train.isdisjoint(splits["validation"])
        assert train.isdisjoint(splits["test"])
        assert train.isdisjoint(splits.get("shift", []))


def test_overlapping_splits_are_rejected():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    config["test_benchmarks"] = config["train_benchmarks"][:1]
    with pytest.raises(ValueError):
        get_splits(config, "compilergym")


def test_features_are_finite_and_log_scaled():
    env = CompilerPassEnv([URI], backend="mock")
    obs, _ = env.reset()
    assert np.all(np.isfinite(obs)) and obs[:AUTOPHASE_DIM].max() < 20  # log1p, not raw counts
    assert instruction_counts(env.raw_env)["current"] == env.initial_count
