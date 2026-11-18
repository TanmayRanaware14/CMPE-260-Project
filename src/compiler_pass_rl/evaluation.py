"""Metrics, PPO evaluation and result summaries.

Primary metric
--------------
``pct_vs_oz`` = 100 * (oz_count - final_count) / oz_count, where the counts are
IR instruction counts. Positive means the final program has FEWER instructions
than what LLVM's -Oz produces; 0 means parity; negative means worse.

This is an instruction-count metric only. It is not runtime and not binary size.
"""

import numpy as np
import pandas as pd

from .environment import instruction_counts


def make_record(counts, init_count, *, method, benchmark, seed, steps,
                total_reward, split="", actions=None):
    """Build one result row from instruction counts."""
    final = counts["current"]
    oz = counts["oz"]
    o3 = counts["o3"]
    return {
        "split": split,
        "benchmark": benchmark,
        "method": method,
        "seed": seed,
        "steps": steps,
        "total_reward": float(total_reward),
        "init_count": int(init_count),
        "final_count": int(final),
        "oz_count": int(oz),
        "o3_count": int(o3),
        "pct_vs_init": 100.0 * (init_count - final) / init_count if init_count else 0.0,
        "pct_vs_oz": 100.0 * (oz - final) / oz if oz else 0.0,
        "pct_vs_o3": 100.0 * (o3 - final) / o3 if o3 else 0.0,
        "ratio_vs_oz": oz / final if final else float("nan"),
        "actions": " ".join(str(a) for a in actions) if actions is not None else "",
    }


def evaluate_ppo(model, env, benchmarks, *, split="", seed=0, deterministic=True,
                 episodes=1, method="ppo"):
    """Evaluate a trained PPO policy on each benchmark.

    A deterministic policy on a fixed program produces identical episodes, so
    ``episodes`` only matters when ``deterministic=False``.
    ``env`` must be a ``CompilerPassEnv``.
    """
    records = []
    for benchmark in benchmarks:
        for _ in range(episodes):
            observation, _ = env.reset(options={"benchmark": benchmark})
            total_reward, actions = 0.0, []
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=deterministic)
                action = int(action)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                actions.append(action)
            records.append(
                make_record(
                    instruction_counts(env.raw_env),
                    env.initial_count,
                    method=method,
                    benchmark=benchmark,
                    seed=seed,
                    steps=len(actions),
                    total_reward=total_reward,
                    split=split,
                    actions=actions,
                )
            )
    return records


def _geomean(values):
    values = np.asarray([v for v in values if v > 0 and np.isfinite(v)], dtype=float)
    return float(np.exp(np.log(values).mean())) if len(values) else float("nan")


def summarize(frame, by=("split", "method")):
    """Aggregate results: mean/std of pct_vs_oz, geomean ratio vs -Oz, win rate."""
    by = list(by)
    grouped = frame.groupby(by)
    summary = grouped.agg(
        n=("pct_vs_oz", "size"),
        mean_pct_vs_oz=("pct_vs_oz", "mean"),
        std_pct_vs_oz=("pct_vs_oz", "std"),
        mean_pct_vs_init=("pct_vs_init", "mean"),
        mean_steps=("steps", "mean"),
    )
    summary["geomean_ratio_vs_oz"] = grouped["ratio_vs_oz"].apply(_geomean)
    summary["beats_oz_rate"] = grouped["pct_vs_oz"].apply(lambda s: float((s > 0).mean()))
    return summary.reset_index()


def generalization_gap(frame, method="ppo", train_split="train", test_split="test"):
    """Mean pct_vs_oz on the train split minus the held-out split."""
    rows = frame[frame["method"] == method]
    train = rows[rows["split"] == train_split]["pct_vs_oz"]
    test = rows[rows["split"] == test_split]["pct_vs_oz"]
    if train.empty or test.empty:
        return float("nan")
    return float(train.mean() - test.mean())


def load_results(*paths):
    """Concatenate result CSVs that exist."""
    frames = [pd.read_csv(p) for p in paths if p is not None and _exists(p)]
    if not frames:
        raise FileNotFoundError("No result CSVs found: " + ", ".join(map(str, paths)))
    return pd.concat(frames, ignore_index=True)


def _exists(path):
    from pathlib import Path

    return Path(path).exists()
