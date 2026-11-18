"""Baseline pass-ordering methods.

All baselines share the PPO agent's budget (``max_steps`` actions per program)
and are scored with the same instruction-count metric (see ``evaluation``).

Methods
-------
* random            - uniformly random actions (one episode per seed)
* random_search_N   - best of N random episodes (a stronger, budget-aware baseline)
* greedy            - at each step, try candidate actions on forked copies of the
                      environment and keep the best immediate reward; stop early
                      when no candidate improves the program
* llvm_oz / llvm_o3 - the compiler's own pipelines (reference only: they use far
                      more than ``max_steps`` passes, so they are NOT budget-matched)
"""

import numpy as np

from .environment import action_count, instruction_counts, reset_compat, step_compat
from .evaluation import make_record


def run_random_episode(env, benchmark, max_steps=20, seed=0, split=""):
    """One episode of uniformly random actions."""
    rng = np.random.default_rng(seed)
    reset_compat(env, benchmark=benchmark)
    init = instruction_counts(env)["current"]
    n = action_count(env)
    total, actions = 0.0, []
    for _ in range(max_steps):
        action = int(rng.integers(0, n))
        _, reward, done, _ = step_compat(env, action)
        total += reward
        actions.append(action)
        if done:
            break
    return make_record(instruction_counts(env), init, method="random",
                       benchmark=benchmark, seed=seed, steps=len(actions),
                       total_reward=total, split=split, actions=actions)


def run_random_search(env, benchmark, max_steps=20, trials=16, seed=0, split=""):
    """Best-of-N random episodes, judged by final instruction count.

    Note: this uses ``trials`` x ``max_steps`` environment steps per program,
    far more than PPO needs at inference time. Report that when comparing.
    """
    rng = np.random.default_rng(seed)
    n = action_count(env)
    best = None
    for _ in range(trials):
        reset_compat(env, benchmark=benchmark)
        init = instruction_counts(env)["current"]
        total, actions = 0.0, []
        for _ in range(max_steps):
            action = int(rng.integers(0, n))
            _, reward, done, _ = step_compat(env, action)
            total += reward
            actions.append(action)
            if done:
                break
        counts = instruction_counts(env)
        if best is None or counts["current"] < best[0]["current"]:
            best = (counts, init, total, actions)
    counts, init, total, actions = best
    return make_record(counts, init, method=f"random_search_{trials}",
                       benchmark=benchmark, seed=seed, steps=len(actions),
                       total_reward=total, split=split, actions=actions)


def run_greedy(env, benchmark, max_steps=20, candidate_actions=None, seed=0, split=""):
    """Step-wise greedy search using ``env.fork()`` to try actions.

    ``candidate_actions``: None tries every action at every step (strongest,
    slowest); an int samples that many actions per step.
    """
    rng = np.random.default_rng(seed)
    reset_compat(env, benchmark=benchmark)
    init = instruction_counts(env)["current"]
    n = action_count(env)
    total, actions = 0.0, []
    for _ in range(max_steps):
        if candidate_actions is None or candidate_actions >= n:
            candidates = range(n)
        else:
            candidates = rng.choice(n, size=candidate_actions, replace=False)
        best_action, best_reward = None, 0.0
        for action in candidates:
            fork = env.fork()
            try:
                _, reward, _, _ = step_compat(fork, int(action))
            finally:
                fork.close()
            if reward > best_reward:
                best_action, best_reward = int(action), reward
        if best_action is None:
            break  # no candidate improves the program
        _, reward, done, _ = step_compat(env, best_action)
        total += reward
        actions.append(best_action)
        if done:
            break
    return make_record(instruction_counts(env), init, method="greedy",
                       benchmark=benchmark, seed=seed, steps=len(actions),
                       total_reward=total, split=split, actions=actions)


def run_compiler_references(env, benchmark, split=""):
    """Rows for LLVM's own -Oz and -O3 pipelines (not budget-matched)."""
    reset_compat(env, benchmark=benchmark)
    counts = instruction_counts(env)
    init = counts["current"]
    rows = []
    for name, key in (("llvm_oz", "oz"), ("llvm_o3", "o3")):
        reference = {"current": counts[key], "oz": counts["oz"], "o3": counts["o3"]}
        rows.append(make_record(reference, init, method=name, benchmark=benchmark,
                                seed=0, steps=0, total_reward=0.0, split=split))
    return rows
