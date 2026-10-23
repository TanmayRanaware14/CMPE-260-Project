
import numpy as np

from .environment import reset_compat, step_compat


def run_random_episode(env, max_steps=20, seed=0):
    """Apply randomly selected actions for a fixed optimization budget."""
    rng = np.random.default_rng(seed)
    reset_compat(env)

    total_reward = 0.0
    actions = []
    rewards = []

    for _ in range(max_steps):
        action = int(rng.integers(0, env.action_space.n))
        _, reward, done, _ = step_compat(env, action)

        actions.append(action)
        rewards.append(reward)
        total_reward += reward

        if done:
            break

    return {
        "total_reward": total_reward,
        "steps": len(actions),
        "actions": actions,
        "rewards": rewards,
    }


def run_greedy_first_action(env, candidate_actions=8, seed=0):
    """Choose the best immediate-reward action from sampled candidates.

    This is a one-step baseline, not a full sequential greedy policy.
    Each candidate is tested from the initial benchmark state.
    """
    rng = np.random.default_rng(seed)
    action_count = env.action_space.n

    candidates = rng.choice(
        action_count,
        size=min(candidate_actions, action_count),
        replace=False,
    )

    best_action = None
    best_reward = -float("inf")

    for action in candidates:
        reset_compat(env)
        _, reward, _, _ = step_compat(env, int(action))

        if reward > best_reward:
            best_reward = reward
            best_action = int(action)

    return {
        "action": best_action,
        "immediate_reward": float(best_reward),
        "candidate_actions": candidates.tolist(),
    }