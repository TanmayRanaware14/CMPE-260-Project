
import numpy as np

try:
    from .environment import ir_instruction_count, reset_compat, step_compat
except ImportError:
    from environment import ir_instruction_count, reset_compat, step_compat


def _session_failure(info):
    """Return a compiler-session failure message carried in step info.

    CompilerGym ends an episode and sets info["error_details"] when a pass
    causes an unrecoverable compiler or service failure. A normal early stop
    has no such message.
    """
    if not isinstance(info, dict):
        return None

    details = info.get("error_details")
    if not details:
        return None

    error_type = info.get("error_type")
    if error_type:
        return f"{error_type}: {details}"
    return str(details)


def run_random_episode(env, max_steps=20, seed=0):
    """Apply randomly selected actions for a fixed optimization budget.

    The return value keeps the original keys. "error" is None when every
    applied pass completed, and a message when CompilerGym reported an
    unrecoverable session failure. Reward accounting is unchanged either way.
    """
    rng = np.random.default_rng(seed)
    reset_compat(env)

    total_reward = 0.0
    actions = []
    rewards = []
    error = None

    for _ in range(max_steps):
        action = int(rng.integers(0, env.action_space.n))
        _, reward, terminated, truncated, info = step_compat(env, action)

        actions.append(action)
        rewards.append(reward)
        total_reward += reward

        error = _session_failure(info)
        if error or terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": len(actions),
        "actions": actions,
        "rewards": rewards,
        "error": error,
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
        _, reward, _, _, _ = step_compat(env, int(action))

        if reward > best_reward:
            best_reward = reward
            best_action = int(action)

    return {
        "action": best_action,
        "immediate_reward": float(best_reward),
        "candidate_actions": candidates.tolist(),
    }


def run_greedy_sequence(env, max_steps=20):
    """Sequential greedy pass selection on IR instruction count.

    At each step every action is tried on a fork of the current state. The
    action with the largest immediate reduction is committed, ties going to
    the lowest action index. Search stops when no action reduces the count or
    the budget is used. "trials" counts every candidate transformation.
    """
    reset_compat(env)
    current = ir_instruction_count(env)
    actions = []
    counts = [current]
    trials = 0
    error = None

    for _ in range(max_steps):
        best_action = None
        best_count = current

        for action in range(env.action_space.n):
            trial = env.fork()
            try:
                _, _, terminated, _, info = step_compat(trial, action)
                trials += 1
                if _session_failure(info) or terminated:
                    continue
                count = ir_instruction_count(trial)
            finally:
                trial.close()

            if count < best_count:
                best_count = count
                best_action = action

        if best_action is None:
            break

        _, _, terminated, _, info = step_compat(env, best_action)
        error = _session_failure(info)
        if error:
            break

        current = ir_instruction_count(env)
        actions.append(best_action)
        counts.append(current)
        if terminated:
            break

    return {
        "steps": len(actions),
        "actions": actions,
        "counts": counts,
        "final_count": current,
        "trials": trials,
        "error": error,
    }
