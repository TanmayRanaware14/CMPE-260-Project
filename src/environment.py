
import numpy as np


def make_compilergym_env(
    benchmark,
    environment_id="llvm-v0",
    observation_space="Autophase",
    reward_space="IrInstructionCountOz",
):
    """Create a CompilerGym LLVM environment."""
    try:
        import compiler_gym
    except ImportError as exc:
        raise RuntimeError(
            "CompilerGym is not installed. Check the README and "
            "the installed Python version."
        ) from exc

    return compiler_gym.make(
        environment_id,
        benchmark=benchmark,
        observation_space=observation_space,
        reward_space=reward_space,
    )


def reset_compat(env, benchmark=None, seed=None):
    """Normalize old Gym and newer Gymnasium reset results."""
    kwargs = {}

    if benchmark is not None:
        kwargs["benchmark"] = benchmark

    if seed is not None:
        kwargs["seed"] = seed

    result = env.reset(**kwargs)

    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
    else:
        observation, info = result, {}

    return np.asarray(observation, dtype=np.float32), info


def step_compat(env, action):
    """Normalize four-value and five-value step APIs."""
    result = env.step(int(action))

    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = terminated or truncated
    elif len(result) == 4:
        observation, reward, done, info = result
    else:
        raise RuntimeError(
            f"Unexpected env.step() result length: {len(result)}"
        )

    return (
        np.asarray(observation, dtype=np.float32),
        float(reward),
        bool(done),
        info or {},
    )


def action_count(env):
    """Return the number of actions for a discrete action space."""
    count = getattr(env.action_space, "n", None)

    if count is None:
        raise RuntimeError(
            "The environment does not expose a discrete action space "
            "with an .n attribute. Inspect its action-space API."
        )

    return int(count)