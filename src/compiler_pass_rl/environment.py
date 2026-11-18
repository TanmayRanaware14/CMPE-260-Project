"""CompilerGym integration layer.

Everything that touches CompilerGym's (archived, Gym-era) API lives here so
the baselines, PPO code and evaluation never depend on its quirks:

* ``make_compilergym_env``  - build the raw environment (real or mock backend)
* ``reset_compat`` / ``step_compat`` - normalize Gym vs Gymnasium return formats
* ``instruction_counts``    - read current / -Oz / -O3 IR instruction counts
* ``CompilerPassEnv``       - Gymnasium wrapper for Stable-Baselines3 that
                              samples a program from a *set* of benchmarks on
                              every reset and normalizes Autophase features
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

AUTOPHASE_DIM = 56


# --------------------------------------------------------------------------
# Raw environment construction and API compatibility helpers
# --------------------------------------------------------------------------
def make_compilergym_env(
    benchmark,
    environment_id="llvm-v0",
    observation_space="Autophase",
    reward_space="IrInstructionCountOz",
    backend="compilergym",
):
    """Create a raw LLVM environment.

    backend="compilergym" uses the real library; backend="mock" uses the
    dependency-free simulator in ``mock_env`` (for tests and development only).
    """
    if backend == "mock":
        from .mock_env import MockLLVMEnv

        return MockLLVMEnv(
            benchmark=benchmark,
            observation_space=observation_space,
            reward_space=reward_space,
        )
    if backend != "compilergym":
        raise ValueError(f"Unknown backend: {backend!r}")
    try:
        import compiler_gym  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "CompilerGym is not installed. See the README for installation "
            "notes, or pass --mock to run the pipeline on the mock backend."
        ) from exc
    return compiler_gym.make(
        environment_id,
        benchmark=benchmark,
        observation_space=observation_space,
        reward_space=reward_space,
    )


def reset_compat(env, benchmark=None, seed=None):
    """Reset and always return ``(observation, info)``.

    Handles old Gym (returns observation only) and Gymnasium (returns a
    2-tuple). ``seed`` is passed through only if the environment accepts it.
    """
    kwargs = {}
    if benchmark is not None:
        kwargs["benchmark"] = benchmark
    if seed is not None:
        kwargs["seed"] = seed
    try:
        result = env.reset(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        result = env.reset(**kwargs)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result
    return result, {}


def step_compat(env, action):
    """Step and always return ``(observation, reward, done, info)``."""
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    elif len(result) == 4:
        observation, reward, done, info = result
        done = bool(done)
    else:
        raise RuntimeError(f"Unexpected step() result of length {len(result)}")
    reward = 0.0 if reward is None else float(reward)
    return observation, reward, done, info


def action_count(env):
    """Number of discrete actions (optimization passes) available."""
    space = env.action_space
    n = getattr(space, "n", None)
    if n is None and hasattr(space, "flags"):
        n = len(space.flags)
    if n is None:
        raise RuntimeError(
            "Action space is not discrete; this project assumes a discrete "
            "pass-selection action space."
        )
    return int(n)


def instruction_counts(env):
    """Current, -Oz and -O3 IR instruction counts for the loaded program."""
    obs = env.observation
    return {
        "current": int(obs["IrInstructionCount"]),
        "oz": int(obs["IrInstructionCountOz"]),
        "o3": int(obs["IrInstructionCountO3"]),
    }


# --------------------------------------------------------------------------
# Gymnasium wrapper for Stable-Baselines3
# --------------------------------------------------------------------------
class CompilerPassEnv(gym.Env):
    """Gymnasium environment over a *set* of benchmark programs.

    Observation (float32 vector):
        log1p(Autophase)                       56 values
        fraction of the step budget used        1 value
        normalized action histogram             n_actions values (optional)

    ``log1p`` compresses the raw instruction counts, which otherwise span
    several orders of magnitude and train poorly with an MLP policy.

    Reward: the CompilerGym reward-space value for each step, optionally
    clipped to ``[-reward_clip, reward_clip]`` to tame programs where the
    normalizing denominator is tiny. Clipping only affects the training
    signal; evaluation reads true instruction counts.

    ``reset(options={"benchmark": uri})`` forces a specific program.
    Otherwise a program is drawn from ``benchmarks`` using ``sampling``:
    "random" (default) or "cycle".
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        benchmarks,
        environment_id="llvm-v0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
        max_episode_steps=20,
        backend="compilergym",
        use_action_histogram=True,
        reward_clip=5.0,
        sampling="random",
        seed=0,
    ):
        super().__init__()
        if not benchmarks:
            raise ValueError("At least one benchmark is required.")
        if sampling not in ("random", "cycle"):
            raise ValueError("sampling must be 'random' or 'cycle'")
        self.benchmarks = list(benchmarks)
        self.max_episode_steps = int(max_episode_steps)
        self.use_action_histogram = use_action_histogram
        self.reward_clip = reward_clip
        self.sampling = sampling
        self._rng = np.random.default_rng(seed)
        self._cycle_index = 0
        self.elapsed_steps = 0
        self.benchmark = self.benchmarks[0]
        self.initial_count = None

        self.raw_env = make_compilergym_env(
            benchmark=self.benchmarks[0],
            environment_id=environment_id,
            observation_space=observation_space,
            reward_space=reward_space,
            backend=backend,
        )
        self.n_actions = action_count(self.raw_env)
        self.action_space = spaces.Discrete(self.n_actions)
        self._action_counts = np.zeros(self.n_actions, dtype=np.float32)

        obs_dim = AUTOPHASE_DIM + 1 + (self.n_actions if use_action_histogram else 0)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # -- helpers ----------------------------------------------------------
    def _pick_benchmark(self, options):
        if options and options.get("benchmark") is not None:
            return options["benchmark"]
        if self.sampling == "cycle":
            uri = self.benchmarks[self._cycle_index % len(self.benchmarks)]
            self._cycle_index += 1
            return uri
        return self.benchmarks[int(self._rng.integers(0, len(self.benchmarks)))]

    def _featurize(self, raw_observation):
        autophase = np.asarray(raw_observation, dtype=np.float64).reshape(-1)
        if autophase.shape[0] != AUTOPHASE_DIM:
            raise ValueError(
                f"Expected {AUTOPHASE_DIM} Autophase features, got {autophase.shape[0]}"
            )
        parts = [
            np.log1p(np.maximum(autophase, 0.0)),
            np.array([self.elapsed_steps / self.max_episode_steps]),
        ]
        if self.use_action_histogram:
            parts.append(self._action_counts / self.max_episode_steps)
        return np.concatenate(parts).astype(np.float32)

    # -- Gymnasium API ----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.benchmark = self._pick_benchmark(options)
        self.elapsed_steps = 0
        self._action_counts[:] = 0.0
        raw_observation, _ = reset_compat(self.raw_env, benchmark=self.benchmark)
        self.initial_count = instruction_counts(self.raw_env)["current"]
        return self._featurize(raw_observation), {"benchmark": self.benchmark}

    def step(self, action):
        action = int(action)
        raw_observation, reward, done, info = step_compat(self.raw_env, action)
        self.elapsed_steps += 1
        self._action_counts[action] += 1.0
        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        terminated = bool(done)
        truncated = (not terminated) and self.elapsed_steps >= self.max_episode_steps
        info = dict(info) if isinstance(info, dict) else {}
        info["benchmark"] = self.benchmark
        return self._featurize(raw_observation), reward, terminated, truncated, info

    def close(self):
        self.raw_env.close()
