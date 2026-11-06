
import compiler_gym
import gymnasium as gym
import numpy as np


def make_compilergym_env(
    benchmark=None,
    environment_id="llvm-v0",
    observation_space="Autophase",
    reward_space="IrInstructionCountOz",
):
    """Create and configure a CompilerGym LLVM environment."""

    env = compiler_gym.make(environment_id)

    if benchmark is not None:
        env.benchmark = benchmark

    env.observation_space = observation_space
    env.reward_space = reward_space

    return env


def reset_compat(env, benchmark=None):
    """Handle CompilerGym and Gymnasium reset return formats."""

    if benchmark is not None:
        env.benchmark = benchmark

    result = env.reset()

    if isinstance(result, tuple):
        observation, info = result
        return observation, info

    return result, {}


def step_compat(env, action):
    """Handle old Gym and new Gymnasium step return formats."""

    result = env.step(action)

    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    else:
        observation, reward, done, info = result
        terminated = done
        truncated = False

    return observation, reward, terminated, truncated, info


def action_count(env):
    """Return the number of available compiler optimization actions."""

    return env.action_space.n


def ir_instruction_count(env):
    """Return the raw IrInstructionCount of the current program state."""

    return int(np.asarray(env.observation["IrInstructionCount"]).reshape(-1)[0])


class CompilerGymWrapper(gym.Env):
    """Adapt CompilerGym to the Gymnasium interface used by SB3 PPO.

    benchmark may be one benchmark id or a list of ids. With a list, each
    reset samples one benchmark uniformly with the environment's RNG.

    With normalized_observation=True the observation is log1p(Autophase)
    followed by the current instruction count divided by the initial count
    and the fraction of the pass budget remaining.

    With normalized_reward=True the step reward is
    (I[t-1] - I[t]) / max(I[0], 1) computed from IrInstructionCount, so the
    undiscounted return equals the fractional reduction from the original
    program. Otherwise the CompilerGym reward_space value is returned.
    """

    def __init__(
        self,
        benchmark,
        environment_id="llvm-v0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
        max_steps=20,
        normalized_observation=False,
        normalized_reward=False,
    ):
        super().__init__()

        if isinstance(benchmark, (list, tuple)):
            self.benchmarks = list(benchmark)
        else:
            self.benchmarks = [benchmark]
        if not self.benchmarks:
            raise ValueError("CompilerGymWrapper needs at least one benchmark.")

        self.env = make_compilergym_env(
            benchmark=self.benchmarks[0],
            environment_id=environment_id,
            observation_space=observation_space,
            reward_space=reward_space,
        )

        self.max_steps = max_steps
        self.normalized_observation = normalized_observation
        self.normalized_reward = normalized_reward
        self.current_step = 0
        self.benchmark = self.benchmarks[0]
        self.initial_count = None
        self.previous_count = None

        self.action_space = gym.spaces.Discrete(action_count(self.env))

        observation_shape = self.env.observation_space.shape
        if normalized_observation:
            observation_shape = (int(np.prod(observation_shape)) + 2,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shape,
            dtype=np.float32,
        )

    def _observation(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        if not self.normalized_observation:
            return raw

        features = np.log1p(np.maximum(raw.reshape(-1), 0.0))
        count_ratio = self.previous_count / max(self.initial_count, 1)
        budget_left = 1.0 - self.current_step / self.max_steps
        extra = np.array([count_ratio, budget_left], dtype=np.float32)
        return np.concatenate([features, extra]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        benchmark = None
        if options and "benchmark" in options:
            benchmark = options["benchmark"]
        elif len(self.benchmarks) > 1:
            benchmark = self.benchmarks[
                int(self.np_random.integers(len(self.benchmarks)))
            ]
        else:
            benchmark = self.benchmarks[0]

        observation, info = reset_compat(self.env, benchmark=benchmark)
        self.benchmark = benchmark
        self.current_step = 0
        self.initial_count = ir_instruction_count(self.env)
        self.previous_count = self.initial_count

        info = dict(info or {})
        info["benchmark"] = benchmark
        info["ir_instruction_count"] = self.initial_count
        return self._observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = step_compat(
            self.env, int(action)
        )

        self.current_step += 1
        info = dict(info or {})

        if self.normalized_reward or self.normalized_observation:
            if terminated and info.get("error_details"):
                count = self.previous_count
            else:
                count = ir_instruction_count(self.env)
            if self.normalized_reward:
                reward = (self.previous_count - count) / max(self.initial_count, 1)
            self.previous_count = count
            info["ir_instruction_count"] = count

        if self.current_step >= self.max_steps:
            truncated = True

        if observation is None:
            observation = np.zeros(self.env.observation_space.shape, dtype=np.float32)

        return self._observation(observation), float(reward or 0.0), bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()
