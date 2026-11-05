
import compiler_gym
import gymnasium as gym
import numpy as np


def make_compilergym_env(
    benchmark=None,
    observation_space="Autophase",
    reward_space="IrInstructionCountOz",
):
    """Create and configure a CompilerGym LLVM environment."""

    env = compiler_gym.make("llvm-v0")

    if benchmark is not None:
        env.benchmark = benchmark

    env.observation_space = observation_space
    env.reward_space = reward_space

    return env


def reset_compat(env):
    """Handle CompilerGym and Gymnasium reset return formats."""

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


class CompilerGymWrapper(gym.Env):
    """Adapt CompilerGym to the Gymnasium interface used by SB3 PPO."""

    def __init__(
        self,
        benchmark,
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
        max_steps=20,
    ):
        super().__init__()

        self.env = make_compilergym_env(
            benchmark=benchmark,
            observation_space=observation_space,
            reward_space=reward_space,
        )

        self.max_steps = max_steps
        self.current_step = 0

        self.action_space = gym.spaces.Discrete(action_count(self.env))

        observation_shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shape,
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation, info = reset_compat(self.env)
        self.current_step = 0

        return np.asarray(observation, dtype=np.float32), info

    def step(self, action):
        observation, reward, terminated, truncated, info = step_compat(
            self.env, action
        )

        self.current_step += 1

        if self.current_step >= self.max_steps:
            truncated = True

        observation = np.asarray(observation, dtype=np.float32)

        return observation, float(reward), terminated, truncated, info

    def close(self):
        self.env.close()
