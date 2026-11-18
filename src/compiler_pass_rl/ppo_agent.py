"""PPO training via Stable-Baselines3."""


def train_ppo(
    env,
    total_timesteps=30_000,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    seed=7,
    ent_coef=0.01,
    gamma=0.99,
    net_arch=(128, 128),
    tensorboard_log=None,
    verbose=1,
):
    """Train a PPO policy (MLP actor-critic) on ``env``.

    ``ent_coef`` > 0 keeps the policy exploring; with ~124 discrete actions
    PPO collapses to a single repeated action very easily without it.
    """
    from stable_baselines3 import PPO

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        policy_kwargs={"net_arch": list(net_arch)},
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def load_ppo(path):
    """Load a saved PPO model."""
    from stable_baselines3 import PPO

    return PPO.load(str(path))
