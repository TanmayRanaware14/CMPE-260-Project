
def train_ppo(
    env,
    total_timesteps=10_000,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    seed=7,
    verbose=1,
    gamma=0.99,
    ent_coef=0.0,
    callback=None,
):
    """Train a PPO policy using Stable-Baselines3."""
    from stable_baselines3 import PPO

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        seed=seed,
        verbose=verbose,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    return model
