
def evaluate_ppo(model, env, episodes=5, max_steps=20):
    """Evaluate PPO deterministically over multiple episodes."""
    results = []

    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                break

        results.append({
            "episode": episode,
            "benchmark": getattr(env, "benchmark", "unknown"),
            "total_reward": total_reward,
            "steps": steps,
        })

    return results
