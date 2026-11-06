
def evaluate_ppo_on_benchmark(model, env, benchmark):
    """Run one deterministic PPO episode on a benchmark.

    env must be a CompilerGymWrapper built with normalized_observation=True
    so that each step reports the raw IR instruction count in info.
    """
    observation, info = env.reset(options={"benchmark": benchmark})
    initial = info["ir_instruction_count"]
    final = initial
    actions = []

    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        actions.append(int(action))
        final = info.get("ir_instruction_count", final)
        if terminated or truncated:
            break

    return {
        "benchmark": benchmark,
        "initial_count": initial,
        "final_count": final,
        "actions": actions,
        "error": info.get("error_details"),
    }


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
