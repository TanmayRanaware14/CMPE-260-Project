
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_returns(csv_path, output_path):
    """Plot episode rewards from a CSV."""
    data = pd.read_csv(csv_path)

    required = {"episode", "total_reward"}
    if not required.issubset(data.columns):
        raise ValueError(
            f"CSV must contain these columns: {required}"
        )

    plt.figure()
    plt.plot(
        data["episode"],
        data["total_reward"],
        marker="o",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("PPO Evaluation Returns")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
