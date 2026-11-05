
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from environment import (
    make_compilergym_env,
    reset_compat,
)


BENCHMARKS = [
    "cbench-v1/qsort",
    "cbench-v1/crc32",
    "cbench-v1/adpcm",
    "cbench-v1/bitcount",
]


def main():
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    rows = []

    env = make_compilergym_env(
        benchmark=BENCHMARKS[0],
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
    )

    try:
        for benchmark in BENCHMARKS:
            try:
                observation, _ = reset_compat(
                    env, benchmark=benchmark
                )
                features = observation.flatten()

                row = {
                    "benchmark": benchmark,
                    "num_features": len(features),
                }

                for index, value in enumerate(features):
                    row[f"feature_{index}"] = float(value)

                rows.append(row)

                print(f"\n{benchmark}")
                print("Feature count:", len(features))
                print("First 10 features:", features[:10])

            except Exception as exc:
                print(f"Could not load {benchmark}: {exc}")

    finally:
        env.close()

    if not rows:
        raise RuntimeError(
            "No benchmarks loaded. Check the environment and benchmark URIs."
        )

    max_features = max(row["num_features"] for row in rows)
    columns = ["benchmark", "num_features"] + [
        f"feature_{i}" for i in range(max_features)
    ]

    with (output_dir / "initial_features.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nSaved {len(rows)} observations to "
        "results/initial_features.csv"
    )


if __name__ == "__main__":
    main()
