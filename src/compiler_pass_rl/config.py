"""Configuration loading and benchmark splits."""

import json
import random
from pathlib import Path

import numpy as np


def load_config(path="configs/default.json"):
    """Load experiment settings from a JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_config_snapshot(config, path, extra=None):
    """Write the config actually used for a run (plus extras) next to its results."""
    snapshot = dict(config)
    if extra:
        snapshot.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2, sort_keys=True)


def get_splits(config, backend="compilergym"):
    """Return ``{split_name: [benchmark URIs]}`` for the chosen backend.

    Real backend: lists from the config ("train", "validation", "test", and
    optionally "shift" for a different benchmark family).
    Mock backend: synthetic programs; "shift" is an unseen program family.
    """
    if backend == "mock":
        return {
            "train": [f"benchmark://mock-v0/{i}" for i in range(0, 12)],  # families 0-2
            "validation": [f"benchmark://mock-v0/{i}" for i in range(12, 14)],
            "test": [f"benchmark://mock-v0/{i}" for i in range(14, 16)],
            "shift": [f"benchmark://mock-v0/{i}" for i in range(12, 16)],
        }
    splits = {
        "train": config["train_benchmarks"],
        "validation": config["validation_benchmarks"],
        "test": config["test_benchmarks"],
    }
    if config.get("shift_benchmarks"):
        splits["shift"] = config["shift_benchmarks"]
    overlap = set(splits["train"]) & (
        set(splits["validation"]) | set(splits["test"]) | set(splits.get("shift", []))
    )
    if overlap:
        raise ValueError(f"Train benchmarks overlap with held-out splits: {sorted(overlap)}")
    return splits


def set_global_seeds(seed):
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
