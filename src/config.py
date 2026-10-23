import json
from pathlib import Path


def load_config(path="configs/default.json"):
    """Load experiment settings from a JSON file."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)