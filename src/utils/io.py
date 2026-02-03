import json
from typing import List


def load_texts(path: str) -> List[str]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of strings.")
        return [t if isinstance(t, str) else "" for t in data]

    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]
