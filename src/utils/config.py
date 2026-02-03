import os
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return payload or {}


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    config_path: str | None = None, default_path: str | None = DEFAULT_CONFIG_PATH
) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if default_path and os.path.exists(default_path):
        config = _load_yaml(default_path)
    if config_path:
        config_path = os.path.abspath(config_path)
        overrides = _load_yaml(config_path)
        config = _deep_update(config, overrides)
    return config
