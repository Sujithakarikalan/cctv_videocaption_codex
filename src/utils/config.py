"""Configuration file loading and basic validation."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from *path*, reporting useful errors for invalid files."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a YAML mapping: {config_path}")
    return config
