"""Load configuration from config.yml."""

from pathlib import Path

import yaml


def load_config() -> dict:
    """Load configuration from src/config/config.yml."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


class Config:
    def __init__(self):
        with open Path(__file__).parent.parent.parent / "config.yml", "r") as f:
            self._config = yaml.safe_load(f)
        self.openai_api_key = self._config["openai_api_key"]
        