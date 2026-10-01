"""Load configuration from config.yml.

config.yml is the single source of truth for non-secret settings — in
particular, every model the app uses (see `models:`). Code reads models
through `model_settings(role)` instead of naming them, so changing a model is
a one-line config edit, not a hunt through the codebase.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yml"


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys.

    Plain YAML silently keeps the LAST value for a repeated key. That is how
    config.yml once lost its `llm_system_role`: a second `llm_config:` block
    further down replaced the first one wholesale, and nothing noticed. Failing
    at load time turns that silent data loss into an immediate, located error.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ValueError(
                    f"Duplicate key {key!r} in {CONFIG_PATH.name} "
                    f"(line {key_node.start_mark.line + 1}) — the later value would "
                    f"silently replace the earlier one."
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_config(path: Optional[Path] = None) -> dict:
    """Load configuration from src/config/config.yml (duplicate keys are an error).

    CONFIG_PATH is looked up at call time, so tests can point it at a copy.
    """
    with open(path or CONFIG_PATH, "r") as f:
        return yaml.load(f, Loader=_UniqueKeyLoader) or {}


def model_settings(role: str) -> Dict[str, Any]:
    """Settings for one model role from config.yml's `models:` section.

    Roles: "chat" (the agent), "rag" (RAG answers), "embedding", "reranker".
    Raises a clear error instead of falling back to a value hidden in code —
    a hidden fallback is exactly what let models drift from the config before.
    """
    models = load_config().get("models") or {}
    if role not in models or not (models[role] or {}).get("name"):
        raise KeyError(
            f"config.yml has no models.{role}.name — every model is configured "
            f"there (see src/config/config.yml)."
        )
    return dict(models[role])
