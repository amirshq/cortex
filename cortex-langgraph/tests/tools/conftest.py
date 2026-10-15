"""Shared setup for the third-party tool integrations (Langfuse, RAGAS, DeepEval, Grafana).

Every test collected under tests/tools/ is marked `tools` automatically, and pytest.ini
deselects that marker by default: these tests call real models and external services, and
need the optional packages in tests/tools/requirements.txt. Run them with:

    .venv/bin/python -m pytest -m tools

Collection still imports deselected modules, so a test file here must start with
`pytest.importorskip("<package>")` for the tool it uses. Without it, a plain `pytest`
run breaks on a machine where the tool isn't installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values, load_dotenv

load_dotenv()
# .env leaves the unused provider settings empty (AZURE_OPENAI_ENDPOINT=, ...). The app reads
# empty as "not set", but DeepEval validates every setting at import and rejects an empty URL.
# Dropping the empty ones is the same thing to the app, and DeepEval is told not to re-read .env.
for key, value in dotenv_values().items():
    if not value:
        os.environ.pop(key, None)
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")  # no usage telemetry to DeepEval's servers

TOOLS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    for item in items:
        if TOOLS_DIR in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.tools)
