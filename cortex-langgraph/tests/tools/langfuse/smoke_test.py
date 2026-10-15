"""Check that the self-hosted Langfuse is up and stores traces (roadmap 5.2).

  1. auth_check(): the server answers and accepts the project keys.
  2. Sends one small trace (no LLM call, nothing from the book), the way the app will.
  3. Reads it back through the public API, which shows the worker processed it into ClickHouse.
     Langfuse v4 runs in "events_only" mode, where the old /traces endpoint is gone: traces are
     read as observations (api.observations.get_many).

Keys and URL come from tests/tools/langfuse/.env, so this works before the app's .env is touched.

Usage (from cortex-langgraph/, with the containers running):
    .venv/bin/python tests/tools/langfuse/smoke_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import dotenv_values

HERE = Path(__file__).parent
BASE_URL = "http://localhost:3002"


def main() -> None:
    from langfuse import Langfuse

    env = dotenv_values(HERE / ".env")
    lf = Langfuse(public_key=env["LANGFUSE_PUBLIC_KEY"], secret_key=env["LANGFUSE_SECRET_KEY"],
                  base_url=BASE_URL)
    if not lf.auth_check():
        sys.exit(f"auth_check failed: is Langfuse running on {BASE_URL}, and are the keys right?")
    print(f"1. auth_check ok ({BASE_URL})")

    with lf.start_as_current_observation(name="cortex-smoke-test", as_type="span",
                                         input={"question": "smoke test"}) as span:
        span.update(output={"answer": "ok"})
        trace_id = lf.get_current_trace_id()
    lf.flush()
    print(f"2. trace sent: {trace_id}")

    for _ in range(30):  # ingestion is asynchronous: web → Redis queue → worker → ClickHouse
        found = lf.api.observations.get_many(trace_id=trace_id, fields="core,basic,io").data
        if found:
            break
        time.sleep(2)
    else:
        sys.exit("3. trace not found after 60 s: check the langfuse-worker logs")
    obs = found[0]
    print(f"3. trace stored: {len(found)} observation(s), name={obs.name!r}, "
          f"input={obs.input!r}, output={obs.output!r}")
    print(f"   open it: {lf.get_trace_url(trace_id=trace_id)}")
    lf.shutdown()


if __name__ == "__main__":
    main()
