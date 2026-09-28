# Evaluation tests

These are **not** unit tests. They call real models, cost money, need API
keys, and are non-deterministic. They are deselected by default via
`addopts = -m "not eval"` in `pytest.ini`.

```bash
# the normal suite (hermetic, no keys, no network)
pytest

# the evals
pytest -m eval -v -s

# one dimension at a time
pytest -m eval tests/evals/test_retrieval_quality.py -v -s
pytest -m eval tests/evals/test_hallucination.py -v -s
pytest -m eval tests/evals/test_latency.py -v -s
```

`-s` matters: each test prints its measured metric, and the number is the
point — the pass/fail threshold is only a floor.

## What each file answers

| File | Question |
|---|---|
| `test_retrieval_quality.py` | Does the right chunk come back, and does the re-ranker improve on raw vector order? |
| `test_hallucination.py` | Does the system refuse to answer what the corpus does not contain? |
| `test_latency.py` | Where does the wall-clock time actually go? |

## The corpus

`data/golden_set.json` describes a fictional company. That is deliberate:
if the model can answer a question about Veldrin Corp without retrieval,
it is fabricating, because no such facts exist in its training data. A
corpus of real facts cannot tell retrieval apart from memorisation.

Extend the JSON, not the test code, when adding cases.

## Thresholds

Thresholds are set as **regression floors, not targets** — deliberately
below current measured performance so that ordinary model drift does not
turn CI red, while a real regression still does. Tighten them as the
system improves. Every threshold is a module-level constant with a
comment explaining what breaking it would mean in production.
