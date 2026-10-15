"""Experiment 3.7: compare vector databases on the SAME vectors.

A vector database decides how fast (and with how much memory and disk) the
nearest chunks are found, not which chunks are nearest: that is the embedding
model's job. So this is a benchmark, not a golden-set run with the LLM:

  1. Load the chunk vectors already stored in an experiment index (no re-embedding)
     and embed the golden questions once.
  2. Compute the EXACT top-k for every question by brute force: the reference.
  3. Build each database from those vectors and measure:
       build time, disk size, search latency (warm, p50 / p95),
       overlap@k  how much of the exact top-k each database returns (approximate
                  indexes like HNSW can miss neighbours),
       golden recall@1/@5/@k and MRR, matched by evidence text as in eval_golden.py.

Databases:
  chroma        what the app uses (HNSW, persistent)
  faiss-flat    FAISS exact inner-product search (in memory)
  faiss-hnsw    FAISS HNSW (in memory, approximate)
  qdrant-local  qdrant-client's embedded local mode. NOTE: local mode is a
                pure-Python, brute-force fallback for development; a Qdrant
                SERVER (Docker/cloud) uses HNSW and is much faster.

--scale N pads the real vectors with random unit vectors up to N, to see how
latency grows beyond one book (golden metrics stay meaningful: the padding
never matches an evidence quote).

Usage (from cortex-langgraph/):
    .venv/bin/python tests/evals/bench_vector_db.py --persist-dir data/experiments/embed-large
    .venv/bin/python tests/evals/bench_vector_db.py --persist-dir data/experiments/embed-large --scale 50000

Needs: pip install faiss-cpu qdrant-client (evaluation only; the app doesn't use them).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_golden import ROOT, evidence_hits, first_hit  # noqa: E402  (also loads .env and the project path)

import numpy as np  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from src.business.core.embedding import create_embedder  # noqa: E402

REPEATS = 20  # timed searches per question, after one warm-up


# ── databases: each returns (search(query_vector) -> list of row ids, disk_dir or None) ──
def build_chroma(X, workdir):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=str(workdir), settings=Settings(anonymized_telemetry=False))
    col = client.create_collection("bench")  # same defaults as the app's collection
    step = client.get_max_batch_size()
    for i in range(0, len(X), step):
        col.add(ids=[str(j) for j in range(i, min(i + step, len(X)))], embeddings=X[i:i + step])
    return lambda q, k: [int(x) for x in col.query(query_embeddings=[q], n_results=k)["ids"][0]], workdir


def build_faiss(X, workdir, hnsw: bool):
    import faiss

    # One thread: we time single queries, and on macOS FAISS's multi-threaded HNSW
    # build crashes when another library has already loaded its own OpenMP runtime.
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(X.shape[1], 32, faiss.METRIC_INNER_PRODUCT) if hnsw else faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(workdir / "index.faiss"))  # only to measure the on-disk size
    return lambda q, k: index.search(q[None, :], k)[1][0].tolist(), workdir


def build_qdrant(X, workdir):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(path=str(workdir))
    client.create_collection("bench", vectors_config=VectorParams(size=X.shape[1], distance=Distance.COSINE))
    client.upload_collection("bench", vectors=X, ids=list(range(len(X))), batch_size=256)
    return lambda q, k: [p.id for p in client.query_points("bench", query=q, limit=k).points], workdir


DATABASES = {
    "chroma": build_chroma,
    "faiss-flat": lambda X, w: build_faiss(X, w, hnsw=False),
    "faiss-hnsw": lambda X, w: build_faiss(X, w, hnsw=True),
    "qdrant-local": build_qdrant,
}


def dir_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persist-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--scale", type=int, help="pad with random unit vectors up to this many")
    parser.add_argument("--golden", default=str(ROOT / "tests/evals/data/ai_engineering_golden.json"))
    parser.add_argument("--label", help="save results to tests/evals/results/<label>.json")
    args = parser.parse_args()

    import chromadb

    persist_dir = Path(args.persist_dir).resolve()
    model = json.loads((persist_dir / "experiment.json").read_text())["embedding_model"]
    stored = chromadb.PersistentClient(path=str(persist_dir)).get_collection("pdf_chunks").get(
        include=["embeddings", "documents"])
    X = np.asarray(stored["embeddings"], dtype="float32")
    texts = list(stored["documents"])
    X /= np.linalg.norm(X, axis=1, keepdims=True)  # OpenAI vectors are unit length already; make it exact
    n_real = len(X)
    if args.scale and args.scale > n_real:
        rng = np.random.default_rng(0)
        pad = rng.standard_normal((args.scale - n_real, X.shape[1]), dtype="float32")
        X = np.vstack([X, pad / np.linalg.norm(pad, axis=1, keepdims=True)])
    k = args.top_k

    cases = [c for c in json.loads(Path(args.golden).read_text())["cases"]]
    Q = np.asarray(create_embedder(model=model).embed_documents([c["question"] for c in cases]), dtype="float32")
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    exact = [np.argsort(-(X @ q))[:k].tolist() for q in Q]  # the reference top-k
    answerable = [i for i, c in enumerate(cases) if c["type"] != "unanswerable"]
    print(f"{len(X):,} vectors × {X.shape[1]} dims ({n_real:,} real), {len(cases)} questions, top_k={k}\n")

    rows = []
    for name, build in DATABASES.items():
        workdir = Path(tempfile.mkdtemp(prefix=f"bench-{name}-"))
        try:
            t0 = time.perf_counter()
            search, disk = build(X, workdir)
            build_s = time.perf_counter() - t0
            results, times = [], []
            for q in Q:
                search(q, k)  # warm-up
                for _ in range(REPEATS):
                    t = time.perf_counter()
                    ids = search(q, k)
                    times.append(time.perf_counter() - t)
                results.append(ids)
            overlap = np.mean([len(set(r) & set(e)) / k for r, e in zip(results, exact)])
            ranks = [first_hit(evidence_hits(cases[i], [Document(page_content=texts[j]) if j < n_real
                                                          else Document(page_content="") for j in results[i]]))
                     for i in answerable]
            ms = sorted(t * 1000 for t in times)
            rows.append({
                "db": name, "build_s": round(build_s, 2), "disk_mb": round(dir_mb(disk), 1),
                "p50_ms": round(ms[len(ms) // 2], 2), "p95_ms": round(ms[int(len(ms) * 0.95)], 2),
                "overlap@k": round(float(overlap), 3),
                "recall@1": round(np.mean([r is not None and r < 1 for r in ranks]), 3),
                "recall@5": round(np.mean([r is not None and r < 5 for r in ranks]), 3),
                f"recall@{k}": round(np.mean([r is not None for r in ranks]), 3),
                "mrr": round(np.mean([1 / (r + 1) if r is not None else 0 for r in ranks]), 3),
            })
            print(f"  {name:13s} built in {build_s:.1f}s", flush=True)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    cols = list(rows[0])
    print("\n" + "  ".join(f"{c:>12}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>12}" for c in cols))
    if args.label:
        out = ROOT / "tests" / "evals" / "results" / "vector_db" / f"{args.label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"label": args.label, "created": time.strftime("%Y-%m-%d %H:%M"),
                                   "vectors": len(X), "real_vectors": n_real, "dims": int(X.shape[1]),
                                   "top_k": k, "embedding_model": model, "rows": rows}, indent=2))
        print(f"\nSaved → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
