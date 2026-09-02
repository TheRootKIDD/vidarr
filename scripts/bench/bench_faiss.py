"""CPU IVF-PQ build time and QPS (`docs/06 §6`, §5.1).

Settles the arithmetic behind `docs/06 §5.1`, which rules per-iteration external
retrieval out of the training loop: a `small` step needs ~3e5 QPS, and IVF-PQ on
12 Zen 2 cores is *expected* to be 1e4-1e5. This measures it.

After ADR-009 the in-loop table is GPU-resident product-key memory, so this
bench no longer gates L8. It sizes two other things: the **offline** neighbour
precompute for L8b (chunk-level, RETRO-style), and the cold-tier ANN of
`docs/01 §6.2` / `docs/02 §7`, whose index sits over $\\mathcal{G}$.

Vectors are random, which is the pessimistic case for IVF: real sentence
embeddings cluster, so recall at a given nprobe is better than this and QPS is
about the same. Recall is reported against exact search so the QPS number is
never read without the accuracy it was bought at.

Run: `python -m scripts.bench.bench_faiss --id 007-faiss --n 1000000`
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import faiss
import numpy as np

from scripts.bench._common import write_result

DIM = 768
"""$d_v$ of `docs/01 §1` — the knowledge-table value dimension."""


def build_index(xb: np.ndarray, nlist: int, m: int, nbits: int) -> tuple[Any, float, float]:
    """Train and populate an IVF-PQ index. Returns (index, train_s, add_s)."""
    quantiser = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFPQ(quantiser, DIM, nlist, m, nbits)
    n_train = min(len(xb), max(50 * nlist, 100_000))

    t0 = time.perf_counter()
    index.train(xb[:n_train])
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    index.add(xb)
    add_s = time.perf_counter() - t0
    return index, train_s, add_s


def measure_qps(
    index: Any,
    xq: np.ndarray,
    nprobe: int,
    k: int,
    truth: np.ndarray | None,
) -> dict[str, Any]:
    """Throughput at one nprobe, plus recall@k against exact search."""
    index.nprobe = nprobe
    index.search(xq[:64], k)  # warm

    t0 = time.perf_counter()
    _, ids = index.search(xq, k)
    secs = time.perf_counter() - t0

    recall = None
    if truth is not None:
        hits = sum(len(set(ids[i]) & set(truth[i])) for i in range(len(xq)))
        recall = hits / (len(xq) * k)

    return {
        "nprobe": nprobe,
        "k": k,
        "n_queries": len(xq),
        "seconds": secs,
        "qps": len(xq) / secs,
        f"recall_at_{k}": recall,
    }


def measure_single_query_latency(index: Any, xq: np.ndarray, nprobe: int, k: int) -> float:
    """Median seconds for one query issued alone.

    Per-iteration retrieval would be latency-bound, not throughput-bound, so the
    batched QPS above overstates what an in-loop lookup could do.
    """
    index.nprobe = nprobe
    times = []
    for i in range(200):
        t0 = time.perf_counter()
        index.search(xq[i : i + 1], k)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="007-faiss")
    ap.add_argument("--n", type=int, default=1_000_000, help="database vectors")
    ap.add_argument("--nq", type=int, default=10_000, help="query vectors")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nlist", type=int, default=None)
    ap.add_argument("--m", type=int, default=96, help="PQ subquantisers (768 = 96 x 8 dims)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-recall", action="store_true", help="skip the exact-search baseline")
    args = ap.parse_args()

    if args.threads:
        faiss.omp_set_num_threads(args.threads)
    threads = faiss.omp_get_max_threads()
    nlist = args.nlist or int(4 * np.sqrt(args.n))

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    xb = rng.standard_normal((args.n, DIM), dtype=np.float32)
    xq = rng.standard_normal((args.nq, DIM), dtype=np.float32)
    gen_s = time.perf_counter() - t0

    truth = None
    exact_qps = None
    if not args.no_recall:
        flat = faiss.IndexFlatL2(DIM)
        flat.add(xb)
        t0 = time.perf_counter()
        _, truth = flat.search(xq[:1000], args.k)
        exact_s = time.perf_counter() - t0
        exact_qps = 1000 / exact_s
        del flat

    index, train_s, add_s = build_index(xb, nlist, args.m, nbits=8)

    rows = [
        measure_qps(index, xq, nprobe, args.k, truth[:1000] if truth is not None else None)
        if truth is None
        else measure_qps(index, xq[:1000], nprobe, args.k, truth)
        for nprobe in (1, 4, 8, 16, 32, 64)
    ]
    latency = {p: measure_single_query_latency(index, xq, p, args.k) for p in (8, 32)}

    payload = {
        "bench": "faiss",
        "n_vectors": args.n,
        "dim": DIM,
        "index": f"IVF{nlist},PQ{args.m}x8",
        "threads": threads,
        "faiss_version": faiss.__version__,
        "generate_s": gen_s,
        "train_s": train_s,
        "add_s": add_s,
        "build_s": train_s + add_s,
        "exact_search_qps": exact_qps,
        "single_query_latency_s": {str(k): v for k, v in latency.items()},
        "results": rows,
    }
    path = write_result(args.id, payload)
    print(f"wrote {path}\n")
    print(f"  {args.n:,} x {DIM}d, IVF{nlist},PQ{args.m}x8, {threads} threads")
    print(f"  build {train_s + add_s:6.1f} s  (train {train_s:.1f} + add {add_s:.1f})")
    if exact_qps:
        print(f"  exact (IndexFlatL2) {exact_qps:8.1f} QPS")
    print(f"\n  {'nprobe':>7} {'QPS':>10} {'recall@' + str(args.k):>10}")
    for r in rows:
        rec = r.get(f"recall_at_{args.k}")
        print(f"  {r['nprobe']:>7} {r['qps']:>10,.0f} {rec if rec is None else f'{rec:>10.3f}'}")
    for p, s in latency.items():
        print(f"  single-query latency @ nprobe={p}: {s * 1e6:.0f} us -> {1 / s:,.0f} QPS")


if __name__ == "__main__":
    main()
