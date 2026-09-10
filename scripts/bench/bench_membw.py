"""Host DRAM bandwidth at STREAM kernels, swept over thread count (`docs/06 §1`).

`docs/06 §1` carried a nominal ≈ 205 GB/s for this CPU for months. That figure is
the controller's DDR4-3200 rating; the installed modules are DDR4-2133 (`022`),
whose 8-channel peak is 136.5 GB/s, and nothing on this rig had ever measured the
real number. `019` ran an ad-hoc probe at one thread count and got 70.8 GB/s copy,
which was enough to tell 8-channel from 4-channel and not enough to enter
`sim/scenarios/local_3060.yaml` — CLAUDE.md requires a scenario input to cite a
result id from `scripts/bench/`.

Why the thread sweep is the point: a single-thread figure measures one core's
outstanding-miss limit, not the memory system. Bandwidth rises with threads and
then plateaus when the channels saturate, and it is the **plateau** that is the
scenario input. Reporting the plateau alongside the curve also shows how many
cores a host-side stage (corpus streaming, KV warm tier, optimiser offload) has
to occupy before it stops getting faster.

Kernels are the four STREAM ones. Bytes counted are the bytes the kernel must
move, counting a written array once for the store; on a write-allocate cache
without non-temporal stores the true DRAM traffic for `copy`, `scale` and `triad`
is higher by the read-for-ownership of the destination, so these figures are
lower bounds on the hardware and upper bounds on what a program can use. numpy
releases the GIL on the elementwise ops, so threads scale past one core.

Run: `python -m scripts.bench.bench_membw --id 0NN-membw`
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from scripts.bench._common import write_result

Kernel = Callable[[int, int], None]


def _bounds(n: int, threads: int) -> list[tuple[int, int]]:
    return [(i * n // threads, (i + 1) * n // threads) for i in range(threads)]


def _best_gbps(
    pool: ThreadPoolExecutor, fn: Kernel, bounds: list[tuple[int, int]], nbytes: int, reps: int
) -> float:
    """Best of `reps` — the maximum is the one least perturbed by scheduling noise."""
    best = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        list(pool.map(lambda s: fn(*s), bounds))
        dt = time.perf_counter() - t0
        best = max(best, nbytes / dt / 1e9)
    return best


def kernels(a: np.ndarray, b: np.ndarray, c: np.ndarray, n: int) -> dict[str, tuple[Kernel, int]]:
    """STREAM's four kernels with the bytes each one must move (8 B per element)."""
    w = 8
    return {
        "copy": (lambda lo, hi: np.copyto(c[lo:hi], a[lo:hi]), 2 * n * w),
        "scale": (lambda lo, hi: np.multiply(b[lo:hi], 3.0, out=c[lo:hi]), 2 * n * w),
        "add": (lambda lo, hi: np.add(a[lo:hi], b[lo:hi], out=c[lo:hi]), 3 * n * w),
        # Not a STREAM triad: `3.0 * b` materialises a temporary, so this moves a
        # fourth array that the byte count does not include. Kept for comparability
        # with the `019` ad-hoc probe, which had the same shape. `copy` is the
        # reference figure -- two arrays, no temporary, no arithmetic.
        "triad": (lambda lo, hi: np.add(a[lo:hi], 3.0 * b[lo:hi], out=c[lo:hi]), 3 * n * w),
    }


def sweep(array_gib: float, thread_counts: list[int], reps: int) -> list[dict[str, Any]]:
    n = int(array_gib * (1 << 30) / 8)
    a = np.ones(n, dtype=np.float64)
    b = np.full(n, 2.0, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    ks = kernels(a, b, c, n)
    rows = []
    for threads in thread_counts:
        pool = ThreadPoolExecutor(threads)
        bounds = _bounds(n, threads)
        row: dict[str, Any] = {"threads": threads}
        for name, (fn, nbytes) in ks.items():
            _best_gbps(pool, fn, bounds, nbytes, 1)  # warm the pages and the pool
            row[name] = _best_gbps(pool, fn, bounds, nbytes, reps)
        pool.shutdown()
        rows.append(row)
        print(f"  {threads:>3} threads: " + "  ".join(f"{k} {row[k]:6.1f}" for k in ks) + "  GB/s")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="025-membw")
    ap.add_argument(
        "--array-gib",
        type=float,
        default=4.0,
        help="per-array size; must far exceed the 64 MB L3 of the 3945WX",
    )
    ap.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8, 12, 16, 24],
        help="thread counts to sweep; the plateau is the scenario input",
    )
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    print(f"host DRAM bandwidth: {args.array_gib:g} GiB per array, best of {args.reps}")
    rows = sweep(args.array_gib, args.threads, args.reps)

    peak = {k: max(r[k] for r in rows) for k in ("copy", "scale", "add", "triad")}
    at = {k: next(r["threads"] for r in rows if r[k] == peak[k]) for k in peak}
    payload = {
        "bench": "membw",
        "array_bytes_per_array": int(args.array_gib * (1 << 30)),
        "dtype": "float64",
        "reps": args.reps,
        "sweep": rows,
        "peak_gbps": peak,
        "peak_at_threads": at,
        "note": (
            "Bytes counted exclude read-for-ownership of the destination, so copy/scale/triad "
            "understate DRAM traffic and state usable bandwidth. Peak is the plateau over the "
            "thread sweep; a single-thread figure measures a core, not the memory system."
        ),
    }
    path = write_result(args.id, payload)
    print(f"wrote {path}")
    for k in ("copy", "scale", "add", "triad"):
        print(f"  {k:>6}: {peak[k]:6.1f} GB/s at {at[k]} threads")


if __name__ == "__main__":
    main()
