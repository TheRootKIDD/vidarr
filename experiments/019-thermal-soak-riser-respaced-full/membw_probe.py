"""Ad-hoc STREAM-like host DRAM bandwidth probe (channel-count sanity check).

Threads over slices of large float64 arrays; numpy releases the GIL on the
elementwise ops, so this scales past one core. Not a benchmark of record --
it exists to tell 4-channel from 8-channel DDR4-2933 on this box.
"""
from __future__ import annotations
import time, os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

N = 1 << 29          # 512 Mi float64 = 4 GiB per array
THREADS = int(os.environ.get("MEMBW_THREADS", "12"))
REPS = 5

a = np.ones(N, dtype=np.float64)
b = np.full(N, 2.0, dtype=np.float64)
c = np.zeros(N, dtype=np.float64)
bounds = [(i * N // THREADS, (i + 1) * N // THREADS) for i in range(THREADS)]
pool = ThreadPoolExecutor(THREADS)

def run(fn, nbytes: int) -> float:
    best = 0.0
    for _ in range(REPS):
        t0 = time.perf_counter()
        list(pool.map(lambda s: fn(*s), bounds))
        dt = time.perf_counter() - t0
        best = max(best, nbytes / dt / 1e9)
    return best

kernels = {
    "copy  (2 arrays)": (lambda lo, hi: np.copyto(c[lo:hi], a[lo:hi]), 2 * N * 8),
    "scale (2 arrays)": (lambda lo, hi: np.multiply(b[lo:hi], 3.0, out=c[lo:hi]), 2 * N * 8),
    "add   (3 arrays)": (lambda lo, hi: np.add(a[lo:hi], b[lo:hi], out=c[lo:hi]), 3 * N * 8),
    "triad (3 arrays)": (lambda lo, hi: np.add(a[lo:hi], 3.0 * b[lo:hi], out=c[lo:hi]), 3 * N * 8),
}
print(f"threads={THREADS}  array={N * 8 / 2**30:.0f} GiB each  reps={REPS}  (best of reps)")
for name, (fn, nb) in kernels.items():
    print(f"  {name}: {run(fn, nb):7.1f} GB/s")
