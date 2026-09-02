"""Memory-tier bandwidth and latency at KV-block sizes (`docs/06 §6`, `02 §7`).

Three tiers, selectable so they can be measured at different times:

* `vram`    — device-local bandwidth, i.e. $\\beta_C$ for `local_3060`, and
              device-to-device copies. Storage-independent.
* `host`    — VRAM ↔ host DRAM over PCIe, pinned and pageable, both directions.
              Storage-independent.
* `storage` — random and sequential reads from a file at KV-block sizes, with
              `O_DIRECT` so the page cache cannot flatter the number. Run this
              only against the drive that will actually hold the cold tier
              (open question I10: the NVMe, once installed — not the SATA SSD).

Block sizes span a single KV block up to a bulk transfer. At `small` one
$\\mathcal{G}$ entry is $2 H_{kv} d_h b_{kv}$ = 1 KiB and a $B_{idx}$ = 16 block
is 16 KiB; the cost model's `persistent_kv_bytes_per_token` gives the exact
figure for any config.

Run: `python -m scripts.bench.bench_tiers --id 011-tiers-mem --tiers vram host`
     `python -m scripts.bench.bench_tiers --id 0NN-tiers-nvme --tiers storage \\
        --storage-path /path/on/nvme/bench.bin --queue-depth 1 4 16 64`
"""

from __future__ import annotations

import argparse
import mmap
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from scripts.bench._common import write_result

BLOCK_SIZES = [4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20]


def _cuda_time(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / 1e3 / iters


def _copy_time(dst: torch.Tensor, src: torch.Tensor, n: int, non_blocking: bool = False) -> float:
    """Time `dst.copy_(src)`; binds its operands so no closure captures a loop variable."""
    return _cuda_time(lambda: dst.copy_(src, non_blocking=non_blocking), 3, _iters(n))


def _iters(nbytes: int) -> int:
    return 200 if nbytes <= 1 << 20 else (50 if nbytes <= 16 << 20 else 20)


def vram_tier(device: int) -> dict[str, Any]:
    """beta_C: on-device copy bandwidth (read + write counted once each)."""
    dev = f"cuda:{device}"
    rows = []
    for n in BLOCK_SIZES + [256 << 20, 1 << 30]:
        src = torch.empty(n, dtype=torch.uint8, device=dev)
        dst = torch.empty(n, dtype=torch.uint8, device=dev)
        s = _copy_time(dst, src, n)
        rows.append(
            {"bytes": n, "seconds": s, "GBps_copy": n / s / 1e9, "GBps_traffic": 2 * n / s / 1e9}
        )
    torch.cuda.empty_cache()
    # A pure read: sum over a large tensor, one pass. Closest to weight streaming.
    read_only = _read_only_gbps(dev)
    torch.cuda.empty_cache()
    return {"device": device, "copy": rows, "read_only_GBps_1GiB": read_only}


def _read_only_gbps(dev: str) -> float:
    big = torch.empty(1 << 30, dtype=torch.uint8, device=dev).view(torch.float32)
    s = _cuda_time(lambda: big.sum(), 3, 20)
    return (1 << 30) / s / 1e9


def d2d_tier(src_dev: int, dst_dev: int) -> list[dict[str, Any]]:
    """Device-to-device copy through host memory (no P2P on GeForce)."""
    rows = []
    for n in [1 << 20, 16 << 20, 256 << 20]:
        src = torch.empty(n, dtype=torch.uint8, device=f"cuda:{src_dev}")
        dst = torch.empty(n, dtype=torch.uint8, device=f"cuda:{dst_dev}")
        s = _copy_time(dst, src, n)
        rows.append(
            {"src": src_dev, "dst": dst_dev, "bytes": n, "seconds": s, "GBps": n / s / 1e9}
        )
    torch.cuda.empty_cache()
    return rows


def host_tier(device: int) -> list[dict[str, Any]]:
    """VRAM <-> host DRAM, pinned and pageable, both directions."""
    dev = f"cuda:{device}"
    rows = []
    for n in BLOCK_SIZES + [256 << 20]:
        gpu = torch.empty(n, dtype=torch.uint8, device=dev)
        for pinned in (True, False):
            host = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
            h2d = _copy_time(gpu, host, n, non_blocking=pinned)
            d2h = _copy_time(host, gpu, n, non_blocking=pinned)
            rows.append(
                {
                    "device": device,
                    "bytes": n,
                    "pinned": pinned,
                    "h2d_s": h2d,
                    "d2h_s": d2h,
                    "h2d_GBps": n / h2d / 1e9,
                    "d2h_GBps": n / d2h / 1e9,
                }
            )
    torch.cuda.empty_cache()
    return rows


def _read_worker(fd: int, n: int, offsets: list[int]) -> list[float]:
    """One reader: issue `offsets` in order into a private page-aligned buffer,
    return the per-request wall times. `os.preadv` releases the GIL, so several
    of these in threads give the drive a real queue depth."""
    buf = mmap.mmap(-1, n)
    times = []
    try:
        for off in offsets:
            t0 = time.perf_counter()
            os.preadv(fd, [buf], off)
            times.append(time.perf_counter() - t0)
    finally:
        buf.close()
    return times


def _run_reads(fd: int, n: int, offsets: list[int], queue_depth: int) -> tuple[list[float], float]:
    """Spread `offsets` round-robin over `queue_depth` threads; return every
    request's latency and the wall time of the whole batch."""
    if queue_depth == 1:
        t0 = time.perf_counter()
        times = _read_worker(fd, n, offsets)
        return times, time.perf_counter() - t0
    slices = [offsets[i::queue_depth] for i in range(queue_depth)]
    with ThreadPoolExecutor(max_workers=queue_depth) as pool:
        t0 = time.perf_counter()
        futs = [pool.submit(_read_worker, fd, n, sl) for sl in slices]
        times = [t for f in futs for t in f.result()]
        wall = time.perf_counter() - t0
    return times, wall


def storage_tier(path: Path, file_bytes: int, queue_depths: list[int]) -> dict[str, Any]:
    """Random + sequential reads with O_DIRECT at KV-block sizes, at each queue depth.

    O_DIRECT bypasses the page cache, so this measures the device, not RAM.
    It requires aligned buffers and offsets, hence the mmap-backed buffers.
    Queue depth > 1 is `queue_depth` Python threads each issuing synchronous
    `preadv`; the GIL bounds the aggregate issue rate (a few 10^5 calls/s), so
    high-QD small-block figures are a lower bound on the drive (I16).
    """
    if not path.exists() or path.stat().st_size < file_bytes:
        with open(path, "wb") as fh:
            chunk = os.urandom(1 << 20)
            for _ in range(file_bytes >> 20):
                fh.write(chunk)
        os.sync()

    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    rng_state = torch.Generator().manual_seed(0)
    rows = []
    try:
        for qd in queue_depths:
            for n in BLOCK_SIZES:
                n_blocks = file_bytes // n
                base = max(20, min(2000, (256 << 20) // n))
                # enough requests per thread to matter, bounded at 4 GiB of traffic
                reps = max(qd * 4, min(base * qd, (4 << 30) // n))
                offsets = (torch.randint(0, n_blocks, (reps,), generator=rng_state) * n).tolist()
                rand_times, rand_wall = _run_reads(fd, n, offsets, qd)

                seq_offsets = [(i * n) % (file_bytes - n) for i in range(reps)]
                _, seq_wall = _run_reads(fd, n, seq_offsets, qd)
                rows.append(
                    {
                        "queue_depth": qd,
                        "bytes": n,
                        "reps": reps,
                        "random_latency_s_median": statistics.median(rand_times),
                        "random_latency_s_p99": statistics.quantiles(rand_times, n=100)[98]
                        if len(rand_times) >= 100 else max(rand_times),
                        "random_GBps": reps * n / rand_wall / 1e9,
                        "random_iops": reps / rand_wall,
                        "sequential_GBps": reps * n / seq_wall / 1e9,
                    }
                )
    finally:
        os.close(fd)
    return {
        "path": str(path),
        "file_bytes": file_bytes,
        "o_direct": True,
        "queue_depths": queue_depths,
        "reader": "python threads, synchronous preadv",
        "results": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--tiers", nargs="+", choices=["vram", "host", "storage"], required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--storage-path", type=Path, default=None)
    ap.add_argument("--storage-file-gib", type=int, default=8)
    ap.add_argument("--queue-depth", type=int, nargs="+", default=[1],
                    help="storage tier: concurrent readers to sweep, e.g. 1 4 16 64")
    args = ap.parse_args()

    payload: dict[str, Any] = {"bench": "tiers", "tiers": args.tiers}
    if "vram" in args.tiers:
        payload["vram"] = vram_tier(args.device)
        n_dev = torch.cuda.device_count()
        if n_dev > 1:
            payload["d2d"] = d2d_tier(args.device, (args.device + 1) % n_dev)
    if "host" in args.tiers:
        payload["host"] = host_tier(args.device)
    if "storage" in args.tiers:
        if args.storage_path is None:
            raise SystemExit("--storage-path is required for the storage tier")
        payload["storage"] = storage_tier(
            args.storage_path, args.storage_file_gib << 30, args.queue_depth
        )

    path = write_result(args.id, payload)
    print(f"wrote {path}\n")

    if "vram" in payload:
        v = payload["vram"]
        big = [r for r in v["copy"] if r["bytes"] >= 256 << 20]
        traffic = statistics.mean(r["GBps_traffic"] for r in big)
        print(f"  VRAM copy @ >=256 MiB: {traffic:6.1f} GB/s traffic  "
              f"| read-only 1 GiB: {v['read_only_GBps_1GiB']:6.1f} GB/s")
        if "d2d" in payload:
            d = payload["d2d"][-1]
            print(f"  D2D gpu{d['src']}->gpu{d['dst']} @ 256 MiB: {d['GBps']:6.2f} GB/s")
    if "host" in payload:
        hdr = f"{'bytes':>10} {'pinned H2D':>11} {'pinned D2H':>11} "
        print(f"\n  {hdr}{'pageable H2D':>13} {'pageable D2H':>13}")
        by = {}
        for r in payload["host"]:
            by.setdefault(r["bytes"], {})[r["pinned"]] = r
        for n, d in sorted(by.items()):
            print(f"  {n:>10} {d[True]['h2d_GBps']:>11.2f} {d[True]['d2h_GBps']:>11.2f} "
                  f"{d[False]['h2d_GBps']:>13.2f} {d[False]['d2h_GBps']:>13.2f}")
    if "storage" in payload:
        print(f"\n  storage {payload['storage']['path']} (O_DIRECT)")
        hdr = f"{'QD':>4} {'bytes':>10} {'rand lat us':>12} {'p99 us':>8} {'rand GB/s':>10} "
        print(f"  {hdr}{'rand IOPS':>10} {'seq GB/s':>9}")
        for r in payload["storage"]["results"]:
            print(f"  {r['queue_depth']:>4} {r['bytes']:>10} "
                  f"{r['random_latency_s_median'] * 1e6:>12.0f} "
                  f"{r['random_latency_s_p99'] * 1e6:>8.0f} "
                  f"{r['random_GBps']:>10.3f} {r['random_iops']:>10,.0f} "
                  f"{r['sequential_GBps']:>9.3f}")

if __name__ == "__main__":
    main()
