"""Collective bandwidth and latency across the four cards (`docs/06 §6`).

Measures all-reduce, all-gather and all-to-all from 1 KB to 1 GB. The cards are
consumer Ampere with no NVLink and no PCIe peer-to-peer, so every transfer is
staged through host memory; `NCCL_P2P_DISABLE=1` is set explicitly so the result
is deterministic rather than dependent on what NCCL decides it can do.

Feeds the `lambda_link` / `beta_link` inputs of `sim/scenarios/local_3060.yaml`
and settles open question I5: whether host-bounced collectives leave enough
all-to-all bandwidth for S0 and S10 to mean anything on four cards.

Run: `python -m scripts.bench.bench_nccl --id 001-nccl`
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.bench._common import write_result

# Bus-bandwidth correction factors: bytes that must cross the link per byte of
# user buffer, for a ring algorithm over `n` ranks. Standard nccl-tests factors.
BUS_FACTOR = {
    "all_reduce": lambda n: 2.0 * (n - 1) / n,
    "all_gather": lambda n: (n - 1) / n,
    "all_to_all": lambda n: (n - 1) / n,
}


def _iters_for(nbytes: int) -> tuple[int, int]:
    """(warmup, timed) iteration counts — many reps when small, few when large."""
    if nbytes <= 1 << 16:
        return 20, 200
    if nbytes <= 1 << 22:
        return 10, 50
    if nbytes <= 1 << 26:
        return 5, 20
    return 3, 8


def _make_call(op: str, buf_bytes: int, rank: int, world: int):
    """Build the collective closure and its buffers for one op at one size.

    `buf_bytes` is the per-rank user buffer, matching nccl-tests convention, so
    all-gather allocates a `world`-times larger output.
    """
    n_elem = buf_bytes // 4
    dev = torch.device(f"cuda:{rank}")
    src = torch.ones(n_elem, dtype=torch.float32, device=dev)

    if op == "all_reduce":
        return lambda: dist.all_reduce(src)
    if op == "all_gather":
        dst = torch.empty(n_elem * world, dtype=torch.float32, device=dev)
        return lambda: dist.all_gather_single(dst, src)
    if op == "all_to_all":
        dst = torch.empty(n_elem, dtype=torch.float32, device=dev)
        return lambda: dist.all_to_all_single(dst, src)
    raise ValueError(op)


def _run_one(op: str, buf_bytes: int, rank: int, world: int) -> list[float]:
    """Time one collective at one size. Returns per-iteration seconds."""
    call = _make_call(op, buf_bytes, rank, world)

    warmup, timed = _iters_for(buf_bytes)
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(timed):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    # Drop the closure (and with it the buffers) before the next, larger size.
    del call
    torch.cuda.empty_cache()
    return times


def _worker(rank: int, world: int, sizes: list[int], ops: list[str], out: dict) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world, device_id=torch.device(f"cuda:{rank}")
    )

    for op in ops:
        for nbytes in sizes:
            times = _run_one(op, nbytes, rank, world)
            # Rank 0 reports; every rank must reach the same collectives in order.
            if rank == 0:
                med = statistics.median(times)
                lo = min(times)
                algbw = nbytes / med
                out[f"{op}:{nbytes}"] = {
                    "op": op,
                    "bytes_per_rank": nbytes,
                    "median_s": med,
                    "min_s": lo,
                    "algbw_GBps": algbw / 1e9,
                    "busbw_GBps": algbw * BUS_FACTOR[op](world) / 1e9,
                }
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="001-nccl")
    ap.add_argument("--max-log2", type=int, default=30, help="largest per-rank buffer, 2^N bytes")
    ap.add_argument("--min-log2", type=int, default=10)
    args = ap.parse_args()

    world = torch.cuda.device_count()
    if world < 2:
        raise SystemExit(f"need >=2 GPUs, found {world}")

    # Explicit rather than implicit: GeForce has no P2P, say so in the record.
    os.environ["NCCL_P2P_DISABLE"] = "1"

    sizes = [1 << e for e in range(args.min_log2, args.max_log2 + 1)]
    ops = ["all_reduce", "all_gather", "all_to_all"]

    mgr = mp.Manager()
    out = mgr.dict()
    mp.spawn(_worker, args=(world, sizes, ops, out), nprocs=world, join=True)

    rows = sorted(dict(out).values(), key=lambda r: (r["op"], r["bytes_per_rank"]))
    payload = {
        "bench": "nccl",
        "world_size": world,
        "backend": "nccl",
        "nccl_p2p_disable": True,
        "nccl_version": ".".join(str(v) for v in torch.cuda.nccl.version()),
        "note": "host-bounced: no NVLink, no PCIe P2P on GeForce",
        "results": rows,
    }
    path = write_result(args.id, payload)
    print(f"wrote {path}\n")

    for op in ops:
        sel = [r for r in rows if r["op"] == op]
        peak = max(sel, key=lambda r: r["busbw_GBps"])
        lat = min(sel, key=lambda r: r["bytes_per_rank"])
        print(f"{op:12s} latency@1KB {lat['median_s']*1e6:8.1f} us   "
              f"peak busbw {peak['busbw_GBps']:6.2f} GB/s @ "
              f"{peak['bytes_per_rank']/2**20:.0f} MiB")


if __name__ == "__main__":
    main()
