"""S0's measured side: bulk-synchronous expert-parallel decode on the four cards (ADR-049).

Rank u is unit u (U = 1). It owns sequences s = u (mod 4) and hosts experts i = u
(mod 4), ADR-043 decision 3. Every rank holds the replicated skeleton, attention,
norms, router and head (managed aggregation, `02 §6`). One **round** is one
middle-block iteration for all z tokens in lockstep:

    owner   DepthNorm, attention, router for its z/4 tokens       (`owner`)
    send    counts, then each (token, expert) row to the expert's rank  (`dispatch`)
    expert  each hosted expert runs once on every row it received (`expert`)
    return  outputs back to the owners                             (`return`)
    owner   gate-weighted sum and residual                         (`post`)

with `torch.cuda.synchronize()` at every phase boundary, so each phase is timed on
its own (the barrier policy needs the synchronisation anyway at the two collectives).
Early blocks run before round 1 and the final blocks and head after round r; both
are counted as owner work (`skel`). The workload is teacher-forced held-out text
(ADR-049 §3): P positions prefilled untimed, then N decoded, the first `warmup` of
them excluded from every statistic.

Run on an idle rig only (CLAUDE.md):
    python -m scripts.bench.s0_ep_decode --id 034-s0-ep-decode --run 132-l5-small-s0
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from model import decode
from model.model import BigMoE
from scripts.analysis.probe_routing import load_run
from scripts.bench._common import write_result
from scripts.train.data import MemmapCorpus
from scripts.train.train import gpu_thermals

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("owner", "dispatch", "expert", "return", "post")


def _sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs), q)) if xs else float("nan")


class Rank:
    def __init__(self, model: BigMoE, rank: int, world: int, dev: torch.device) -> None:
        self.m, self.rank, self.world, self.dev = model, rank, world, dev
        self.n_e = model.cfg.router.n_experts
        self.k = model.cfg.router.k
        self.d = model.cfg.d
        self.hosted = [i for i in range(self.n_e) if i % world == rank]

    # ---- one round (one middle-block iteration of every token) ---------------------------
    def round(
        self,
        j: int,
        c: torch.Tensor,
        pos: torch.Tensor,
        buf: decode.KVBuffer,
        rec: dict[str, Any] | None,
    ) -> torch.Tensor:
        t0 = _sync_time()
        u, a, route = decode.owner_pre(self.m, j, c, pos, buf)
        flat = u.reshape(-1, self.d)
        n = flat.shape[0]
        expert_of = route.idx.reshape(-1)  # [n k]
        token_of = torch.arange(n, device=self.dev).repeat_interleave(self.k)
        dest = expert_of % self.world
        order = torch.argsort(dest, stable=True)
        send_counts = torch.bincount(dest, minlength=self.world)
        # payload = the normalised vector plus the expert id as one extra column
        # (ids < 256 are exact in bf16), so dispatch is two collectives: counts, rows
        payload = torch.cat([flat[token_of[order]], expert_of[order, None].to(flat.dtype)], dim=1)
        t1 = _sync_time()
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        sc, rc = send_counts.tolist(), recv_counts.tolist()
        recv = payload.new_empty(sum(rc), self.d + 1)
        dist.all_to_all_single(recv, payload, rc, sc)
        t2 = _sync_time()
        x, ids = recv[:, : self.d], recv[:, self.d].round().long()
        y = torch.empty_like(x)
        sizes = []
        for i in self.hosted:
            sel = (ids == i).nonzero(as_tuple=True)[0]
            sizes.append(int(sel.numel()))
            if sel.numel():
                y[sel] = self.m.middle.moe.experts[i](x[sel]).to(y.dtype)
        t3 = _sync_time()
        back = y.new_empty(sum(sc), self.d)
        dist.all_to_all_single(back, y, sc, rc)
        t4 = _sync_time()
        out = torch.zeros_like(flat)
        gate = route.gate.reshape(-1)[order, None]
        out.index_add_(0, token_of[order], (back * gate).to(out.dtype))
        c = decode.owner_post(c, a, out)
        t5 = _sync_time()
        if rec is not None:
            for name, dt in zip(PHASES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4), strict=True):
                rec[name].append(dt)
            rec["iter"].append(t5 - t0)
            rec["batch"].extend(sizes)
            off = sum(s for r_, s in enumerate(sc) if r_ != self.rank)
            row = flat.element_size()
            rec["wire_bytes"] += off * ((self.d + 1) * row + self.d * row)
            rec["hist"][j - 1] += torch.bincount(expert_of, minlength=self.n_e).cpu()
        return c

    # ---- one decode position of every owned sequence --------------------------------------
    def position(
        self,
        tok: torch.Tensor,
        target: torch.Tensor,
        cache: decode.DecodeCache,
        rec: dict[str, Any] | None,
    ) -> float:
        t0 = _sync_time()
        c, pos = decode.early(self.m, tok, cache)
        t1 = _sync_time()
        for j, buf in enumerate(cache.middle, start=1):
            c = self.round(j, c, pos, buf, rec)
        t2 = _sync_time()
        logits = decode.final(self.m, c, pos, cache)
        t3 = _sync_time()
        if rec is not None:
            rec["skel"].append((t1 - t0) + (t3 - t2))
            rec["token"].append(t3 - t0)
            rec["nll"] += float(F.cross_entropy(logits[:, -1].float(), target, reduction="sum"))
            rec["n_nll"] += target.numel()
        return t3 - t0


class ThermalSampler:
    """`gpu_thermals()` every `period` seconds on a thread of rank 0, so the timed loop never
    waits on `nvidia-smi` (each call takes tens of milliseconds)."""

    def __init__(self, period: float = 5.0) -> None:
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, args=(period,), daemon=True)

    def _loop(self, period: float) -> None:
        while not self._stop.is_set():
            self.samples.append(gpu_thermals())
            self._stop.wait(period)

    def __enter__(self) -> ThermalSampler:
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join()
        self.samples.append(gpu_thermals())


def _empty_rec(n_e: int, r: int) -> dict[str, Any]:
    rec: dict[str, Any] = {p: [] for p in PHASES}
    rec.update(iter=[], batch=[], skel=[], token=[], wire_bytes=0, nll=0.0, n_nll=0)
    rec["hist"] = torch.zeros(r, n_e, dtype=torch.long)
    return rec


def _run_z(
    rk: Rank,
    windows: list[torch.Tensor],
    z: int,
    prefill: int,
    n_dec: int,
    warmup: int,
    chunk: int,
    prefill_rows: int,
) -> dict[str, Any]:
    world, rank, dev = rk.world, rk.rank, rk.dev
    mine = [s for s in range(z) if s % world == rank]
    seqs = torch.cat([windows[s] for s in mine]).to(dev)  # [B, P + N + 1]
    b = seqs.shape[0]
    r = rk.m.cfg.middle.r_max
    cache = decode.new_cache(rk.m, b, prefill + n_dec, r, dev, torch.bfloat16)
    torch.cuda.reset_peak_memory_stats(dev)
    t_pre = _sync_time()
    with torch.no_grad():
        for b0 in range(0, b, prefill_rows):
            sub = cache.rows(b0, min(b, b0 + prefill_rows))
            for p0 in range(0, prefill, chunk):
                rk.m.prefill(seqs[b0 : b0 + prefill_rows, p0 : min(prefill, p0 + chunk)], sub)
        cache.set_length(prefill)
    dist.barrier()
    t_pre = _sync_time() - t_pre
    rec = _empty_rec(rk.n_e, r)
    sampler = ThermalSampler() if rank == 0 else None
    if sampler is not None:
        sampler.__enter__()
    with torch.no_grad():
        dist.barrier()
        t_start = 0.0
        for n in range(n_dec):
            if n == warmup:
                dist.barrier()
                t_start = _sync_time()
            p = prefill + n
            live = rec if n >= warmup else None
            rk.position(seqs[:, p : p + 1], seqs[:, p + 1], cache, live)
        dist.barrier()
        wall = _sync_time() - t_start
    therm: list[dict[str, float]] = []
    if sampler is not None:
        sampler.__exit__()
        therm = sampler.samples
    timed = n_dec - warmup
    local = {
        "rank": rank,
        "sequences": b,
        **{f"{p}_s_p50": _pct(rec[p], 50) for p in PHASES},
        **{f"{p}_s_mean": statistics.fmean(rec[p]) for p in PHASES},
        "iter_s": rec["iter"],
        "token_s": rec["token"],
        "skel_s_mean": statistics.fmean(rec["skel"]),
        "batch": rec["batch"],
        "wire_bytes": rec["wire_bytes"],
        "nll": rec["nll"],
        "n_nll": rec["n_nll"],
        "hist": rec["hist"],
        "busy_s": sum(rec["owner"]) + sum(rec["expert"]) + sum(rec["post"]) + sum(rec["skel"]),
        "peak_gib": torch.cuda.max_memory_allocated(dev) / 2**30,
        "cache_gib": cache.nbytes() / 2**30,
    }
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, local)
    if rank != 0:
        return {}
    iters = [x for g in gathered for x in g["iter_s"]]
    hist = sum(g["hist"] for g in gathered)
    tokens = z * timed
    return {
        "z": z,
        "prefill_positions": prefill,
        "decode_positions": n_dec,
        "warmup_positions": warmup,
        "mean_keys_per_decode_token": prefill + warmup + (timed + 1) / 2,
        "prefill_s": t_pre,
        "wall_s": wall,
        "throughput_tok_s": tokens / wall,
        "iteration_s": {
            "p50": _pct(iters, 50),
            "p99": _pct(iters, 99),
            "mean": statistics.fmean(iters),
        },
        "token_s": {
            "p50": _pct(gathered[0]["token_s"], 50),
            "p99": _pct(gathered[0]["token_s"], 99),
            "mean": statistics.fmean(gathered[0]["token_s"]),
        },
        "per_rank": [
            {
                k: v
                for k, v in g.items()
                if k not in ("iter_s", "token_s", "batch", "hist", "nll", "n_nll")
            }
            | {
                "iteration_s_p50": _pct(g["iter_s"], 50),
                "busy_fraction": g["busy_s"] / wall,
                "expert_batch": {
                    "mean": statistics.fmean(g["batch"]) if g["batch"] else 0.0,
                    "p50": _pct(g["batch"], 50),
                    "min": min(g["batch"], default=0),
                    "max": max(g["batch"], default=0),
                },
            }
            for g in gathered
        ],
        "wire_bytes_per_token_iteration": sum(g["wire_bytes"] for g in gathered)
        / (tokens * hist.shape[0]),
        "logical_bytes_per_token_iteration": 2 * rk.k * rk.d * 2,
        "routing_hist": hist.tolist(),
        "routing_share": (hist.sum(0).double() / hist.sum()).tolist(),
        "teacher_forced_nll": sum(g["nll"] for g in gathered) / sum(g["n_nll"] for g in gathered),
        "thermal": therm,
        "n_thermal_max": max((t.get("n_thermal", 0.0) for t in therm), default=float("nan")),
    }


def _worker(rank: int, world: int, args: argparse.Namespace, out: Any) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    model, cfg, step = load_run(ROOT / "experiments" / args.run, dev)
    model = model.to(torch.bfloat16).eval()
    corpus = MemmapCorpus(Path(args.corpus))
    windows = [w[:, : args.prefill + args.decode + 1] for w in corpus.val_windows(cfg.context)]
    rk = Rank(model, rank, world, dev)
    results = []
    for z in args.z:
        if z % world or z > len(windows):
            raise ValueError(f"z = {z} must be a multiple of {world} and <= {len(windows)}")
        res = _run_z(
            rk, windows, z, args.prefill, args.decode, args.warmup, args.chunk, args.prefill_rows
        )
        if rank == 0:
            res["checkpoint_step"] = step
            results.append(res)
            print(
                f"z {z:5d}: {res['throughput_tok_s']:9.1f} tok/s, iteration p50 "
                f"{res['iteration_s']['p50'] * 1e3:7.3f} ms, "
                f"p99 {res['iteration_s']['p99'] * 1e3:7.3f} ms, "
                f"nll {res['teacher_forced_nll']:.4f}, n_thermal {res['n_thermal_max']}",
                flush=True,
            )
        torch.cuda.empty_cache()
    if rank == 0:
        out["results"] = results
    dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", required=True, help="result id, e.g. 034-s0-ep-decode")
    ap.add_argument("--run", default="132-l5-small-s0", help="checkpoint run under experiments/")
    ap.add_argument("--corpus", default="/mnt/nvme/corpus/fineweb-edu/tok-mistral32k")
    ap.add_argument("--z", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--prefill", type=int, default=1536)
    ap.add_argument("--decode", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=512, help="prefill positions per pass")
    ap.add_argument("--prefill-rows", type=int, default=16, help="sequences per prefill pass")
    ap.add_argument("--scenario", default="sim/scenarios/local_3060.yaml")
    ap.add_argument("--dry", action="store_true", help="print, do not write a result")
    args = ap.parse_args()
    world = torch.cuda.device_count()
    if world != 4:
        raise SystemExit(f"S0 needs the four cards, found {world}")
    if args.warmup >= args.decode:
        raise SystemExit("warmup must be shorter than decode")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # deterministic host bounce, as bench_nccl
    mgr = mp.Manager()
    out = mgr.dict()
    t0 = time.time()
    mp.spawn(_worker, args=(world, args, out), nprocs=world, join=True)
    results = out["results"]
    scen = ROOT / args.scenario
    payload = {
        "bench": "s0_ep_decode",
        "adr": "ADR-049",
        "run": args.run,
        "policy": "round (bulk-synchronous; phase-synchronised timing)",
        "placement": "expert i on rank i mod 4; sequence s owned by rank s mod 4",
        "precision": "bf16 weights and caches",
        "scenario": args.scenario,
        "scenario_sha256": hashlib.sha256(scen.read_bytes()).hexdigest(),
        "args": vars(args),
        "wall_s_total": time.time() - t0,
        "results": results,
    }
    if args.dry:
        print({k: v for k, v in payload.items() if k != "results"})
        return
    print(write_result(args.id, payload))


if __name__ == "__main__":
    main()
