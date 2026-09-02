"""End-to-end training-step throughput for the `small` dense config (`docs/06 §6`).

Replaces the "8 TFLOPS per card dense / 5 TFLOPS recurrent" *assumptions* of
`docs/06 §4` with a measured tokens/s. This is a benchmark, not a training run:
random tokens, a few hundred steps, no checkpoint, no eval. It is deliberately
self-contained and lives in `scripts/bench/`, not `model/` — `model/` stays a
skeleton until Phase 1 and its reference implementation will be built for
readability, not for this number.

The dense model is the L0 baseline of `docs/06 §3`: 12 layers, d = 768, 12/4
heads (GQA), SwiGLU d_ff = 2048, V = 32 k, context 2048. MFU is reported against
`costmodel.dense_baseline_flops_per_token` so the throughput number and the
budget it feeds use the same accounting.

The recurrent config is a stub: it needs the middle block of `docs/01 §4`, which
is Phase-1 work and is not something to hand-roll here.

Run: `python -m scripts.bench.bench_train_step --id 009-train-step`
     `python -m scripts.bench.bench_train_step --id 010-train-step-ddp --ddp`
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from costmodel import Attention, dense_baseline_flops_per_token, dense_baseline_params
from scripts.bench._common import write_result


@dataclass(frozen=True)
class Shape:
    """`docs/06 §3` `small` dense anchor."""

    layers: int = 12
    d: int = 768
    heads: int = 12
    kv_heads: int = 4
    head_dim: int = 64
    d_ff: int = 2048
    vocab: int = 32_000
    context: int = 2048


class Block(nn.Module):
    """Pre-norm attention (GQA, causal, SDPA) + SwiGLU. Nothing exotic."""

    def __init__(self, s: Shape) -> None:
        super().__init__()
        self.s = s
        self.n1 = nn.RMSNorm(s.d)
        self.wq = nn.Linear(s.d, s.heads * s.head_dim, bias=False)
        self.wk = nn.Linear(s.d, s.kv_heads * s.head_dim, bias=False)
        self.wv = nn.Linear(s.d, s.kv_heads * s.head_dim, bias=False)
        self.wo = nn.Linear(s.heads * s.head_dim, s.d, bias=False)
        self.n2 = nn.RMSNorm(s.d)
        self.gate = nn.Linear(s.d, s.d_ff, bias=False)
        self.up = nn.Linear(s.d, s.d_ff, bias=False)
        self.down = nn.Linear(s.d_ff, s.d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        s = self.s
        h = self.n1(x)
        q = self.wq(h).view(b, t, s.heads, s.head_dim).transpose(1, 2)
        k = self.wk(h).view(b, t, s.kv_heads, s.head_dim).transpose(1, 2)
        v = self.wv(h).view(b, t, s.kv_heads, s.head_dim).transpose(1, 2)
        rep = s.heads // s.kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(a.transpose(1, 2).reshape(b, t, -1))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Dense(nn.Module):
    def __init__(self, s: Shape) -> None:
        super().__init__()
        self.emb = nn.Embedding(s.vocab, s.d)
        self.blocks = nn.ModuleList(Block(s) for _ in range(s.layers))
        self.norm = nn.RMSNorm(s.d)
        self.head = nn.Linear(s.d, s.vocab, bias=False)
        self.head.weight = self.emb.weight  # tied, as the docs/06 §3 count assumes

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


def build(rung: str | None, s: Shape) -> tuple[nn.Module, int | None, str]:
    """`rung` = a `docs/03 §2` ladder step built from `model/` (Phase 1); None = the
    Phase-0 dense reference above (records 009/010)."""
    if rung is None:
        return Dense(s), None, "small_dense"
    from model.config import ladder
    from model.model import BigMoE

    cfg = ladder(rung)
    r = None if cfg.halting.mode == "act" else cfg.middle.r_max
    return BigMoE(cfg), r, rung


def flops_per_token(s: Shape) -> dict[str, float]:
    """Training FLOPs/token, from the cost model. Two conventions reported."""
    fwd_no_head = dense_baseline_flops_per_token(
        s.layers, s.d, s.d_ff, s.heads * s.head_dim, s.kv_heads * s.head_dim,
        n_keys=(s.context + 1) // 2, vocab=s.vocab, include_head=False,
    )
    fwd_head = dense_baseline_flops_per_token(
        s.layers, s.d, s.d_ff, s.heads * s.head_dim, s.kv_heads * s.head_dim,
        n_keys=(s.context + 1) // 2, vocab=s.vocab, include_head=True,
    )
    return {"train_no_head": 3 * fwd_no_head, "train_with_head": 3 * fwd_head}


def run_steps(
    rank: int,
    world: int,
    s: Shape,
    micro_batch: int,
    steps: int,
    warmup: int,
    ddp: bool,
    rung: str | None = None,
) -> dict[str, Any]:
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    torch.manual_seed(0)

    model, r, _ = build(rung, s)
    model = model.to(dev)
    if ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], broadcast_buffers=False
        )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=True)

    def step() -> None:
        idx = torch.randint(0, s.vocab, (micro_batch, s.context + 1), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if rung is None:
                logits = model(idx[:, :-1])
                # No .float() here: a fp32 copy of the logits (mb*ctx*V*4 B = 2 GiB at
                # mb=8) is what pushes this over 10 GiB; cross_entropy upcasts internally.
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, s.vocab), idx[:, 1:-1].reshape(-1)
                )
            else:
                loss = model(idx, r=r).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(dev)

    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    tokens_per_step = micro_batch * s.context
    med = statistics.median(times)
    return {
        "rank": rank,
        "micro_batch_seqs": micro_batch,
        "tokens_per_step_per_gpu": tokens_per_step,
        "step_s_median": med,
        "step_s_min": min(times),
        "step_s_p90": sorted(times)[int(0.9 * len(times))],
        "tokens_per_s_per_gpu": tokens_per_step / med,
        "peak_mem_bytes": torch.cuda.max_memory_allocated(dev),
    }


def _ddp_worker(
    rank: int, world: int, s: Shape, mb: int, steps: int, warmup: int, out: dict, rung: str | None
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29518")
    os.environ["NCCL_P2P_DISABLE"] = "1"
    dist.init_process_group(
        "nccl", rank=rank, world_size=world, device_id=torch.device(f"cuda:{rank}")
    )
    out[rank] = run_steps(rank, world, s, mb, steps, warmup, ddp=True, rung=rung)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="009-train-step")
    ap.add_argument("--micro-batch", type=int, default=8, help="sequences per GPU per step")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--ddp", action="store_true", help="all GPUs, DDP, host-bounced NCCL")
    ap.add_argument(
        "--rung",
        nargs="*",
        default=None,
        help="ladder steps from model/ (e.g. L5 L5d), optionally rung:micro_batch; default dense",
    )
    args = ap.parse_args()
    if args.rung:
        _main_rungs(args)
        return

    s = Shape()
    config_name = "small_dense"
    params = dense_baseline_params(
        s.layers, s.d, Attention(s.heads, s.kv_heads, s.head_dim), s.d_ff, s.vocab
    )
    fpt = flops_per_token(s)

    if args.ddp:
        world = torch.cuda.device_count()
        mgr = mp.Manager()
        out = mgr.dict()
        mp.spawn(
            _ddp_worker,
            args=(world, s, args.micro_batch, args.steps, args.warmup, out, args.rung),
            nprocs=world,
            join=True,
        )
        per_gpu = [dict(out[r]) for r in range(world)]
    else:
        world = 1
        per_gpu = [
            run_steps(0, 1, s, args.micro_batch, args.steps, args.warmup, ddp=False, rung=args.rung)
        ]

    # Aggregate: the slowest rank sets DDP step time.
    slowest = max(r["step_s_median"] for r in per_gpu)
    tokens_per_step = sum(r["tokens_per_step_per_gpu"] for r in per_gpu)
    agg_tps = tokens_per_step / slowest
    achieved = {k: agg_tps * v for k, v in fpt.items()}
    phi_measured = 27.1e12  # 003-gemm gpu0; the scenario file carries the citation

    payload = {
        "bench": "train_step",
        "config": config_name,
        "shape": s.__dict__,
        "params": params,
        "flops_per_token": fpt,
        "world_size": world,
        "ddp": args.ddp,
        "per_gpu": per_gpu,
        "aggregate_tokens_per_s": agg_tps,
        "achieved_tflops_total": {k: v / 1e12 for k, v in achieved.items()},
        "achieved_tflops_per_gpu": {k: v / 1e12 / world for k, v in achieved.items()},
        "mfu_vs_measured_phi": {k: v / world / phi_measured for k, v in achieved.items()},
        "note": "rungs are built from model/ (Phase 1); FLOPs/token is the matched budget",
    }
    path = write_result(args.id, payload)
    print(f"wrote {path}\n")
    print(f"  small dense: {params['total'] / 1e6:.1f} M params "
          f"({params['non_embedding'] / 1e6:.1f} M non-embedding)")
    print(f"  {world} GPU(s), micro-batch {args.micro_batch} x {s.context} tokens")
    for r in per_gpu:
        print(f"    gpu{r['rank']}: {r['step_s_median'] * 1e3:7.1f} ms/step  "
              f"{r['tokens_per_s_per_gpu']:8,.0f} tok/s  "
              f"peak {r['peak_mem_bytes'] / 2**30:.2f} GiB")
    print(f"\n  aggregate {agg_tps:,.0f} tokens/s")
    for k in fpt:
        print(f"    {k:16s}: {payload['achieved_tflops_per_gpu'][k]:5.2f} TFLOPS/GPU  "
              f"MFU {100 * payload['mfu_vs_measured_phi'][k]:4.1f}%")
    # docs/06 §4 budget check
    for name, toks in (("screen", 1e9), ("small", 2.5e9)):
        print(f"    {name:6s} ({toks:.1e} tok): {toks / agg_tps / 3600:5.1f} h at this rate")
    _ = math  # keep import for future use without lint noise


def _measure(rung: str, mb: int, steps: int, warmup: int, ddp: bool, s: Shape) -> dict[str, Any]:
    if ddp:
        world = torch.cuda.device_count()
        mgr = mp.Manager()
        out = mgr.dict()
        mp.spawn(
            _ddp_worker, args=(world, s, mb, steps, warmup, out, rung), nprocs=world, join=True
        )
        per_gpu = [dict(out[r]) for r in range(world)]
    else:
        world = 1
        per_gpu = [run_steps(0, 1, s, mb, steps, warmup, ddp=False, rung=rung)]
    slowest = max(r["step_s_median"] for r in per_gpu)
    tokens_per_step = sum(r["tokens_per_step_per_gpu"] for r in per_gpu)
    model, _, _ = build(rung, s)
    n_params = sum(p.numel() for p in model.parameters())
    del model
    return {
        "rung": rung,
        "micro_batch": mb,
        "world_size": world,
        "params_total": n_params,
        "per_gpu": per_gpu,
        "aggregate_tokens_per_s": tokens_per_step / slowest,
        "peak_mem_gib_max": max(r["peak_mem_bytes"] for r in per_gpu) / 2**30,
    }


def _main_rungs(args: argparse.Namespace) -> None:
    """Phase-1 rungs from model/: one payload, one row per rung, at the matched budget."""
    s = Shape()
    fpt = flops_per_token(s)
    rows = []
    for spec in args.rung:
        rung, _, mb = spec.partition(":")
        mb_i = int(mb) if mb else args.micro_batch
        row = _measure(rung, mb_i, args.steps, args.warmup, args.ddp, s)
        rows.append(row)
        print(
            f"  {rung:9s} mb {mb_i} x{row['world_size']}: "
            f"{row['aggregate_tokens_per_s']:8,.0f} tok/s aggregate "
            f"({row['per_gpu'][0]['step_s_median'] * 1e3:.0f} ms/step), "
            f"peak {row['peak_mem_gib_max']:.2f} GiB, {row['params_total'] / 1e6:.1f} M params, "
            f"screen {1e9 / row['aggregate_tokens_per_s'] / 3600:.1f} h",
            flush=True,
        )
    payload = {
        "bench": "train_step_rungs",
        "shape": s.__dict__,
        "flops_per_token_budget": fpt,  # matched budget (dense anchor), same for every rung
        "ddp": args.ddp,
        "rungs": rows,
    }
    path = write_result(args.id, payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
