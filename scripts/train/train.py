"""Train one ladder rung (`docs/03 §2`) on the ADR-014 corpus with DDP over the four cards.

    python -m scripts.train.train --rung L0 --id 100-l0-screen --tokens 1e9
    python -m scripts.train.train --rung L5d --id 105-l5d-screen --tokens 1e9 --micro-batch 4

Conventions (docs/06 §3–§5): plain DDP, all experts replicated, ≤ 10 GB per
GPU, host-bounced NCCL, 0.5 M-token batches by gradient accumulation,
checkpoints ≤ 30 min apart, every run leaves `config.yaml`, `log.jsonl` and a
`results.md` with the numbers in `experiments/<id>/` (append-only: an existing
id is refused unless `--resume`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch import nn

from model.config import ModelCfg, ladder
from model.model import BigMoE
from model.sparsity import apply_fixed_24_masks, sparsity_fraction
from scripts.bench._common import RESULTS_DIR, provenance
from scripts.train.data import MemmapCorpus


@dataclass
class TrainArgs:
    rung: str
    id: str
    tokens: float
    batch_tokens: int = 524_288
    micro_batch: int = 4
    lr: float = 6e-4
    min_lr_frac: float = 0.1
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    clip: float = 1.0
    seed: int = 0
    r_schedule: str = "fixed"  # fixed = r_max every step; uniform = r ~ U[r_min, r_max] per step
    eval_every: int = 100
    eval_tokens: int = 2_000_000  # subset during training; full val at the end
    ckpt_minutes: float = 30.0
    corpus: str = "/mnt/nvme/corpus/fineweb-edu/tok-mistral32k"
    out_dir: str | None = None  # default experiments/<id>
    gpus: int = 0  # 0 = all
    resume: bool = False
    smoke_steps: int = 0  # > 0: run this many steps into out_dir and stop (no experiment record)


# ---- schedule and optimiser ---------------------------------------------------------
def lr_at(step: int, total: int, a: TrainArgs) -> float:
    warm = max(1, int(a.warmup_frac * total))
    if step < warm:
        return a.lr * (step + 1) / warm
    prog = min(1.0, (step - warm) / max(1, total - warm))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def make_optimizer(model: nn.Module, a: TrainArgs) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # no decay on norms, biases, depth/stream embeddings, router keys (PEER practice, 02 §13.6)
        if p.dim() < 2 or "router.keys" in name or "depth" in name or "stream_tag" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": a.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=a.lr, betas=a.betas, fused=True)


# ---- thermals (docs/06 §7: a throttled card invalidates throughput numbers) ---------------
# NVML packs every reason into one bitmask, and they are not equivalent. I23 is about
# *thermal* slowdown; hitting the 170 W software power cap is the card working as
# configured (`015`: these cards are thermally, not power, limited) and a well-cooled
# card reaches it more often, not less. Counting them together makes better cooling
# look like more throttling, so `n_thermal` is the field the acceptance criterion and
# the abort rule read; `n_throttled` is kept so records before 024 stay comparable.
SW_POWER_CAP = 0x4
THERMAL_REASONS = 0x20 | 0x40  # SwThermalSlowdown | HwThermalSlowdown


def gpu_thermals() -> dict[str, float]:
    """min SM clock, max temperature and the active throttle reasons across all cards."""
    import subprocess

    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except Exception:  # noqa: BLE001 — telemetry must never stop training
        return {}
    rows = [line.split(", ") for line in out]
    clocks = [float(r[0]) for r in rows]
    temps = [float(r[1]) for r in rows]
    reasons = [int(r[2], 16) for r in rows]
    throttled = sum(r not in (0, 1) for r in reasons)  # 0x1 = GPU idle, not a throttle
    return {
        "sm_mhz_min": min(clocks),
        "temp_c_max": max(temps),
        "n_throttled": float(throttled),
        "n_thermal": float(sum(bool(r & THERMAL_REASONS) for r in reasons)),
        "n_power_cap": float(sum(bool(r & SW_POWER_CAP) for r in reasons)),
    }


# ---- evaluation ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: BigMoE,
    corpus: MemmapCorpus,
    cfg: ModelCfg,
    rank: int,
    world: int,
    max_tokens: int | None,
    dev: torch.device,
    r: int | None,
) -> dict[str, float]:
    model.eval()
    windows = corpus.val_windows(cfg.context, max_tokens)[rank::world]
    tot = torch.zeros(3, device=dev)  # sum loss, count, sum r
    mtp_sums: dict[str, float] = {}
    for i in range(0, len(windows), 8):
        toks = torch.cat(windows[i : i + 8]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.loss(toks, r=r)
        n = toks.shape[0]
        tot[0] += out.main.float() * n
        tot[1] += n
        tot[2] += out.metrics.get("r_mean", float("nan")) * n
        for k, v in out.metrics.items():
            if k.startswith("mtp_acc"):
                mtp_sums[k] = mtp_sums.get(k, 0.0) + v * n
    if world > 1:
        dist.all_reduce(tot)
        for k in sorted(mtp_sums):
            t = torch.tensor(mtp_sums[k], device=dev)
            dist.all_reduce(t)
            mtp_sums[k] = float(t)
    model.train()
    res = {"val_loss": float(tot[0] / tot[1]), "val_windows": float(tot[1])}
    if cfg.arch == "recurrent":
        res["val_r_mean"] = float(tot[2] / tot[1])
    for k, v in mtp_sums.items():
        res["val_" + k] = v / float(tot[1])
    return res


# ---- the worker -------------------------------------------------------------------
def worker(rank: int, world: int, a: TrainArgs, cfg: ModelCfg, out: Path) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29531")
        os.environ["NCCL_P2P_DISABLE"] = "1"
        dist.init_process_group(
            "nccl", rank=rank, world_size=world, device_id=torch.device(f"cuda:{rank}")
        )
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    torch.manual_seed(a.seed)
    is_main = rank == 0

    corpus = MemmapCorpus(Path(a.corpus))
    model = BigMoE(cfg).to(dev)
    raw = model
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], broadcast_buffers=False
        )
    opt = make_optimizer(raw, a)

    accum = a.batch_tokens // (world * a.micro_batch * cfg.context)
    if accum < 1:
        raise SystemExit("batch_tokens smaller than one micro-step across the world")
    step_tokens = accum * world * a.micro_batch * cfg.context
    total_steps = a.smoke_steps or int(a.tokens // step_tokens)
    sparsity_step = int(0.01 * total_steps) if cfg.numerics.sparsity_2_4 else None

    step, tokens_seen = 0, 0
    ckpt_path = out / "ckpt" / "latest.pt"
    if a.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=dev, weights_only=False)
        raw.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, tokens_seen = state["step"], state["tokens"]
        if state.get("masked") and sparsity_step is not None:
            apply_fixed_24_masks(raw)
            raw.load_state_dict(state["model"])
        if is_main:
            print(f"resumed at step {step}, {tokens_seen:,} tokens", flush=True)
    rng = np.random.default_rng(a.seed * 1000 + rank + step * 7919)
    r_rng = np.random.default_rng(a.seed + 17)
    for _ in range(step):
        r_rng.integers(cfg.middle.r_min, cfg.middle.r_max + 1)  # replay the r schedule

    log = open(out / "log.jsonl", "a") if is_main else None  # noqa: SIM115 — long-lived
    t_start, t_ckpt = time.time(), time.time()
    masked = sparsity_step is not None and step > sparsity_step
    model.train()
    while step < total_steps:
        if sparsity_step is not None and step == sparsity_step and not masked:
            n = apply_fixed_24_masks(raw)
            masked = True
            if is_main:
                print(
                    f"step {step}: fixed 2:4 masks on {n} expert matrices "
                    f"(sparsity {sparsity_fraction(raw):.3f})",
                    flush=True,
                )
        lr = lr_at(step, total_steps, a)
        for g in opt.param_groups:
            g["lr"] = lr
        r_val = (
            cfg.middle.r_max
            if a.r_schedule == "fixed"
            else int(r_rng.integers(cfg.middle.r_min, cfg.middle.r_max + 1))
        )
        r = None if cfg.halting.mode == "act" else r_val
        sums = torch.zeros(5, device=dev)  # loss, main, mtp, aux, ponder
        metric_sums: dict[str, float] = {}
        load_sums: list[torch.Tensor] = []
        t0 = time.time()
        for micro in range(accum):
            toks = corpus.train_batch(rng, a.micro_batch, cfg.context).to(dev, non_blocking=True)
            ctx = model.no_sync() if (world > 1 and micro < accum - 1) else _nullctx()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(toks, r=r)
                (o.loss / accum).backward()
            sums += torch.stack(
                [
                    o.loss.detach(),
                    o.main.detach(),
                    o.mtp.detach(),
                    o.router_aux.detach(),
                    o.ponder.detach(),
                ]
            ).float()
            for k, v in o.metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            loads = [
                m.last_route.load
                for m in raw.modules()
                if hasattr(m, "last_route") and m.last_route is not None
            ]
            if loads:
                load_sums = (
                    [ls + ld for ls, ld in zip(load_sums, loads, strict=True)]
                    if load_sums
                    else loads
                )
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        # aux-loss-free load controller on the world-averaged load per router (one per MoE)
        if load_sums:
            moes = [m for m in raw.modules() if hasattr(m, "last_route")]
            for moe, ls in zip(moes, load_sums, strict=True):
                ld = ls / accum
                if world > 1:
                    dist.all_reduce(ld)
                    ld /= world
                moe.router.update_bias(ld)
        step += 1
        tokens_seen += step_tokens
        torch.cuda.synchronize(dev)  # the step time must include queued GPU work (I22)
        dt = time.time() - t0
        if world > 1:
            dist.all_reduce(sums)
            sums /= world
        if is_main:
            rec: dict[str, Any] = {
                "step": step,
                "tokens": tokens_seen,
                "lr": lr,
                "r": r_val,
                "loss": float(sums[0] / accum),
                "main": float(sums[1] / accum),
                "mtp": float(sums[2] / accum),
                "router_aux": float(sums[3] / accum),
                "ponder": float(sums[4] / accum),
                "grad_norm": float(gnorm),
                "step_s": dt,
                "tok_s": step_tokens / dt,
                "peak_gib": torch.cuda.max_memory_allocated(dev) / 2**30,
            }
            for k, v in metric_sums.items():
                rec[k] = v / accum
            if load_sums:
                ld = torch.stack([ls / accum for ls in load_sums])  # [n_moe, n_exp]
                rec["load_max"] = float(ld.max())
                rec["load_min"] = float(ld.min())
                # `load_max`/`load_min` are extremes over *every* router at once, so they
                # cannot tell one collapsed layer from mild spread everywhere. Per-router
                # entropy can: exp(H) is the effective expert count, directly comparable
                # to n_experts, and `route_eff_min` is the worst router in the model.
                # Tiny tensor ops on already-detached loads, after the optimiser step —
                # they cannot perturb training. Added 2026-09-11, so absent from 106/107;
                # `scripts/analysis/probe_routing.py` recovers the endpoint from any
                # checkpoint. n_exp = 1 (L5-ne1) has no entropy to speak of, hence the guard.
                n_exp = ld.shape[-1]
                if n_exp > 1:
                    p = ld / ld.sum(-1, keepdim=True).clamp_min(1e-12)
                    ent = -(p * p.clamp_min(1e-12).log()).sum(-1)  # [n_moe], nats
                    rec["route_ent_mean"] = float(ent.mean() / math.log(n_exp))
                    rec["route_ent_min"] = float(ent.min() / math.log(n_exp))
                    rec["route_eff_min"] = float(ent.min().exp())
                    rec["route_dead"] = float((p < 0.1 / n_exp).sum())
            if step % 10 == 0 or step == total_steps or step <= 3:
                print(
                    f"step {step:5d}/{total_steps} tok {tokens_seen / 1e9:6.3f}B "
                    f"loss {rec['loss']:.4f} main {rec['main']:.4f} lr {lr:.2e} "
                    f"gn {rec['grad_norm']:.2f} {rec['tok_s']:8,.0f} tok/s "
                    f"{rec['peak_gib']:.2f} GiB",
                    flush=True,
                )
        t_eval = time.time()
        if step % a.eval_every == 0 or step == total_steps:
            ev = evaluate(
                raw,
                corpus,
                cfg,
                rank,
                world,
                None if step == total_steps else a.eval_tokens,
                dev,
                r,
            )
            if is_main:
                rec.update(ev)
                print(
                    f"  eval step {step}: val_loss {ev['val_loss']:.4f} "
                    f"({int(ev['val_windows'])} windows)",
                    flush=True,
                )
        if is_main:
            rec["eval_s"] = time.time() - t_eval
            rec["wall"] = time.time()  # absolute, to reconcile step time with wall-clock (I22)
            if step % 10 == 0:
                rec.update(gpu_thermals())  # I23: throttling is the wall-clock gap
            due = (time.time() - t_ckpt) / 60 >= a.ckpt_minutes
            if due or step == total_steps:
                t_save = time.time()
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": raw.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step,
                        "tokens": tokens_seen,
                        "masked": masked,
                        "config": cfg.to_dict(),
                    },
                    ckpt_path,
                )
                t_ckpt = time.time()
                rec["ckpt_s"] = time.time() - t_save
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if world > 1:
            dist.barrier()

    if is_main:
        wall = time.time() - t_start
        final = json.loads(Path(out / "log.jsonl").read_text().splitlines()[-1])
        summary = {
            "steps": step,
            "tokens": tokens_seen,
            "wall_s_this_session": wall,
            "final": final,
            "params": raw.n_params(),
            "world": world,
            "accum": accum,
            "step_tokens": step_tokens,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        _write_results_md(out, a, cfg, summary)
        log.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


class _nullctx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def _write_results_md(out: Path, a: TrainArgs, cfg: ModelCfg, s: dict[str, Any]) -> None:
    f = s["final"]
    lines = [
        f"# {a.id} — {a.rung} at `{'screen' if a.tokens <= 1.1e9 else 'small'}`",
        "",
        f"`python -m scripts.train.train --rung {a.rung} --id {a.id} --tokens {a.tokens:g} "
        f"--micro-batch {a.micro_batch} --seed {a.seed}` · {s['world']} × RTX 3060, DDP, "
        f"host-bounced NCCL · batch {s['step_tokens']:,} tokens "
        f"({s['accum']} × {a.micro_batch} × {cfg.context} × {s['world']}) · "
        f"config in `config.yaml`, per-step log in `log.jsonl`.",
        "",
        "**Bears on:** (fill in: H numbers and verdict per ADR-013).",
        "",
        "## Numbers",
        "",
        "| | |",
        "|---|---|",
        f"| params | {s['params']['total'] / 1e6:.1f} M total, "
        f"{s['params']['non_embedding'] / 1e6:.1f} M non-embedding |",
        f"| tokens | {s['tokens'] / 1e9:.3f} B in {s['steps']} steps |",
        f"| final val loss (full held-out, {int(f.get('val_windows', 0))} windows) | "
        f"**{f.get('val_loss', float('nan')):.4f}** nats |",
        f"| final train loss (last step, main head) | {f['main']:.4f} |",
        f"| throughput (last step) | {f['tok_s']:,.0f} tokens/s aggregate |",
        f"| peak memory per GPU | {f['peak_gib']:.2f} GiB |",
        f"| wall-clock (this session) | {s['wall_s_this_session'] / 3600:.2f} h |",
        f"| seed | {a.seed} |",
    ]
    for k in sorted(f):
        if k.startswith("val_mtp_acc") or k == "val_r_mean":
            lines.append(f"| {k} | {f[k]:.4f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "(one paragraph; state Δ vs the matched baseline, σ, T and the verdict rule)",
        "",
    ]
    (out / "results.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    for f_ in TrainArgs.__dataclass_fields__.values():
        if f_.name in ("rung", "id"):
            ap.add_argument(f"--{f_.name}", required=True)
        elif f_.name == "tokens":
            ap.add_argument("--tokens", type=float, required=True)
        elif f_.type == "bool" or isinstance(f_.default, bool):
            ap.add_argument(f"--{f_.name.replace('_', '-')}", action="store_true")
        elif f_.name == "betas":
            ap.add_argument("--betas", type=float, nargs=2, default=list(f_.default))
        else:
            typ = type(f_.default) if f_.default is not None else str
            ap.add_argument(f"--{f_.name.replace('_', '-')}", type=typ, default=f_.default)
    ap.add_argument(
        "--set", nargs="*", default=[], help="model config overrides key=value (dotted)"
    )
    ns = ap.parse_args()
    d = vars(ns)
    overrides = d.pop("set")
    d["betas"] = tuple(d["betas"])
    a = TrainArgs(**d)

    cfg = ladder(a.rung)
    if overrides:
        doc = cfg.to_dict()
        for kv in overrides:
            key, val = kv.split("=", 1)
            node = doc
            parts = key.split(".")
            for p_ in parts[:-1]:
                node = node[p_]
            node[parts[-1]] = yaml.safe_load(val)
        cfg = ModelCfg.from_dict(doc)
        cfg.validate()

    out = Path(a.out_dir) if a.out_dir else RESULTS_DIR / a.id
    if a.smoke_steps == 0 and (out / "config.yaml").exists() and not a.resume:
        sys.exit(
            f"refusing to overwrite {out} — experiments are append-only, pick a new id or --resume"
        )
    out.mkdir(parents=True, exist_ok=True)
    if not a.resume:
        doc = {"model": cfg.to_dict(), "train": asdict(a), "provenance": provenance()}
        doc = json.loads(json.dumps(doc, default=str))  # plain types only (TorchVersion etc.)
        (out / "config.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    world = a.gpus or torch.cuda.device_count()
    if world == 1:
        worker(0, 1, a, cfg, out)
    else:
        mp.spawn(worker, args=(world, a, cfg, out), nprocs=world, join=True)


if __name__ == "__main__":
    main()
