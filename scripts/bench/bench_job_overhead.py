"""Fixed cost per job of S0's decode stack (ADR-050 §3): time vs rows for each job kind, and
the latency of the harness's collectives; and the owner kinds' efficiency against batch
(ADR-051 §2-3). Feeds the whole of `local_3060.yaml`'s `software` section.

Every card runs the same sweep at once (one process each), so CPU contention matches
`s0_ep_decode`. Each job is the exact code of one harness phase (`scripts/bench/s0_ep_decode.py`,
`Rank.round` / `Rank.position`), timed the same way, between two `cuda.synchronize()` calls:

    skel_in   decode.early: embedding + early blocks, b sequences
    attn      decode.owner_pre + building the dispatch payload, b sequences
    post      the gate-weighted index_add + owner_post, b sequences (k rows each)
    skel_out  decode.final: final blocks + head, b sequences
    expert    one hosted expert's selection and call on b rows

The attention-reading kinds see `--keys` keys (S0's mean, 1792). Owner kinds run up to
`--max-owner` sequences per card (256 = S0's largest, z = 1024 over four owners): at 1024 their
caches alone would pass the 10 GiB ceiling. The expert runs the whole row sweep.
Collectives: a counts
`all_to_all_single` (4 int64) and a rows `all_to_all_single` at S0's per-rank sizes. The roofline
of each kind is `sim/`'s own (decisions 1, 6, 12 with t0 = 0), so `compose` ∈ {max, sum} is chosen
by which reproduces this bench's own curve, never by S0. Then, under that rule, each owner kind's
efficiency is η(n) = min(1, roofline(n) / measured work(n)), the work being t(n) - t0 under `sum`
and t(n) under `max` (ADR-051 §2).

Run on an idle rig only: `python -m scripts.bench.bench_job_overhead --id 037-job-overhead-eta`
(`035-job-overhead` was the ADR-050 run, on powers of two only.)
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model import decode
from model.model import BigMoE
from scripts.analysis.probe_routing import load_run
from scripts.bench._common import write_result
from scripts.bench.s0_ep_decode import ThermalSampler
from scripts.train.train import gpu_thermals

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("skel_in", "attn", "post", "skel_out", "expert")
OWNER_KINDS = KINDS[:4]
# ADR-051 §3: the owner grid (1-256 sequences, denser where η moves) joined with the expert's
# powers of two to 1024 rows
ROWS = sorted({1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256} | {2**i for i in range(11)})


def _sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _median_time(fn: Any, reps: int, warm: int) -> float:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = _sync()
        fn()
        ts.append(_sync() - t0)
    return statistics.median(ts)


def _jobs(m: BigMoE, b: int, keys: int, dev: torch.device) -> dict[str, Any]:
    """One closure per kind at b rows/sequences, each a copy of the harness's phase code."""
    cfg = m.cfg
    d, k, world = cfg.d, cfg.router.k, 4
    cache = decode.new_cache(m, b, keys, 1, dev, torch.bfloat16)  # one middle buffer suffices
    cache.set_length(keys - 1)  # each call appends the keys-th position, then is rewound
    tok = torch.randint(0, cfg.vocab, (b, 1), device=dev)
    c = torch.randn(b, 1, d, device=dev, dtype=torch.bfloat16)
    pos = torch.tensor([keys - 1], device=dev)

    def rewind() -> None:
        cache.set_length(keys - 1)

    def skel_in() -> None:
        decode.early(m, tok, cache)
        rewind()

    def attn() -> None:
        u, a, route = decode.owner_pre(m, 1, c, pos, cache.middle[0])
        flat = u.reshape(-1, d)
        n = flat.shape[0]
        expert_of = route.idx.reshape(-1)
        token_of = torch.arange(n, device=dev).repeat_interleave(k)
        dest = expert_of % world
        order = torch.argsort(dest, stable=True)
        torch.bincount(dest, minlength=world)
        torch.cat([flat[token_of[order]], expert_of[order, None].to(flat.dtype)], dim=1)
        rewind()

    route = m.middle.moe.router(c.reshape(-1, d), decode.depth_embed(m, 1, dev))
    back = torch.randn(b * k, d, device=dev, dtype=torch.bfloat16)
    order = torch.arange(b * k, device=dev)
    token_of = torch.arange(b, device=dev).repeat_interleave(k)
    a_out = torch.randn_like(c)

    def post() -> None:
        flat = c.reshape(-1, d)
        out = torch.zeros_like(flat)
        gate = route.gate.reshape(-1)[order, None]
        out.index_add_(0, token_of[order], (back * gate).to(out.dtype))
        decode.owner_post(c, a_out, out)

    def skel_out() -> None:
        decode.final(m, c, pos, cache)
        rewind()

    return {
        "skel_in": skel_in,
        "attn": attn,
        "post": post,
        "skel_out": skel_out,
        "expert": _expert_job(m, b, dev),
    }


def _expert_job(m: BigMoE, b: int, dev: torch.device) -> Any:
    x = torch.randn(b, m.cfg.d, device=dev, dtype=torch.bfloat16)
    ids = torch.zeros(b, dtype=torch.long, device=dev)
    expert0 = m.middle.moe.experts[0]

    def expert() -> None:
        y = torch.empty_like(x)
        sel = (ids == 0).nonzero(as_tuple=True)[0]
        int(sel.numel())  # the harness reads the batch size on the host
        if sel.numel():
            y[sel] = expert0(x[sel]).to(y.dtype)

    return expert


def _collectives(z_values: list[int], k: int, d: int, reps: int, dev: torch.device) -> list[dict]:
    """The harness's three collectives per round, at S0's sizes (rows spread evenly)."""
    out = []
    counts = torch.ones(4, dtype=torch.long, device=dev)
    recv_counts = torch.empty_like(counts)
    for z in z_values:
        rows = z // 4 * k  # rows one rank sends per round
        split = [rows // 4 + (1 if i < rows % 4 else 0) for i in range(4)]
        send = torch.randn(rows, d + 1, device=dev, dtype=torch.bfloat16)
        recv = torch.empty_like(send)
        ret = torch.randn(rows, d, device=dev, dtype=torch.bfloat16)
        back = torch.empty_like(ret)

        def run(fn: Any) -> float:
            dist.barrier()
            return _median_time(fn, reps, 10)

        send_counts = torch.tensor(split, dtype=torch.long, device=dev)

        def dispatch_phase(send_counts=send_counts, send=send) -> None:  # harness code
            rc_t = torch.empty_like(send_counts)
            dist.all_to_all_single(rc_t, send_counts)
            sc, rc = send_counts.tolist(), rc_t.tolist()
            r = send.new_empty(sum(rc), d + 1)
            dist.all_to_all_single(r, send, rc, sc)

        def return_phase(ret=ret, split=split) -> None:  # the harness's return code
            b_ = ret.new_empty(sum(split), d)
            dist.all_to_all_single(b_, ret, split, split)

        out.append(
            {
                "z": z,
                "rows_per_rank": rows,
                "counts_s": run(lambda: dist.all_to_all_single(recv_counts, counts)),
                "rows_out_s": run(
                    lambda s=send, r=recv, sp=split: dist.all_to_all_single(r, s, sp, sp)
                ),
                "rows_back_s": run(
                    lambda s=ret, r=back, sp=split: dist.all_to_all_single(r, s, sp, sp)
                ),
                "dispatch_phase_s": run(dispatch_phase),
                "return_phase_s": run(return_phase),
                "rows_out_bytes_off_rank": sum(split[1:]) * (d + 1) * 2,
                "rows_back_bytes_off_rank": sum(split[1:]) * d * 2,
            }
        )
    return out


def _worker(rank: int, world: int, args: argparse.Namespace, out: Any) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    m, cfg, _ = load_run(ROOT / "experiments" / args.run, dev)
    m = m.to(torch.bfloat16).eval()
    sampler = ThermalSampler() if rank == 0 else None
    if sampler is not None:
        sampler.__enter__()
    curves: dict[str, list[tuple[int, float]]] = {kd: [] for kd in KINDS}
    with torch.no_grad():
        for b in args.rows:
            dist.barrier()
            kinds = KINDS if b <= args.max_owner else ("expert",)
            jobs = _jobs(m, min(b, args.max_owner), args.keys, dev)
            if b > args.max_owner:  # expert rows only: rebuild its closure at b rows
                jobs = {"expert": _expert_job(m, b, dev)}
            for kd in kinds:
                curves[kd].append((b, _median_time(jobs[kd], args.reps, args.warm)))
            del jobs
            torch.cuda.empty_cache()
        coll = _collectives(args.z, cfg.router.k, cfg.d, args.reps, dev)
    therm = []
    if sampler is not None:
        sampler.__exit__()
        therm = sampler.samples
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, {"rank": rank, "curves": curves, "collectives": coll})
    if rank == 0:
        out["per_rank"] = gathered
        out["thermal"] = therm
        out["peak_gib"] = torch.cuda.max_memory_allocated(dev) / 2**30
    dist.destroy_process_group()


Curve = dict[str, list[tuple[int, float]]]


def _sim_rooflines(meas: Curve, keys: int) -> Curve:
    """`sim/`'s roofline for each kind at the measured row counts, t0 = 0 (decisions 1, 6, 12).
    The scenario is read without its `software` section's effect: t0 is forced to zero."""
    from sim.params import SOFTWARE_KINDS, load_params
    from sim.system import Simulation

    zero = {f"t0_{kd}_s": 0.0 for kd in SOFTWARE_KINDS}
    bare = load_params(
        "local_3060", "L5", context_tokens=keys, compose="sum", owner_eff=(), **zero
    )  # peak rooflines: no fixed cost and η = 1, whatever the scenario holds
    sim = Simulation(bare)
    roof: Curve = {}
    for kd, pts in meas.items():
        step = sim.expert_step_s if kd == "expert" else lambda b, kd=kd: sim.owner_step_s(kd, b)
        roof[kd] = [(b, step(b)[0]) for b, _ in pts]
    return roof


def _compose_fit(meas: Curve, roof: Curve) -> dict[str, Any]:
    """t0 = time at the smallest row count; worst relative error of each compose rule."""
    res: dict[str, Any] = {"t0_s": {kd: meas[kd][0][1] for kd in KINDS}, "worst_rel_error": {}}
    for rule in ("max", "sum"):
        per = {}
        for kd in KINDS:
            t0 = meas[kd][0][1]
            pred = [max(t0, r) if rule == "max" else t0 + r for _, r in roof[kd]]
            per[kd] = max(abs(p / t - 1) for p, (_, t) in zip(pred, meas[kd], strict=True))
        res["worst_rel_error"][rule] = per
    worst = {rule: max(v.values()) for rule, v in res["worst_rel_error"].items()}
    res["compose"] = min(worst, key=worst.__getitem__)
    res["worst_overall"] = worst
    return res


def _owner_efficiency(meas: Curve, roof: Curve, fit: dict[str, Any]) -> dict[str, dict[int, float]]:
    """η per owner kind and sequence count (ADR-051 §2), under the compose rule already chosen."""
    out: dict[str, dict[int, float]] = {}
    for kd in OWNER_KINDS:
        t0 = fit["t0_s"][kd]
        table = {}
        for (n, t), (_, r) in zip(meas[kd], roof[kd], strict=True):
            w = t - t0 if fit["compose"] == "sum" else t
            table[n] = 1.0 if w <= 0 else min(1.0, r / w)
        out[kd] = table
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", required=True)
    ap.add_argument("--run", default="132-l5-small-s0")
    ap.add_argument("--rows", type=int, nargs="+", default=ROWS)
    ap.add_argument("--keys", type=int, default=1792)
    ap.add_argument("--max-owner", type=int, default=256, help="largest owner batch")
    ap.add_argument("--z", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    world = torch.cuda.device_count()
    if world != 4:
        raise SystemExit(f"needs the four cards, found {world}")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    mgr = mp.Manager()
    out = mgr.dict()
    t0 = time.time()
    mp.spawn(_worker, args=(world, args, out), nprocs=world, join=True)
    per_rank = out["per_rank"]
    # per kind and row count: the median over the four cards (they ran at once)
    meas: Curve = {
        kd: [
            (b, statistics.median(g["curves"][kd][i][1] for g in per_rank))
            for i, (b, _) in enumerate(per_rank[0]["curves"][kd])
        ]
        for kd in KINDS
    }
    roof = _sim_rooflines(meas, args.keys)
    fit = _compose_fit(meas, roof)
    eta = _owner_efficiency(meas, roof, fit)
    coll = per_rank[0]["collectives"]
    # t_coll (ADR-050 §2): one collective as the harness runs it, at the smallest size: the
    # return phase at the smallest z, median over the cards. The dispatch phase is reported
    # beside it as the check of "two collectives" (it also reads the counts back to the host)
    t_coll = statistics.median(g["collectives"][0]["return_phase_s"] for g in per_rank)
    lam = 3.59e9  # local_3060 lambda_link (002-nccl), for the check only
    coll_check = [
        {
            "z": c["z"],
            "dispatch_measured_s": c["dispatch_phase_s"],
            "dispatch_model_s": 2 * t_coll + c["rows_out_bytes_off_rank"] / lam,
            "return_measured_s": c["return_phase_s"],
            "return_model_s": t_coll + c["rows_back_bytes_off_rank"] / lam,
        }
        for c in coll
    ]
    therm = list(out["thermal"])
    payload = {
        "bench": "bench_job_overhead",
        "adr": "ADR-050, ADR-051",
        "run": args.run,
        "stack": "PyTorch eager + model/decode.py (reference), bf16, NCCL_P2P_DISABLE=1",
        "args": vars(args),
        "rows": args.rows,
        "measured_s": meas,
        "sim_roofline_s": roof,
        "fit": fit,
        "owner_efficiency": eta,
        "t_coll_s": t_coll,
        "collectives_rank0": coll,
        "collective_model_check": coll_check,
        "per_rank": per_rank,
        "thermal": therm,
        "n_thermal_max": max((t.get("n_thermal", 0.0) for t in therm), default=float("nan")),
        "peak_gib_rank0": out["peak_gib"],
        "wall_s": time.time() - t0,
        "end_thermal": gpu_thermals(),
    }
    for kd in KINDS:
        print(kd, [(b, round(t * 1e3, 3)) for b, t in meas[kd]])
    t0_ms = {kd: round(v * 1e3, 3) for kd, v in fit["t0_s"].items()}
    print("fit", fit["compose"], fit["worst_overall"], "t0 ms", t0_ms)
    print("t_coll ms", round(t_coll * 1e3, 4))
    for kd, table in eta.items():
        print("eta", kd, {n: round(e, 3) for n, e in table.items()})
    for c in coll_check:
        print({k: (round(v * 1e3, 4) if isinstance(v, float) else v) for k, v in c.items()})
    if args.dry:
        return
    print(write_result(args.id, payload))


if __name__ == "__main__":
    main()
