"""Per-router expert-load diagnostic from a finished run's checkpoint (`docs/01 §4.4`).

The trainer logs only `load_max` and `load_min` taken over *every* router at once,
which answers "is anything badly off" but not "what does the distribution look
like, per layer". This probe rebuilds a finished model from its checkpoint, runs
held-out batches through it, and reports the full picture per router.

The headline number is the **effective expert count**, exp(H) of the load
distribution: 8.0 means perfectly uniform over 8 experts, 1.0 means total
collapse onto one. It is the entropy expressed in units a reader can compare
against `n_experts` without doing arithmetic.

This reads a checkpoint and never touches the training path, so it is safe to
run against a rung while later rungs are still training. Default device is CPU
for exactly that reason -- a GPU run would contend with the ladder and perturb
its throughput numbers. `latest.pt` is overwritten every `ckpt_minutes`, so this
measures the *converged* routing of a finished run; the trajectory needs the
trainer's own `route_ent_*` fields, added 2026-09-11 and therefore absent from
`106` and `107`.

Run: `python -m scripts.analysis.probe_routing --id 106-l1-screen`
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from model.config import ModelCfg
from model.model import BigMoE
from scripts.train.data import MemmapCorpus

RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments"


def load_run(run_dir: Path, device: torch.device) -> tuple[BigMoE, ModelCfg, int]:
    ckpt = torch.load(run_dir / "ckpt" / "latest.pt", map_location="cpu", weights_only=False)
    doc = ckpt["config"]
    cfg = ModelCfg.from_dict(doc.get("model", doc))
    model = BigMoE(cfg)
    # The trainer saves a DDP-wrapped state dict on some paths; tolerate both.
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    return model.to(device).eval(), cfg, int(ckpt["step"])


@torch.no_grad()
def collect_loads(
    model: BigMoE,
    cfg: ModelCfg,
    corpus: MemmapCorpus,
    batches: int,
    batch: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Sum each router's load vector over `batches` held-out batches.

    Loads are already normalised per forward pass, so summing and dividing by the
    number of passes gives the mean share -- equal weighting per batch, which is
    what we want when every batch has the same token count.
    """
    windows = corpus.val_windows(cfg.context, None)
    totals: list[torch.Tensor] | None = None
    used = 0
    for i in range(0, min(batches * batch, len(windows)), batch):
        toks = torch.cat(windows[i : i + batch]).to(device)
        if toks.shape[0] == 0:
            break
        _, ex = model.forward_hidden(toks[:, :-1])
        if not ex.loads:
            return []
        loads = [x.detach().float().cpu() for x in ex.loads]
        totals = loads if totals is None else [a + b for a, b in zip(totals, loads, strict=True)]
        used += 1
    return [t / used for t in (totals or [])]


def describe(load: torch.Tensor, dead_frac: float) -> dict[str, Any]:
    """Entropy, effective expert count and the tails, for one router."""
    p = (load / load.sum().clamp_min(1e-12)).double()
    nz = p[p > 0]
    ent = float(-(nz * nz.log()).sum())
    n = int(p.numel())
    return {
        "n_experts": n,
        "entropy_nats": ent,
        "entropy_normalised": ent / math.log(n) if n > 1 else 1.0,
        "effective_experts": math.exp(ent),
        "max_share": float(p.max()),
        "min_share": float(p.min()),
        "dead_experts": int((p < dead_frac / n).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="run id under experiments/")
    ap.add_argument("--corpus", default="/mnt/nvme/corpus/fineweb-edu/tok-mistral32k")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4, help="sequences per batch")
    ap.add_argument(
        "--device", default="cpu", help="cpu by default so a running ladder is untouched"
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch CPU threads; 0 = all. The probe saturated ~10 of 12 cores on 106, "
        "which would dent a running ladder's throughput (the trainer needs cores for "
        "memmap loading and host-staged NCCL). Cap it, or better, wait for a gap — "
        "each run keeps its own final checkpoint, so the endpoint never expires.",
    )
    ap.add_argument(
        "--dead-frac",
        type=float,
        default=0.1,
        help="an expert is 'dead' below this fraction of its uniform share",
    )
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    run_dir = RESULTS_DIR / args.id
    device = torch.device(args.device)
    model, cfg, step = load_run(run_dir, device)
    corpus = MemmapCorpus(Path(args.corpus))

    print(
        f"{args.id}: step {step}, {cfg.arch} / ff={cfg.ff}, {args.batches} x {args.batch} "
        f"held-out sequences of {cfg.context} on {device}"
    )
    loads = collect_loads(model, cfg, corpus, args.batches, args.batch, device)
    if not loads:
        print("  no routers in this model (dense) — nothing to report")
        return

    rows = [describe(x, args.dead_frac) for x in loads]
    n_exp = rows[0]["n_experts"]
    print(f"  {len(rows)} routers, {n_exp} experts each, uniform share {1 / n_exp:.4f}")
    hdr = (
        f"  {'router':>7s} {'eff.experts':>12s} {'H/Hmax':>8s} {'max':>7s} {'min':>7s} {'dead':>5s}"
    )
    print(hdr)
    for i, r in enumerate(rows):
        print(
            f"  {i:7d} {r['effective_experts']:12.2f} {r['entropy_normalised']:8.3f} "
            f"{r['max_share']:7.4f} {r['min_share']:7.4f} {r['dead_experts']:5d}"
        )
    worst = min(rows, key=lambda r: r["effective_experts"])
    print(
        f"  worst router: {worst['effective_experts']:.2f} of {n_exp} effective experts, "
        f"{worst['dead_experts']} dead"
    )

    out = run_dir / "routing.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out} — experiments are append-only")
    out.write_text(
        json.dumps(
            {
                "run_id": args.id,
                "step": step,
                "arch": cfg.arch,
                "ff": cfg.ff,
                "batches": args.batches,
                "batch": args.batch,
                "device": str(device),
                "dead_frac": args.dead_frac,
                "per_router": rows,
                "loads": [x.tolist() for x in loads],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
