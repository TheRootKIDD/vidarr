"""L5g: held-out loss against the selection size k_sel at eval (ADR-070 §2).

Rebuilds a trained L5g model and scores held-out windows with the sparse iterations keeping
k_sel ∈ {64, 128, 256} (and the trained value) of the G keys picked at each token's last full
iteration; reports the loss and the global keys read per token from `sim`'s counter. Read-only on
a checkpoint; run with the GPUs idle:
`python -m scripts.analysis.probe_ksel --id 194-l5g-fsss-screen`
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import time
from pathlib import Path

import torch

from scripts.analysis.eval_context import CORPUS, token_losses
from scripts.analysis.probe_routing import RESULTS_DIR, load_run
from scripts.train.data import MemmapCorpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--k-sel", default="64,128,256")
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device)
    run_dir = RESULTS_DIR / a.id
    model, cfg, step = load_run(run_dir, dev)
    if not cfg.attention.schedule or "S" not in cfg.attention.schedule:
        raise SystemExit("needs an L5g model with sparse iterations")
    windows = MemmapCorpus(Path(CORPUS)).val_windows(cfg.context, None)[: a.windows]
    n_g = cfg.context - cfg.attention.window  # |G| at the last position
    res: dict[str, object] = {"id": a.id, "step": step, "trained_k_sel": cfg.attention.k_sel}
    rows = {}
    t0 = time.time()
    for k in sorted({*(int(x) for x in a.k_sel.split(",")), cfg.attention.k_sel}):
        model.middle.cfg = dc.replace(cfg, attention=dc.replace(cfg.attention, k_sel=k))
        tot, n = 0.0, 0
        for i in range(0, len(windows), a.batch):
            toks = torch.cat(windows[i : i + a.batch]).to(dev)
            tot += float(token_losses(model, toks).double().sum())
            n += toks[:, 1:].numel()
        sched = cfg.attention.schedule
        reads = sum(n_g if c == "F" else min(k, n_g) if c == "S" else 0 for c in sched)
        rows[str(k)] = {"loss": tot / n, "global_keys_per_token_at_last_position": reads}
    model.middle.cfg = cfg
    res.update({"windows": len(windows), "k_sel": rows, "wall_s": round(time.time() - t0, 1)})
    print(json.dumps(res, indent=2))
    out = run_dir / "probe_ksel.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
