"""Copy probe (ADR-066 §2): can the model copy a span it can only reach through the global range?

In each held-out window of 2048 tokens, positions 1792..1855 are overwritten with a copy of
positions 256..319. The source is ~1500 tokens back, beyond the local window W = 512, so an
L5d-family model can serve the copy only through its global cache G. Reported: the main head's
loss on copied tokens 2..64 (the first copied token is not predictable), against the same
positions in the unmodified window (the control). The gap control - copy is how much of the
span the model retrieves. A stand-in for the needle/copy evals H15b names, until they exist.

Reads a checkpoint, never touches the training path. Run on idle GPUs only (ADR-066 §2):
`python -m scripts.analysis.probe_copy --id 162-l5d-segmean-screen --device cuda`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.analysis.eval_context import CORPUS, token_losses
from scripts.analysis.probe_routing import RESULTS_DIR, load_run
from scripts.train.data import MemmapCorpus

SRC, DST, SPAN = 256, 1792, 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device)
    run_dir = RESULTS_DIR / a.id
    model, cfg, step = load_run(run_dir, dev)
    if cfg.context < DST + SPAN + 1:
        raise SystemExit(f"context {cfg.context} too short for the probe")
    ctx = cfg.context
    windows = MemmapCorpus(Path(CORPUS)).val_windows(ctx, None)[: a.windows]
    sel = slice(DST, DST + SPAN - 1)  # predictions of copied tokens 2..64
    copy_sum = ctrl_sum = 0.0
    n_tok = 0
    t0 = time.time()
    for i in range(0, len(windows), a.batch):
        toks = torch.cat(windows[i : i + a.batch]).to(dev)  # [B, ctx + 1]
        alt = toks.clone()
        alt[:, DST : DST + SPAN] = toks[:, SRC : SRC + SPAN]
        ctrl_sum += float(token_losses(model, toks)[:, sel].double().sum())
        copy_sum += float(token_losses(model, alt)[:, sel].double().sum())
        n_tok += toks.shape[0] * (SPAN - 1)
    res = {
        "id": a.id,
        "step": step,
        "context": ctx,
        "windows": len(windows),
        "src": [SRC, SRC + SPAN],
        "dst": [DST, DST + SPAN],
        "scored_tokens": n_tok,
        "copy_loss": copy_sum / n_tok,
        "control_loss": ctrl_sum / n_tok,
        "gap": (ctrl_sum - copy_sum) / n_tok,
        "wall_s": round(time.time() - t0, 1),
        "device": a.device,
    }
    print(json.dumps(res, indent=2))
    out = run_dir / "probe_copy.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
