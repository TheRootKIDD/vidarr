"""ReLU^2 global attention: sparsity per iteration and head, and loss under a top-K cap.

(ADR-068 §3.)
On held-out windows of the training context:
- exact-zero fraction of the global weights, per iteration j and query head, and the V-skip rate
  (keys whose weight is zero for every query head of their KV group), per iteration;
- held-out loss with an eval-time cap that keeps the top-K global keys by pre-ReLU score,
  K in {64, 128, 256} and K = |G|max (must equal the uncapped loss: the check that capping to
  the model's own support changes nothing).

Read-only on a checkpoint. Run with the GPUs idle:
`python -m scripts.analysis.probe_relu2 --id 189-l9-relu2-screen`
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--caps", default="64,128,256")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device)
    run_dir = RESULTS_DIR / a.id
    model, cfg, step = load_run(run_dir, dev)
    if cfg.attention.global_scoring != "relu2":
        raise SystemExit("needs a ReLU^2-scored model (L9-relu2)")
    attn = model.middle.attn
    r, h = cfg.middle.r_max, cfg.attention.heads
    n_seg = cfg.context // cfg.attention.window
    windows = MemmapCorpus(Path(CORPUS)).val_windows(cfg.context, None)[: a.windows]
    t0 = time.time()

    def run(cap: int) -> tuple[float, torch.Tensor]:
        attn.eval_cap = cap
        attn.zero_stats.clear()
        tot, n, stats = 0.0, 0, []
        for i in range(0, len(windows), a.batch):
            toks = torch.cat(windows[i : i + a.batch]).to(dev)
            attn.zero_stats.clear()
            tot += float(token_losses(model, toks).double().sum())
            n += toks[:, 1:].numel()
            # calls are ordered (segment 1..n_seg-1, iteration 1..r): the first segment has no G
            stats.append(torch.stack(attn.zero_stats).view(n_seg - 1, r, -1).mean(0).cpu())
        attn.eval_cap = 0
        return tot / n, torch.stack(stats).mean(0)  # [r, H + Hkv]

    base, st = run(0)
    max_g = (n_seg - 1) * cfg.attention.window
    caps = {}
    for k in [int(x) for x in a.caps.split(",")] + [max_g]:
        loss_k, _ = run(k)
        caps[str(k)] = {"loss": loss_k, "delta": loss_k - base}
    res = {
        "id": a.id,
        "step": step,
        "windows": len(windows),
        "loss_uncapped": base,
        "zero_frac": float(st[:, :h].mean()),
        "vskip_frac": float(st[:, h:].mean()),
        "zero_frac_per_iteration": [float(x) for x in st[:, :h].mean(1)],
        "zero_frac_per_head": [float(x) for x in st[:, :h].mean(0)],
        "vskip_frac_per_iteration": [float(x) for x in st[:, h:].mean(1)],
        "cap": caps,
        "wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps(res, indent=2))
    out = run_dir / "probe_relu2.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
