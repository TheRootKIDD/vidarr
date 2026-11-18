"""Post-training round-to-nearest 4-bit weights (ADR-073 §2, rig `053`).

Loads a bf16 checkpoint, scores the full held-out set as trained, then with the attention and
SwiGLU weights fake-quantised (FP4 E2M1 and INT4, groups of 64, bf16 scales), all under the same
bf16 autocast, and reports each loss and its delta. Read-only; run with the GPUs idle:
`python -m scripts.analysis.probe_quant --id 100-l0-screen`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model.quant import set_qat, swap_linears
from scripts.analysis.eval_context import CORPUS, token_losses
from scripts.analysis.probe_routing import RESULTS_DIR, load_run
from scripts.train.data import MemmapCorpus


def held_out(model: torch.nn.Module, windows: list[torch.Tensor], dev: torch.device) -> float:
    tot, n = 0.0, 0
    for i in range(0, len(windows), 8):
        toks = torch.cat(windows[i : i + 8]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += float(token_losses(model, toks).double().sum())
        n += toks[:, 1:].numel()
    return tot / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--formats", default="fp4_e2m1,int4")
    ap.add_argument("--max-windows", type=int, default=None, help="default: all 5606")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    run_dir = RESULTS_DIR / a.id
    model, cfg, step = load_run(run_dir, dev)
    if cfg.numerics.qat_format != "none":
        raise SystemExit("post-training rounding is for bf16-trained checkpoints")
    windows = MemmapCorpus(Path(CORPUS)).val_windows(cfg.context, None)[: a.max_windows]
    t0 = time.time()
    res = {"id": a.id, "step": step, "windows": len(windows), "bf16": held_out(model, windows, dev)}
    swap_linears(model, "fp4_e2m1", 64)
    for fmt in a.formats.split(","):
        set_qat(model, True, fmt)
        loss = held_out(model, windows, dev)
        res[fmt] = {"loss": loss, "delta": loss - res["bf16"]}
    res["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=2))
    (run_dir / "probe_quant.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
