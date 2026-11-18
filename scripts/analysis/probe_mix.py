"""L5f: which sources each iteration's input mix reads (ADR-072 §4).

Mean softmax weight on each source (the middle block's input v_0, then the updates u^(1..j-1))
for every iteration j's mix and the final blocks' mix, over held-out windows. Read-only on a
checkpoint; run with the GPUs idle:
`python -m scripts.analysis.probe_mix --id 198-l5f-screen`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.analysis.eval_context import CORPUS
from scripts.analysis.probe_routing import RESULTS_DIR, load_run
from scripts.train.data import MemmapCorpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    run_dir = RESULTS_DIR / a.id
    model, cfg, step = load_run(run_dir, dev)
    if cfg.middle.input_mix != "attention_residual":
        raise SystemExit("needs an L5f model")
    windows = MemmapCorpus(Path(CORPUS)).val_windows(cfg.context, None)[: a.windows]
    sums: list[torch.Tensor] | None = None
    n = 0
    with torch.no_grad():
        for i in range(0, len(windows), a.batch):
            toks = torch.cat(windows[i : i + a.batch]).to(dev)
            model.forward_hidden(toks[:, :-1])
            w = [x.cpu() for x in model.last_mix_weights]
            sums = w if sums is None else [s + x for s, x in zip(sums, w, strict=True)]
            n += 1
    assert sums is not None
    names = [f"iter{j}" for j in range(1, len(sums))] + ["final"]
    res = {
        "id": a.id,
        "step": step,
        "windows": len(windows),
        "sources": "v0 = middle input, then u1..u(j-1)",
        "mean_weights": {k: [round(float(x) / n, 4) for x in s] for k, s in zip(names, sums,
                                                                                strict=True)},
    }
    print(json.dumps(res, indent=2))
    (run_dir / "probe_mix.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
