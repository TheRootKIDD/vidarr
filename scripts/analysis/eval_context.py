"""Held-out loss of a finished run at a context length other than its training one.

The trainer evaluates only at `cfg.context`. H15a (`docs/03 §1`) states its range as
2 k–8 k, and the `small` rungs train at 2 k, so the 8 k end is owed as an eval. This
probe rebuilds a model from its checkpoint and evaluates it on `val.bin` in windows
of `--context` tokens, reporting the mean loss overall and per position bucket.

Read the buckets, not only the mean: a model trained at 2 k has never seen a RoPE
offset past 2047 (L5, global per-iteration attention) or a final-vector memory
longer than 2 k (L5d), so positions >= the training context measure *length
extrapolation*, not the cost of the attention design at a trained 8 k. Positions
below it are the in-distribution control and must reproduce the trainer's number.

Like `probe_routing`, this reads a checkpoint and never touches the training path.
Default device is CPU so that a running queue keeps its GPUs and its throughput
numbers; the trainer's DDP workers use one core each.

Run: `python -m scripts.analysis.eval_context --id 134-l5d-small-s0 --context 8192`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model.model import BigMoE
from scripts.analysis.probe_routing import RESULTS_DIR, load_run
from scripts.train.data import MemmapCorpus

CORPUS = "/mnt/nvme/corpus/fineweb-edu/tok-mistral32k"
EDGES = (0, 512, 1024, 2048, 4096, 8192)


@torch.no_grad()
def token_losses(model: BigMoE, toks: torch.Tensor) -> torch.Tensor:
    """Per-position cross-entropy [B, T] of the main head; toks is [B, T+1]."""
    h, _ = model.forward_hidden(toks[:, :-1])
    tgt = toks[:, 1:]
    out = torch.empty(tgt.shape, dtype=torch.float32)
    chunk = model.cfg.ce_chunk_tokens
    flat_h, flat_t, flat_o = h.reshape(-1, h.shape[-1]), tgt.reshape(-1), out.view(-1)
    for i in range(0, flat_h.shape[0], chunk):
        logits = model.head(flat_h[i : i + chunk]).float()
        flat_o[i : i + chunk] = F.cross_entropy(logits, flat_t[i : i + chunk], reduction="none")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=None, help="default: all of val.bin")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    run_dir = RESULTS_DIR / a.id
    model, cfg, step = load_run(run_dir, dev)
    windows = MemmapCorpus(Path(CORPUS)).val_windows(a.context, a.max_tokens)
    edges = [e for e in EDGES if e < a.context] + [a.context]
    sums = torch.zeros(len(edges) - 1, dtype=torch.float64)
    t0 = time.time()
    n = 0
    for i in range(0, len(windows), a.batch):
        toks = torch.cat(windows[i : i + a.batch]).to(dev)
        loss = token_losses(model, toks).sum(0).double()  # [T], summed over the batch
        for b in range(len(edges) - 1):
            sums[b] += loss[edges[b] : edges[b + 1]].sum()
        n += toks.shape[0]
        if i % (50 * a.batch) == 0:
            print(
                f"{n}/{len(windows)} windows, {n * a.context / (time.time() - t0):,.0f} tok/s",
                flush=True,
            )
    widths = torch.tensor(
        [edges[b + 1] - edges[b] for b in range(len(edges) - 1)], dtype=torch.float64
    )
    res = {
        "id": a.id,
        "step": step,
        "train_context": cfg.context,
        "eval_context": a.context,
        "windows": n,
        "tokens": n * a.context,
        "val_loss": float(sums.sum() / (n * a.context)),
        "buckets": {
            f"{edges[b]}-{edges[b + 1]}": float(sums[b] / (n * widths[b]))
            for b in range(len(edges) - 1)
        },
        "wall_s": round(time.time() - t0, 1),
        "device": a.device,
        "threads": a.threads,
    }
    print(json.dumps(res, indent=2))
    out = run_dir / f"eval_ctx{a.context}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
