"""Speculative-decode acceptance per MTP head, as ADR-013 actually defines it.

**Why this exists.** ADR-013 sets H13's threshold as *"greedy acceptance of heads
2/3/4 **against the main head** on held-out >= 70 / 55 / 45 %"*, and ADR-012 repeats
it: *"acceptance per head is measured greedily against head 1"*. The trainer's
`mtp_acc_h{j}` does not measure that. `model/mtp.py` computes
`logits.argmax(-1) == target` where `target` is the **ground-truth token**, i.e.
top-1 accuracy, not acceptance.

Those are different quantities and acceptance is normally the larger one: a draft
is accepted when it **agrees with the verifier**, not when it is right. A model
whose main head is itself wrong most of the time can still have its drafts
accepted, because draft and verifier share the same biases. So `108`'s reported
21.5 / 12.6 / 8.9 % is not the number H13's 70 / 55 / 45 % thresholds were written
against, and the H13 conclusion cannot be read off it.

CLAUDE.md: if code and doc disagree, the doc wins. This probe implements the doc.

**What acceptance means here.** Head 1 at position t predicts token t+1. Head j at
position t drafts token t+j. In speculative decode the draft is verified by the
main head one step later, so head j's draft of token t+j is checked against head
1's prediction of token t+j -- which head 1 makes from position t+j-1. Hence
`accept_j[t] = (pred_j[t] == pred_1[t+j-1])`. Ground-truth accuracy is reported
beside it so the two are never confused again.

Offline from a checkpoint, CPU by default, so it cannot perturb a running ladder.

Run: `python -m scripts.analysis.probe_mtp_acceptance --id 108-l3-screen --threads 4`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.analysis.probe_routing import RESULTS_DIR, load_run
from scripts.train.data import MemmapCorpus


@torch.no_grad()
def _argmax_over_vocab(hidden: torch.Tensor, emb: torch.Tensor, chunk: int) -> torch.Tensor:
    """argmax of `hidden @ emb.T` without ever holding the full [N, V] logits."""
    out = torch.empty(hidden.shape[0], dtype=torch.long)
    for i in range(0, hidden.shape[0], chunk):
        out[i : i + chunk] = (hidden[i : i + chunk] @ emb.t()).argmax(-1)
    return out


@torch.no_grad()
def measure(
    model, cfg, corpus: MemmapCorpus, batches: int, batch: int, chunk: int, device: torch.device
) -> dict:
    m = cfg.mtp.m
    if m <= 1 or model.mtp is None:
        return {}
    emb = model.emb.weight
    windows = corpus.val_windows(cfg.context, None)
    hits = {j: 0 for j in range(2, m + 1)}
    correct = {j: 0 for j in range(2, m + 1)}
    n = {j: 0 for j in range(2, m + 1)}
    main_correct = 0
    main_n = 0

    for i in range(0, min(batches * batch, len(windows)), batch):
        toks = torch.cat(windows[i : i + batch]).to(device)
        if toks.shape[0] == 0:
            break
        inp = toks[:, :-1]
        h, _ = model.forward_hidden(inp)
        b, t, d = h.shape
        flat = h.reshape(-1, d)
        pred1 = _argmax_over_vocab(flat, emb, chunk).reshape(b, t)  # pred1[t] -> token t+1
        main_correct += int((pred1 == toks[:, 1:]).sum())
        main_n += pred1.numel()

        valid = t - m  # mirrors model/mtp.py's own valid range
        for j in range(2, m + 1):
            adapter = model.mtp.adapters[j - 2]
            predj = _argmax_over_vocab(adapter(flat), emb, chunk).reshape(b, t)
            draft = predj[:, :valid]  # head j's guess at token t+j
            verify = pred1[:, j - 1 : j - 1 + valid]  # head 1's guess at token t+j
            truth = toks[:, j : j + valid]
            hits[j] += int((draft == verify).sum())
            correct[j] += int((draft == truth).sum())
            n[j] += draft.numel()

    return {
        "main_head_top1_accuracy": main_correct / max(1, main_n),
        "per_head": {
            f"h{j}": {
                "acceptance_vs_head1": hits[j] / max(1, n[j]),
                "top1_accuracy_vs_truth": correct[j] / max(1, n[j]),
                "positions": n[j],
            }
            for j in range(2, m + 1)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--corpus", default="/mnt/nvme/corpus/fineweb-edu/tok-mistral32k")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=2048, help="rows per logits chunk")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="0 = all; cap it if a run is live")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    run_dir = RESULTS_DIR / args.id
    device = torch.device(args.device)
    model, cfg, step = load_run(run_dir, device)
    corpus = MemmapCorpus(Path(args.corpus))

    print(
        f"{args.id}: step {step}, m = {cfg.mtp.m}, {args.batches} x {args.batch} held-out "
        f"sequences of {cfg.context} on {device}"
    )
    res = measure(model, cfg, corpus, args.batches, args.batch, args.chunk, device)
    if not res:
        print("  no MTP heads in this model — nothing to report")
        return

    print(f"  main head top-1 vs truth: {100 * res['main_head_top1_accuracy']:.1f}%")
    print(f"  {'head':>5s} {'ACCEPTANCE':>11s} {'top-1 truth':>12s}   (H13 wants 70/55/45)")
    for k, v in res["per_head"].items():
        print(
            f"  {k:>5s} {100 * v['acceptance_vs_head1']:10.1f}% "
            f"{100 * v['top1_accuracy_vs_truth']:11.1f}%"
        )

    out = run_dir / "mtp_acceptance.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out} — experiments are append-only")
    out.write_text(json.dumps({"run_id": args.id, "step": step, **res}, indent=2) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
