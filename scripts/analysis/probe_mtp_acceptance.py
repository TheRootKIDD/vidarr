"""Speculative-decode acceptance per MTP head, as ADR-013 actually defines it.

**Why this exists.** ADR-013 sets H13's threshold as *"greedy acceptance of heads
2/3/4 **against the main head** on held-out >= 70 / 55 / 45 %"*, and ADR-012 repeats
it: *"acceptance per head is measured greedily against head 1"*. The trainer's
`mtp_acc_h{j}` did not measure that (until I27's fix on 2026-09-11 it computed
`logits.argmax(-1) == target` where `target` is the **ground-truth token**, i.e.
top-1 accuracy, not acceptance; the trainer now logs `mtp_top1_h{j}` and
`mtp_accept_h{j}` separately, and this probe stays as the offline reference).

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

**Two more numbers, added for I27's experiments 1 and 2.** (1) *Expected accepted run
length*: what fixes speculative-decode speed-up is not the per-head marginals but how
many drafts survive per verify pass, and acceptance in a run is sequential -- the
window closes at the first rejection. So `run_length` is the mean over positions of
the number of consecutive accepted heads starting at head 2, and the *joint* is
reported next to the independence estimate built from the marginals (the joint is
lower when the same hard positions defeat every head). Tokens per verify pass is
1 + run_length. (2) *Acceptance vs head-1 confidence*: positions are bucketed by the
entropy of head 1's distribution at the draft position t -- the number a decoder has
in hand when it decides whether to draft at all -- and acceptance per head is reported
per bucket. If acceptance is high where head 1 is confident, the aggregate understates
what speculative decode actually gets.

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

ENTROPY_EDGES = [0.5, 1.0, 2.0, 3.0, 4.0]  # nats; buckets are (-inf,0.5], (0.5,1], ..., (4,inf)


@torch.no_grad()
def _argmax_over_vocab(hidden: torch.Tensor, emb: torch.Tensor, chunk: int) -> torch.Tensor:
    """argmax of `hidden @ emb.T` without ever holding the full [N, V] logits."""
    out = torch.empty(hidden.shape[0], dtype=torch.long)
    for i in range(0, hidden.shape[0], chunk):
        out[i : i + chunk] = (hidden[i : i + chunk] @ emb.t()).argmax(-1)
    return out


@torch.no_grad()
def _argmax_and_entropy(
    hidden: torch.Tensor, emb: torch.Tensor, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """argmax and softmax entropy (nats) of `hidden @ emb.T`, chunked."""
    n = hidden.shape[0]
    am = torch.empty(n, dtype=torch.long)
    ent = torch.empty(n, dtype=torch.float32)
    for i in range(0, n, chunk):
        logits = (hidden[i : i + chunk] @ emb.t()).float()
        am[i : i + chunk] = logits.argmax(-1)
        logp = torch.log_softmax(logits, -1)
        ent[i : i + chunk] = -(logp.exp() * logp).sum(-1)
    return am, ent


def _bucket(ent: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(ent.contiguous(), torch.tensor(ENTROPY_EDGES))


def _bucket_label(k: int) -> str:
    lo = "0" if k == 0 else f"{ENTROPY_EDGES[k - 1]:g}"
    hi = "inf" if k == len(ENTROPY_EDGES) else f"{ENTROPY_EDGES[k]:g}"
    return f"H1_in_({lo},{hi}]"


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
    nb = len(ENTROPY_EDGES) + 1
    bucket_n = torch.zeros(nb, dtype=torch.long)
    bucket_hits = {j: torch.zeros(nb, dtype=torch.long) for j in range(2, m + 1)}
    run_sum = 0.0  # sum over positions of consecutive accepted heads from head 2
    run_n = 0
    run_hist = torch.zeros(m, dtype=torch.long)  # run_hist[L] = positions with run length L

    for i in range(0, min(batches * batch, len(windows)), batch):
        toks = torch.cat(windows[i : i + batch]).to(device)
        if toks.shape[0] == 0:
            break
        inp = toks[:, :-1]
        h, _ = model.forward_hidden(inp)
        b, t, d = h.shape
        flat = h.reshape(-1, d)
        pred1, ent1 = _argmax_and_entropy(flat, emb, chunk)
        pred1 = pred1.reshape(b, t)  # pred1[t] -> token t+1
        ent1 = ent1.reshape(b, t)
        main_correct += int((pred1 == toks[:, 1:]).sum())
        main_n += pred1.numel()

        valid = t - m  # mirrors model/mtp.py's own valid range
        bk = _bucket(ent1[:, :valid])  # confidence of head 1 at the draft position t
        bucket_n += torch.bincount(bk.reshape(-1), minlength=nb)
        alive = torch.ones(b, valid, dtype=torch.bool)  # run still unbroken at this position
        run_len = torch.zeros(b, valid, dtype=torch.long)
        for j in range(2, m + 1):
            adapter = model.mtp.adapters[j - 2]
            predj = _argmax_over_vocab(adapter(flat), emb, chunk).reshape(b, t)
            draft = predj[:, :valid]  # head j's guess at token t+j
            verify = pred1[:, j - 1 : j - 1 + valid]  # head 1's guess at token t+j
            truth = toks[:, j : j + valid]
            acc = draft == verify
            hits[j] += int(acc.sum())
            correct[j] += int((draft == truth).sum())
            n[j] += draft.numel()
            bucket_hits[j] += torch.bincount(bk[acc].reshape(-1), minlength=nb)
            alive &= acc
            run_len += alive.long()
        run_sum += float(run_len.sum())
        run_n += run_len.numel()
        run_hist += torch.bincount(run_len.reshape(-1), minlength=m)[:m]

    marg = [hits[j] / max(1, n[j]) for j in range(2, m + 1)]
    indep, p = 0.0, 1.0
    for a in marg:  # E[run] under independence = sum_L prod_{j<=L} a_j
        p *= a
        indep += p
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
        "run_length": {
            "expected_joint": run_sum / max(1, run_n),
            "expected_if_independent": indep,
            "tokens_per_verify_pass": 1.0 + run_sum / max(1, run_n),
            "histogram": {str(L): int(run_hist[L]) for L in range(m)},
            "positions": run_n,
        },
        "by_head1_entropy": {
            _bucket_label(k): {
                "positions": int(bucket_n[k]),
                "share": float(bucket_n[k]) / max(1, int(bucket_n.sum())),
                **{
                    f"acceptance_h{j}": float(bucket_hits[j][k]) / max(1, int(bucket_n[k]))
                    for j in range(2, m + 1)
                },
            }
            for k in range(nb)
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
    ap.add_argument(
        "--out",
        default="mtp_acceptance.json",
        help="output file name inside the run directory (never overwritten)",
    )
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

    rl = res["run_length"]
    print(
        f"  expected accepted run length: {rl['expected_joint']:.3f} joint vs "
        f"{rl['expected_if_independent']:.3f} if independent -> "
        f"{rl['tokens_per_verify_pass']:.3f} tokens per verify pass; "
        f"histogram {rl['histogram']}"
    )
    heads = range(2, cfg.mtp.m + 1)
    cols = " ".join(f"{f'acc h{j}':>7s}" for j in heads)
    print(f"  {'head-1 entropy':>16s} {'share':>6s} {cols}")
    for k, v in res["by_head1_entropy"].items():
        print(
            f"  {k:>16s} {100 * v['share']:5.1f}% "
            + " ".join(f"{100 * v[f'acceptance_h{j}']:6.1f}%" for j in heads)
        )

    out = run_dir / args.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out} — experiments are append-only")
    out.write_text(json.dumps({"run_id": args.id, "step": step, **res}, indent=2) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
