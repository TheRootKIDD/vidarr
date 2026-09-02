# 009-train-step — `small` dense, one GPU

2026-09-02 · `scripts/bench/bench_train_step.py --micro-batch 4` · gpu0 ·
numbers in `result.json`. **010-train-step-ddp** is the 4-GPU run; read both.

**Bears on:** no hypothesis. Replaces the "8 TFLOPS per card dense" *assumption*
of `docs/06 §4` (ADR-015) with a measurement. Benchmark, not a training run:
random tokens, 40 timed steps after 10 warm-up, no accumulation, no checkpoint.

Model: the L0 anchor of `docs/06 §3` — 12 layers, $d$ = 768, 12/4 heads (GQA),
SwiGLU $d_{ff}$ = 2048, $V$ = 32 k, context 2048, tied embeddings, AdamW (fused),
BF16 autocast with FP32 master weights, SDPA causal attention.
**100.1 M parameters, 75.5 M non-embedding** — the GQA count of
`costmodel/README.md §C`, not the doc's 85 M.

## Measured

| | |
|---|---|
| step (4 × 2048 tokens) | 370.4 ms median |
| throughput | **22 117 tokens/s** |
| achieved, non-embedding accounting | **12.52 TFLOPS** = 46.2 % of measured $\phi$ (27.1) |
| achieved, incl. output head | 15.78 TFLOPS = 58.2 % |
| peak memory | **7.62 GiB** |

Micro-batch 8 (16 k tokens) **does not fit**: OOM at 11.6 GiB, ~2 GiB short,
in the loss — the 32 k-vocab logits and their `log_softmax` buffer at
16 k tokens are ~4 GiB on their own.

## Interpretation

**12.5 TFLOPS per card, against an assumed 8.** `docs/06 §4` budgeted 31 % of
nominal for dense; the rig delivers 46 % of *measured* peak (49 % of nominal)
on a plain implementation with no fused attention beyond SDPA, no fused
cross-entropy, no `torch.compile`. Two accounting conventions are reported
because the anchors in `docs/06 §3` are quoted without the output head; the
with-head figure is the honest MFU, the without-head figure is what plugs into
the §4 table. Either way the assumption was conservative by 1.5–2×.

**Memory is the binding constraint at this shape, not compute.** 7.6 GiB at
4 × 2048 leaves headroom under the 10 GB rule, but 8 × 2048 does not fit. The
recurrent variants add expert weights and the depth-embedding path, so
`docs/06 §3`'s "≈ 3.2 GB of states per GPU + activations" needs the activation
term measured, not assumed — at `small` the activations, not the states, decide
the micro-batch. A chunked cross-entropy would recover most of the 2 GiB and is
worth doing in Phase 1's `model/` but is out of scope for a bench.

**What 22 k tokens/s means for the programme**, single card: `screen` 12.6 h,
`small` 31 h. The four-card numbers are in 010.
