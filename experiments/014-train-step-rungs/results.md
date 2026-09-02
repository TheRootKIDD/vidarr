# 014-train-step-rungs — training step of every Phase-1 rung at `small`

2026-09-02 · `scripts/bench/bench_train_step.py --ddp --rung L0:4 L1:2 L2:2 L3:2 L5:4
L5d:4 L5-ne128:4 L5-ne1:4 L7b:4` · 4 × RTX 3060, DDP, host-bounced NCCL,
`NCCL_P2P_DISABLE=1` · random tokens, 10 warm-up + 40 timed micro-steps, **an
all-reduce every micro-step (no accumulation)** · models from `model/` (Phase 1)
at the ADR-025/027 widths · numbers in `result.json`.

**Bears on:** no hypothesis. Replaces the "5 TFLOPS assumed" variant column of
`docs/06 §4` and sets the micro-batch per rung for `scripts/train/train.py`.

## Measured (no accumulation — the pessimistic bound)

| rung | params | micro-batch / GPU | ms / micro-step | tokens/s aggregate | peak GiB |
|---|---|---|---|---|---|
| L0 dense | 100.1 M | 4 | 694 | 47.2 k | 5.91 |
| L1 layered MoE | 271.2 M | 2 | 1013 | 16.2 k | 7.85 |
| L2 + parallel form | 271.1 M | 2 | 1363 | 12.0 k | 7.70 |
| L3 + MTP (m = 4) | 272.9 M | 2 | 1938 | 8.5 k | 7.87 |
| L5 recurrent, $N_e$ = 8 | 77.5 M | 4 | 2526 | 13.0 k | 3.59 |
| L5d final-vector global | 77.5 M | 4 | 3779 | 8.7 k | 3.63 |
| L5 $N_e$ = 128, $k$ = 4 | 193.1 M | 4 | 5862 | 5.6 k | 5.33 |
| L5 $N_e$ = 1 (dense recurrent) | 65.1 M | 4 | 2652 | 12.4 k | 3.40 |
| L7b ($L_e$ = 2) | 77.5 M | 4 | 2678 | 12.2 k | 3.59 |

Single-GPU micro-step times for reference (same code, no DDP, `model/` path):
L0 446 ms, L1 (mb 2) 324, L3 673, L5 651, L5d 903, L5-$N_e$128 1544 ms.

With the trainer's gradient accumulation (4 micro-steps per optimiser step in
the smoke runs; 16 in a real `screen` step), measured aggregate rates were:
**L0 81 k, L3 (mb 2) 43 k, L5 47 k, L5d 33 k tokens/s** — i.e. `screen` (1 B
tokens) ≈ 3.4 h / 6.5 h / 5.9 h / 8.4 h.

## Interpretation

**The recurrent rungs pay far more for the per-micro-step all-reduce than the
dense one.** DDP adds 250 ms to L0's 446 ms micro-step (the 010 figure, plus
`model/`'s chunked-CE recompute) but ≈ 1.9 s to L5's 651 ms and ≈ 2.9 s to
L5d's 903 ms — 8× the dense overhead for a gradient 25 % smaller (77 M vs
100 M parameters). The shared middle block's parameters are used in every
iteration, so their gradient is complete only at the very end of backward and
DDP cannot overlap its buckets with compute; the experts' many small matrices
and the checkpoint recompute may add to it. Recorded as **I21**: measure the
reducer's timeline (`TORCH_DISTRIBUTED_DEBUG`, NCCL trace) before any
throughput claim about the recurrent design on this rig. With accumulation the
cost is paid once per 16 micro-steps and the rates above hold; the `screen`
budget of `docs/06 §4` (≈ 8 h per variant) stands.

**Memory is not the constraint.** Per-iteration checkpointing and the chunked
head keep every recurrent rung under 6 GiB at micro-batch 4; the layered MoE
baselines are parameter-heavy (271 M, matched-FLOPs experts of width 1024 per
layer) and need micro-batch 2 to stay under the 10 GB rule with DDP buckets.

**`model/` costs ≈ 20 % over the Phase-0 dense reference** (446 vs 370 ms
single-GPU): the reference implementation's explicit streams, rotary in
Python and the checkpointed cross-entropy. Accepted (CLAUDE.md: readability
over speed).
