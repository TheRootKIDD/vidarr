# 010-train-step-ddp — `small` dense, 4 GPUs, DDP over host-bounced NCCL

2026-09-02 · `scripts/bench/bench_train_step.py --ddp --micro-batch 4` ·
`NCCL_P2P_DISABLE=1` · numbers in `result.json`. Extends **009-train-step**.

**Bears on:** no hypothesis. This is the number that replaces `docs/06 §4`'s
throughput assumptions and sizes the whole programme.

## Measured

| | per GPU | aggregate |
|---|---|---|
| step (4 × 2048 tokens per GPU) | 485.7 ms (all four identical) | — |
| throughput | 16 867 tok/s | **67 466 tokens/s** |
| achieved, non-embedding | 9.55 TFLOPS (35.2 % MFU) | 38.2 TFLOPS |
| achieved, incl. head | 12.04 TFLOPS (44.4 %) | 48.2 TFLOPS |
| peak memory | 7.99 GiB | — |

## Interpretation

**DDP costs 31 % of step time at this micro-batch, and that is the all-reduce.**
485.7 ms vs 370.4 ms single-card. DDP all-reduces FP32 gradients — 100 M × 4 B
= 400 MB — and at the measured λ_link of 3.59 GB/s with the ring's 1.5×
bytes-on-wire factor that is ≈ 170 ms un-overlapped; the observed +115 ms says
roughly a third of it overlaps with the backward pass. The number is consistent
with `002-nccl` to within the overlap, which is the cross-check that matters.

**`docs/06 §3`'s "gradient all-reduce ≈ 0.4 GB per step is ≪ step time even
host-bounced" is true only with gradient accumulation.** Per *micro*-step it is
31 %, not ≪. A `small` step processes ≈ 0.5 M tokens (`docs/06 §5.1`), which at
4 × 8 k tokens per micro-step is ~15 micro-steps per optimiser step, so the
all-reduce amortises to ~2 % and the doc's claim holds at the real batch. The
bench does no accumulation — deliberately, so the communication cost is
visible — and the programme numbers below should be read as the pessimistic
per-micro-step case, with the accumulated case approaching 4 × 22 k = 88 k tok/s.

**Programme cost, replacing `docs/06 §4`:**

| Run | doc assumed (8 TF/card) | measured, no accumulation | with accumulation (≈ 88 k tok/s) |
|---|---|---|---|
| `screen` (1 B tok) | ≈ 5 h | **4.1 h** | ≈ 3.2 h |
| `small` (2.5 B tok) | ≈ 13 h | **10.3 h** | ≈ 7.9 h |

The dense assumption was conservative. The "5 TFLOPS for MoE / recurrent
variants" figure is *not* replaced here — the recurrent stub is deliberately
unimplemented (see the script's docstring) and that number waits for Phase 1's
`model/`; its ratio to dense is the thing to measure, not to assume.

All four cards report identical step times to 0.1 ms, which is DDP's barrier
doing its job and — with 003-gemm's 2.5 % $\phi$ spread — closes the DDP half of
I11: the GA104/GA106 mix does not cost a visible step-time penalty.
