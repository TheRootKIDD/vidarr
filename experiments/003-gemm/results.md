# 003-gemm — BF16 GEMM throughput over the expert-step grid

2026-09-02 · `scripts/bench/bench_gemm.py` · numbers in `result.json`

**Bears on:** no hypothesis directly. Supplies $\phi$, $s_{min}$ and the MoE MFU
assumption to `sim/scenarios/local_3060.yaml`, replacing the nominal values of
`docs/06 §1` (ADR-015). Settles open question **I11**.

Grid: $d \in \{768, 1024\}$, $d_{ff} \in \{128 … 4096\}$, $b \in \{16 … 8192\}$,
BF16 with FP32 accumulate, CUDA-event timing, median of 10–50 iterations, run
independently on all four cards.

## Measured

| | gpu0 | gpu1 | gpu2 | gpu3 |
|---|---|---|---|---|
| $\phi$ (peak achieved TFLOPS) | 27.10 | 26.44 | 26.91 | 26.91 |
| $s_{min}$ (smallest $d_{ff}$ at ≥ 80 % of $\phi$) | 256 | 256 | 256 | 256 |
| empirical batch floor (≥ 80 % of $\phi$) | 512 | 512 | 512 | 512 |

Grouped GEMM, $d$ = 768, $d_{ff}$ = 1665, 8192 tokens split over $N_e$ experts:

| tokens/expert | 4096 | 2048 | 1024 | 512 | 256 | 128 | 64 |
|---|---|---|---|---|---|---|---|
| % of one dense GEMM | 102 | 102 | 102 | 102 | 101 | 98 | **67** |

## Interpretation

**$\phi$ = 27.1 TFLOPS, slightly *above* the 25.5 nominal.** `docs/06 §1` takes
25.5 from the reference boost clock; the cards evidently boost higher, and
25.5 × (actual/reference clock) lands in this range. The nominal figure was
conservative rather than optimistic, which is the pleasant direction. Every
budget in `docs/06 §4` may use 27.1 — but see 004 before choosing widths.

**$s_{min}$ = 256, at the top of the 128–256 the doc expected.** So the sizing
invariant $d_{ff} \ge U s_{min}$ (`docs/01 §4.5`) is 1024 at $U$ = 4, not the
512 that `docs/06 §3` and ADR-020 quote from $s_{min}$ = 128. **This raises the
tile floor by 2×** and therefore tightens the two-stream infeasibility recorded
in `costmodel/README.md §B`: at $U$ = 4 the floor is now 1024, and the
two-stream `small` solve wants 362.

**I11 is closed: the GA104/GA106 mix does not matter for compute.** Per-card
$\phi$ spread is 1.025×, i.e. 2.5 %, well inside run-to-run variation. All four
cards report 28 SMs and identical $s_{min}$ and batch floors. DDP step time will
be set by the slowest card by ~2 %, which is not worth modelling. Memory
bandwidth is not covered here and is measured by `bench_tiers`.

**The empirical batch floor is 512, seven times the roofline $b_{min}$ of 75.**
This is the more consequential number of the two. `docs/02 §2` derives
$b_{min} = (\phi/\beta_C)(b_w/2)$ — 75 tokens at the measured $\phi$ and the
nominal 360 GB/s — and says "the simulator's per-expert queue uses $b_{min}$ as
its *enough tokens* threshold". At 75 tokens per expert this rig delivers well
under half of $\phi$; 80 % needs 512. The roofline threshold is a *necessary*
condition (weights are streamed once), the occupancy floor is what actually
binds, and the scenario must carry both with the queue using the larger.
Whether that is a property of GA106-class hardware or of the roofline argument
itself is a question for `note64`, where the ratio may differ; recorded as a new
open question rather than generalised from one rig.

**Grouped GEMM is free down to 128 tokens per expert, then falls off a cliff.**
101–102 % of dense from 4096 down to 256 tokens per expert (the >100 % is the
batched kernel beating one large GEMM at this shape, not a measurement error),
98 % at 128, and **67 % at 64**. So the MoE MFU assumption is: no dispatch
penalty provided every expert receives ≥ 128 tokens. Combined with the batch
floor above, the operating rule for the simulator's queue is *≥ 512 tokens per
expert for throughput, ≥ 128 to avoid an additional grouping penalty*.

Caveat on the baseline: the grouped comparison uses $d_{ff}$ = 1665, which
004-gemm-alignment shows is a ~20 % depressed shape. The *ratios* are unaffected
(numerator and denominator share the shape) but the absolute 19.5 TFLOPS is not
$\phi$.
