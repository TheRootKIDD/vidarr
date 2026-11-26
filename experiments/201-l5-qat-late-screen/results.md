# 201-l5-qat-late-screen — L5 with FP4 weight fake-quant switched on for the last 2 % of steps (ADR-073): 3.3493 nats in quantised form, **+0.0165 vs `110` (5.6σ, above 4σ → H3's FP4 clause *weakened* at `screen` for this arm)**; 39 steps of QAT recover 59 % of post-training rounding's +0.0403, the same share as on L0

`python -m scripts.train.train --rung L5-qat-late --id 201-l5-qat-late-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the L5 preset with `numerics.qat_format: fp4_e2m1`, `qat_group: 64`, `qat_start_frac: 0.98`: fake-quant from step 1869 of 1907), per-step log in `log.jsonl`. Code at the `queue_screen_25` gate commit `d4d0dfd`.

**Bears on:** **H3**, FP4 clause ("FP4 weights add ≤ 0.5 %", read as $T$ = 2σ under ADR-013), by
**ADR-073 §3** under ADR-066 §1. Quality only: Ampere has no FP4.

| rule (ADR-073 §3) | measured | reading |
|---|---|---|
| Δ = quantised arm − its bf16 base `110`; within 4σ = 0.0119 silent, above weakens at `screen` | **+0.0165** (+0.50 %, 5.6σ) | **weakens** (one seed, 0.0046 over the line) |
| run-to-run control: this run's bf16 loss before the switch against `110` at the same step | +0.0011 at step 1800 (0.4σ) | the two runs agree until step 1869 |
| the comparison that matters: late vs post-training vs from step 0 | post-training +0.0403 (rig `053`), late **+0.0165**, from 0: `202`, running | late QAT recovers 59 % of the rounding loss |

**H3's FP4 clause at `screen`, L5, late QAT: weakened**, at one seed. Three trained arms now sit
between 4.8σ and 5.6σ (`199`, `200` on L0; this one on L5). `202` (QAT from step 0 on L5) is
running. No `small` pair follows automatically (ADR-073 §3).

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1; σ multiples use the unrounded 0.00296.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.90 M non-embedding (identical to `110`) |
| tokens | 1.000 B in 1907 steps; fake-quant on for steps 1869–1907 (39 steps, 20.4 M tokens) |
| final val loss (full held-out, 5606 windows), scored quantised | **3.3493** nats |
| final train loss (last step, main head, quantised) | 3.3174 (`110` 3.3016) |
| throughput (median step) | 49,278 tokens/s aggregate before the switch, 39,941 after (fake-quant costs 19 % of the step; 6.8 % on L0) |
| peak memory per GPU | 3.86 GiB (as `110`) |
| wall-clock (this session) | 5.73 h |
| seed | 0 |
| $r$ | fixed 8 |
| routing at the end | entropy 0.9999 of $\ln 8$, 8.00 effective experts, 0 dead (as `110`) |

## Against `110`, the post-training arm, and L0

| | val loss | Δ vs its base | reading |
|---|---|---|---|
| `110-l5-screen`, bf16 | 3.3328 | — | base |
| `110` rounded to FP4 after training (rig `053`) | 3.3731 | +0.0403 (13.6σ) | weakens |
| `110` rounded to INT4 after training (side reading) | 3.3590 | +0.0262 (8.8σ) | |
| **`201`: FP4 fake-quant for the last 2 % of steps** | **3.3493** | **+0.0165** (5.6σ) | **weakens** |
| `202`: FP4 fake-quant from step 0 | running | | |
| for comparison, L0: `199` late / `200` from 0, vs `100` | 3.3501 / 3.3507 | +0.0142 / +0.0149 | weaken, just past 4σ |

L5's late-QAT cost is 0.0023 above L0's (0.8σ): the weight-tied recurrence pays about the same
as the stack once the quantiser is in the loop, where round-to-nearest cost it 0.0060 more.

**Curve** (976-window evals; last row full held-out). The fake-quant switches on at step 1869.

| step | `110` | `201` | `201` − `110` |
|---|---|---|---|
| 100 | 5.6159 | 5.5739 | −0.0420 |
| 500 | 3.8804 | 3.8939 | +0.0135 |
| 1000 | 3.4995 | 3.5020 | +0.0025 |
| 1500 | 3.3697 | 3.3714 | +0.0017 |
| 1800 | 3.3356 | 3.3367 | +0.0011 |
| 1900 | 3.3288 | 3.3473 | +0.0184 |
| 1907 (full) | 3.3328 | 3.3493 | +0.0165 |

**Train loss after the switch** (`201` − `110`, main head, same seed and data order, so
step-matched): mean +0.0006 over steps 1800–1868; **+0.033 over steps 1870–1874**, +0.025 over
1875–1884, +0.020 over 1885–1897, +0.018 over the last ten steps. It starts higher than on L0
(+0.029) and is still falling at the end. On L0 the same fall ended on the level QAT from
step 0 reaches (`200`); whether L5's does is what `202` shows.

## Run health

| | |
|---|---|
| wall-clock | 5.73 h; summed `step_s` 5.651 h + evals 0.075 h + checkpoints 0.003 h → **0.1 % unaccounted** |
| throughput | 49.3 k tok/s median before the switch, 39.9 k after; flat within each part |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 78 °C, min clock 1897 MHz |

## Interpretation

L5 with FP4 fake-quant for its last 39 steps ends 0.0165 nats above its bf16 twin `110`, 5.6σ:
over ADR-073's 4σ line, so this arm weakens H3's FP4 clause at `screen`. As on L0, the late
quantiser recovers 59 % of what rounding the finished checkpoint loses (0.0403), and the
recurrence, which was the numerics worry in `00 §P3`, pays only 0.002 more than the dense stack
here, inside noise. The step-matched train-loss gap was still shrinking at the end, more
visibly than on L0; `200` showed that on L0 this decay stops at the level a fully
quantisation-trained model has, so no further gain should be assumed until `202` reports. A
side cost worth recording: fake-quant takes 19 % of L5's step in the reference implementation
(the same weights are re-quantised at each of eight iterations), against 7 % on L0; it is a
training-time cost of the simulation, not a property of FP4 hardware. The control holds
(+0.0011 before the switch). One seed at `screen`, quality only.
