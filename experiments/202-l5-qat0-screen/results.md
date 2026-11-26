# 202-l5-qat0-screen — L5 with FP4 weight fake-quant from step 0 (ADR-073): 3.3590 nats in quantised form, **+0.0262 vs `110` (8.9σ → H3's FP4 clause *weakened* at `screen` for this arm)** and **+0.0097 worse than late QAT `201` (3.3σ)**: on the weight-tied recurrence, training with the quantiser throughout is worse than switching it on for the last 2 %

`python -m scripts.train.train --rung L5-qat0 --id 202-l5-qat0-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the L5 preset with `numerics.qat_format: fp4_e2m1`, `qat_group: 64`, `qat_start_frac: 0.0`: fake-quant at every step), per-step log in `log.jsonl`. Code at the `queue_screen_25` gate commit `d4d0dfd`.

**Bears on:** **H3**, FP4 clause ("FP4 weights add ≤ 0.5 %", read as $T$ = 2σ under ADR-013), by
**ADR-073 §3** under ADR-066 §1. Quality only: Ampere has no FP4. This is the last of ADR-073's
arms.

| rule (ADR-073 §3) | measured | reading |
|---|---|---|
| Δ = quantised arm − its bf16 base `110`; within 4σ = 0.0119 silent, above weakens at `screen` | **+0.0262** (+0.79 %, 8.9σ) | **weakens** |
| the comparison that matters: late vs post-training vs from step 0 | post-training +0.0403 (rig `053`), late +0.0165 (`201`), from 0 **+0.0262** | on L5 late QAT is the best arm; from step 0 recovers 35 % of the rounding loss, late 59 % |

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1; σ multiples use the unrounded 0.00296.

## ADR-073's table, complete

Δ is each arm minus its own bf16 base (`100` = 3.3358 for L0, `110` = 3.3328 for L5).

| arm | L0 | L5 |
|---|---|---|
| post-training round-to-nearest FP4 (rig `053`) | +0.0343 (11.6σ) | +0.0403 (13.6σ) |
| FP4 fake-quant for the last 2 % of steps | `199` +0.0142 (4.8σ) | `201` +0.0165 (5.6σ) |
| FP4 fake-quant from step 0 | `200` +0.0149 (5.0σ) | **`202` +0.0262 (8.9σ)** |
| from 0 − late | +0.0007 (0.2σ) | **+0.0097 (3.3σ)** |
| post-training INT4 (side reading) | +0.0208 | +0.0262 |

**H3's FP4 clause at `screen`: weakened on every arm**, one seed each. The best arm on both
bases is late QAT, at +0.014 (L0) and +0.017 (L5), 0.43 % and 0.50 % of the loss: past the 4σ
line ADR-013 gives the clause, and at the edge of the note's literal "0.5 %". By ADR-073 §3 no
`small` pair follows automatically; whether one late-QAT pair is worth its ≈ 1.4 days is the
user's call.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.90 M non-embedding (identical to `110`) |
| tokens | 1.000 B in 1907 steps, fake-quant on throughout |
| final val loss (full held-out, 5606 windows), scored quantised | **3.3590** nats |
| final train loss (last step, main head, quantised) | 3.3270 (`110` 3.3016, `201` 3.3174) |
| throughput (median step) | 39,690 tokens/s aggregate (0.805× `201`'s bf16 steps) |
| peak memory per GPU | 3.86 GiB (as `110`) |
| wall-clock (this session) | 7.08 h |
| seed | 0 |
| $r$ | fixed 8 |
| routing at the end | entropy 0.9999 of $\ln 8$, 8.00 effective experts, 0 dead |

**Curve** (976-window evals; last row full held-out). `201` is bf16 until step 1869.

| step | `110` | `201` | `202` | `202` − `110` |
|---|---|---|---|---|
| 100 | 5.6159 | 5.5739 | 5.6757 | +0.0598 |
| 300 | 4.3664 | 4.3489 | 4.3890 | +0.0226 |
| 500 | 3.8804 | 3.8939 | 3.8966 | +0.0162 |
| 700 | 3.6601 | 3.6706 | 3.6853 | +0.0252 |
| 1000 | 3.4995 | 3.5020 | 3.5274 | +0.0279 |
| 1500 | 3.3697 | 3.3714 | 3.3979 | +0.0282 |
| 1800 | 3.3356 | 3.3367 | 3.3627 | +0.0271 |
| 1900 | 3.3288 | 3.3473 | 3.3563 | +0.0275 |
| 1907 (full) | 3.3328 | 3.3493 | 3.3590 | +0.0262 |

**Train loss** (`202` − `110`, main head, same seed and data order, step-matched means): +0.013
over steps 100–500, +0.021 over 500–1000, +0.027 over 1000–1500, +0.026 over 1500–1868, +0.026
over the last ten steps. The gap is flat at +0.026 from step 1000. `201`, after its switch,
fell from +0.033 to +0.018 in 39 steps: it passed below this level within ten steps and kept
going.

## Run health

| | |
|---|---|
| wall-clock | 7.08 h; summed `step_s` 6.987 h + evals 0.088 h + checkpoints 0.004 h → **0.1 % unaccounted** |
| throughput | 39.7 k tok/s median, flat (first and last 300 steps agree to 0.2 %) |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 78 °C, min clock 1897 MHz |

## Interpretation

L5 trained with FP4 fake-quant from its first step ends 0.0262 nats above bf16 `110`, 8.9σ,
which weakens H3's FP4 clause at `screen` for this arm. Unlike on L0, where training with the
quantiser throughout and switching it on for the last 39 steps gave the same loss (`200` vs
`199`, 0.0007 apart), here the full-length arm is 0.0097 worse than the late one (3.3σ), with
a step-matched gap that is constant from the middle of training. So the two bases differ in
kind, not only in size: the dense stack has one FP4 level, reached by either route, while the
weight-tied recurrence trained under quantisation settles at a worse solution than the one a
bf16 run reaches and then adapts in 2 % of its steps. A plausible reading, not tested here, is
that the straight-through gradient through eight applications of the same quantised weights is
a poorer training signal than one application each of eight layers' weights. The practical
result for the design is the same on both bases: train in bf16, quantise late; that costs
0.014–0.017 nats (0.43–0.50 %) at 1 B tokens against 0.034–0.040 for rounding alone. That is
still over the 4σ line, so the clause is weakened at `screen` on all arms, at one seed each.
Quality only; fake-quant's 19 % step cost on L5 is a property of the simulation.
