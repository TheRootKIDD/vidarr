# 200-l0-qat0-screen — L0 with FP4 weight fake-quant from step 0 (ADR-073): 3.3507 nats in quantised form, **+0.0149 vs `100` (5.0σ, above 4σ → H3's FP4 clause *weakened* at `screen` for this arm)**; the same loss as late QAT `199` (+0.0007), so a full run with the quantiser buys nothing over its last 39 steps

`python -m scripts.train.train --rung L0-qat0 --id 200-l0-qat0-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the L0 preset with `numerics.qat_format: fp4_e2m1`, `qat_group: 64`, `qat_start_frac: 0.0`: fake-quant at every step), per-step log in `log.jsonl`. Code at the `queue_screen_25` gate commit `d4d0dfd`.

**Bears on:** **H3**, FP4 clause ("FP4 weights add ≤ 0.5 %", read as $T$ = 2σ under ADR-013), by
**ADR-073 §3** under ADR-066 §1. Quality only: Ampere has no FP4.

| rule (ADR-073 §3) | measured | reading |
|---|---|---|
| Δ = quantised arm − its bf16 base `100`; within 4σ = 0.0119 silent, above weakens at `screen` | **+0.0149** (+0.45 %, 5.0σ) | **weakens** (one seed, 0.0030 over the line) |
| the comparison that matters: late vs post-training vs from step 0 | post-training +0.0343 (rig `053`), late +0.0142 (`199`), from 0 **+0.0149** | QAT from step 0 and late QAT agree to 0.0007 (0.2σ); both recover ≈ 58 % of the rounding loss |

**H3's FP4 clause at `screen`, L0: weakened** by both trained arms, each at one seed and each
within one σ of the 4σ line, but they are two runs landing on the same number. The L5 arms are
`201` (running) and `202`. No `small` pair follows automatically (ADR-073 §3).

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1; σ multiples use the unrounded 0.00296.

## Numbers

| | |
|---|---|
| params | 100.09 M total, 75.52 M non-embedding (identical to `100`) |
| tokens | 1.000 B in 1907 steps, fake-quant on throughout |
| final val loss (full held-out, 5606 windows), scored quantised | **3.3507** nats |
| final train loss (last step, quantised) | 3.3180 (`100` 3.3048, `199` 3.3181) |
| throughput (median step) | 66,950 tokens/s aggregate (0.931× the bf16 steps of `199`) |
| peak memory per GPU | 6.25 GiB (as `100`) |
| wall-clock (this session) | 4.22 h |
| seed | 0 |
| $r$ | fixed 8 (L0: a stack, no router) |

## The three FP4 arms on L0

| | val loss | Δ vs `100` | reading |
|---|---|---|---|
| `100-l0-screen`, bf16 ($L_{ref}$) | 3.3358 | — | base |
| `100` rounded to FP4 after training (rig `053`) | 3.3702 | +0.0343 (11.6σ) | weakens |
| `199`: FP4 fake-quant for the last 2 % of steps | 3.3501 | +0.0142 (4.8σ) | weakens, just past 4σ |
| **`200`: FP4 fake-quant from step 0** | **3.3507** | **+0.0149** (5.0σ) | **weakens, just past 4σ** |

**Curve** (976-window evals; last row full held-out). `199` is bf16 until step 1869.

| step | `100` | `199` | `200` | `200` − `100` |
|---|---|---|---|---|
| 100 | 6.0833 | 6.0815 | 6.1241 | +0.0408 |
| 300 | 4.7840 | 4.7898 | 4.8021 | +0.0181 |
| 500 | 4.0987 | 4.1058 | 4.1071 | +0.0084 |
| 700 | 3.7763 | 3.7791 | 3.7876 | +0.0113 |
| 1000 | 3.5507 | 3.5527 | 3.5689 | +0.0182 |
| 1500 | 3.3844 | 3.3840 | 3.3993 | +0.0149 |
| 1800 | 3.3420 | 3.3416 | 3.3560 | +0.0140 |
| 1900 | 3.3336 | 3.3484 | 3.3480 | +0.0144 |
| 1907 (full) | 3.3358 | 3.3501 | 3.3507 | +0.0149 |

**Train loss** (`200` − `100`, same seed and data order, step-matched means): +0.029 over steps
100–500, +0.015 over 500–1000, +0.016 over 1000–1500, +0.015 over 1500–1868, **+0.015 over the
last ten steps**. Against `199` after its switch: −0.0001 over the last ten steps. The gap is
flat from the middle of training on; nothing in it is still closing.

## Run health

| | |
|---|---|
| wall-clock | 4.22 h; summed `step_s` 4.149 h + evals 0.065 h + checkpoints 0.003 h → **0.1 % unaccounted** |
| throughput | 66.9 k tok/s median, flat (first and last 300 steps agree to 3 tok/s) |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 77 °C, min clock 1897 MHz |

## Interpretation

Training L0 with FP4 fake-quant from the first step ends 0.0149 nats above the bf16 run, 5.0σ
and so past ADR-073's 4σ line: this arm weakens H3's FP4 clause at `screen`. The more useful
result is the comparison with `199`. That run trained in bf16 for 98 % of its steps and met the
quantiser only for the last 39; it ends 0.0007 below this one, and its step-matched train loss
over the last ten steps is the same to 0.0001. `199`'s write-up left open that its gap was
still shrinking and a longer window might close it. This run answers that: the gap of a model
that has never seen anything but FP4 weights sits at +0.015 from step 500 onward, and `199` was
falling onto that same level. So about 0.015 nats (0.45 %) is what FP4 E2M1 with one scale per
64 weights costs this dense model at 1 B tokens however it is trained, against 0.034 for
rounding without training; and late QAT gets all of the available recovery for 2 % of the
fake-quant overhead (6.9 % of a step). Whether the weight-tied recurrence behaves the same is
`201`/`202`. One seed per arm at `screen`, each 0.002–0.003 over the line; quality only.
