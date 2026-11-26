# 199-l0-qat-late-screen — L0 with FP4 weight fake-quant switched on for the last 2 % of steps (ADR-073): 3.3501 nats in quantised form, **+0.0142 vs `100` (4.8σ, above 4σ → H3's FP4 clause *weakened* at `screen` for this arm)**; 39 steps of QAT recover 59 % of post-training rounding's +0.0343 and the loss was still falling when the run ended

`python -m scripts.train.train --rung L0-qat-late --id 199-l0-qat-late-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the L0 preset with `numerics.qat_format: fp4_e2m1`, `qat_group: 64`, `qat_start_frac: 0.98`: fake-quant from step 1869 of 1907), per-step log in `log.jsonl`. Code at the `queue_screen_25` gate commit `d4d0dfd`.

**Bears on:** **H3**, FP4 clause ("FP4 weights add ≤ 0.5 %", read as $T$ = 2σ under ADR-013), by
**ADR-073 §3** under ADR-066 §1. Quality only: Ampere has no FP4.

| rule (ADR-073 §3) | measured | reading |
|---|---|---|
| Δ = quantised arm − its bf16 base `100`; within 4σ = 0.0119 silent, above weakens at `screen` | **+0.0142** (+0.43 %, 4.8σ) | **weakens** (one seed, 0.0023 over the line) |
| run-to-run control: this run's bf16 loss before the switch against `100` at the same step | −0.0004 at step 1800 | the two runs are the same run until step 1869 |
| the comparison that matters: late vs post-training vs from step 0 | post-training +0.0343 (rig `053`), late **+0.0142**, from 0: `200`, running | late QAT recovers 59 % of the rounding loss |

**H3's FP4 clause at `screen`, L0, late QAT: weakened**, narrowly and at one seed. The clause is
read per arm; the picture ADR-073 asks for needs `200` (QAT from step 0, running) and the L5
arms `201`/`202`. No `small` pair follows automatically (ADR-073 §3).

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1; σ multiples use the unrounded 0.00296.

## Numbers

| | |
|---|---|
| params | 100.09 M total, 75.52 M non-embedding (identical to `100`) |
| tokens | 1.000 B in 1907 steps; fake-quant on for steps 1869–1907 (39 steps, 20.4 M tokens) |
| final val loss (full held-out, 5606 windows), scored quantised | **3.3501** nats |
| final train loss (last step, quantised) | 3.3181 (`100` 3.3048) |
| throughput (median step) | 71,876 tokens/s aggregate before the switch, 66,959 after (fake-quant costs 6.8 % of the step) |
| peak memory per GPU | 6.25 GiB (as `100`) |
| wall-clock (this session) | 3.94 h |
| seed | 0 |
| $r$ | fixed 8 (L0: a stack, no router) |

## Against `100` and the post-training arm

| | val loss | Δ vs `100` | reading |
|---|---|---|---|
| `100-l0-screen`, bf16 ($L_{ref}$) | 3.3358 | — | base |
| `100` rounded to FP4 after training (rig `053`) | 3.3702 | +0.0343 (11.6σ) | weakens |
| `100` rounded to INT4 after training (side reading) | 3.3567 | +0.0208 (7.0σ) | |
| **`199`: FP4 fake-quant for the last 2 % of steps** | **3.3501** | **+0.0142** (4.8σ) | **weakens, just past 4σ** |
| `200`: FP4 fake-quant from step 0 | running | | |

**Curve** (976-window evals; last row full held-out). The fake-quant switches on at step 1869.

| step | `100` | `199` | `199` − `100` |
|---|---|---|---|
| 100 | 6.0833 | 6.0815 | −0.0018 |
| 500 | 4.0987 | 4.1058 | +0.0071 |
| 1000 | 3.5507 | 3.5527 | +0.0020 |
| 1500 | 3.3844 | 3.3840 | −0.0003 |
| 1800 | 3.3420 | 3.3416 | −0.0004 |
| 1900 | 3.3336 | 3.3484 | +0.0148 |
| 1907 (full) | 3.3358 | 3.3501 | +0.0142 |

**Train loss after the switch** (`199` − `100`, same seed and data order, so step-matched): mean
−0.0002 over steps 1800–1868; **+0.029 at step 1872**, +0.019 by 1880, +0.017 around 1890, and
+0.015 over the last ten steps. The quantiser's cost halves within about ten steps and then
creeps down; it had not flattened by the end. The learning rate is at its floor (6 × 10⁻⁵, a
tenth of peak) for these steps.

## Run health

| | |
|---|---|
| wall-clock | 3.94 h; summed `step_s` 3.869 h + evals 0.060 h + checkpoints 0.002 h → **0.1 % unaccounted** |
| throughput | 71.9 k tok/s median before the switch, 67.0 k after; flat within each part |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 77 °C, min clock 1897 MHz |

## Interpretation

Switching FP4 fake-quant on for the last 39 steps of an otherwise unchanged L0 run leaves the
quantised model 0.0142 nats above its bf16 twin. That is much better than rounding the finished
checkpoint (0.0343): 2 % of training with the quantiser in the loop buys back three fifths of
the damage. It is still over the 4σ line, by 0.0023, and well over the 2σ that H3's FP4 clause
allows, so by ADR-073 this arm weakens the clause at `screen`. Two cautions about how firmly.
The margin over the line is under one σ at a single seed. And the cost was still shrinking when
the run stopped, at the lowest learning rate of the schedule, so "the last 2 %" may be too
short a window more than FP4 is too coarse; `200`, which trains with the quantiser throughout,
is the test of that and is running now. The control holds: until the switch this run and `100`
agree to 0.0004. One seed at `screen`, quality only.
