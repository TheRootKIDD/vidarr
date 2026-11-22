# 190-eqb-iter-screen — L5 with Exact Quantile Balancing per iteration, $N_e$ = 8 (ADR-069): 3.3351 nats, **+0.0023 vs `110` (0.8σ: matches the ADR-004 balancer)**; no auxiliary loss, same throughput; H26's clauses wait on `191` (per step) and the routing probe

`python -m scripts.train.train --rung L5-eqb-iter --id 190-eqb-iter-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`router.balance: eqb_iteration`; the auxiliary loss is off under EQB, `aux_loss_coef` is unused), per-step log in `log.jsonl`. Code at the `queue_screen_21` gate commit `414f7f9`.

**Bears on:** **H26** (EQB per iteration flattens the per-iteration expert histogram and costs
≥ 1 % loss against EQB per step; EQB per step matches or beats ADR-004's balancer), by
**ADR-069 §6** at $N_e$ = 8. This run alone decides none of the three clauses:

| clause | rule | state after `190` |
|---|---|---|
| 1 | `eqb_iteration`'s per-iteration entropy within 5 % of $\ln N_e$ at every $j$ | waits on `probe_routing` (end of `queue_screen_21`) |
| 2 | $\Delta$(`190` − `191`) ≥ 0.0334 | waits on `191` (running, ETA ≈ 19:45 today) |
| 3 | $\Delta$(`191` − `110`) ≤ 2σ | waits on `191` |

**H26 at `screen` after this run: silent.** What it does measure: $\Delta$(`190` − `110`) =
**+0.0023 (0.8σ)**, inside 2σ, so per-iteration EQB *matches* the bias controller with its
auxiliary loss at $N_e$ = 8. It does not beat the comparator by 4σ, so no `small` pair follows
from this arm (ADR-069 §6, ADR-031). One consequence for clause 2 is already visible: for the
clause to hold, `191` would have to land at ≤ 3.3017, which is 0.031 below `110`.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.90 M non-embedding (identical to `110`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3351** nats |
| final train loss (last step, main head) | 3.3031 (`110`: 3.3095 including its aux term of 0.0079) |
| router aux | 0 (off under EQB) |
| throughput (median step) | 48,534 tokens/s aggregate (**1.00× `110`'s 48,553**) |
| peak memory per GPU | 3.88 GiB (`110` 3.86) |
| wall-clock (this session) | 5.78 h (`110` 5.79 h) |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `110` and the ladder

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `110-l5-screen` (L5, ADR-004 bias update + aux loss) | 3.3328 | — | comparator |
| **`190-eqb-iter-screen` (EQB per iteration, no aux)** | **3.3351** | **+0.0023 vs `110`** (+0.07 %, 0.8σ) | **inside 2σ: matches** |
| | | −0.0007 vs $L_{ref}$ | |

**Curve** (976-window evals; last row full held-out):

| step | `110` | `190` | `190` − `110` |
|---|---|---|---|
| 100 | 5.6159 | 5.6566 | +0.0407 |
| 300 | 4.3664 | 4.4014 | +0.0350 |
| 500 | 3.8804 | 3.9065 | +0.0261 |
| 700 | 3.6601 | 3.6726 | +0.0125 |
| 1000 | 3.4995 | 3.5046 | +0.0051 |
| 1500 | 3.3697 | 3.3729 | +0.0032 |
| 1900 | 3.3288 | 3.3319 | +0.0031 |
| 1907 (full) | 3.3328 | 3.3351 | +0.0023 |

EQB starts 0.04 behind and closes steadily; the gap is still shrinking at the end and is inside
1σ there.

## Routing, from the training log (the probe's per-iteration table comes with `193`/`203`)

| | `110` | `190` |
|---|---|---|
| worst effective experts (of 8) | 4.17 at step 20 | 4.06 at step 19 |
| dead experts, max / last step with one | 1 / step 37 | 1 / step 38 |
| end: effective experts | 7.997 | 7.997 |
| end: load min – max (ideal 0.125) | 0.118 – 0.129 | 0.119 – 0.130 |

The two balancers give the same early collapse, the same recovery and the same end state in the
logged aggregates. The logged load is the realised one at the step, with biases computed one step
earlier (ADR-069 §1), so it is not exactly flat.

**Solver fidelity (`eqb_err`, the largest relative share error of the new biases on the step they
were computed from; ADR-069 §2 tests for ≤ 1 %):** median 0.5 % over the run, but **above 1 % in
132 of 1907 steps**: 95 of the first 100 (max 34 % at step 19, while the router collapses and
recovers), 30 of steps 101–500 (max 3.1 %), 7 of steps 501–1907 (max 1.2 %). So the cutoff as
built is not exact at every step, most visibly while the logits are nearly tied early on. The
cause (bisection depth, bf16 ties) is not diagnosed here; it is recorded as an as-built limit of
this arm and applies to `191`–`193` as well.

## Run health

| | |
|---|---|
| wall-clock | 5.78 h; summed `step_s` 5.708 h + evals 0.070 h + checkpoints 0.003 h → **0.1 % unaccounted** |
| throughput | 48.5 k tok/s median, flat; the EQB update costs nothing measurable at $N_e$ = 8 (rig `051`: 3.2 % at $N_e$ = 128) |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 79 °C, min clock 1897 MHz |

## Interpretation

Replacing ADR-004's bias controller and auxiliary loss with per-iteration exact quantile
balancing changes nothing measurable at eight experts: 3.3351 against 3.3328 (0.8σ), with the
same throughput, memory and routing end state, after a slower start that has closed by the end.
That is a null in the useful direction for the mechanism (one fewer loss term and no step size
for the same result) and a weak hint against H26's second clause, which expected forcing every
iteration to be balanced to *cost* something: this arm already sits on the comparator, so the
per-step arm would have to beat L5 by 0.031 for the clause to hold. $N_e$ = 8 is also the pool
size where L5 shows no depth partition (cross-iteration correlation 0.75), so per-iteration
balance may simply not bind here; `192`/`193` at $N_e$ = 128, where the router does partition by
depth, are the arms that can show a cost. The solver's share error exceeds its 1 % test in 7 %
of steps, nearly all in the first 500. One seed at `screen`; no clause is decided.
