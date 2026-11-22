# 191-eqb-step-screen — L5 with Exact Quantile Balancing per step, $N_e$ = 8 (ADR-069): 3.3286 nats, **−0.0042 vs `110` (1.4σ: matches → H26 clause 3 met at $N_e$ = 8)**; **$\Delta$(`190` − `191`) = +0.0065 (2.2σ) against the 0.0334 clause → clause 2 silent, pointing against**; the logged iteration is left unbalanced (5.6 of 8 experts, two dead)

`python -m scripts.train.train --rung L5-eqb-step --id 191-eqb-step-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`router.balance: eqb_step`; no auxiliary loss), per-step log in `log.jsonl`. Code at the `queue_screen_21` gate commit `414f7f9`.

**Bears on:** **H26** (EQB per iteration flattens the per-iteration expert histogram and costs
≥ 1 % loss against EQB per step; EQB per step matches or beats ADR-004's balancer), by
**ADR-069 §6** at $N_e$ = 8:

| clause | rule | measured | reading |
|---|---|---|---|
| 1 | `eqb_iteration`'s per-iteration entropy within 5 % of $\ln N_e$ at every $j$ | — | waits on `probe_routing` (end of `queue_screen_21`) |
| 2 | $\Delta$(`190` − `191`) ≥ 1 % $L_{ref}$ = 0.0334; inside 4σ silent, near zero points against | **+0.0065** (2.2σ, a fifth of the threshold) | **silent; recorded as pointing against** |
| 3 | $\Delta$(`191` − `110`) ≤ 2σ = 0.0059 | **−0.0042** (1.4σ) | **met: matches** |

**H26 at `screen`, $N_e$ = 8, one seed: clause 3 supported; clause 2 silent and pointing against
(per-iteration balance costs 0.2 % here, not ≥ 1 %); clause 1 open.** Neither EQB arm beats `110`
by 4σ, so no `small` pair follows at this pool size (ADR-069 §6, ADR-031). The $N_e$ = 128 arms
(`192` running, `193` next) read the same clauses against `112`.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.90 M non-embedding (identical to `110`, `190`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3286** nats |
| final train loss (last step, main head) | 3.2974 |
| router aux | 0 (off under EQB) |
| throughput (median step) | 48,384 tokens/s aggregate (1.00× `110`'s 48,553) |
| peak memory per GPU | 3.88 GiB |
| wall-clock (this session) | 5.81 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `110`, `190` and the ladder

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `110-l5-screen` (ADR-004 bias update + aux loss) | 3.3328 | — | comparator |
| `190-eqb-iter-screen` (EQB per iteration) | 3.3351 | +0.0023 vs `110` (0.8σ) | matches |
| **`191-eqb-step-screen` (EQB per step)** | **3.3286** | **−0.0042 vs `110`** (−0.13 %, 1.4σ) | **≤ 2σ: matches (clause 3)** |
| | | **−0.0065 vs `190`** (2.2σ) | clause 2 needs ≥ 0.0334: silent, pointing against |
| | | −0.0072 vs $L_{ref}$ | |

**Curve** (976-window evals; last row full held-out):

| step | `110` | `190` (iter) | `191` (step) | `191` − `110` | `190` − `191` |
|---|---|---|---|---|---|
| 100 | 5.6159 | 5.6566 | 5.5530 | −0.0629 | +0.1036 |
| 300 | 4.3664 | 4.4014 | 4.3421 | −0.0243 | +0.0593 |
| 500 | 3.8804 | 3.9065 | 3.8535 | −0.0269 | +0.0530 |
| 700 | 3.6601 | 3.6726 | 3.6487 | −0.0113 | +0.0239 |
| 1000 | 3.4995 | 3.5046 | 3.4940 | −0.0055 | +0.0105 |
| 1500 | 3.3697 | 3.3729 | 3.3657 | −0.0040 | +0.0072 |
| 1900 | 3.3288 | 3.3319 | 3.3252 | −0.0036 | +0.0067 |
| 1907 (full) | 3.3328 | 3.3351 | 3.3286 | −0.0042 | +0.0065 |

The per-step arm is ahead of both from the first eval, and the per-iteration arm behind both.
The iter − step gap is 0.10 at step 100 and shrinks to a steady ≈ 0.007 from step 1500: the order
is the one H26 predicted at every eval, the end-of-run size is not.

## Routing, from the training log

The trainer logs one router call per step (the last call of the checkpointed replay, which is
iteration 1 by I43), so these are one iteration's loads, not the step's pool.

| | `110` | `190` (iter) | `191` (step) |
|---|---|---|---|
| worst effective experts (of 8) | 4.17 at step 20 | 4.06 at step 19 | 3.67 at step 12 (four dead) |
| effective experts at the end | 7.997 | 7.997 | **5.61** |
| dead experts at the end (share < 0.1/$N_e$) | 0 | 0 | **2** |
| load min – max at the end (ideal 0.125) | 0.118 – 0.129 | 0.119 – 0.130 | **0.0006 – 0.249** |

Under per-step EQB the logged iteration settles on about 5.6 of 8 experts from step ≈ 500
(effective experts 6.75 / 6.41 / 5.89 / 5.65 / 5.55 / 5.61 at steps 100 / 300 / 500 / 1000 /
1500 / 1907), with two experts almost unused there. The step's pool is balanced by construction
(`eqb_err` below), so those two are used at other iterations: **at $N_e$ = 8 the per-step
balancer lets the router specialise experts by depth**, which ADR-004's controller (correlation
0.75 in `110`) and per-iteration EQB do not. The full per-iteration histograms and the
cross-iteration correlation are `probe_routing`'s and come at the end of the queue; this
paragraph rests on one logged iteration.

**Solver fidelity (`eqb_err`, ADR-069 §2's ≤ 1 % test):** median 0.17 %; above 1 % in 45 of 1907
steps, 44 of them in the first 100 (max 12.8 % at step 16), one at 1.01 % in steps 101–500, none
after 500 (max 0.49 %). Better than `190`'s 132 steps: one pooled quantile over eight times as
many pairs.

## Run health

| | |
|---|---|
| wall-clock | 5.81 h; summed `step_s` 5.734 h + evals 0.070 h + checkpoints 0.003 h → **0.1 % unaccounted** |
| throughput | 48.4 k tok/s median, flat |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 79 °C, min clock 1897 MHz |

## Interpretation

EQB over the whole step does what H26's third clause says: it matches the bias controller with
its auxiliary loss (−0.0042, 1.4σ, on the good side) with no loss term and no step size, at the
same speed. The second clause is not borne out at eight experts: forcing every iteration to be
balanced costs 0.0065 nats against balancing the step (2.2σ), the predicted direction at every
eval but a fifth of the 1 % the clause asks for, and inside the 4σ band where one seed is silent.
The routing log gives a reason the two arms can differ at all: per-step EQB leaves individual
iterations free, and the logged one ends on 5.6 effective experts with two nearly dead, so the
router has started to partition experts by depth where neither the ADR-004 controller nor
per-iteration EQB lets it. Whether that partition is what buys the 0.0065 is not shown by these
runs. Three balancers within 0.0065 of each other at $N_e$ = 8 mostly says the balancer is not
where L5's loss is decided at this pool size; `192`/`193` at $N_e$ = 128, where depth
partitioning already exists under ADR-004, are the stronger test of clause 2. One seed at
`screen`.
