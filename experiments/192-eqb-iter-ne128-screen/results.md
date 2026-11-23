# 192-eqb-iter-ne128-screen — L5-ne128 with Exact Quantile Balancing per iteration (ADR-069): 3.2229 nats, **−0.0003 vs `112` (0.1σ: matches the ADR-004 balancer at $N_e$ = 128)**; H26's clauses at $N_e$ = 128 wait on `193` (per step) and the routing probe; the solver misses its 1 % share test at every step (median 3.9 %)

`python -m scripts.train.train --rung L5-ne128-eqb-iter --id 192-eqb-iter-ne128-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the L5-ne128 preset, $k$ = 4, width 448, `router.balance: eqb_iteration`; no auxiliary loss), per-step log in `log.jsonl`. Code at the `queue_screen_21` gate commit `414f7f9`.

**Bears on:** **H26** (EQB per iteration flattens the per-iteration expert histogram and costs
≥ 1 % loss against EQB per step; EQB per step matches or beats ADR-004's balancer), by
**ADR-069 §6** at $N_e$ = 128. This run alone decides none of the three clauses:

| clause | rule | state after `192` |
|---|---|---|
| 1 | `eqb_iteration`'s per-iteration entropy within 5 % of $\ln N_e$ at every $j$ | waits on `probe_routing` (end of `queue_screen_21`) |
| 2 | $\Delta$(`192` − `193`) ≥ 0.0334 | waits on `193` (running, ETA ≈ 07:20 on 2026-11-24) |
| 3 | $\Delta$(`193` − `112`) ≤ 2σ | waits on `193` |

**H26 at `screen`, $N_e$ = 128, after this run: silent.** What it measures:
$\Delta$(`192` − `112`) = **−0.0003 (0.1σ)**: per-iteration EQB *matches* the bias controller with
its auxiliary loss at 128 experts, as `190` did at 8 (+0.0023). It does not beat the comparator,
so no `small` pair follows from this arm. For clause 2 to hold, `193` would have to reach
≤ 3.1895, 0.034 below `112`.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 193.10 M total, 168.53 M non-embedding (identical to `112`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.2229** nats |
| final train loss (last step, main head) | 3.1924 (`112`: 3.1988 including its aux term of 0.0071) |
| router aux | 0 (off under EQB) |
| throughput (median step) | 15,754 tokens/s aggregate (**0.953× `112`'s 16,529**) |
| peak memory per GPU | 6.26 GiB (`112` 6.01; as rig `051` measured) |
| wall-clock (this session) | 17.77 h (`112` 16.95 h) |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `112` and the ladder

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `106-l1-screen` (L1) | 3.2049 | | |
| `112-l5-ne128-screen` (ADR-004 bias update + aux loss) | 3.2232 | — | comparator |
| **`192-eqb-iter-ne128-screen` (EQB per iteration, no aux)** | **3.2229** | **−0.0003 vs `112`** (−0.01 %, 0.1σ) | **inside 2σ: matches** |
| | | −0.1129 vs $L_{ref}$ (−3.38 %) | |

**Curve** (976-window evals; last row full held-out):

| step | `112` | `192` | `192` − `112` |
|---|---|---|---|
| 100 | 5.5305 | 5.5802 | +0.0497 |
| 300 | 4.2642 | 4.2861 | +0.0218 |
| 500 | 3.7629 | 3.7664 | +0.0035 |
| 700 | 3.5583 | 3.5625 | +0.0042 |
| 1000 | 3.3959 | 3.3960 | +0.0001 |
| 1500 | 3.2619 | 3.2628 | +0.0009 |
| 1900 | 3.2203 | 3.2200 | −0.0004 |
| 1907 (full) | 3.2232 | 3.2229 | −0.0003 |

As at $N_e$ = 8, EQB starts behind (0.05 at step 100); here it is level with the comparator from
step 1000.

## Routing, from the training log

One router call is logged per step (iteration 1 by I43), so these are one iteration's loads.

| | `112` | `192` |
|---|---|---|
| worst effective experts (of 128) | 15.1 at step 25 | 33.4 at step 25 |
| dead experts, max / last step with one | 90 / step 364 | 54 / step 155 |
| effective experts at step 100 / 300 / 500 | 93.3 / 115.4 / 121.3 | **124.0 / 127.5 / 127.7** |
| effective experts at the end | 127.70 | 127.84 |
| load min – max at the end (ideal 0.0078) | 0.0065 – 0.0092 | 0.0068 – 0.0086 |

Per-iteration EQB balances the logged iteration much sooner (124 of 128 effective experts by
step 100, against 93 with 15 dead for the controller) and ends slightly flatter. The early
collapse is shallower and shorter. None of that shows in the loss. Whether every iteration is
this flat is clause 1, and is the probe's to say.

**Solver fidelity (`eqb_err`, ADR-069 §2's ≤ 1 % test): not met at this pool size.** The largest
relative share error of the new biases on their own step is **above 1 % at all 1907 steps**:
median 33 % over the first 100 (max 192 % at step 29), 3.9 % over steps 101–500 (max 30 %), 3.9 %
over steps 501–1907 (max 7.7 %). At $N_e$ = 8 the same arm (`190`) had a median of 0.5 %. The
number is a maximum over $r \times N_e$ = 1024 cutoffs, each set from about 16 k (token,
iteration) pairs, so it is a worst case, and the realised loads above are tight; but the cutoff
as built is not "exact" to ADR-069 §2's tolerance here. The cause (bisection depth, bf16 ties) is
not diagnosed; recorded as an as-built limit.

## Run health

| | |
|---|---|
| wall-clock | 17.77 h; summed `step_s` 17.632 h + evals 0.110 h + checkpoints 0.026 h → **0.0 % unaccounted** |
| throughput | 15.75 k tok/s median, flat; EQB costs 4.7 % of the step against `112` (rig `051` measured 3.2 % on 6 steps) |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 67 °C, min clock 1912 MHz |

## Interpretation

At 128 experts, where the ADR-004 router is known to partition experts by depth (cross-iteration
correlation 0.34 in `112`), forcing each iteration to use all experts equally was the arm H26
expected to pay. It does not, against the controller: 3.2229 against 3.2232, with curves that
coincide from step 1000, at 4.7 % more wall-clock and 0.25 GiB. The balancer reaches a flat load
several hundred steps earlier than the controller and that buys nothing either, so at this budget
the early routing collapse of `112` is not what limits its loss. This says nothing yet about
clause 2, which compares against per-step EQB (`193`, running): at $N_e$ = 8 the per-step arm was
0.0065 better than the per-iteration one, and a gap of 0.0334 is what the clause needs. The
solver's worst-case share error stays near 4 % throughout, well over its 1 % test, which should
be kept in mind when reading "exact" for this arm. One seed at `screen`; no clause is decided.
