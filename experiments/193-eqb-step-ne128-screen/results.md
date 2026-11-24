# 193-eqb-step-ne128-screen — L5-ne128 with Exact Quantile Balancing per step (ADR-069): 3.2113 nats, **−0.0119 vs `112` (4.0σ, on the 4σ line → H26 clause 3 met; "beats" depends on rounding)**; **$\Delta$(`192` − `193`) = +0.0116 (3.9σ) against the 0.0334 clause → clause 2 silent, pointing against**; the logged iteration uses 40 of 128 experts (59 dead there)

`python -m scripts.train.train --rung L5-ne128-eqb-step --id 193-eqb-step-ne128-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the L5-ne128 preset, $k$ = 4, width 448, `router.balance: eqb_step`; no auxiliary loss), per-step log in `log.jsonl`. Code at the `queue_screen_21` gate commit `414f7f9`.

**Bears on:** **H26** (EQB per iteration flattens the per-iteration expert histogram and costs
≥ 1 % loss against EQB per step; EQB per step matches or beats ADR-004's balancer), by
**ADR-069 §6** at $N_e$ = 128:

| clause | rule | measured | reading |
|---|---|---|---|
| 1 | `eqb_iteration`'s per-iteration entropy within 5 % of $\ln N_e$ at every $j$ | — | waits on `probe_routing` (after `203`, ≈ 13:15 today) |
| 2 | $\Delta$(`192` − `193`) ≥ 1 % $L_{ref}$ = 0.0334; inside 4σ silent, near zero points against | **+0.0116** (3.9σ, a third of the threshold) | **silent; recorded as pointing against** |
| 3 | $\Delta$(`193` − `112`) ≤ 2σ = 0.0059 | **−0.0119** | **met** (matches or beats) |

**H26 at `screen`, $N_e$ = 128, one seed: clause 3 supported; clause 2 silent and pointing
against (per-iteration balance costs 0.36 % here, not ≥ 1 %); clause 1 open.**

**The 4σ line.** $\Delta$(`193` − `112`) = 3.21133 − 3.22324 = **−0.01191**. ADR-066 §1 fixes
4σ = 0.0119 from σ = 0.0030; the unrounded pooled σ is 0.00296 (4σ = 0.01184, so 4.02σ), and
4 × 0.0030 is 0.0120 (3.97σ). The result sits on the line to the fifth decimal, and the reading
"beats its comparator by > 4σ" turns on rounding. ADR-069 §6 gives an arm past that line a `small`
pair by ADR-031. I record it as *on the line* and do not decide it: whether the L5-ne128 EQB-step
`small` pair runs (two seeds, ≈ 45 h each at this arm's rate, against `138`/`142`) is the user's
call, and nothing is queued. The curve argues the gain is real rather than a last-eval
fluctuation: the gap is −0.012 ± 0.001 at every eval from step 700.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 193.10 M total, 168.53 M non-embedding (identical to `112`, `192`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.2113** nats |
| final train loss (last step, main head) | 3.1813 |
| router aux | 0 (off under EQB) |
| throughput (median step) | 15,675 tokens/s aggregate (0.948× `112`'s 16,529) |
| peak memory per GPU | 6.26 GiB (`112` 6.01) |
| wall-clock (this session) | 17.87 h (`112` 16.95 h) |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `112`, `192` and the ladder

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `106-l1-screen` (L1) | 3.2049 | | |
| `112-l5-ne128-screen` (ADR-004 bias update + aux loss) | 3.2232 | — | comparator |
| `192-eqb-iter-ne128-screen` (EQB per iteration) | 3.2229 | −0.0003 vs `112` (0.1σ) | matches |
| **`193-eqb-step-ne128-screen` (EQB per step)** | **3.2113** | **−0.0119 vs `112`** (−0.37 %, 4.0σ) | **clause 3 met; on the 4σ line** |
| | | **−0.0116 vs `192`** (3.9σ) | clause 2 needs ≥ 0.0334: silent, pointing against |
| | | +0.0064 vs `106` (L1; `112` was +0.0183) | |
| | | −0.1245 vs $L_{ref}$ (−3.73 %) | |

**Curve** (976-window evals; last row full held-out):

| step | `112` | `192` (iter) | `193` (step) | `193` − `112` | `192` − `193` |
|---|---|---|---|---|---|
| 100 | 5.5305 | 5.5802 | 5.4736 | −0.0569 | +0.1066 |
| 300 | 4.2642 | 4.2861 | 4.2486 | −0.0156 | +0.0374 |
| 500 | 3.7629 | 3.7664 | 3.7529 | −0.0100 | +0.0136 |
| 700 | 3.5583 | 3.5625 | 3.5452 | −0.0131 | +0.0173 |
| 1000 | 3.3959 | 3.3960 | 3.3837 | −0.0122 | +0.0122 |
| 1500 | 3.2619 | 3.2628 | 3.2504 | −0.0115 | +0.0124 |
| 1900 | 3.2203 | 3.2200 | 3.2077 | −0.0127 | +0.0123 |
| 1907 (full) | 3.2232 | 3.2229 | 3.2113 | −0.0119 | +0.0116 |

The per-step arm leads both others at every eval, by a steady 0.012 from step 700. The pattern
is the one seen at $N_e$ = 8 (`191` − `110` = −0.0042, `190` − `191` = +0.0065), about twice the
size.

## Routing, from the training log

One router call is logged per step (iteration 1 by I43), so these are one iteration's loads, not
the step's pool.

| | `112` | `192` (iter) | `193` (step) |
|---|---|---|---|
| effective experts (of 128) at step 100 / 500 / 1000 | 93.3 / 121.3 / 126.5 | 124.0 / 127.7 / 127.8 | **69.1 / 47.4 / 40.9** |
| effective experts at the end | 127.70 | 127.84 | **39.73** |
| experts dead in the logged iteration at the end (share < 0.1/$N_e$) | 0 | 0 | **59** |
| load max at the end (ideal 0.0078) | 0.0092 | 0.0086 | **0.0635** |

Under per-step EQB the logged iteration narrows over training to about 40 effective experts, with
59 of 128 unused there and one expert taking 6.4 % of that iteration's selections. The step's
pool is balanced (`eqb_err` below), so the other experts work at other iterations: **per-step EQB
at $N_e$ = 128 produces a strong depth partition**, far beyond what the ADR-004 controller allows
in its logged iteration (127.7 effective). `112`'s own partition (cross-iteration correlation
0.34) lives within balanced per-iteration loads; this one does not. The per-iteration histograms,
correlations and co-activation statistics are `probe_routing`'s and come after `203`; this
paragraph rests on one logged iteration.

**Solver fidelity (`eqb_err`, ADR-069 §2's ≤ 1 % test):** median 1.0 %; above 1 % in all of the
first 100 steps (max 49 %), 185 of steps 101–500 (max 3.2 %), 933 of steps 501–1907 (max 1.9 %,
median 1.07 %). Better than `192` (median 3.9 %), since one pooled quantile has eight times the
pairs, but still around the tolerance and not inside it.

## Run health

| | |
|---|---|
| wall-clock | 17.87 h; summed `step_s` 17.723 h + evals 0.123 h + checkpoints 0.020 h → **0.0 % unaccounted** |
| throughput | 15.68 k tok/s median, flat; EQB costs 5.2 % of the step against `112` |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 67 °C, min clock 1912 MHz |

## Interpretation

This is the first EQB arm that moves the loss. Balancing the whole step and leaving each
iteration free gives L5-ne128 3.2113 against 3.2232 for ADR-004's controller and 3.2229 for
per-iteration EQB: 0.012 nats, steady from a third of the way through the run, and it closes two
thirds of `112`'s gap to L1. The routing log says what the freedom is used for: the router
concentrates each iteration on a subset of experts (40 effective of 128 in the logged one), so
the experts specialise by depth much more than a per-iteration-balanced router lets them. That
agrees with H26's direction (forcing per-iteration balance costs something) but not its size:
0.0116 is a third of the 1 % the second clause names, and inside 4σ, so the clause is silent and
points against at both pool sizes. The third clause holds at both. Against the controller the
result lands exactly on the 4σ line, so whether it "beats" by the letter is a rounding question
and the `small` pair it would trigger is left to the user; the reason to want it is that ADR-064
(EQB per step as the default) waits on H26 at `small`, and L5's `screen` results have moved at
`small` before. Costs: 5 % of wall-clock, 0.25 GiB, and a solver whose worst-case share error
sits at its 1 % tolerance instead of inside it. One seed at `screen`.
