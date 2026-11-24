# 203-l5-balfix-screen — L5 with the bias controller fed every iteration's load (ADR-074, I43's control): 3.3315 nats, **−0.0013 vs `110` (0.4σ, inside 4σ → I43 closes as immaterial, the default stays `last_call`)**; with it, `queue_screen_21`'s routing probes: **H26 clause 1 met at both pool sizes**, and per-step EQB partitions experts by depth in every iteration

`python -m scripts.train.train --rung L5-balfix --id 203-l5-balfix-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the L5 preset with `router.bias_load: all_calls`; auxiliary loss on, as `110`), per-step log in `log.jsonl`. Code at the `queue_screen_21` gate commit `414f7f9`.

This write-up also closes `queue_screen_21`: it carries the `probe_routing` outputs for `190`–`193`
and `203` (`routing.json` in each run's directory, run on GPU 13:05–13:06 on 2026-11-24, 128
held-out sequences of 2048 each) and with them H26's first clause.

**Bears on:** no numbered hypothesis directly. It is the control for **I43** (the middle router's
bias controller has only ever seen iteration 1's load), read by **ADR-074 §3** under ADR-066 §1,
and it touches ADR-004. The probes bear on **H26** (ADR-069 §6, clause 1).

| | rule | measured | reading |
|---|---|---|---|
| I43 (ADR-074 §3) | $|\Delta$(`203` − `110`)$|$ ≤ 4σ = 0.0119 → immaterial, default stays | **−0.0013** (0.4σ) | **I43 closes as immaterial; `bias_load` stays `last_call`** |
| H26 clause 1, $N_e$ = 8 | `190`'s per-iteration entropy within 5 % of $\ln N_e$ at every $j$ | 0.9998–0.9999 of $\ln 8$ | **met** |
| H26 clause 1, $N_e$ = 128 | `192`'s, likewise | 0.9997–0.9998 of $\ln 128$ | **met** |

**H26 at `screen`, one seed, all three clauses now read:** clause 1 (per-iteration EQB flattens
every iteration's histogram) **supported** at both pool sizes; clause 2 (that flatness costs ≥ 1 %
against per-step EQB) **silent, pointing against** at both (+0.0065 and +0.0116 against 0.0334);
clause 3 (per-step EQB matches or beats ADR-004's balancer) **supported** at both (−0.0042,
−0.0119). The `small` pair for `193`, which sits on the 4σ line, remains the user's call.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.90 M non-embedding (identical to `110`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3315** nats |
| final train loss (last step, main head) | 3.2998 (`110` 3.3016) |
| router aux (last step) | 0.0081 (`110` 0.0079) |
| throughput (median step) | 49,363 tokens/s aggregate (1.02× `110`'s 48,553) |
| peak memory per GPU | 3.86 GiB |
| wall-clock (this session) | 5.69 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `110`

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `110-l5-screen` (controller fed iteration 1's load) | 3.3328 | — | comparator |
| **`203-l5-balfix-screen` (fed the mean over all eight calls)** | **3.3315** | **−0.0013** (−0.04 %, 0.4σ) | **inside 4σ: immaterial** |
| | | −0.0043 vs $L_{ref}$ | |

**Curve** (976-window evals; last row full held-out):

| step | `110` | `203` | `203` − `110` |
|---|---|---|---|
| 100 | 5.6159 | 5.5891 | −0.0268 |
| 300 | 4.3664 | 4.3641 | −0.0023 |
| 500 | 3.8804 | 3.8645 | −0.0159 |
| 700 | 3.6601 | 3.6566 | −0.0034 |
| 1000 | 3.4995 | 3.4967 | −0.0027 |
| 1500 | 3.3697 | 3.3675 | −0.0022 |
| 1900 | 3.3288 | 3.3274 | −0.0014 |
| 1907 (full) | 3.3328 | 3.3315 | −0.0013 |

`203` is ahead at every eval, by an amount that shrinks to under half a σ.

## Routing per iteration (`probe_routing`, checkpoint at step 1907)

Entropy is each iteration's expert-load entropy over $\ln N_e$. "Dead" counts experts with a share
below 0.1/$N_e$ in that iteration. Correlation is the mean Pearson correlation between the load
vectors of two different iterations (28 pairs), computed here from `routing.json`'s `loads`; by
this computation `110` gives 0.71 (earlier entries quote 0.75) and `112` 0.34.

**$N_e$ = 8**

| | `110` (ADR-004) | `203` (all calls) | `190` (EQB per iteration) | `191` (EQB per step) |
|---|---|---|---|---|
| entropy / $\ln 8$, iteration 1 | 0.9996 | 0.9972 | 0.9999 | 0.820 |
| entropy / $\ln 8$, range over iterations 1–8 | 0.954–0.9996 | 0.993–0.999 | 0.9998–0.9999 | 0.800–0.971 |
| effective experts, range | 7.26–7.99 | 7.89–7.98 | 8.00 | 5.27–7.53 |
| largest share in any iteration (ideal 0.125) | 0.266 | 0.159 | 0.132 | 0.331 |
| dead experts per iteration | 0 | 0 | 0 | 2, 0, 0, 1, 0, 1, 1, 1 |
| pooled over iterations: effective experts | 7.79 | 8.00 | 8.00 | 8.00 |
| cross-iteration correlation | 0.71 | −0.09 | 0.92 | −0.14 |

**$N_e$ = 128**

| | `112` (ADR-004) | `192` (EQB per iteration) | `193` (EQB per step) |
|---|---|---|---|
| entropy / $\ln 128$, range over iterations | 0.928–0.9993 | 0.9997–0.9998 | **0.759–0.888** |
| effective experts per iteration 1–8 | 127.6, 112.4, 113.5, 102.5, 111.4, 103.5, 90.7, 90.2 | 127.8 at each | **39.8, 63.6, 74.2, 61.5, 49.4, 43.4, 41.8, 51.1** |
| dead experts per iteration 1–8 | 0, 0, 0, 1, 0, 1, 5, 8 | 0 | **60, 28, 26, 42, 55, 54, 62, 47** |
| largest share in any iteration (ideal 0.0078) | 0.038 | 0.0093 | 0.064 |
| pooled over iterations: effective experts | 118.3 | 127.9 | 127.9 |
| experts live in 8 / 5–7 / 1–4 iterations | 117 / 11 / 0 | 128 / 0 / 0 | **14 / 64 / 50** |
| cross-iteration correlation | 0.34 | 0.79 | **−0.14** |

What the tables say:

- **I43 is visible in `110`.** Its controller saw only iteration 1, and iteration 1 is its
  flattest (0.9996); later iterations drift to 0.954 with one expert at 27 % in iteration 8. Fed
  every call, `203` holds all eight iterations at 0.993 or better and the pool exactly flat. The
  loss does not care (−0.0013).
- **`203` also lowers the cross-iteration correlation from 0.71 to −0.09.** With loads this flat
  the correlation is taken over deviations of about ±0.02 around 1/8, so it describes small
  residuals, not a depth partition: every expert is live in every iteration.
- **Clause 1 holds with room to spare.** Per-iteration EQB keeps every iteration within 0.03 % of
  $\ln N_e$ at both pool sizes, and makes the iterations' loads alike (correlation 0.92 and 0.79).
- **Per-step EQB is a depth partition, in every iteration.** The reading taken from one logged
  iteration in `191` and `193` is confirmed: at $N_e$ = 128 each iteration uses 40 to 74 effective
  experts and leaves 26 to 62 unused, the pool over iterations is flat (127.9), only 14 experts
  work at all eight depths, and 50 work at four or fewer. At $N_e$ = 8 four of the eight experts
  work at every depth and the other four miss one or two.

So the three balancers at $N_e$ = 128 order by how much depth specialisation they permit
(correlation 0.79 / 0.34 / −0.14 for `192` / `112` / `193`), while the loss separates only the
last (3.2229 / 3.2232 / 3.2113). That order is suggestive of the partition buying `193`'s 0.012;
these runs do not isolate it.

## Routing during training (logged iteration) and run health

Worst point 3.08 of 8 effective experts at step 19 with three dead (`110`: 4.17 at step 20, one
dead); last dead expert at step 61 (`110`: 37); 7.86 / 7.98 / 7.99 effective at steps 100 / 300 /
500 (`110`: 7.75 / 7.90 / 7.91); 7.998 at the end, loads 0.120–0.128. The early collapse is deeper
with the pooled load and the recovery to a flat load is quicker.

| | |
|---|---|
| wall-clock | 5.69 h; summed `step_s` 5.613 h + evals 0.070 h + checkpoints 0.003 h → **0.1 % unaccounted** |
| throughput | 49.4 k tok/s median, flat; recording every call's load costs nothing measurable |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 78 °C, min clock 1897 MHz |

## Interpretation

Feeding the bias controller the load of all eight iterations, in place of the first one it has
been seeing since the recurrent model was built, changes the loss by −0.0013 nats (0.4σ). The
mechanism does what it should: every iteration is balanced where `110` let the later ones drift.
But the auxiliary loss had already kept those iterations healthy enough, and the result is
immaterial under ADR-074 §3. I43 closes, `bias_load` stays `last_call`, and no earlier comparison
needs a new baseline. The probes finish H26's reading at `screen`: per-iteration EQB flattens
every iteration as claimed (clause 1), per-step EQB matches or beats the controller (clause 3),
and the cost of forced per-iteration balance is real in direction but a fifth to a third of the
1 % claimed (clause 2 silent, pointing against). They also confirm that per-step EQB turns the
128-expert pool into depth specialists. One seed at `screen` throughout.
