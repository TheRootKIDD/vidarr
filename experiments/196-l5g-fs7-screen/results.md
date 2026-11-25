# 196-l5g-fs7-screen — L5g with the global schedule `[F,S,S,S,S,S,S,S]`, $K_{sel}$ = 128 (ADR-070, on L5d): 3.3531 nats, **+0.0088 vs `162` (3.0σ, inside 4σ and inside $T$)**, +0.0056 vs `194`, −0.0015 vs `195`; a curve point, H21 unchanged. With it, `queue_screen_22`'s probes: the second full iteration is what `194` buys, $K_{sel}$ can be halved at eval time for ≤ 0.002, no arm copies

`python -m scripts.train.train --rung L5g-fs7 --id 196-l5g-fs7-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the L5d preset with `attention.schedule: [F,S,S,S,S,S,S,S]`, `k_sel: 128`), per-step log in `log.jsonl`. Code at the `queue_screen_22` gate commit `7e50b5c`.

This write-up also closes `queue_screen_22`: it carries the eval-time $K_{sel}$ sweep
(`probe_ksel.json`), the copy probe (`probe_copy.json`) and the per-position eval
(`eval_ctx2048.json`) for `194`, `195` and `196`, run on GPU 15:20–15:54 on 2026-11-25.

**Bears on:** **H21**, as a curve point only. ADR-070 §5 puts H21's verdict on `194`
(`[F,S,S,S] × 2`: +0.0032 vs `162`, loss clause met at `screen`); `195` and `196` "are points on
the loss-vs-reads curve, not verdicts". **H21: unchanged by this run and by the probes (silent as
a verdict at one seed, leaning supports; the `small` pair `209`/`210` under ADR-078 decides).**
For the record, under ADR-066 §1's bands this arm is inside 4σ of `162` and inside $T$ = 0.0334.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.91 M non-embedding (as `162`, `194`, `195`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3531** nats |
| final train loss (last step, main head) | 3.3203 |
| router aux (last step) | 0.0080 |
| throughput (median step) | 31,519 tokens/s aggregate (0.929× `162`'s 33,913; `194` 31,394, `195` 33,858) |
| peak memory per GPU | 3.91 GiB |
| wall-clock (this session) | 8.92 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## The loss-vs-reads curve, complete

Global keys read per token, summed over the 8 iterations, by ADR-070 §2's counting
($|\mathcal{G}|$ = $n$ − 512 at F, min(128, $|\mathcal{G}|$) at S, 0 at L); the window mean is my
arithmetic over the positions of a 2 k window, as in `194`.

| | schedule | val loss | Δ vs `162` | reads at position 2048 | reads, 2 k window mean |
|---|---|---|---|---|---|
| `162` (L5d) | F × 8 | 3.3443 | — | 12,288 (1.00×) | 1.00× |
| `194` | `[F,S,S,S] × 2` | 3.3475 | +0.0032 (1.1σ) | 3,840 (0.31×) | 0.37× |
| **`196`** | **`[F,S,S,S,S,S,S,S]`** | **3.3531** | **+0.0088** (+0.26 %, 3.0σ) | **2,432 (0.20×)** | **0.26×** |
| `195` | `[F,S,S,S,L,L,L,L]` | 3.3546 | +0.0103 (3.4σ) | 1,920 (0.16×) | 0.18× |

The two changes `195` made to `194` now separate:

| step | Δ | reading |
|---|---|---|
| drop the second full iteration (`196` − `194`) | **+0.0056** (1.9σ) | most of the cost |
| then make iterations 5–8 local-only (`195` − `196`) | +0.0015 (0.5σ) | nothing measurable |

All three arms are inside 4σ of `162` at one seed, and so are the steps between them.

**Curve** (976-window evals; last row full held-out):

| step | `162` | `194` | `195` | `196` | `196` − `162` | `196` − `194` | `196` − `195` |
|---|---|---|---|---|---|---|---|
| 100 | 5.5372 | 5.5286 | 5.5257 | 5.5392 | +0.0021 | +0.0106 | +0.0135 |
| 300 | 4.3094 | 4.3067 | 4.3405 | 4.3224 | +0.0130 | +0.0158 | −0.0181 |
| 500 | 3.8551 | 3.8579 | 3.8922 | 3.8769 | +0.0217 | +0.0189 | −0.0154 |
| 700 | 3.6599 | 3.6644 | 3.6858 | 3.6735 | +0.0136 | +0.0091 | −0.0123 |
| 1000 | 3.5084 | 3.5121 | 3.5225 | 3.5178 | +0.0095 | +0.0058 | −0.0047 |
| 1500 | 3.3835 | 3.3863 | 3.3918 | 3.3911 | +0.0077 | +0.0048 | −0.0006 |
| 1900 | 3.3436 | 3.3464 | 3.3517 | 3.3513 | +0.0077 | +0.0049 | −0.0004 |
| 1907 (full) | 3.3443 | 3.3475 | 3.3546 | 3.3531 | +0.0088 | +0.0056 | −0.0015 |

`196` falls behind early like `195`, by about half as much (+0.022 at step 500 against +0.037),
and the two meet from step 1500. Against `194` the gap settles at +0.005.

## Per-position loss (`eval_context`, 2 k, full held-out)

$\mathcal{G}$ is empty for positions below 512, so the schedule cannot act there directly.

| positions | `162` | `194` − `162` | `196` − `162` | `195` − `162` |
|---|---|---|---|---|
| 0–512 | 3.4106 | +0.0011 | +0.0060 | +0.0029 |
| 512–1024 | 3.3224 | +0.0035 | +0.0083 | +0.0117 |
| 1024–2048 | 3.3219 | +0.0041 | +0.0104 | +0.0133 |
| all | 3.3442 | +0.0032 | +0.0088 | +0.0103 |

In every arm the cost grows with position, which is where the global range is: `194` pays 0.001
where there is nothing to select and 0.004 at the far end. `195` puts nearly all of its cost
beyond 512 (+0.003 → +0.013). `196` differs: it is already +0.0060 at positions 0–512, where its
schedule is the same function as L5d's, and adds only 0.002–0.004 beyond. So about two thirds of
`196`'s gap is not a far-context effect at these positions. One seed; a 0.006 difference in a
bucket that the mechanism cannot touch is also the size of seed noise at `screen` (2σ).

## Eval-time $K_{sel}$ sweep (`probe_ksel`, 512 held-out windows, trained at 128)

Losses are on the probe's 512 windows, so they compare within an arm, not with the tables above.

| | $K_{sel}$ = 64 | 128 (trained) | 256 | Δ at 64 | Δ at 256 | reads at 2048 with 64 |
|---|---|---|---|---|---|---|
| `194` | 3.3302 | 3.3294 | 3.3294 | +0.0008 | +0.0000 | 3,456 (0.28×) |
| `195` | 3.3367 | 3.3354 | 3.3349 | +0.0012 | −0.0005 | 1,728 (0.14×) |
| `196` | 3.3365 | 3.3345 | 3.3339 | +0.0019 | −0.0006 | 1,984 (0.16×) |

Halving the selection at eval time costs 0.001–0.002 nats and doubling it buys at most 0.0006:
the models barely use keys ranked beyond 64, and the trained value is not a cliff. The cost of 64
is largest where seven iterations depend on one selection (`196`). By ADR-078 the `small` pair keeps the screened $K_{sel}$ = 128.

## Copy probe (`probe_copy`, 512 windows, source 256–320 repeated at 1792–1856)

| | copy loss | control loss | control − copy |
|---|---|---|---|
| `162` | 3.9118 | 3.2506 | −0.661 |
| `194` | 3.9222 | 3.2622 | −0.660 |
| `195` | 3.9221 | 3.2639 | −0.658 |
| `196` | 3.9300 | 3.2712 | −0.659 |

No arm copies a span from 1.5 k tokens back (the repeated span is 0.66 nats *harder* than the
control in all four), as for every L5d-family arm so far. The schedule changes nothing here, so
this probe cannot tell the arms apart.

## Routing and run health

Worst point 3.38 of 8 effective experts at step 18, two dead; last dead expert at step 37 (`162`:
22); 7.99 of 8 at the end, loads 0.118–0.131.

| | |
|---|---|
| wall-clock | 8.92 h; summed `step_s` 8.813 h + evals 0.097 h + checkpoints 0.004 h → **0.0 % unaccounted** |
| throughput | 31.5 k tok/s median, flat; 7.1 % slower than `162`, as `194`: seven masked iterations pay the top-$K$ selection and the reference skips no reads |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 74 °C, min clock 1905 MHz |

## Interpretation

One full global iteration followed by seven that reuse its top-128 keys costs L5d 0.0088 nats:
0.26 % of $L_{ref}$, inside the 4σ band and a quarter of the 1 % H21 allows, for 0.20× of the
global reads at the end of a 2 k context. Set beside the other two arms it orders the schedule's
parts. Refreshing the selection with a second full iteration (`194`) is worth about 0.006;
letting iterations 5–8 see the stale selection instead of nothing (`196` against `195`) is worth
0.0015, which is nothing at one seed. So among the cheap schedules `195` dominates `196` (fewer
reads, same loss, and no selection overhead in half its iterations), and `194` remains the point
that matches L5d. The sweep says the selection could be half as large at eval time for under
0.002. The per-position eval supports reading `194`'s and `195`'s costs as far-context costs;
`196`'s sits partly where the schedule cannot act, which I cannot explain from one seed and do
not chase. None of this is a verdict by ADR-070: H21 rests on `194` at `screen` and on the
`209`/`210` pair at `small`. One seed at `screen`.
