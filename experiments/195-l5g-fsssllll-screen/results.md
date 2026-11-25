# 195-l5g-fsssllll-screen — L5g with the global schedule `[F,S,S,S,L,L,L,L]`, $K_{sel}$ = 128 (ADR-070, on L5d): 3.3546 nats, **+0.0103 vs `162` (3.4σ, inside 4σ and inside $T$)**, +0.0071 vs `194` (2.4σ); a point on H21's loss-vs-reads curve, not a verdict: global keys read 0.16× of L5d's at position 2048

`python -m scripts.train.train --rung L5g-fsssllll --id 195-l5g-fsssllll-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the L5d preset with `attention.schedule: [F,S,S,S,L,L,L,L]`, `k_sel: 128`), per-step log in `log.jsonl`. Code at the `queue_screen_22` gate commit `7e50b5c`.

**Bears on:** **H21**, as a curve point only. ADR-070 §5 puts H21's verdict on `194`
(`[F,S,S,S] × 2`: +0.0032 vs `162`, loss clause met); `195` and `196` "are points on the
loss-vs-reads curve, not verdicts". **H21: unchanged by this run (silent).** For the record,
under ADR-066 §1's bands the arm is inside 4σ of `162` and inside $T$ = 0.0334.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.91 M non-embedding (as `162`, `194`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3546** nats |
| final train loss (last step, main head) | 3.3228 |
| router aux (last step) | 0.0081 |
| throughput (median step) | 33,858 tokens/s aggregate (1.00× `162`'s 33,913; `194` 31,394) |
| peak memory per GPU | 3.91 GiB |
| wall-clock (this session) | 8.30 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## The loss-vs-reads curve so far

Global keys read per token, summed over the 8 iterations, by ADR-070 §2's counting
($|\mathcal{G}|$ = $n$ − 512 at F, min(128, $|\mathcal{G}|$) at S, 0 at L); the window mean is my
arithmetic over the positions of a 2 k window, as in `194`.

| | schedule | val loss | Δ vs `162` | reads at position 2048 | reads, 2 k window mean |
|---|---|---|---|---|---|
| `162` (L5d) | F × 8 | 3.3443 | — | 12,288 (1.00×) | 1.00× |
| `194` | `[F,S,S,S] × 2` | 3.3475 | +0.0032 (1.1σ) | 3,840 (0.31×) | 0.37× |
| **`195`** | **`[F,S,S,S,L,L,L,L]`** | **3.3546** | **+0.0103** (+0.31 %, 3.4σ) | **1,920 (0.16×)** | **0.18×** |
| `196` | `[F,S,S,S,S,S,S,S]` | running | | 2,432 (0.20×) | 0.26× |

`195` − `194` = **+0.0071 (2.4σ)**: dropping the second full iteration *and* all global reads in
iterations 5–8 costs about twice what `194` cost, for half of `194`'s reads. Both steps are
inside 4σ at one seed.

**Curve** (976-window evals; last row full held-out):

| step | `162` | `194` | `195` | `195` − `162` | `195` − `194` |
|---|---|---|---|---|---|
| 100 | 5.5372 | 5.5286 | 5.5257 | −0.0114 | −0.0029 |
| 300 | 4.3094 | 4.3067 | 4.3405 | +0.0311 | +0.0338 |
| 500 | 3.8551 | 3.8579 | 3.8922 | +0.0371 | +0.0343 |
| 700 | 3.6599 | 3.6644 | 3.6858 | +0.0259 | +0.0215 |
| 1000 | 3.5084 | 3.5121 | 3.5225 | +0.0141 | +0.0104 |
| 1500 | 3.3835 | 3.3863 | 3.3918 | +0.0083 | +0.0054 |
| 1900 | 3.3436 | 3.3464 | 3.3517 | +0.0081 | +0.0053 |
| 1907 (full) | 3.3443 | 3.3475 | 3.3546 | +0.0103 | +0.0071 |

Unlike `194`, this arm falls 0.03–0.04 behind early (steps 300–500) and recovers most of it; the
gap is still closing slowly on the 976-window evals at the end (+0.008).

## Routing and run health

Worst point 3.30 of 8 effective experts at step 24, two dead; last dead expert at step 46 (`162`:
22); 7.99 of 8 at the end, loads 0.116–0.132.

| | |
|---|---|
| wall-clock | 8.30 h; summed `step_s` 8.205 h + evals 0.086 h + checkpoints 0.004 h → **0.0 % unaccounted** |
| throughput | 33.9 k tok/s median, flat. The four local iterations skip the global scores, which pays back the selection overhead that made `194` 7 % slower; no read saving beyond that is implemented |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 72 °C, min clock 1905 MHz |

## Interpretation

With one full global iteration, three sparse ones and four that see only the local window, L5d
loses 0.0103 nats: 0.31 % of $L_{ref}$, inside the 1 % that H21 allows its own schedule and just
inside the 4σ band, for 0.16× of the global reads at the end of a 2 k context. Against `194` the
extra cost is 0.0071, so the last four iterations' view of far context is worth something, but
little; the early deficit and slow recovery suggest the model takes longer to learn to carry what
it read in iterations 1–4 through the local-only half. This is a curve point by ADR-070 and
decides nothing on its own. `196` (one full iteration, seven sparse) separates the two changes
made here, and the per-position eval at the end of the queue says whether the cost sits beyond
position 512, where $\mathcal{G}$ acts. One seed at `screen`.
