# 194-l5g-fsss-screen — L5g with the global schedule `[F,S,S,S] × 2`, $K_{sel}$ = 128 (ADR-070, on L5d): 3.3475 nats, **+0.0032 vs `162` (1.1σ, far inside $T$ = 0.0334 → H21's loss clause met at `screen`; one seed, silent as a verdict)**; global keys read 0.31× of L5d's at position 2048 by arithmetic; the `small` pair needs its own ADR

`python -m scripts.train.train --rung L5g-fsss --id 194-l5g-fsss-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the L5d preset with `attention.schedule: [F,S,S,S,F,S,S,S]`, `k_sel: 128`, `rope_scope: all`, `segment_aux: mean`), per-step log in `log.jsonl`. Code at the `queue_screen_22` gate commit `7e50b5c`.

**Bears on:** **H21** (full global attention at every 4th iteration, with the other iterations
attending to the last full iteration's top-$K_{sel}$ keys, matches full global attention at every
iteration within 1 % loss; global reads per token drop ≈ 4×), by **ADR-070 §5** under ADR-066 §1:

| clause | rule | measured | reading |
|---|---|---|---|
| loss | $\Delta$(`194` − `162`) against $T$ = 1 % $L_{ref}$ = 0.0334; weakens above $T$ + 2σ = 0.0393; $\lvert\Delta\rvert$ ≤ 4σ silent at one seed | **+0.0032** (+0.10 %, 1.1σ) | **inside 2σ: matches `162`; within $T$ by a factor of ten** |
| reads | global keys read, against L5d | 0.3125× at position 2048 (3.2× fewer); 0.37× averaged over a 2 k window | arithmetic, not measured (the reference masks, it does not skip) |

**H21 at `screen`, one seed: the loss clause is met; by ADR-066 §1 a result inside 4σ is silent
as a verdict, leaning supports.** The read saving at 2 k is 3.2×, not the ≈ 4× of the hypothesis,
because $K_{sel}$ = 128 is a twelfth of $|\mathcal{G}|$ here and not negligible; the ratio falls
towards 0.25 as the context grows. **What follows (ADR-070 §5):** `194` lands within $T$, which
is the condition for a `small` pair *by its own ADR*. That is the user's call; nothing is queued.
`195` (running) and `196` are the other two points on the loss-vs-reads curve. The $K_{sel}$
sweep {64, 256}, the copy probe and the per-position eval run at the end of `queue_screen_22` and
are written up with `196`.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.91 M non-embedding (as `162`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3475** nats |
| final train loss (last step, main head) | 3.3145 |
| router aux (last step) | 0.0081 |
| throughput (median step) | 31,394 tokens/s aggregate (**0.926× `162`'s 33,913**) |
| peak memory per GPU | 3.94 GiB (`162` 3.89) |
| wall-clock (this session) | 8.95 h (`162` 8.28 h) |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `162` and the ladder

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `110-l5-screen` (L5) | 3.3328 | | |
| `162-l5d-segmean-screen` (L5d, full $\mathcal{G}$ at all 8 iterations) | 3.3443 | — | comparator |
| **`194-l5g-fsss-screen` (full at iterations 1 and 5, top-128 at the other six)** | **3.3475** | **+0.0032 vs `162`** (+0.10 %, 1.1σ) | **matches; inside $T$** |
| | | +0.0147 vs `110` | |
| | | +0.0117 vs $L_{ref}$ (+0.35 %) | |

**Curve** (976-window evals; last row full held-out):

| step | `162` | `194` | `194` − `162` |
|---|---|---|---|
| 100 | 5.5372 | 5.5286 | −0.0086 |
| 300 | 4.3094 | 4.3067 | −0.0027 |
| 500 | 3.8551 | 3.8579 | +0.0028 |
| 700 | 3.6599 | 3.6644 | +0.0045 |
| 1000 | 3.5084 | 3.5121 | +0.0037 |
| 1500 | 3.3835 | 3.3863 | +0.0029 |
| 1900 | 3.3436 | 3.3464 | +0.0028 |
| 1907 (full) | 3.3443 | 3.3475 | +0.0032 |

A steady +0.003 from step 500: a small cost that neither grows nor closes, about one σ.

## Global reads, by arithmetic (ADR-070 §2)

Keys of $\mathcal{G}$ read per token per iteration: $|\mathcal{G}|$ = $n$ − 512 at F,
min($K_{sel}$, $|\mathcal{G}|$) at S, 0 at L (the counting `sim/params.py`'s `attention_reads`
uses).

| | L5d (`162`) | `194` | ratio |
|---|---|---|---|
| at position 2048 ($|\mathcal{G}|$ = 1536), summed over 8 iterations | 12,288 | 2 × 1536 + 6 × 128 = 3,840 | **0.3125** |
| mean over the positions of a 2 k window (my arithmetic: mean $|\mathcal{G}|$ 575.6, mean sparse read 92.0) | 4,605 | 1,703 | 0.370 |

Nothing here is a measured saving: the reference implementation applies the selection as a mask
over the full score matrix, so the run is 7.4 % *slower* than `162` (the top-$K$ selection is
extra work) and uses 0.05 GiB more.

## Routing and run health

Worst point 4.23 of 8 effective experts at step 24 with two dead (`162`: 4.06 at step 19); no dead
expert after step 24; 7.997 of 8 at the end, loads 0.119–0.131.

| | |
|---|---|
| wall-clock | 8.95 h; summed `step_s` 8.846 h + evals 0.097 h + checkpoints 0.004 h → **0.0 % unaccounted** |
| throughput | 31.4 k tok/s median, flat |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 74 °C, min clock 1905 MHz |

## Interpretation

Restricting six of the eight iterations to the 128 global keys that the last full iteration
scored highest costs 0.0032 nats against L5d, about one σ and a tenth of the 1 % H21 allows. The
gap is constant from a quarter of the way through training, so it reads as a small real cost
more than as noise, but it cannot be told from zero at one seed. In exchange the design reads
0.31× of L5d's global keys at the end of a 2 k context, which is the point of the schedule for
S6; the reference implementation shows none of that as speed. So H21's loss clause holds at
`screen` with a wide margin, and its "≈ 4×" is 3.2× at this context with $K_{sel}$ = 128. Whether
the selection itself matters (a smaller or larger $K_{sel}$ at eval time), whether local-only
iterations are affordable (`195`) and whether one full iteration is enough (`196`) are the rest
of this queue. One seed at `screen`.
