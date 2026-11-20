# 185-l5h-jb2-screen — L5h, $\mathcal{G}$ bridged from iteration $j_b$ = 2 through its own projections (ADR-067): 3.3292 nats, **−0.0151 vs `162` (5.0σ, beats L5d)**, −0.0001 vs `184` ($j_b$ = 1) → H20's loss clause met again; the gain does not depend on $j_b$ between 1 and 2; attribution waits on the untied control `187`

`python -m scripts.train.train --rung L5h-jb2 --id 185-l5h-jb2-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`attention.global_source: bridged`, `bridge_iteration: 2`), per-step log in `log.jsonl`. Code at the `queue_screen_19` gate commit `15bd3b3`.

**Bears on:** **H20** (a global cache written from an early iteration costs little, so prefill can
stop early), by **ADR-067 §5**: the loss clause is non-inferiority with $T$ = 2 % $L_{ref}$ =
0.0667 on $\Delta$(arm − `162`), read by ADR-066 §1. **Result: $\Delta$ = −0.0151 = 5.0σ below
`162`**: inside the allowance and past the 4σ = 0.0119 line on the good side, as `184` was.
**H20's loss clause: supported in direction at `screen`, one seed.** Against `184` the difference
is −0.0001 (0.03σ): the two bridge points are the same run to within noise. What the gain is
*from* is still not known: L5h differs from L5d in the early source and in the untied global
projections with their own norm (+0.39 M parameters), and `187` ($j_b$ = 8, untied only)
separates them. $\Delta$(arm − `187`) and $\Delta$(`187` − `162`) wait for `187` (≈ 10:20 on
2026-11-21).

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.87 M total, 53.30 M non-embedding (as `184`; +0.39 M vs `162`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3292** nats |
| final train loss (last step, main head) | 3.2973 |
| router aux (last step) | 0.0079 |
| throughput (median step) | 33,636 tokens/s aggregate (0.99× `162`) |
| peak memory per GPU | 3.90 GiB |
| wall-clock (this session) | 8.35 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `162`, `184` and the ladder

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `110-l5-screen` (L5, global per-iteration attention) | 3.3328 | | |
| `162-l5d-segmean-screen` (L5d, $\mathcal{G}$ from $c^{(r)}$, tied projections) | 3.3443 | — | comparator |
| `184-l5h-jb1-screen` ($\mathcal{G}$ from $c^{(1)}$, own projections) | 3.3293 | −0.0150 vs `162` | beats L5d |
| **`185-l5h-jb2-screen` ($\mathcal{G}$ from $c^{(2)}$, own projections)** | **3.3292** | **−0.0151 vs `162`** (−0.45 %, 5.0σ) | **$\Delta \le T$ = 0.0667 and $\Delta$ < −4σ: beats L5d** |
| | | −0.0001 vs `184` (0.03σ) | no dependence on $j_b$ between 1 and 2 |
| | | −0.0036 vs `110` (1.2σ) | |
| | | −0.0066 vs $L_{ref}$ | |

**Curve** (976-window evals; last row full held-out):

| step | `162` | `184` | `185` | `185` − `162` | `185` − `184` |
|---|---|---|---|---|---|
| 100 | 5.5372 | 5.5346 | 5.5671 | +0.0299 | +0.0325 |
| 300 | 4.3094 | 4.3274 | 4.3386 | +0.0292 | +0.0112 |
| 500 | 3.8551 | 3.8690 | 3.8612 | +0.0061 | −0.0078 |
| 700 | 3.6599 | 3.6565 | 3.6570 | −0.0029 | +0.0005 |
| 1000 | 3.5084 | 3.4942 | 3.4952 | −0.0132 | +0.0010 |
| 1500 | 3.3835 | 3.3667 | 3.3675 | −0.0160 | +0.0008 |
| 1900 | 3.3436 | 3.3263 | 3.3272 | −0.0164 | +0.0009 |
| 1907 (full) | 3.3443 | 3.3293 | 3.3292 | −0.0151 | −0.0001 |

Behind `162` until step ≈ 600, ahead from 700, lead steady at ≈ 0.016 over the last third. From
step 700 on `185` and `184` are within 0.001 of each other at every eval.

## Prefill arithmetic (ADR-067 §3; reported, not measured)

With $j_b$ = 2 a 2048-token prompt in 512-token segments costs $(L − S)\,j_b + S\,r$ = 3072 + 4096
iterations against 16,384: **0.44×** of L5d's prefill iterations (`184`: 0.34×). TTFT itself is
S5's to measure.

## Routing and run health

Worst point 4.23 of 8 effective experts at step 13, up to three dead experts (step 11); last dead
expert at step 98 (`184`: step 52); 7.99 of 8 at the end, loads 0.117–0.132 — as `162` and `184`.

| | |
|---|---|
| wall-clock | 8.35 h; summed `step_s` 8.259 h + evals 0.083 h + checkpoints 0.004 h → **0.0 % unaccounted** |
| throughput | 33.6 k tok/s median, flat |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 75 °C, min clock 1905 MHz |

## Post-run evals

The copy probe and `eval_context` (ADR-066 §2) run for `184`–`187` at the end of
`queue_screen_19`; they are written up with `187`.

## Interpretation

Bridging the global cache from the second middle iteration gives the same loss as bridging from
the first: 3.3292 against 3.3293, with curves that coincide from step 700. Both beat L5d by 5.0σ
and sit level with L5 (`110`). So within $j_b$ ∈ {1, 2} the bridge point buys nothing in loss, and
by ADR-067 §5 the smaller one stays the candidate: $j_b$ = 1 passes and has the cheaper prefill
(0.34× against 0.44× at 2 k). That the two arms agree this closely makes it more likely that the
gain over `162` comes from what they share, the untied global projections and their norm, than
from where the cache is read; `148` and L5c showed in this ladder that untied attention weights
alone are worth several σ. That is a guess until `187` ($j_b$ = 8, untied only) is in: if `187`
also lands near 3.329, the early source is free and the untying is the cause. `186` ($j_b$ = 4) is
running. One seed at `screen`; nothing here is a verdict.
