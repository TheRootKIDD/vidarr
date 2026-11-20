# 184-l5h-jb1-screen — L5h, $\mathcal{G}$ bridged from iteration $j_b$ = 1 through its own projections (ADR-067): 3.3293 nats, **−0.0150 vs `162` (5.0σ, beats L5d)** → H20's loss clause met at $j_b$ = 1 with room to spare; attribution waits on the untied control `187`

`python -m scripts.train.train --rung L5h-jb1 --id 184-l5h-jb1-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`attention.global_source: bridged`, `bridge_iteration: 1`), per-step log in `log.jsonl`. Code at the `queue_screen_19` gate commit `15bd3b3`.

**Bears on:** **H20** (a global cache written from an early iteration costs little, so prefill can
stop early), by **ADR-067 §5**: the loss clause is non-inferiority with $T$ = 2 % $L_{ref}$ =
0.0667 on $\Delta$(arm − `162`), read by ADR-066 §1. **Result: $\Delta$ = −0.0150 = 5.0σ below
`162`.** The arm is not merely inside the allowance, it *beats* L5d by more than 4σ = 0.0119, at
the most aggressive bridge point. **H20's loss clause: supported in direction at `screen`, one
seed.** What the gain is *from* is not known yet: L5h differs from L5d in two ways (the early
source, and untied global projections with their own norm, +0.39 M parameters), and `187`
($j_b$ = 8, untied only) separates them. ADR-067 reports $\Delta$(arm − `187`) and
$\Delta$(`187` − `162`) beside this number; both wait for `187` (≈ 10:20 on 2026-11-21).

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.87 M total, 53.30 M non-embedding (+0.39 M vs `162`: $W_K^b$, $W_V^b$, norm) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3293** nats |
| final train loss (last step, main head) | 3.2979 |
| router aux (last step) | 0.0081 |
| throughput (median step) | 33,696 tokens/s aggregate (0.99× `162`) |
| peak memory per GPU | 3.90 GiB |
| wall-clock (this session) | 8.34 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `162` and the ladder

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `110-l5-screen` (L5, global per-iteration attention) | 3.3328 | | |
| `162-l5d-segmean-screen` (L5d, $\mathcal{G}$ from $c^{(r)}$, tied projections) | 3.3443 | — | comparator |
| **`184-l5h-jb1-screen` ($\mathcal{G}$ from $c^{(1)}$, own projections)** | **3.3293** | **−0.0150 vs `162`** (−0.45 %, 5.0σ) | **$\Delta \le T$ = 0.0667 and $\Delta$ < −4σ: beats L5d** |
| | | −0.0035 vs `110` (1.2σ) | L5d's whole cost against L5 is recovered |
| | | −0.0065 vs $L_{ref}$ | |

**Curve** (976-window evals; last row full held-out):

| step | `162` | `184` | `184` − `162` |
|---|---|---|---|
| 100 | 5.5372 | 5.5346 | −0.0026 |
| 300 | 4.3094 | 4.3274 | +0.0180 |
| 500 | 3.8551 | 3.8690 | +0.0139 |
| 700 | 3.6599 | 3.6565 | −0.0034 |
| 1000 | 3.5084 | 3.4942 | −0.0142 |
| 1500 | 3.3835 | 3.3667 | −0.0168 |
| 1900 | 3.3436 | 3.3263 | −0.0173 |
| 1907 (full) | 3.3443 | 3.3293 | −0.0150 |

Behind until step ≈ 600, ahead from 700, and the lead is steady at 0.015–0.017 over the last third.

## Prefill arithmetic (ADR-067 §3; reported, not measured)

With $j_b$ = 1 a 2048-token prompt in 512-token segments costs $(L − S)\,j_b + S\,r$ = 1536 + 4096
iterations against 16,384: **0.34×** of L5d's prefill iterations. TTFT itself is S5's to measure.

## Routing and run health

Worst point 4.20 of 8 effective experts with two dead at step 14; last dead expert at step 52;
8.00 of 8 at the end, loads 0.121–0.130 — as `162`.

| | |
|---|---|
| wall-clock | 8.34 h; summed `step_s` 8.246 h + evals 0.082 h + checkpoints 0.004 h → **0.0 % unaccounted** |
| throughput | 33.7 k tok/s median, flat |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 73 °C, min clock 1905 MHz |

## Post-run evals

The copy probe and `eval_context` (ADR-066 §2) run for `184`–`187` at the end of
`queue_screen_19`; they are written up with `187`.

## Interpretation

Writing the global cache from the first middle iteration, through projections of its own, does
not cost L5d anything at 1 B tokens: it gains 0.0150 nats (5.0σ) and brings L5d level with L5's
global per-iteration attention (−0.0035 vs `110`, inside noise) while cutting the prefill
iteration count for a 2 k prompt to 0.34×. H20 asked only that the cost stay inside 2 % of
$L_{ref}$; that is met with the sign reversed. The obvious caution is ADR-067's own: the arm also
unties the global projections, and `148`/L5c showed in this ladder that untied attention weights
alone can be worth several σ. If `187` shows the same gain, the early source is free but not the
cause; if `187` sits at `162`, the early source itself helps. Either way $j_b$ = 1 is the smallest
bridge point and it passes, so by ADR-067 §5 it is the candidate for the `small` pair (which needs
its own ADR), and by ADR-066 §1 an L5-family arm that beats its comparator by > 4σ earns one.
`185` and `186` now show whether the gain depends on $j_b$. One seed at `screen`: the L5 = dense
tie at this budget did not survive `small`, so nothing here is a verdict.
