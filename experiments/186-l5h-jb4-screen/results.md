# 186-l5h-jb4-screen — L5h, $\mathcal{G}$ bridged from iteration $j_b$ = 4 through its own projections (ADR-067): 3.3355 nats, **−0.0088 vs `162` (2.9σ, inside 4σ: matches L5d)**, +0.0062 vs `184` and +0.0063 vs `185` (2.1σ) → H20's loss clause met; $j_b$ = 4 is not better than $j_b$ = 1 or 2; attribution waits on the untied control `187`

`python -m scripts.train.train --rung L5h-jb4 --id 186-l5h-jb4-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`attention.global_source: bridged`, `bridge_iteration: 4`), per-step log in `log.jsonl`. Code at the `queue_screen_19` gate commit `15bd3b3`.

**Bears on:** **H20** (a global cache written from an early iteration costs little, so prefill can
stop early), by **ADR-067 §5**: the loss clause is non-inferiority with $T$ = 2 % $L_{ref}$ =
0.0667 on $\Delta$(arm − `162`), read by ADR-066 §1. **Result: $\Delta$ = −0.0088 = 2.9σ below
`162`**: inside the allowance, and inside the 4σ = 0.0119 band, so by ADR-066 §1 this arm
*matches* L5d and does not beat it (`184` and `185` did, at 5.0σ). **H20's loss clause: supported
in direction at `screen`, one seed.** The arm is 0.0062 / 0.0063 behind `184` / `185` (2.1σ, just
past 2σ, inside 4σ): a later bridge point is not better and may be slightly worse, which one seed
cannot decide. $\Delta$(arm − `187`) and $\Delta$(`187` − `162`) wait for `187` (≈ 10:20 today).

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.87 M total, 53.30 M non-embedding (as `184`; +0.39 M vs `162`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3355** nats |
| final train loss (last step, main head) | 3.3040 |
| router aux (last step) | 0.0081 |
| throughput (median step) | 33,676 tokens/s aggregate (0.99× `162`) |
| peak memory per GPU | 3.90 GiB |
| wall-clock (this session) | 8.34 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `162`, the other bridge points and the ladder

| | val loss | Δ vs `162` | reading |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `110-l5-screen` (L5, global per-iteration attention) | 3.3328 | | |
| `162-l5d-segmean-screen` (L5d, $\mathcal{G}$ from $c^{(r)}$, tied projections) | 3.3443 | — | comparator |
| `184-l5h-jb1-screen` ($j_b$ = 1) | 3.3293 | −0.0150 (5.0σ) | beats L5d |
| `185-l5h-jb2-screen` ($j_b$ = 2) | 3.3292 | −0.0151 (5.0σ) | beats L5d |
| **`186-l5h-jb4-screen` ($j_b$ = 4)** | **3.3355** | **−0.0088** (−0.26 %, 2.9σ) | **$\Delta \le T$ = 0.0667; $|\Delta| \le$ 4σ: matches L5d** |

Also: +0.0062 vs `184`, +0.0063 vs `185` (2.1σ); +0.0027 vs `110` (0.9σ); −0.0003 vs $L_{ref}$.

**Curve** (976-window evals; last row full held-out):

| step | `162` | `184` | `185` | `186` | `186` − `162` | `186` − `185` |
|---|---|---|---|---|---|---|
| 100 | 5.5372 | 5.5346 | 5.5671 | 5.5768 | +0.0396 | +0.0097 |
| 300 | 4.3094 | 4.3274 | 4.3386 | 4.3583 | +0.0489 | +0.0197 |
| 500 | 3.8551 | 3.8690 | 3.8612 | 3.8668 | +0.0117 | +0.0056 |
| 700 | 3.6599 | 3.6565 | 3.6570 | 3.6593 | −0.0006 | +0.0023 |
| 1000 | 3.5084 | 3.4942 | 3.4952 | 3.5038 | −0.0046 | +0.0086 |
| 1500 | 3.3835 | 3.3667 | 3.3675 | 3.3730 | −0.0105 | +0.0055 |
| 1900 | 3.3436 | 3.3263 | 3.3272 | 3.3332 | −0.0104 | +0.0060 |
| 1907 (full) | 3.3443 | 3.3293 | 3.3292 | 3.3355 | −0.0088 | +0.0063 |

Level with `162` at step 700, ahead by ≈ 0.010 over the last quarter. Behind `185` at every eval,
by 0.005–0.009 from step 1000 to the end: the gap to the earlier bridge points is steady, not an
end-of-run fluctuation.

## Prefill arithmetic (ADR-067 §3; reported, not measured)

With $j_b$ = 4 a 2048-token prompt in 512-token segments costs $(L − S)\,j_b + S\,r$ = 6144 + 4096
iterations against 16,384: **0.63×** of L5d's prefill iterations (`184` 0.34×, `185` 0.44×).

## Routing and run health

Worst point 3.33 of 8 effective experts at step 14, up to three dead experts (step 10); last dead
expert at step 34; 7.99 of 8 at the end, loads 0.117–0.133 — as `162`, `184`, `185`.

| | |
|---|---|
| wall-clock | 8.34 h; summed `step_s` 8.254 h + evals 0.082 h + checkpoints 0.004 h → **0.0 % unaccounted** |
| throughput | 33.7 k tok/s median, flat |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 74 °C, min clock 1905 MHz |

## Post-run evals

The copy probe and `eval_context` (ADR-066 §2) run for `184`–`187` at the end of
`queue_screen_19`; they are written up with `187`.

## Interpretation

Bridging from the fourth iteration passes H20's loss clause like the two earlier points, but with
a smaller margin: 0.0088 below `162` (2.9σ, which ADR-066 §1 reads as matching) against 0.0150 for
$j_b$ = 1 and 2. The 0.006 gap to those two is 2.1σ and is there at every eval from step 1000, so
it looks like a real ordering, but it is inside the 4σ band and from one seed, so it is recorded
and not claimed. Nothing in the design predicted that a deeper source would be worse; the handoff's
worry ran the other way. It also complicates the simple reading of `185` (that the whole gain is
the untied projections): if untying were everything, `186` should sit at 3.329 too. `187`
($j_b$ = 8, untied, running) is the test: near 3.329 would make `186` the outlier; near 3.336 or
above would say the early source itself helps and the gain fades as $j_b$ grows. For the ladder the
practical outcome does not change: $j_b$ = 1 is the smallest passing bridge point, has the best
loss and the cheapest prefill (0.34× against 0.63× here), and stays the `small`-pair candidate by
its own ADR.
