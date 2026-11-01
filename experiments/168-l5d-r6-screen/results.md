# 168-l5d-r6-screen — L5d at fixed $r$ = 6 (ADR-041 §3): 3.3492 nats, **+0.0049 vs `162` ($r$ = 8, 0.8σ), −0.0109 vs `167` ($r$ = 4)** → the L5d depth curve bends: iterations 5–6 are worth twice iterations 7–8; the middle point of the curve that scores L6/L6c

`python -m scripts.train.train --rung L5d --id 168-l5d-r6-screen --tokens 1e+09 --micro-batch 4 --seed 0 --set middle.r_min=6 middle.r_max=6 experts.d_ff=896` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`router.segment_aux: mean`), per-step log in `log.jsonl`. Code at the `queue_screen_12` gate commit `5942ee1`.

**Bears on:** **H7**, as a control (I25, ADR-041 §3), like `167`. The only change against `162` is
$r$ = 6 instead of 8, with $d_{ff}$ pinned at 896. **It is not a verdict on a feature.** At $r$ = 6
= 0.75 $r_{max}$ it does not reach ADR-013's H7 depth clause ($\bar r \le 0.7\,r_{max}$ = 5.6), so it
says nothing against H7's own wording. Its job is to complete the curve: `165` L6 and `166` L6c are
scored against it in their own `results.md`.

σ = 0.0030 over twelve pairs (`161`; 2σ 0.0059, 4σ 0.0119).

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (identical to `162`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3492** nats |
| final train loss (last step, main head) | 3.3172 |
| router aux (last step) | 0.0061 (summed over iterations: `167` 0.0041 at 4, `162` 0.0076 at 8) |
| throughput (median step) | 41,282 tokens/s aggregate (`162` 33,913; `167` 53,054) |
| peak memory per GPU | 3.84 GiB |
| wall-clock (this session) | 6.81 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 6 / 6.0000 |

## The L5d fixed-$r$ curve (ADR-041 §3), complete

| $r$ | run | val loss | Δ vs `162` | marginal value of the last two iterations |
|---|---|---|---|---|
| 4 | `167` | 3.3601 | +0.0158 (1.3× 4σ) | — |
| **6** | **`168`** | **3.3492** | **+0.0049 (0.8σ)** | **iterations 5–6: 0.0109** |
| 8 | `162` | 3.3443 | — | iterations 7–8: 0.0049 |
| 8 | `111` (aux summed; not the baseline) | 3.3477 | +0.0034 | |

The curve is concave: iterations 5–6 buy 0.0109 nats and iterations 7–8 buy 0.0049, less than half.
On L5 the 4 → 8 step was worth 0.0239 and 8 → 12 nothing (`114`/`110`/`115`), so both bases flatten
around $r$ = 8. L5d flattens sooner. Every point is one seed, and the 6 → 8 difference (0.0049) is
inside 2σ, so "the curve bends" rests on the 4 → 6 step being more than twice the 6 → 8 step, not on
the 6 → 8 step being resolved.

**Curve over training** (val loss):

| step | `167` ($r$ = 4) | `168` ($r$ = 6) | `162` ($r$ = 8) |
|---|---|---|---|
| 100 | **5.3543** | 5.5017 | 5.5372 |
| 300 | **4.2735** | 4.3169 | 4.3094 |
| 500 | **3.8532** | 3.8859 | 3.8551 |
| 700 | 3.6681 | 3.6737 | **3.6599** |
| 1000 | 3.5186 | 3.5141 | **3.5084** |
| 1500 | 3.3985 | 3.3871 | **3.3835** |
| 1907 | 3.3601 | 3.3492 | **3.3443** |

The depth ordering settles from step ≈ 1000 (0.5 B tokens). Before that, shallower is faster, and $r$ = 6
sits *behind* both neighbours at step 500 (one seed; probably noise in the early curve).

## Routing

Worst point 4.10 of 8 effective experts at step 22, up to three dead experts in the first 44 steps,
**8.00 of 8 at the end with loads 0.119–0.129**. Same endpoint as `162` and `167`.

## Run health

| | |
|---|---|
| wall-clock | 6.81 h; summed `step_s` 6.728 h + evals 0.070 h → **0.2 % unaccounted** |
| throughput | 41.3 k tok/s median; per-step time is linear in $r$ to within 2 % (`167` 9.9 s, `168` 12.7 s, `162` 15.5 s → ≈ 1.4 s per iteration + 4.3 s fixed) |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 74 °C |

## Interpretation

L5d at six fixed iterations lands 0.0049 nats (0.8σ) behind eight and 0.0109 ahead of four. The
fixed-depth curve that ADR-041 §4 scores L6 and L6c against is now complete: 3.3601 / 3.3492 / 3.3443
at $r$ = 4 / 6 / 8. It is concave, so a learned policy has the most to win below $r$ = 6 and almost
nothing to win between 6 and 8. **H7: no verdict from this run.** It is a control, and at 0.75
$r_{max}$ it is outside H7's depth clause. The scoring of `165` (L6, $\bar r$ 7.38) and `166` (L6c,
$\bar r$ 5.52) is in their own write-ups.
