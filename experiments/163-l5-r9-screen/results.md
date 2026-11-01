# 163-l5-r9-screen — L5 at fixed $r$ = 9, FLOP-matched to L5e (ADR-040): 3.3349 nats, **+0.0021 vs `110`** (the extra iteration buys nothing) → **`151` − `163` = −0.0404, 3.4× the 4σ line: L5e's gain is not its compute**; I32 resolved at one seed

`python -m scripts.train.train --rung L5 --id 163-l5-r9-screen --tokens 1e+09 --micro-batch 4 --seed 0 --set middle.r_min=9 middle.r_max=9 experts.d_ff=896` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. The `114`/`115` construction ($d_{ff}$ pinned at 896 so only $r$ changes). Code at the `queue_screen_11` gate commit `22d1cdd`.

**Bears on:** **H6** and `08 §4`'s HRM-Text claim (two weight sets beat one shared loop), through
**I32** and **ADR-040**. L5e (`151`) adds a dense H block with its own weights and +7.9 % forward
FLOPs; one more L5 iteration adds +7.4 % FLOPs and no parameters. ADR-040's reading: `151` − `163`
< −4σ → the second weight set beats equal compute (HRM-Text supported in mechanism); |`151` − `163`|
≤ 2σ → L5e's gain is compute; in between → inconclusive. **Result: −0.0404, 3.4× 4σ → the second
weight set beats equal compute, at one seed.** H6 itself is **silent** here (no matched-params
comparison is made).

σ = 0.0030 over twelve pairs (`161`; 2σ 0.0059, 4σ 0.0119).

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (identical to `110`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3349** nats |
| final train loss (last step, main head) | 3.3033 |
| throughput (median step) | 45,817 tokens/s aggregate (`110` 48,553: −5.6 %; `151` 47,368) |
| peak memory per GPU | 3.88 GiB |
| wall-clock (this session) | 6.13 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 9 / 9.0000 |

## ADR-040's reading

| | val loss | Δ vs `110` | params | training FLOPs vs `110` |
|---|---|---|---|---|
| `110-l5-screen` (L5, $r$ = 8) | 3.3328 | — | 77.5 M | — |
| **`163-l5-r9-screen` (L5, $r$ = 9)** | **3.3349** | **+0.0021** (0.7σ) | 77.5 M | +7.4 % |
| `151-l5e-screen` (L5e, H block after iteration 4) | 3.2945 | −0.0383 | +6.3 M | +7.9 % |
| `148-l5-ne13-screen` (L5, $N_e$ = 13) | 3.3043 | −0.0285 | +10.3 M | ≈ 0 |

- **What the compute buys:** $\Delta_9$ = `163` − `110` = **+0.0021**, inside 2σ. A ninth iteration
  is worth nothing at `screen`.
- **`151` − `163` = −0.0404**, 3.4× the 4σ line (0.0119) → **L5e's gain is the second weight set,
  not its compute.** I32's question is answered at one seed.
- What it does **not** answer: whether the H block's *form* matters, or just its parameters. `151` −
  `148` = −0.0098 (between 2σ and 4σ, inconclusive) with `148`'s extra parameters as experts at no
  added FLOPs. The cleaner statement is "added weights help and added depth does not"; whether a
  dense block with its own weights beats the same weights spent on experts stays open.

## The L5 depth curve, now with $r$ = 9

| $r$ | run | val loss | Δ vs $r$ = 8 |
|---|---|---|---|
| 4 | `114` | 3.3567 | +0.0239 |
| 8 | `110` | 3.3328 | — |
| 9 | **`163`** | **3.3349** | **+0.0021** |
| 12 | `115` | 3.3370 | +0.0042 |

The curve is flat past 8: $r$ = 9 and 12 both sit within 2σ of 8. `115`'s inference that depth alone
buys little is now measured at the FLOP-matched point.

**Curve against `110` and `151`:**

| step | `110` ($r$ 8) | `163` ($r$ 9) | `151` (L5e) |
|---|---|---|---|
| 100 | 5.6159 | 5.7138 | 5.6039 |
| 300 | 4.3664 | 4.4021 | 4.3266 |
| 500 | 3.8804 | 3.8864 | 3.8217 |
| 1000 | 3.4995 | 3.5043 | 3.4595 |
| 1500 | 3.3697 | 3.3734 | 3.3320 |
| 1907 | 3.3328 | 3.3349 | 3.2945 |

The deeper loop starts slower (+0.10 at step 100, a longer path through the shared block) and catches
up to `110` without passing it; L5e is ahead from step 100 to the end.

## Routing

Worst point 3.57 of 8 effective experts early; **8.00 of 8 at the end, loads 0.121–0.128**, as `110`.
The logged router aux is 0.0090 against `110`'s 0.0079, which is the expected ≈ 9/8 for one more
iteration's aux term.

## Run health

| | |
|---|---|
| wall-clock | 6.13 h; summed `step_s` 6.052 h + evals 0.074 h → **0.0 % unaccounted** |
| throughput | 45.8 k tok/s median, −5.6 % vs `110` for +7.4 % FLOPs |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 80 °C, min clock 1890 MHz |

## Interpretation

A ninth iteration of the shared block, which costs what L5e's extra block costs in FLOPs, moves
held-out loss by +0.0021 nats: nothing. L5e is 0.0404 below it, 3.4× the 4σ line, so L5e's gain over
L5 is not the extra compute; it comes from the second set of weights, which is the mechanism
`08 §4`'s HRM-Text claim names. That claim is supported in mechanism at one seed, with one caveat: the
same parameter budget spent on more experts (`148`) gets most of the way there (−0.0285 vs L5e's
−0.0383), and the 0.0098 difference is inconclusive. I32 is resolved; whether to spend a `small` pair
separating L5e from `148`'s construction is the user's call. The run also extends the L5 depth curve:
flat past $r$ = 8 at $r$ = 9 and 12, steep below it.
