# 198-l5f-screen — L5f, attention residuals over iterations (ADR-072, on L5): 3.3148 nats, **−0.0180 vs `110` (6.0σ: beats L5 by more than 4σ)**, closing **14 % of the L5–L1 gap against the 25 % H25 names → H25 silent at `screen`**; a `small` pair follows by ADR-031 (its own ADR, the user's call); the lead is still growing at 1 B tokens

`python -m scripts.train.train --rung L5f --id 198-l5f-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the L5 preset with `middle.input_mix: attention_residual`), per-step log in `log.jsonl`. Code at the `queue_screen_24` gate commit `d4d0dfd`.

**Bears on:** **H25** (attention residuals over iterations close ≥ 25 % of the L5-vs-L1 gap at
matched FLOPs), by **ADR-072 §4** under ADR-066 §1; touches H6.

| rule (ADR-072 §4) | threshold | measured | reading |
|---|---|---|---|
| H25 supported: closes 25 % of `110` − `106` = 0.1280 | Δ(`198` − `110`) ≤ −0.0320 | **−0.0180** (14.1 % of the gap) | not reached |
| H25 silent: beats L5 but short of 25 % | −0.0320 < Δ < −4σ = −0.0119 | −0.0180 (6.0σ) | **this band** |
| beats L5 by > 4σ → `small` pair (ADR-031) | Δ < −0.0119 | −0.0180 | **met** |

**H25 at `screen`, one seed: silent.** L5f beats L5 by 6.0σ and closes 14 % of the gap to L1,
a little over half of what the hypothesis claims. ADR-072 §4: "either way, beating L5 by > 4σ
brings a `small` pair". As for the other Embla arms, that pair needs its own ADR and is the
user's call; nothing is queued.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.53 M total, 52.96 M non-embedding (+50.7 k vs `110`: $w$ and $W_q$, as ADR-072 §2 estimated) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3148** nats |
| final train loss (last step, main head) | 3.2843 (`110` 3.3016) |
| router aux (last step) | 0.0073 |
| throughput (median step) | 43,479 tokens/s aggregate (**0.895× `110`'s 48,553**) |
| peak memory per GPU | **5.66 GiB** (`110` 3.86: +1.80, the iterations' updates are kept for the mix) |
| wall-clock (this session) | 6.46 h (`110` 5.79 h) |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `110` and the ladder

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `106-l1-screen` (L1) | 3.2048 | | |
| `110-l5-screen` (L5, residual sum over iterations) | 3.3328 | — | comparator |
| **`198-l5f-screen` (learned per-token softmax mix over $v_0$ and earlier updates)** | **3.3148** | **−0.0180 vs `110`** (−0.54 %, 6.0σ) | **beats L5; 14.1 % of the gap** |
| | | +0.1100 vs `106` (the gap was 0.1280) | |
| | | −0.0210 vs $L_{ref}$ (−0.63 %) | |

**Curve** (976-window evals; last row full held-out):

| step | `106` (L1) | `110` (L5) | `198` | `198` − `110` |
|---|---|---|---|---|
| 100 | 6.2088 | 5.6159 | 5.9610 | +0.3451 |
| 300 | 4.6641 | 4.3664 | 4.6118 | +0.2454 |
| 500 | 3.9453 | 3.8804 | 3.9550 | +0.0746 |
| 700 | 3.6339 | 3.6601 | 3.6909 | +0.0308 |
| 1000 | 3.4184 | 3.4995 | 3.4982 | −0.0012 |
| 1500 | 3.2543 | 3.3697 | 3.3553 | −0.0144 |
| 1900 | 3.2016 | 3.3288 | 3.3114 | −0.0174 |
| 1907 (full) | 3.2048 | 3.3328 | 3.3148 | −0.0180 |

L5f starts far behind (0.35 nats at step 100: the mix begins uniform, so early on each iteration
reads an average, not a running sum), crosses `110` at step ≈ 1000 and is still pulling away at
the end (−0.014 → −0.018 over the last 400 steps). The 1 B-token number is therefore a point on
a moving gap; L5's own `screen` results have moved at `small` before.

## Mix weights per iteration (`probe_mix`, 256 held-out windows, checkpoint at step 1907)

Mean $\alpha_{j,s}$ over tokens. Sources: $v_0$ = the middle block's input, $u_i$ = iteration
$i$'s update. Rows sum to 1.

| reads | $v_0$ | $u_1$ | $u_2$ | $u_3$ | $u_4$ | $u_5$ | $u_6$ | $u_7$ | $u_8$ |
|---|---|---|---|---|---|---|---|---|---|
| iteration 1 | 1.000 | | | | | | | | |
| iteration 2 | 0.174 | **0.826** | | | | | | | |
| iteration 3 | 0.114 | 0.156 | **0.730** | | | | | | |
| iteration 4 | 0.090 | 0.071 | 0.135 | **0.704** | | | | | |
| iteration 5 | 0.077 | 0.043 | 0.054 | 0.137 | **0.690** | | | | |
| iteration 6 | 0.068 | 0.029 | 0.042 | 0.055 | 0.150 | **0.656** | | | |
| iteration 7 | 0.061 | 0.034 | 0.024 | 0.036 | 0.046 | 0.134 | **0.664** | | |
| iteration 8 | 0.076 | 0.038 | 0.036 | 0.036 | 0.044 | 0.054 | 0.170 | **0.545** | |
| final blocks | 0.112 | 0.100 | 0.076 | 0.079 | 0.084 | 0.086 | 0.114 | 0.140 | **0.209** |

Every iteration reads mostly the update just before it (0.83 falling to 0.55 with depth), then
the one before that (0.13–0.17), and keeps 6–17 % on the block's input. Older updates get
2–5 % each. The final blocks are different: their mix is close to flat (0.08–0.21), an average
over all eight updates and the input with a tilt to the last two. So the learned structure is a
chain with a short skip inside the loop and a broad read-out at the end, not a deep iteration
reaching back to an early one.

## Routing and run health

Worst point 2.30 of 8 effective experts at step 19 with five dead (`110`: 4.17 at step 20); last
dead expert at step 43 (`110`: 37); 7.998 of 8 at the end, loads 0.119–0.130. The early collapse
is deeper than L5's and recovers as quickly.

| | |
|---|---|
| wall-clock | 6.46 h; summed `step_s` 6.373 h + evals 0.085 h + checkpoints 0.003 h → **0.1 % unaccounted** |
| throughput | 43.5 k tok/s median, flat; the mix costs 10.5 % of L5's step and 1.80 GiB in this reference implementation, against ≈ 0.1 MFLOP per token on paper |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 78 °C, min clock 1897 MHz |

## Interpretation

Replacing L5's residual sum over iterations with a learned per-token mix of the block's input
and the earlier updates lowers the loss by 0.0180 nats, six σ, for 51 k parameters. That is a
real gain by the ladder's bands and the largest of the Embla follow-ups on L5 at this size, but
it is 14 % of the distance to L1, not the 25 % H25 states, so H25 is silent at `screen` and not
supported. The mix it learns is mostly local in depth, with a flat read-out into the final
blocks; whether the gain comes from that read-out, from the short skip, or from the input being
an average and not a growing sum is not separated by this run. Two things argue for the `small`
pair that ADR-072 calls for: the lead is still growing at the end of the run, after a slow
start, and L5's results at 1 B tokens have changed at 2.5 B in both directions. The costs to
carry into that decision are 10 % of throughput and 1.8 GiB per GPU. One seed at `screen`.
