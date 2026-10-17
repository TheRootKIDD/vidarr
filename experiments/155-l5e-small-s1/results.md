# 155-l5e-small-s1 — L5e at `small`, seed 1: 3.0428 nats; **L5e pair 3.0454, −0.96 % vs the L5 pair** (2.7× the 4σ line) at +6.3 M params and +7 % FLOPs; a flat −0.029 from 0.4 B tokens, but 0.023 behind the L5c pair; H6 silent; I32 unresolved; σ re-pooled over ten pairs = 0.0027

`python -m scripts.train.train --rung L5e --id 155-l5e-small-s1 --tokens 2.5e+09 --micro-batch 4 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Last run of `queue_screen_9.sh` (2026-10-17 04:07 → 18:57), on the same tree as `154` (`3839aa4`, docs-only on top of the gate `36257dc`). Seed 1 of `154`: L5 with one dense H block ($d_{ff}^H$ = 2048, own weights) after iteration 4 of 8, $N_e$ = 8, $k$ = 2, $d_{ff}$ pinned at 896, fixed $r$ = 8 (ADR-034).

**Bears on:** **H6** and `08 §4`'s HRM-Text claim that two weight sets beat one shared loop (I32).
Comparator: the L5 pair (`132`/`133`) 3.0748, +6.29 M parameters (+11.9 % non-embedding), +7.0 %
training FLOPs, reported, not traded (ADR-031). Under ADR-013 at two seeds, with **σ re-pooled over
ten pairs = 0.0027** (0.00274, this pair included; was 0.00264 over nine). 2σ = 0.0055, 4σ = 0.0110.

## Numbers

| | |
|---|---|
| params | 83.77 M total, 59.20 M non-embedding (L5: 77.48 / 52.90) |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0428** nats |
| final train loss (last step): main / total | 3.0267 / 3.0342 (router aux 0.0075) |
| throughput (median step) | 47,245 tokens/s aggregate (`step_s` median 11.097 s, p90/median 1.003) |
| peak memory per GPU | 4.25 GiB |
| wall-clock (this session) | 14.84 h |
| seed | 1 |
| routing at the last step | `load_max` 0.131 / `load_min` 0.121, `route_ent_mean` 0.9998, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 81 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 40 rows |

Wall-clock accounting: summed `step_s` 14.665 h + evals 0.156 h + checkpoints 0.008 h = 14.829 h of
14.84 h, **< 0.1 % unaccounted**.

## The pair against the comparators

| comparator | loss | Δ (nats) | Δ % | ADR-013: matches ≤ 0.0055, resolvable > 0.0110 |
|---|---|---|---|---|
| **L5e pair (`154`/`155`) vs L5 pair (`132`/`133`)** | 3.0454 vs 3.0748 | **−0.0294** | **−0.96 %** | **resolvable, 2.7× the 4σ line** |
| `155` alone vs L5 pair | 3.0428 | −0.0320 | −1.04 % | resolvable |
| `154` alone vs L5 pair | 3.0479 | −0.0269 | −0.87 % | resolvable |
| L5e pair vs L5c pair (`149`/`150`), +11.0 M params, L5's FLOPs | 3.0225 | **+0.0229** | +0.76 % | resolvable (L5c better) |
| L5e pair vs L0 pair (`116`/`117`), dense | 2.9958 | +0.0496 | +1.65 % | |
| L5e pair vs L5-ne128 pair (`138`/`142`) | 2.9587 | +0.0867 | +2.93 % | |
| L5e pair vs L1 pair (`130`/`119`), stacked MoE | 2.8839 | +0.1615 | +5.60 % | |

All four seed-wise differences against the L5 seeds are negative (−0.024 to −0.035). The seed gap
is 0.0051, the fourth-largest of the ten pairs.

**σ re-pooled.** Over ten pairs (L0, L1, L2, L3, L5, L5d, L5-ne128, L7b, L5c, L5e),
σ = √(mean of (a − b)²/2) = 0.00274. No earlier pair verdict moves under the slightly wider bands. The
closest pair reading is L7b's +0.0178, which stays above 4σ = 0.0110. One `screen` reading moves onto
the line: `153` L5-pm vs L1, −0.0106, which is 4.0σ under the nine-pair σ and 3.9σ under this one.
Silent either way (see `153`).

## Trajectory (976-window evals, pair means)

| step | tokens | L5e pair | − L5 pair | − L5c pair | − L1 pair |
|---|---|---|---|---|---|
| 400 | 0.21 B | 4.0213 | −0.063 | +0.003 | −0.098 |
| 800 | 0.42 B | 3.5549 | −0.034 | +0.023 | +0.066 |
| 1600 | 0.84 B | 3.2955 | −0.030 | +0.025 | +0.121 |
| 2400 | 1.26 B | 3.1795 | −0.029 | +0.025 | +0.141 |
| 3200 | 1.68 B | 3.1068 | −0.029 | +0.024 | +0.152 |
| 4000 | 2.10 B | 3.0617 | −0.030 | +0.024 | +0.158 |
| 4700 | 2.46 B | 3.0422 | −0.029 | +0.024 | +0.162 |

The gain over L5 is flat at −0.029 to −0.030 from 0.4 B tokens to the end. The deficit to L5c is
flat at +0.024 over the same range. Both are fixed offsets, the same shape as L5c's gain.

## Across budgets and against the other additions

| added to L5 | Δ non-emb params | Δ FLOPs | Δ `screen` (1 B) | Δ `small` (2.5 B, pair) | nats per M params at `small` |
|---|---|---|---|---|---|
| **L5e: one dense H block after iteration 4** | +6.29 M | +7.0 % | −0.0383 (`151`) | **−0.0294** | 0.0047 |
| L5c: 7 more attention blocks | +11.0 M | 0 | −0.0560 (`146`) | −0.0523 | 0.0047 |
| L5-ne13: 5 more experts | +10.3 M | 0 | −0.0285 (`148`) | — | — |

L5e keeps 77 % of its `screen` gain at `small`; L5c keeps 93 %. Per added parameter the two are now
equal (0.0047 nats per M). But L5e also pays +7 % FLOPs and 3.8 % throughput (47.3 k vs L5's 49.3 k
tok/s), and L5c pays neither. Against L5's deficit to the L1 pair (+0.1908), H recovers 15 %; L5c
recovered 27 %.

## Interpretation

**Resolvable at two seeds: one dense block with its own weights after iteration 4 improves L5 by
0.0294 nats (−0.96 %, 2.7× ADR-013's 4σ line), for +6.29 M parameters and +7 % FLOPs.** The gain is a
fixed offset from 0.4 B tokens on, and it is 23 % smaller than at `screen` (−0.0383). **H6: silent on
the clause** (L5e is matched to L1 on neither parameters nor FLOPs). **HRM-Text's claim (`08 §4`):
supported in direction, not in mechanism.** A second weight set inside the loop helps durably. But
I32 stays open: no run separates the second weight set from the +7 % compute. The `148` comparison
(inconclusive at `screen`) has no `small` counterpart, and the FLOP-matched control (L5 at ≈ $r$ = 9)
has not run. As a design addition L5e is dominated by L5c, which is 0.0229 nats better at two seeds
(2.1× 4σ) for no added FLOPs. If both are kept they are separate arms: ADR-034 refuses L5e with
L5b/L5c, so whether H on top of untied attention still pays is a new question, not a reading of
this pair. This pair is the one ADR-031's "> 4σ → `small` pair" rule asked for, and the rule schedules
nothing further. Rig: 14.8 h, zero `n_thermal`, ≤ 81 °C.
