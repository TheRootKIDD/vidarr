# 144-l7b-small-s1 — L7b ($L_e$ = 2) at `small`, seed 1: 3.0909 nats; **L7b pair 3.0926, +0.58 % vs the L5 pair** — 1.6× ADR-013's H8 line (4σ): **H8 weakened at two seeds**

`python -m scripts.train.train --rung L7b --id 144-l7b-small-s1 --tokens 2.5e+09 --micro-batch 4 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Only run of `queue_small_7.sh` (2026-10-09 16:18, pick-up after the 2026-10-03 pause; the user asked to pause after it, so nothing is chained behind it). Seed 1 of `143`: L5 with `ExpertsCfg(layers=2)` — each of the $N_e$ = 8 experts ($k$ = 2) is a two-layer SwiGLU stack at $d_{ff}$ = 448 instead of one layer at 896, same params and FLOPs.

**Bears on:** **H8** — *$L_e$ = 2 halves fabric bytes per FLOP at ≤ 0.5 % loss* (quality clause, arm
A; the bytes half is S3's arithmetic). Comparator: the L5 pair (`132`/`133`) 3.0748, same params and
FLOPs, $L_e$ = 1. **Threshold, per ADR-013** (binding): "0.5 %" is below the floor and becomes "no
detectable regression", **$T$ = 2σ**; with σ = 0.0028 over eight pairs (this pair included; 0.00278,
was 0.00283 over seven): *supports* if Δ ≤ 0.0056, *weakens* if Δ > $T$ + 2σ = **0.0111**, silent
between. Under the misapplied 0.5 %-of-$L$ reading of `137`–`143` (`docs/04` 2026-10-03) it is
given for completeness only.

## Numbers

| | |
|---|---|
| params | 77.49 M total, 52.91 M non-embedding (L5: 77.48 / 52.90) |
| expert shape | $N_e$ = 8, $k$ = 2, **$L_e$ = 2, $d_{ff}$ = 448** (L5: $L_e$ = 1, $d_{ff}$ = 896) |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0909** nats |
| final train loss (last step): main / total | 3.0750 / 3.0831 (router aux 0.0081) |
| throughput (median step) | 45,078 tokens/s aggregate (`step_s` median 11.631 s, p90/median 1.007) — same as `143` (45.1 k), 8 % below L5 at equal FLOPs |
| peak memory per GPU | 3.86 GiB |
| wall-clock (this session) | 15.60 h |
| seed | 1 |
| routing at the last step (trainer) | `load_max` 0.131 / `load_min` 0.118, `route_ent_mean` 1.000, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 77 °C, `sm_mhz_min` ≥ 1905; `n_power_cap` 28 rows |

Wall-clock accounting: summed `step_s` 15.423 h + evals 0.161 h + checkpoints 0.008 h = 15.592 h of
15.601 h, **0.06 % unaccounted**.

## Against the comparator

| comparator | loss | Δ (nats) | Δ % | ADR-013 ($T$ = 2σ): supports ≤ 0.0056, weakens > 0.0111 | 0.5 %-of-$L$ reading: supports ≤ 0.0154, weakens > 0.0210 |
|---|---|---|---|---|---|
| **L7b pair (`143`/`144`) vs L5 pair** | 3.0926 vs 3.0748 | **+0.0178** | **+0.58 %** | **weakens** (1.6× the line) | silent (0.0032 under the line) |
| `144` alone vs L5 pair | 3.0909 | +0.0161 | +0.52 % | weakens | silent |
| `143` alone vs L5 pair | 3.0943 | +0.0196 | +0.64 % | weakens | silent |
| L7b pair vs L0 pair (`116`/`117`), dense | 2.9958 | +0.0968 | +3.23 % | | |

Seed gap `143` − `144` = +0.0034 nats, inside the range of the seven earlier `small` pairs
(|Δ| ≤ 0.0061). Every one of the four seed-wise L7b − L5 differences is positive (+0.0131 to
+0.0226).

Trajectory against the L5 pair (nats):

| step (tokens) | 500 (0.26 B) | 1000 (0.52 B) | 1900 (1.0 B) | 2900 (1.5 B) | 3800 (2.0 B) | 4700 (2.46 B) | 4768 (final) |
|---|---|---|---|---|---|---|---|
| `144` − L5 pair | +0.043 | +0.023 | +0.020 | +0.018 | +0.016 | +0.016 | +0.016 |
| `143` − L5 pair | +0.007 | +0.022 | +0.023 | +0.022 | +0.021 | +0.020 | +0.020 |
| **L7b pair − L5 pair** | +0.025 | +0.023 | +0.022 | +0.020 | +0.019 | +0.018 | **+0.018** |

The pair's gap narrows by ≈ 0.004 nats over the last 1.5 B tokens and has flattened by 2.46 B (both
seeds unchanged from step 3800 to 4700 within 0.001). `137` measured +0.019 at `screen` (1 B
tokens), so the cost is ≈ 0.02 nats at both budgets. A slow closing at larger budgets cannot be
excluded, but nothing at this scale points to it.

## Interpretation

**H8's quality clause is *weakened* at two seeds (ADR-013).** Splitting each expert into two
half-width layers ($L_e$ = 2) at equal params and FLOPs costs **+0.0178 nats, +0.58 %** pair against
pair, 1.6× ADR-013's weakening line (4σ = 0.0111) and 3.2× its "no detectable regression" band.
Seed 1 lands 0.0034 nats under seed 0, an ordinary seed gap, so it confirms `143`'s provisional
verdict and does not reverse it. Under the looser 0.5 %-of-$L$ reading that `137`–`143` had misapplied,
the pair would be silent, so no reading supports H8. The cost is ≈ 0.02 nats at 1 B tokens (`screen`) and 2.5 B
(`small`), flattening by the end of the run. **What S3 should carry:** halving fabric bytes per FLOP
via $L_e$ = 2 costs ≈ 0.6 % loss, plus ≈ 8 % GEMM throughput on Ampere at these widths (45.1 k vs
49.2 k tok/s, a kernel-shape cost, not FLOPs). Whether the bytes saving is worth 0.6 % is a systems
question for S3 to answer, not this run. The note's "≤ 0.5 %" is not met even at face value
(+0.58 %). Rig: 15.6 h, zero `n_thermal`, ≤ 77 °C. Queue paused after this run on the user's request.
