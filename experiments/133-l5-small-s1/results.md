# 133-l5-small-s1 — L5 at `small`, seed 1: 3.0717 nats; **the L5 pair = 3.0748, +2.64 % vs $L_{ref}$(small)** — the $N_e$ = 8 shared block trails dense at 2.5 B tokens, two seeds; σ re-pooled over five pairs to 0.0034

`python -m scripts.train.train --rung L5 --id 133-l5-small-s1 --tokens 2.5e+09 --micro-batch 4 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Third run of `queue_small_3.sh`; the rerun of the burnt `125`. Shared middle block, $r$ = 8, $N_e$ = 8, $k$ = 2 (ADR-018 local default), as `110`/`132`. Driver 615.71.09.

**Bears on:** **H6**, and — as `110` and `132` — **gives no H6 verdict**: the arm matches L0's FLOPs
with 52.9 M non-embedding params (70 % of L0's, 21 % of L1's), and H6 is stated at matched params
*and* FLOPs. The matched arm is L5-ne128 at `small`, queued as `138`/`139` (`queue_small_4.sh`, I29).
This run completes the **L5 pair**, which is the recurrent design's position at `small` and the
comparator for H15a (`134`/`135`) and for every later recurrent rung.

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0717** nats |
| final train loss (last step): main / total | 3.0556 / 3.0635 (router aux 0.0079) |
| throughput (median step) | 49,172 tokens/s aggregate (`step_s` median 10.662 s, p90 10.704, p90/median 1.004) — `132`: 49,232 |
| peak memory per GPU | 3.86 GiB |
| wall-clock (this session) | 14.26 h |
| seed | 1 |
| depth | fixed $r$ = 8 (`r_mean` 8.0 train and eval) |
| routing at the last step | `load_max` 0.132 / `load_min` 0.121, `route_ent_mean` 0.9998, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 79 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 29 rows |

Wall-clock accounting: summed `step_s` 14.091 h + evals 0.148 h + checkpoints 0.007 h = 14.246 h of
14.256 h, **0.07 % unaccounted**. (`n_throttled` = 4 on every row is the 615.71 `0x400` bit, `030`.)

## The L5 pair

| | seed 0 (`132`) | seed 1 (`133`) | **pair** |
|---|---|---|---|
| full held-out loss | 3.0778 | 3.0717 | **3.0748** |

$|Δ_{seed}|$ = **0.0061 nats** ($s$ = 0.00429), the same size as the L1 pair's (0.0061) and L2's
(0.0054); over the last 20 paired evals seed 0 − seed 1 = +0.0061 ± 0.0006 (sd), a stable offset.
The 77 M recurrent model has the same seed spread as the 271 M MoE rungs; the dense L0 (0.0026) and
L3 (0.0018) pairs are the tight ones.

| comparator | loss | Δ (nats) | Δ % | vs 4σ |
|---|---|---|---|---|
| $L_{ref}$(small), L0 pair, 75.5 M non-emb | 2.9958 | **+0.0790** | **+2.64 %** | 5.9× (4σ = 0.0134) |
| L1 pair, 246.6 M non-emb | 2.8839 | +0.1909 | +6.62 % | 14× |
| at `screen`: `110` vs `100` | 3.3328 vs 3.3358 | −0.0030 | −0.10 % | silent |

Pair against pair along the trajectory (976-window evals), L5 − L0:

| step (tokens) | 500 (0.26 B) | 1000 (0.52 B) | 1900 (1.0 B) | 2900 (1.5 B) | 3800 (2.0 B) | 4700 (2.46 B) |
|---|---|---|---|---|---|---|
| L5 pair − L0 pair (nats) | −0.108 | +0.011 | +0.053 | +0.070 | +0.076 | +0.079 |

Both seeds tell `132`'s story: an early lead for the shared block, a crossing at ≈ 0.5 B tokens, and
a gap that widens and flattens near +0.08. **The `screen` tie (`110`) was a property of the 1 B
budget and its schedule, confirmed at two seeds.** No ADR-013 threshold names this arm against L0, so
no verdict is recorded; the difference is 5.9× the silent band with two seeds each side.

## σ re-pooled over five pairs

Pair sample sds: L0 0.00184, L1 0.00431, L2 0.00382, L3 0.00125, **L5 0.00429** → rms = **0.0034
nats** (was 0.0031 over four). Under ADR-013's 0.01 third-seed trigger; two seeds stand. Bands at
`small`: 2σ = **0.0067**, 4σ = **0.0134**, 1 % = 0.030 (weakens above 0.0367), 2 % = 0.060 (weakens
above 0.0667). **No verdict flips**: H2 (L2 − L1 = +0.0012) inside 2σ; H13 (L3 − L2 = +0.0831) 6.2×
the weakening line; at `screen`, `112` (H6, +0.0184) weakens by 0.0050 over the 0.0134 line, `111`
(H15a, +0.0149) sits at 1.1× the silent band and still supports, `110`/`115` silent, `114`
resolvable.

**For H15a (`134`/`135`):** the comparator is this pair, 3.0748; $T$ = 2 % = 0.060 → *supports* if
the L5d pair ≤ 3.1348, *weakens* above 3.1415 ($T$ + 2σ).

## Interpretation

**No H6 verdict (not the matched arm); the position result is now two seeds.** At matched FLOPs and
70 % of the dense model's non-embedding parameters, the $N_e$ = 8 shared block is **+2.64 % above
dense at 2.5 B tokens**, where it tied at 1 B. Whether that is capacity (the reading in `132`) or
something the recurrence itself costs is exactly what the matched arm separates: L5-ne128 at `small`
(`138`/`139`) against the L1 pair. Rig: 14.26 h, zero `n_thermal`, GPU 0 ≤ 79 °C, throughput within
0.1 % of `132`. `134-l5d-small-s0` started 08:37.
