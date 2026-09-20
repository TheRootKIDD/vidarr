# 132-l5-small-s0 — L5 at `small`, seed 0: 3.0778 nats, **+2.74 % vs $L_{ref}$(small)** — the `screen` tie with dense does not survive 2.5 B tokens (one seed; `133` completes the pair)

`python -m scripts.train.train --rung L5 --id 132-l5-small-s0 --tokens 2.5e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Second run of `queue_small_3.sh`; the rerun of the burnt `124`. Shared middle block, $r$ = 8, $N_e$ = 8, $k$ = 2 (ADR-018 local default), as `110-l5-screen`. Driver 615.71.09.

**Bears on:** **H6**, and — as `110` — **gives no H6 verdict**: H6 is stated *at matched params and
FLOPs*, and this arm matches L0's FLOPs with **52.9 M non-embedding params, 70 % of L0's 75.5 M and
21 % of L1's 246.6 M**. The parameter-matched arm is L5-ne128 (I29), which is not in this queue. What
the run is for: (1) the recurrent design's position at `small` against the dense baseline, and (2)
the comparator for the L5d pair (`134`/`135`, H15a) and for every later recurrent rung at `small`.
One seed; the reading below is provisional until `133-l5-small-s1` lands.

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0778** nats |
| final train loss (last step): main / total | 3.0810 / 3.0889 (router aux 0.0079) |
| throughput (median step) | 49,232 tokens/s aggregate (`step_s` median 10.649 s, p90 10.697, p90/median 1.004) |
| peak memory per GPU | 3.86 GiB |
| wall-clock (this session) | 14.25 h |
| seed | 0 |
| depth | fixed $r$ = 8 (`r_mean` 8.0 train and eval) |
| routing at the last step | `load_max` 0.129 / `load_min` 0.119, `route_ent_mean` 0.9999, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 80 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 40 rows |

Wall-clock accounting: summed `step_s` 14.084 h + evals 0.149 h + checkpoints 0.007 h = 14.239 h of
14.249 h, **0.07 % unaccounted**. (`n_throttled` = 4 on every row is the 615.71 `0x400` bit, `030`.)

## Against the dense baseline

| comparator | loss | Δ (nats) | Δ % | vs 4σ = 0.0123 |
|---|---|---|---|---|
| $L_{ref}$(small), L0 pair (`116`/`117`), 75.5 M non-emb | 2.9958 | **+0.0820** | **+2.74 %** | 6.7× |
| L1 pair (`130`/`119`), 246.6 M non-emb | 2.8839 | +0.1939 | +6.72 % | 16× |
| at `screen`: `110` vs `100` | 3.3328 vs 3.3358 | −0.0030 | −0.10 % | silent |

**The `screen` tie was a budget effect.** Along this run's own trajectory against the L0 pair mean
(same schedule, paired evals on the 976-window subset):

| step (tokens) | 500 (0.26 B) | 1000 (0.52 B) | 1900 (1.0 B) | 2900 (1.5 B) | 3800 (2.0 B) | 4700 (2.46 B) |
|---|---|---|---|---|---|---|
| L5 − L0 pair (nats) | −0.129 | +0.011 | +0.056 | +0.073 | +0.079 | +0.082 |

The shared block learns faster early (−0.13 nats at 0.26 B), the curves cross at ≈ 0.5 B tokens,
and the gap then widens and flattens towards ≈ +0.08. Under the `small` schedule L5 is already
+0.056 behind at 1 B tokens, where `110` under the `screen` schedule tied — the same schedule
dependence `120` found for L2 ("improves" at `screen`, non-inferior at `small`). With 70 % of the
dense model's non-embedding parameters, the likeliest reading is that L5 at $N_e$ = 8 is
capacity-limited once the budget passes ≈ 10 tokens per parameter (the crossing point) — a reading,
not a measurement; an $N_e$ sweep at `small` would test it. It says nothing about H6's
matched-params claim.

**Reading under ADR-013.** No threshold names this arm (as L1 against L0), so no verdict. The
difference is resolvable at one seed — 6.7× the 4σ band — and pending `133` only in its third
decimal: the largest `small` seed gap so far is 0.0061.

## Thermals and throughput

Zero thermal-slowdown rows over 14.25 h; GPU 0 (`GPU-4673cc7d`) reached 80 °C in the afternoon
(77 °C overnight) at ≥ 1890 MHz, the warmest long-run reading since the intake fix, with no thermal
event (40 power-cap rows; `131` had 35). Throughput 49.2 k tok/s, flat over the run (p90/median
1.004).

## Interpretation

**No H6 verdict (not the matched arm); resolvable position result.** At 2.5 B tokens the $N_e$ = 8
shared block sits **+2.74 % above dense** at matched FLOPs and 70 % of its params, where at 1 B
tokens under the `screen` schedule it tied. Two consequences: the `screen` statement "L5 is
indistinguishable from dense at a fifth of L1's params" is a statement about 1 B tokens only, and
I29's matched arm (L5-ne128 at `small`) matters more than before, because the $N_e$ = 8 arm now
clearly cannot carry H6 by itself. The run is the seed-0 comparator for H15a (`134`/`135`: Δ ≤ 2 % =
0.060 against the L5 pair). `133-l5-small-s1` started 18:22.
