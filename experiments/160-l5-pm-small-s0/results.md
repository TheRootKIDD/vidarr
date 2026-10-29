# 160-l5-pm-small-s0 — L5-pm at `small`, seed 0: 2.9349 nats, **+0.0510 vs the L1 pair (+1.77 %)**, −0.0238 vs the L5-ne128 pair; the pair verdict is in `161`

`python -m scripts.train.train --rung L5-pm --id 160-l5-pm-small-s0 --tokens 2.5e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. First run of `queue_small_10.sh` (2026-10-20 11:47 → 2026-10-23 01:52), on the tree at `ae08987` (docs only on top of the gate commit `36257dc`; `model/`, `scripts/` and `costmodel/` unchanged through `4dc091e`). Seed 0 of the L5-pm `small` pair (ADR-035 §2): L5-ne128 extended to $N_e$ = 204, $k$ = 4, $d_{ff}$ 448, at L1's non-embedding parameter count and L5-ne128's FLOPs.

**Bears on:** **H6** at matched params and FLOPs (I29's arm). Comparators: the L1 pair (`130`/`119`)
2.8839 (246.6 M non-embedding), the L5-ne128 pair (`138`/`142`) 2.9587 and $L_{ref}$(small) 2.9958.
Read under ADR-013 with σ = 0.0030 re-pooled over twelve pairs including this one (see `161`; 2σ
0.0059, 4σ 0.0119).

## Numbers

| | |
|---|---|
| params | 271.6 M total, 247.0 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| final val loss (full held-out, 5606 windows) | **2.9349** nats |
| final train loss (last step, main head) | 2.9395 |
| throughput (median step) | 11,264 tokens/s aggregate (`step_s` median 46.544 s, p90/median 1.004); last step 11,275 |
| peak memory per GPU | 7.56 GiB |
| wall-clock (this session) | 62.07 h |
| seed | 0 |
| val_r_mean | 8.0000 |
| routing at the last step | `load_max` 0.0066 / `load_min` 0.0040 (uniform 0.0049), `route_ent_mean` 0.9995, `route_eff_min` 203.5 of 204, no dead experts (pooled over iterations) |
| routing probe, per iteration (`routing.json`, GPU, 8 × 4 held-out batches) | effective experts 203 → 164 → 172 → 149 → 159 → 143 → 137 → 133; near-dead 0, 4, 3, 6, 3, 8, 8, 19; mean cross-iteration load correlation **0.20** |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 65 °C, `sm_mhz_min` ≥ 1920; `n_power_cap` 0 rows |

Wall-clock accounting: summed `step_s` 61.656 h + evals 0.309 h + checkpoints 0.093 h = 62.058 h of
62.067 h, **0.01 % unaccounted**.

## Interpretation

Seed 0 alone: +0.0510 nats vs the L1 pair (+1.77 %), 8.6× the 2σ "matches" band, and −0.0238 vs
the L5-ne128 pair. It beats dense by 0.0609 (−2.03 %). The `screen` lead over L1 (`153`, −0.33 %)
has reversed by 0.58 B tokens. The pair verdict, the trajectory and the routing comparison are in
`161`. Rig: 62.1 h, zero `n_thermal`, ≤ 65 °C, 7.56 GiB per GPU.
