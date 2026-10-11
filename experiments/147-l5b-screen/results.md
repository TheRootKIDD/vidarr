# 147-l5b-screen — L5b, rank-16 per-iteration adapters on attention: 3.3286 nats, **−0.13 % vs L5** — inside ADR-013's "matches" band; recovers none of L5c's gain; no `small` pair

`python -m scripts.train.train --rung L5b --id 147-l5b-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Second and last run of `queue_screen_6.sh` (2026-10-10 20:40 → 2026-10-11 02:40). This is `110-l5-screen` with `MiddleCfg(per_iter_lora_rank=16)` (ADR-032): the shared attention gets one low-rank adapter per iteration on each of $W_q, W_k, W_v, W_o$, $W + A_j B_j$ with $r_a$ = 16 and $B_j$ = 0 at init, so the run starts identical to L5. The expert pool is unchanged ($N_e$ = 8, $k$ = 2, $d_{ff}$ pinned at 896), with fixed $r$ = 8.

**Bears on:** **H6**, through `00 §P6` caveat 2. This is the cheap version of L5c (`146`), at
+0.66 M parameters (+1.2 % non-embedding) and ≈ +0.65 % training FLOPs. Comparator: `110`, under
ADR-013 at one seed, with σ = 0.0028 (2σ 0.0056, 4σ 0.0111).

## Numbers

| | |
|---|---|
| params | 78.14 M total, 53.56 M non-embedding (`110`: 77.48 / 52.90) |
| adapters | $r_a$ = 16 on q/k/v/o × 8 iterations |
| tokens | 1.000 B in 1907 steps |
| **final val loss** (full held-out, 5606 windows) | **3.3286** nats |
| final train loss (last step): main / total | 3.2965 / 3.3045 (router aux 0.0080) |
| throughput (median step) | 46,861 tokens/s aggregate (`step_s` median 11.188 s, p90/median 1.002). That is 5 % below L5's ≈ 49.3 k for +0.65 % FLOPs: 8 extra rank-16 matmuls per iteration (two per projection), a kernel-shape cost on this GPU |
| peak memory per GPU | 3.87 GiB |
| wall-clock (this session) | 5.99 h |
| seed | 0 |
| routing at the last step | `load_max` 0.131 / `load_min` 0.120, `route_ent_mean` 0.9998, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 190 telemetry rows**; `temp_c_max` ≤ 79 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 15 rows |

Wall-clock accounting: summed `step_s` 5.912 h + evals 0.073 h + checkpoints 0.003 h = 5.988 h of
5.992 h, **0.06 % unaccounted**.

## Loss against the comparators

| comparator | loss | Δ (nats) | Δ % | ADR-013: matches ≤ 0.0056, resolvable > 0.0111 |
|---|---|---|---|---|
| **`110` L5, shared attention** | 3.3328 | **−0.0042** | **−0.13 %** | matches (inside 2σ) |
| `146` L5c, one attention per iteration | 3.2768 | +0.0518 | +1.58 % | resolvable |

Along the trajectory (976-window evals) the gap to `110` is −0.004 to −0.007 from step 600 to
1100, then −0.002 to −0.003 from step 1300 to 1900. It has stayed negative since step 400 but
inside the 2σ band, and it is narrowing rather than growing. By contrast, L5c's gap was −0.05 by
step 800.

## Interpretation

**Matches L5 at one seed (Δ = −0.0042 nats, −0.13 %, inside ADR-013's 2σ band): rank-16
per-iteration adapters on attention buy no detectable improvement.** Read with `146`, this is
ADR-032's "L5c wins, L5b does not" outcome. The 0.056 nats that full per-iteration attention
recovers does not fit in a rank-16 delta on q/k/v/o. Either the iterations need attention that
differs at high rank, or L5c's gain is partly capacity (+11.0 M parameters), which I31's
parameter-matched control is to separate. A rank sweep (ADR-032 point 1) is now justified only
between 16 and full rank. It stays unscheduled until L5c's `small` pair and I31 show which reading
holds. For H6 this run is **silent**. ADR-031's `small` rule does not trigger, so L5b closes at
`screen`. The adapters also cost 5 % throughput for +0.65 % FLOPs on this GPU. Rig: 6.0 h, zero
`n_thermal`, ≤ 79 °C.
