# 154-l5e-small-s0 — L5e at `small`, seed 0: 3.0479 nats, **−0.87 % vs the L5 pair** (2.4× the 4σ line); the `screen` gain holds at 2.5 B tokens, a little smaller; the pair verdict is in `155`

`python -m scripts.train.train --rung L5e --id 154-l5e-small-s0 --tokens 2.5e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. First training run of `queue_screen_9.sh` (2026-10-16 13:19 → 2026-10-17 04:07), after the `031` rerun on the same queue. It imported the tree at `3839aa4` (docs-only changes on top of the gate commit `36257dc`; the queue aborts if `model/`, `scripts/` or `costmodel/` change). `git_dirty` is the untracked experiment directories. Seed 0 of the L5e `small` pair (ADR-031, after `151`): L5 with one dense H block ($d_{ff}^H$ = 2048, own weights) after iteration 4 of 8, $N_e$ = 8, $k$ = 2, $d_{ff}$ pinned at 896, fixed $r$ = 8 (ADR-034).

**Bears on:** **H6** and `08 §4`'s HRM-Text claim (I32). Comparator: the L5 pair (`132`/`133`) 3.0748,
+6.29 M parameters, +7.0 % FLOPs (reported, not traded; ADR-031). Read under ADR-013 with σ = 0.0027
re-pooled over ten pairs including this one (see `155`; 2σ 0.0055, 4σ 0.0110).

## Numbers

| | |
|---|---|
| params | 83.77 M total, 59.20 M non-embedding (L5: 77.48 / 52.90) |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0479** nats |
| final train loss (last step): main / total | 3.0515 / 3.0596 (router aux 0.0081) |
| throughput (median step) | 47,376 tokens/s aggregate (`step_s` median 11.067 s, p90/median 1.003), as at `screen` (`151`: 47,368) |
| peak memory per GPU | 4.25 GiB |
| wall-clock (this session) | 14.79 h |
| seed | 0 |
| routing at the last step | `load_max` 0.128 / `load_min` 0.121, `route_ent_mean` 0.9999, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 80 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 45 rows |

Wall-clock accounting: summed `step_s` 14.613 h + evals 0.156 h + checkpoints 0.008 h = 14.777 h of
14.79 h, **< 0.1 % unaccounted**.

## Loss against the comparators

| comparator | loss | Δ (nats) | Δ % |
|---|---|---|---|
| **L5 pair (`132`/`133`)** | 3.0748 | **−0.0269** | **−0.87 %** |
| `132` (L5 seed 0) alone | 3.0778 | −0.0299 | −0.97 % |
| L5c pair (`149`/`150`) | 3.0225 | +0.0254 | +0.84 % |
| L0 pair (`116`/`117`), dense | 2.9958 | +0.0521 | +1.74 % |
| L1 pair (`130`/`119`) | 2.8839 | +0.1640 | +5.69 % |

## Interpretation

Seed 0 alone: −0.0269 nats against the L5 pair, 2.4× ADR-013's 4σ line, at +7 % FLOPs. The `screen`
gain (−0.0383, `151`) holds at 2.5 B tokens, a little smaller. The pair verdict, the trajectory and
the I32 reading are in `155`. Rig: 14.8 h, zero `n_thermal`, ≤ 80 °C.
