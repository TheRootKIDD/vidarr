# 149-l5c-small-s0 — L5c at `small`, seed 0: 3.0230 nats, **−1.69 % vs the L5 pair** (4.9× the 4σ line); the `screen` gain holds at 2.5 B tokens; the pair verdict is in `150`

`python -m scripts.train.train --rung L5c --id 149-l5c-small-s0 --tokens 2.5e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. First run of `queue_small_8.sh` (2026-10-13 19:29 → 2026-10-14 09:45), with the unit suite passing on `4aef0c1` before launch on idle GPUs. Seed 0 of the L5c `small` pair (ADR-031, ADR-033): `146`'s config at 2.5 B tokens, i.e. L5 with `MiddleCfg(shared_attention=False)`. Each of the 8 iterations has its own attention, while the norms, router, $N_e$ = 8 expert pool ($k$ = 2, $d_{ff}$ pinned at 896) and halting head stay shared. Fixed $r$ = 8.

**Bears on:** **H6**, through `00 §P6` caveat 2 (sharing attention across iterations). Comparator:
the L5 pair (`132`/`133`) 3.0748, same FLOPs, +11.0 M parameters. This is read under ADR-013 with σ =
0.0026 re-pooled over nine pairs including this one (see `150`; 2σ 0.0053, 4σ 0.0105).

## Numbers

| | |
|---|---|
| params | 88.49 M total, 63.91 M non-embedding (L5: 77.48 / 52.90) |
| attention | 8 modules, one per iteration (L5: 1, shared) |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0230** nats |
| final train loss (last step): main / total | 3.0270 / 3.0346 (router aux 0.0076) |
| throughput (median step) | 49,171 tokens/s aggregate (`step_s` median 10.663 s, p90/median 1.003), the same as `146` (49.2 k) and L5 |
| peak memory per GPU | 4.07 GiB |
| wall-clock (this session) | 14.26 h |
| seed | 0 |
| routing at the last step | `load_max` 0.129 / `load_min` 0.119, `route_ent_mean` 0.9998, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 78 °C, `sm_mhz_min` ≥ 1897; `n_power_cap` 35 rows |

Wall-clock accounting: summed `step_s` 14.096 h + evals 0.148 h + checkpoints 0.008 h = 14.252 h of
14.261 h, **0.07 % unaccounted**.

## Against the comparators

| comparator | loss | Δ (nats) | Δ % | ADR-013: matches ≤ 0.0053, resolvable > 0.0105 |
|---|---|---|---|---|
| **L5 pair (`132`/`133`)** | 3.0748 | **−0.0518** | **−1.69 %** | resolvable, 4.9× the 4σ line |
| `132` (L5 seed 0) | 3.0778 | −0.0548 | −1.78 % | |
| `133` (L5 seed 1) | 3.0717 | −0.0487 | −1.59 % | |
| L0 pair (`116`/`117`), dense | 2.9958 | +0.0272 | +0.91 % | |
| L5-ne128 pair (`138`/`142`) | 2.9587 | +0.0643 | +2.17 % | |
| L1 pair (`130`/`119`) | 2.8839 | +0.1391 | +4.82 % | |

Along the trajectory (976-window evals, against `132`) the gap is −0.035 to −0.057 up to step 600,
then −0.050 to −0.060 from step 600 to 4768. It has one sign at every eval. It is flat from ≈ 1 B
tokens on (−0.056 at step 1900, −0.055 at 4700), so it is not a 1 B-token effect the way L5's tie
with L0 at `screen` was.

## Interpretation

**Seed 0 holds `146`'s `screen` gain at 2.5 B tokens: −0.0518 nats (−1.69 %) vs the L5 pair, 4.9×
ADR-013's 4σ line, at the same FLOPs and +20.8 % non-embedding parameters.** At `screen` it was
−0.0560. The gap opens within the first 0.3 B tokens and then stays flat. Untying attention across
iterations is therefore a durable gain, not a small-budget effect. The pair verdict, re-pooled σ and
the reading for H6 are in `150`. Rig: 14.3 h, zero `n_thermal`, ≤ 78 °C.
