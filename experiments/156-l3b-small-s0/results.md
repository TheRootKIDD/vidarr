# 156-l3b-small-s0 — L3b at `small`, seed 0: 2.9473 nats on the main head, **+2.16 % vs the L2 pair**, −0.70 % vs the L3 pair; acceptance 41.8 / 23.9 / 15.6 %; the pair verdict is in `157`

`python -m scripts.train.train --rung L3b --id 156-l3b-small-s0 --tokens 2.5e+09 --micro-batch 2 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (32 × 2 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. First run of `queue_small_9.sh` (2026-10-17 18:57 → 2026-10-18 10:48), on the tree at `3839aa4` (docs-only on top of the gate commit `36257dc`) (the queue aborts if `model/`, `scripts/` or `costmodel/` change). Seed 0 of the L3b `small` pair (ADR-036 §3): L3 with the forward MTP curriculum, four equal phases of 1192 steps (≈ 625 M tokens each), heads 2..$i$ active in phase $i$.

**Bears on:** **H13**, both clauses. Comparators: the L2 pair (`120`/`121`) 2.8851 and the L3 pair
(`122`/`131`) 2.9682. Read under ADR-013 with σ = 0.0029 re-pooled over eleven pairs including this
one (see `157`; 2σ 0.0057, 4σ 0.0114).

## Numbers

| | |
|---|---|
| params | 272.92 M total, 248.34 M non-embedding (as L3) |
| tokens | 2.500 B in 4768 steps |
| **final val loss, main head** (full held-out, 5606 windows) | **2.9473** nats |
| acceptance vs main head, heads 2 / 3 / 4 (held-out) | **41.8 / 23.9 / 15.6 %** |
| head top-1 vs ground truth, heads 2 / 3 / 4 (held-out) | 24.3 / 14.0 / 9.0 % |
| final train loss (last step): main / total | 2.9510 / 7.4572 (router aux 0.0122) |
| throughput (median step) | 42,825 tokens/s aggregate (`step_s` median 12.243 s, p90/median 1.010); ≈ 50.8 k in phase 1 |
| peak memory per GPU | 9.35 GiB (as `152`) |
| wall-clock (this session) | 15.84 h |
| seed | 0 |
| routing at the last step | `load_max` 0.135 / `load_min` 0.112, `route_ent_mean` 0.9998, `route_eff_min` 7.99 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 78 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 26 rows |

Wall-clock accounting: summed `step_s` 15.608 h + evals 0.198 h + checkpoints 0.023 h = 15.829 h of
15.839 h, **0.06 % unaccounted**.

## Interpretation

Seed 0 alone: +0.0622 nats vs the L2 pair (+2.16 %), −0.0209 vs the L3 pair. Acceptance is about one
point under the L3 pair's at head 2 and two points under at heads 3 and 4, which is ADR-036's stated
trade. Both of H13's clauses miss by wide margins. The pair verdict and the trajectory by phase are
in `157`. Rig: 15.8 h, zero `n_thermal`, ≤ 78 °C, 9.35 GiB per GPU.
