# 116-l0-small-s0 — L0 at `small`, seed 0: $L_{ref}$(small, s0) = 2.9971 nats; GPU 0 held 80 °C for 9.8 h

`python -m scripts.train.train --rung L0 --id 116-l0-small-s0 --tokens 2.5e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** no hypothesis directly — this is the dense baseline, the first half of the L0 pair that
ADR-013 takes σ from. It supplies $L_{ref}$ at `small` for seed 0. No verdict is possible until `117`
(seed 1) lands: σ is the pooled spread of the pair, and every `small` threshold is a fraction of the
pair's mean. First run of the `small` programme (`docs/04` 2026-09-15); same model as `100-l0-screen`
(100.1 M params, GQA dense, ADR-006/025), 2.5× the tokens on the same cosine schedule.

## Numbers

| | |
|---|---|
| params | 100.1 M total, 75.5 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| final val loss (full held-out, 5606 windows) | **2.9971** nats |
| final train loss (last step, main head) | 2.9997 |
| throughput (median step) | 71,700 tokens/s aggregate (`step_s` median 7.314 s, p90 7.316, p90/median 1.0003) |
| peak memory per GPU | 6.25 GiB |
| wall-clock (this session) | 9.82 h |
| seed | 0 |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` 75–80 °C, `sm_mhz_min` 1860–1897; `n_power_cap` 124 rows (the 170 W cap, GPU 0) |

## Against `100-l0-screen`, the same model at 1 B tokens

| | tokens | val loss | wall-clock | tok/s |
|---|---|---|---|---|
| `100-l0-screen` | 1.0 B | 3.3358 | 18.0 h (throttled, `06 §7`) | — |
| `116` at its 1 B-token eval (step 1900) | 1.0 B | 3.2206 | — | — |
| **`116` final** | **2.5 B** | **2.9971** | 9.82 h | 71.7 k |

The 1 B-token row is not a like-for-like comparison with `100` (the cosine schedule is 2.5× longer,
so the learning rate at step 1900 is 4.5e-4, not the 6e-5 `100` ended at); it is recorded so the
budget effect can be read off later as a difference, as ADR-030 asks for H13. Going from 1 B to 2.5 B
tokens takes the dense baseline down **0.339 nats (10.2 %)**, which sets the scale against which
every architecture delta on the `screen` ladder (0.02–0.19 nats) should be read.

## Thermals over 9.8 hours: the intake fix holds

This is the first hours-long load on GPU 0 (`GPU-4673cc7d`) since the riser cable under its slot was
moved (`027`/`028`), and the reading the 2026-09-15 log asked for. **Zero thermal-slowdown samples on
any card across 476 telemetry rows**; the hottest card peaked at 80 °C and the slowest clock at
1860 MHz; the step time is flat to 0.03 % between median and p90. Against `110`/`114`/`115`, where
GPU 0 sat at 84–86 °C with `n_thermal` set on ≈ 185 of 190 rows and a 4–5 % clock trim, this is the
end of that regime. The 124 power-cap rows are GPU 0 drawing its 170 W, which is the card working as
configured (`024`). **The slot-vs-card question closes as moot**: the deferred slot swap is not
needed while this holds. Throughput here (71.7 k tok/s, 9.82 h for 2.5 B) is a valid number.

Wall-clock accounting: summed `step_s` 9.683 h + evals 0.123 h + checkpoints 0.006 h = 9.812 h of
9.82 h, **0.1 % unaccounted**.

## Interpretation

Silent on H1–H19, by design: L0 is the comparator, not a hypothesis. Δ is not defined for this run;
σ and $T$ wait on `117`. What it establishes is $L_{ref}$(small, seed 0) = 2.9971 nats, a clean
71.7 k tok/s dense rate for the `small` budget (9.8 h per L0-class run, so the twelve-run programme is
≈ 8 days as planned), and that the rig now holds four cards at full clock for ten hours.
