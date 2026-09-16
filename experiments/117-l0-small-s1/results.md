# 117-l0-small-s1 — L0 at `small`, seed 1: 2.9945 nats; the pair gives $L_{ref}$(small) = 2.9958 and **σ = 0.0018 nats**

`python -m scripts.train.train --rung L0 --id 117-l0-small-s1 --tokens 2.5e+09 --micro-batch 4 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** no hypothesis directly — the second half of the L0 pair. With `116` it fixes the two
numbers every `small` verdict is written against: **$L_{ref}$(small) = 2.9958 nats** (mean of the pair)
and **σ = 0.0018 nats** (ADR-013's pooled seed spread, here the sample standard deviation of the pair,
$|L_0 - L_1|/\sqrt{2}$). Same command as `116` with `--seed 1`; seed changes initialisation and the
data order, nothing else.

## Numbers

| | |
|---|---|
| params | 100.1 M total, 75.5 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| final val loss (full held-out, 5606 windows) | **2.9945** nats |
| final train loss (last step, main head) | 2.9778 |
| throughput (median step) | 71,700 tokens/s aggregate (`step_s` median 7.311 s, p90 7.312) |
| peak memory per GPU | 6.25 GiB |
| wall-clock (this session) | 9.82 h |
| seed | 1 |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 81 °C, `sm_mhz_min` ≥ 1860; `n_power_cap` 137 rows (GPU 0 at its 170 W cap) |

## The L0 pair

| | seed 0 (`116`) | seed 1 (`117`) | pair |
|---|---|---|---|
| final val loss, full held-out | 2.9971 | 2.9945 | **mean 2.9958**, $|Δ_{seed}|$ = 0.0026 |
| val loss at the 1 B-token eval (step 1900) | 3.2206 | 3.2200 | $|Δ_{seed}|$ = 0.0005 |
| wall-clock | 9.82 h | 9.82 h | 71.7 k tok/s both |

**σ = 0.0018 nats.** ADR-013 defines σ as the pooled seed spread of the L0 pair; with one pair that is
the sample standard deviation, $0.0026/\sqrt{2}$. It is below the 0.005–0.01 nats `06 §4` assumed
and well below the 0.01 nats at which ADR-013 would demand a third seed for every "1 %" test.

Two points on how much to trust a two-point estimate. First, the estimate is corroborated by the
whole trajectory: over the last 20 paired evals (2 M-token subset, same steps in both runs) the
seed-0 − seed-1 difference is **+0.0023 ± 0.0006 nats** (mean ± sd), i.e. a stable offset between
the seeds rather than eval noise, so σ is measuring a real between-seed spread and not the eval set.
Second, two seeds give no useful confidence interval on σ itself; the number is what ADR-013 asked
for and it is used as written. If a later pair at `small` (L1, L2, L3, L5, L5d all run at two seeds
in `queue_small_1`) shows a materially larger spread, σ is re-pooled over every pair and the verdicts
recomputed, in a new log entry.

**The bands this fixes for every `small` verdict** (fractions of $L_{ref}$ = 2.9958):

| threshold | nats | verdict rule (ADR-013) |
|---|---|---|
| 2σ ("matches", "no detectable regression") | **0.0037** | supports if $|Δ| \le$ 0.0037 |
| 4σ (`screen` silent band) | **0.0073** | a one-seed `screen` run is silent unless $|Δ|$ > 0.0073 |
| 1 % $L_{ref}$ | **0.0300** | supports if $Δ \le$ 0.030; weakens if $Δ$ > 0.0336 |
| 2 % $L_{ref}$ | **0.0599** | supports if $Δ \le$ 0.060; weakens if $Δ$ > 0.0636 |

## Thermals

Same as `116`: zero `n_thermal` rows over 9.8 h, GPU 0 at 80–81 °C and ≥ 1860 MHz, step time flat
(p90/median 1.0002). Wall-clock: 9.683 h of steps + 0.123 h of evals + 0.006 h of checkpoints =
9.812 of 9.82 h, 0.1 % unaccounted.

## Interpretation

Silent on H1–H19 as a run; decisive as a calibration. $L_{ref}$(small) = 2.9958 nats and
σ = 0.0018 nats are now the ruler. The immediate consequence is for the `screen` ladder: ADR-013
calls a one-seed run silent unless $|Δ|$ > 4σ = 0.0073 nats, and most `screen` deltas are ten times
that, so several `screen` results that were recorded as "silent for want of σ" now carry verdicts —
re-scored in `docs/04` (2026-09-16). The caveat is that σ is measured at 2.5 B tokens and applied to
1 B-token runs; the pair's spread at its own 1 B-token eval (0.0005 nats) says the 1 B spread is no
larger, so the 4σ band is, if anything, conservative there.
