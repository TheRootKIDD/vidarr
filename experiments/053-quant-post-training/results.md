# 053-quant-post-training — rig for ADR-073 §2: round-to-nearest FP4 of the finished checkpoints costs **+0.0343 nats on L0 (`100`) and +0.0403 on L5 (`110`)**, 11.6σ and 13.6σ → H3's FP4 clause *weakened* for post-training quantisation at `screen`; INT4 costs less than FP4 on both (+0.0208 / +0.0262)

Eval-only: `scripts.analysis.probe_quant` on the step-1907 checkpoints of `100-l0-screen` and
`110-l5-screen`, full held-out set (5606 windows of 2048 tokens), one RTX 3060 each, run by
`queue_screen_25` on 2026-11-25 23:18–23:31 (10.2 and 13.1 min). Code at gate commit `d4d0dfd`;
commands in `config.yaml`; outputs are `probe_quant.json` in each base run's directory. No
training, so no seed beyond the checkpoints' own (seed 0).

**Bears on:** **H3**, FP4 clause ("FP4 weights add ≤ 0.5 %", read as $T$ = 2σ under ADR-013), by
**ADR-073 §3** under ADR-066 §1: Δ = quantised − its bf16 base; within 4σ silent, above 4σ
weakens at `screen`. This is the post-training arm of three; the comparison ADR-073 cares about
(late QAT vs post-training vs QAT from step 0) completes with `199`–`202`.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| base | bf16 (re-scored) | FP4 E2M1 | Δ | INT4 (side reading) | Δ |
|---|---|---|---|---|---|
| `100-l0-screen` (L0, dense) | 3.3358 | 3.3702 | **+0.0343** (+1.03 %, 11.6σ) | 3.3567 | +0.0208 (7.0σ) |
| `110-l5-screen` (L5, weight-tied recurrence) | 3.3328 | 3.3731 | **+0.0403** (+1.21 %, 13.6σ) | 3.3590 | +0.0262 (8.8σ) |

Both bf16 re-scores reproduce the runs' final evals (3.3358, 3.3328). σ multiples use the
unrounded σ = 0.00296.

## Reading

- **Post-training FP4 weakens the clause on both bases:** three to four times the 4σ line, about
  1 % of the loss where the hypothesis allows 2σ.
- **The recurrence costs a little more, not much more:** L5 loses 0.0060 more than L0 (2.0σ).
  `00 §P3` names FP4 on a weight-tied recurrence as the numerics risk; at round-to-nearest the
  error does compound over eight applications of the same weights, by about a sixth.
- **INT4 is the gentler format here** (0.61× and 0.65× of FP4's cost): with one absmax scale per
  group of 64, uniform levels fit these weights better than E2M1's log-spaced ones. It stays a
  side reading; H3 is stated for FP4.

## Interpretation

Rounding trained bf16 weights to FP4 without any training costs 0.034 nats on the dense baseline
and 0.040 on L5, far outside the band H3's FP4 clause allows, so if FP4 is to hold it must be
through training with the quantiser in the loop. That is what `199`–`202` test, and these two
numbers are the floor they are compared against. One checkpoint per base, `screen`, quality only
(Ampere has no FP4; nothing here is a speed or memory measurement).
