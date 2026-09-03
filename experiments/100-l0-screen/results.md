# 100-l0-screen — L0 at `screen`

`python -m scripts.train.train --rung L0 --id 100-l0-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** no hypothesis directly — this is the **dense baseline L0**, the author's requested plain-transformer comparator (`docs/04 §Author's answers`, Baseline) and the $L_{ref}$ of every ADR-013 verdict at `screen`. One seed: `screen` verdicts are silent unless $|\Delta| > 4\sigma$; $\sigma$ comes from the L0 pair at `small`.

## Numbers

| | |
|---|---|
| params | 100.1 M total, 75.5 M non-embedding |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3358** nats |
| final train loss (last step, main head) | 3.3048 |
| throughput (last step) | 63,308 tokens/s aggregate |
| peak memory per GPU | 6.25 GiB |
| wall-clock | 17.99 h (20:06 → 14:05) — **of which 4.28 h in timed steps**; see I22 |
| seed | 0 |

## Interpretation

**$L_{ref}$ = 3.336 nats at `screen`** (1.0 B tokens, 100 M params, seed 0). The curve
is monotone and unremarkable: 6.08 at step 100, 4.41 at 400, 3.55 at 1000,
3.38 at 1500, 3.334 at 1900 on the 2 M-token subset, 3.336 on the full 11.5 M
held-out; train loss 3.30 at the end, no spike, gradient norm 0.2–0.3 through
the second half. For ADR-013 this fixes the thresholds at `screen`: "1 %" =
0.033 nats, "2 %" = 0.067 nats; the seed spread $\sigma$ is still to be measured
(L0 pair at `small`).

**Compute matched the bench, wall-clock did not.** Timed steps average 8.1 s
(≈ 65 k tokens/s, `014`'s L0 figure with accumulation) and sum to 4.28 h, but
the run occupied the cards for 18 h. The first checkpoint landed 30 min after
launch at step ≈ 220, i.e. wall-clock tracked step time early on; the loss was
somewhere later, outside the timed region (evaluation, checkpointing to the
SATA `/home` volume, or the end-of-step barrier). Recorded as **I22**; the
trainer now logs absolute time and eval/checkpoint durations per step, and
`101-l1-screen` is being watched with timestamps. The loss figures are
unaffected — every token was trained and evaluated as configured.
