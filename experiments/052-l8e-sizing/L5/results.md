# 052-l8e-sizing/L5 — L5 at `screen`

`python -m scripts.train.train --rung L5 --id 052-l8e-sizing/L5 --tokens 3.2e+06 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** (fill in: H numbers and verdict per ADR-013).

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding |
| tokens | 0.003 B in 6 steps |
| final val loss (full held-out, 5606 windows) | **8.4160** nats |
| final train loss (last step, main head) | 8.4930 |
| throughput (last step) | 48,251 tokens/s aggregate |
| peak memory per GPU | 3.86 GiB |
| wall-clock (this session) | 0.04 h |
| seed | 0 |
| val_r_mean | 8.0000 |

## Interpretation

(one paragraph; state Δ vs the matched baseline, σ, T and the verdict rule)
