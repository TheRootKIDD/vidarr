# 052-l8e-sizing/L8e — L8e at `screen`

`python -m scripts.train.train --rung L8e --id 052-l8e-sizing/L8e --tokens 3.2e+06 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** (fill in: H numbers and verdict per ADR-013).

## Numbers

| | |
|---|---|
| params | 78.1 M total, 53.5 M non-embedding |
| tokens | 0.003 B in 6 steps |
| final val loss (full held-out, 5606 windows) | **8.4281** nats |
| final train loss (last step, main head) | 8.5042 |
| throughput (last step) | 30,150 tokens/s aggregate |
| peak memory per GPU | 4.62 GiB |
| wall-clock (this session) | 0.05 h |
| seed | 0 |
| val_r_mean | 8.0000 |

## Interpretation

(one paragraph; state Δ vs the matched baseline, σ, T and the verdict rule)
