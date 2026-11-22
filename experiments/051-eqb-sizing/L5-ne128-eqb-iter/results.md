# 051-eqb-sizing/L5-ne128-eqb-iter — L5-ne128-eqb-iter at `screen`

`python -m scripts.train.train --rung L5-ne128-eqb-iter --id 051-eqb-sizing/L5-ne128-eqb-iter --tokens 3.2e+06 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** H26, as one of rig `051`'s three sizing smokes (ADR-069 §3); no verdict. Read `../results.md`.

## Numbers

| | |
|---|---|
| params | 193.1 M total, 168.5 M non-embedding |
| tokens | 0.003 B in 6 steps |
| final val loss (full held-out, 5606 windows) | **8.4001** nats |
| final train loss (last step, main head) | 8.4749 |
| throughput (last step) | 15,151 tokens/s aggregate |
| peak memory per GPU | 6.26 GiB |
| wall-clock (this session) | 0.09 h |
| seed | 0 |
| val_r_mean | 8.0000 |

## Interpretation

(one paragraph; state Δ vs the matched baseline, σ, T and the verdict rule)
