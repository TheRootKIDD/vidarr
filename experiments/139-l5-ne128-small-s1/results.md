# 139-l5-ne128-small-s1 — burnt: killed at step 850 of 4768 (0.45 B tokens) for a planned pause, 2026-09-25 16:30

`python -m scripts.train.train --rung L5-ne128 --id 139-l5-ne128-small-s1 --tokens 2.5e+09 --micro-batch 4 --seed 1` · started 2026-09-25 08:47 as the second run of `queue_small_4.sh`; the seed-1 partner of `138`.

**Bears on:** nothing. Partial record only: the user asked for a pause at 16:26, the trainer and its
four DDP workers were killed by PID at step 850 (7.7 h of ≈ 43 h; last 976-window eval at step 800:
3.4884 nats, 0.001 below `138` at the same step), and `queue_small_4.sh` and the armed
`queue_small_5.sh` were stopped. Ids are append-only and checkpoints are never resumed (`docs/04`),
so **L5-ne128 seed 1 reruns from scratch as `142-l5-ne128-small-s1`** in `queue_small_6.sh`, which
also carries the L7b pair (`143`/`144`, ex-`140`/`141`; ids 140 and 141 were reserved by
`queue_small_5.sh` but never started and are left unused). `log.jsonl` holds the 850 steps that ran;
`n_thermal` was 0 on all 85 telemetry rows.
