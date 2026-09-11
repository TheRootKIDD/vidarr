# 109-l5-screen — L5 (shared middle block) at `screen` — **ABORTED at step 170 / 1907 by a power outage**

2026-09-11 · trainer `--rung L5 --id 109-l5-screen --tokens 1e9 --micro-batch 4 --seed 0` ·
4 × RTX 3060, DDP · started 18:03 by `/mnt/nvme/queue_screen_2.sh` after `108` finished ·
killed at ≈ 18:36 by a mains power outage (the host rebooted at 19:32 and again at 19:42).
Last checkpoint `ckpt/latest.pt` at step 165 (18:33); `log.jsonl` runs to step 170.

**Bears on:** nothing yet. 0.089 B of 1.0 B tokens. Held-out 5.6553 nats at step 100
(L1 at the same step: 5.7 range; not comparable across rungs this early). Not a result;
kept as the append-only record of the attempt.

**Run health while it ran, for the record:** 47.8 → 49.0 k tok/s aggregate, above `018`'s
44.5 k L5 projection; 3.86 GiB peak per GPU; min SM clock 1890 MHz; `n_thermal` = 0
throughout; GPU 0 at 80–83 °C. Routing healthy: 7.8 of 8 effective experts at the worst
router, 0 dead, by step 170. The thermal watch's "4 of 6 records flagged" rule tripped at
step 10 and was widened to sustained clock loss / 90 °C the same evening (`04` session log);
the flags were the software power cap on a card at full clock, not thermal slowdown.

**Not resumed, by choice.** The trainer's `--resume` path is sound for loss numbers (model,
optimiser, step, token count and the $r$ schedule are all restored), but the ladder is
append-only and the precedent from `101` is a fresh id. The rerun is **`110-l5-screen`**
(`/mnt/nvme/queue_screen_3.sh`), followed by `111-l5d-screen`. Cost of the choice: ≈ 30 min
of compute.
