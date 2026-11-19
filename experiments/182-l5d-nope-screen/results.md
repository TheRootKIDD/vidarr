# 182-l5d-nope-screen — L5d-nope (RoPE local only, NoPE on $\mathcal{G}$) at `screen` — **ABORTED at step 117 / 1907 by a power outage**

2026-11-18/19 · trainer `--rung L5d-nope --id 182-l5d-nope-screen --tokens 1e9 --micro-batch 4 --seed 0` ·
4 × RTX 3060, DDP · code `c072e86` (the `embla` merge; gate passed 23:59:35) · started 23:59:35 on
2026-11-18 by `/mnt/nvme/queue_screen_18.sh` after `queue_medium_1` · killed at ≈ 00:36 on 2026-11-19
by a mains power outage (last journal line 00:36:39; the host came back at 06:23).
Last checkpoint `ckpt/latest.pt` at step 98 (00:29); `log.jsonl` runs to step 117.

**Bears on:** H22 (ADR-066) — **silent; not a result.** 0.061 B of 1.0 B tokens. Held-out 5.5224 nats
at step 100 (976 windows); train loss 5.35 at step 117. Nothing here is compared against `162` or
$L_{ref}$; kept as the append-only record of the attempt.

**Run health while it ran, for the record:** 27.9–28.9 k tok/s aggregate (18.4 s per step, which
projects ≈ 9.7 h for the rung against the queue header's ≈ 8.3 h), 3.89 GiB peak per GPU. Routing
healthy at step 117: 7.75 of 8 effective experts at the worst router, 0 dead, load 0.062–0.158.

**Not resumed, by choice.** The trainer's `--resume` path is sound for loss numbers, but the ladder
is append-only and the precedent from `101`/`109` is a fresh id. Ids `183`–`203` are reserved by
ADR-066–074, so the rerun takes the next free id: **`204-l5d-nope-screen`**, same preset, seed and
token budget, started 06:38 on 2026-11-19 by `/mnt/nvme/queue_screen_18b.sh`. **Wherever ADR-066–068
say `182`, read `204`** (ADR-066's verdict against `162`; ADR-066 §7's base choice for `188`/`189`,
which `queue_screen_20.sh` now reads from `204`'s summary). `204`'s first three steps reproduce this
run's losses to four decimals (10.5239 / 10.4830 / 10.3777). Cost of the choice: ≈ 36 min of compute.
