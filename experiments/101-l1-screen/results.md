# 101-l1-screen — L1 (layered MoE baseline) at `screen` — **ABORTED at step 180 / 1907**

2026-09-03 · trainer `--rung L1 --id 101-l1-screen --tokens 1e9 --micro-batch 2 --seed 0` ·
4 × RTX 3060, DDP · started 14:06, stopped 16:25 by the user because three cards were
thermally throttling (I23: 91–93 °C, SM clocks down to 270 MHz; ≈ 48 s per 0.5 M-token
step against ≈ 12 s expected).

**Bears on:** nothing yet. 0.094 B of 1.0 B tokens. Held-out 6.216 nats at step 100
(L0: 6.083 at the same step). Not a result; kept as the append-only record of the
attempt. The rerun after the cooling fix gets a **new id** (106+); do not resume this
one — its checkpoint predates the timer and thermal-logging fixes, and the
throughput column of `log.jsonl` is CPU-side and invalid.
