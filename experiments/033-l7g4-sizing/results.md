# 033-l7g4-sizing — L7-g4 under ADR-047 (8 of 32 sub-experts, width 224) fits easily (3.86 GiB) and runs at 30.6 k tok/s, **0.63× L5** → `172-l7-g4-screen` ≈ 9.2 h

2026-11-02 15:30–15:34 UTC+1 · idle rig, by hand, between queues · tree `09d68be` (ADR-047 code; ruff and
the 119-test suite passed on it just before) · driver 615.71.09 ·
`python -m scripts.train.train --rung L7-g4 --id 033-l7g4-sizing --tokens 1e9 --micro-batch 4 --seed 0 --smoke-steps 6 --out-dir /mnt/nvme/ckpt/033-l7g4-sizing`.
Six optimiser steps of 524,288 tokens (16 × 4 × 2048 × 4), DDP on 4 × RTX 3060, then one full
held-out eval. `config.yaml`, `log.jsonl` and `summary.json` are copied here (`train.log` is gitignored and stays at `/mnt/nvme/ckpt/033-l7g4-sizing/`). The
checkpoint stays off-repo.

**Bears on:** nothing directly. This is rig sizing for ADR-047 §4 (H4's arm). Rig series id `033`.

| | |
|---|---|
| config (derived) | expert width 896 → 4 sub-experts of 224 per unit, 32 sub-experts, **8 active** ($k g$), $N_e$ = 8 units |
| params | 77,483,393 total, 52,907,393 non-embedding: **+3,072 vs `110` L5** (52,904,321), the router keys for 24 extra sub-experts (ADR-047 §1) |
| step time | 17.04–17.15 s at steps 2–6 (step 1 16.89 s) |
| throughput | **30.6 k tok/s** aggregate (`110` L5 median 48,553 → **0.63×**) |
| peak memory per GPU | **3.86 GiB** (limit 10 GiB) |
| wall-clock | 186 s including startup and one full eval |
| thermal | no thermal record falls inside 6 steps (the logger writes one every 10). Throughput only, so this costs nothing |

**Estimate for `172`:** 1907 steps × ≈ 17.1 s ≈ 9.1 h, plus ≈ 0.1 h of evals → **≈ 9.2 h**. The
slowdown is the reference implementation's Python loop over experts (32 gathers and launches instead of
8, each a quarter of the width). The FLOPs are matched (ADR-047 §1), so it is not a FLOPs effect. As
with every throughput figure, it is a property of `model/`, not of the design.
