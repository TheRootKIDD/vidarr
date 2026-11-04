# 043-l8q-sizing — L8q under ADR-055 fits (5.72 GiB per GPU) and runs at 36.1 k tok/s, **0.88× L8** (0.74× L5). The teacher takes ≈ 16 % of step time, and $\mathcal{L}_q$ starts at ln 1024 and falls → `174-l8q-screen` ≈ 8.0 h

2026-11-04 11:18–11:21 UTC+1 · idle rig, by hand, before `queue_screen_16` · tree `3262716` (ruff and the test suite passed on it, including the GPU teacher test against `040`) · driver 615.71.09 ·
`python -m scripts.train.train --rung L8q --id 043-l8q-sizing --tokens 1e9 --micro-batch 4 --seed 0 --smoke-steps 6 --out-dir /mnt/nvme/ckpt/043-l8q-sizing`.
Six optimiser steps of 524,288 tokens, DDP on 4 × RTX 3060, then one held-out eval. `config.yaml`, `log.jsonl` and `summary.json` are copied here. The checkpoint stays off-repo.

**Bears on:** nothing directly. This is rig sizing for ADR-055 §7c (H17's L8q arm). Rig series id `043`.

| | |
|---|---|
| config (derived) | as `173` (expert width 704, 2²⁰ cells, $k_m$ 32), plus `memory.query_aux`: $\lambda_q$ 0.1, chunk 64, teacher BGE-base at `040`'s revision, ≤ 128 BGE tokens |
| params | 74,686,849 total, 50,110,849 non-embedding: **identical to `173`**. The teacher is not a model parameter |
| step time | 14.45–14.63 s at steps 2–6 (`041` L8: 12.76 s) |
| throughput | **36.1 k tok/s** aggregate (`041` 41.1 k → **0.88×**; `110` L5 48,553 → 0.74×) |
| teacher | **0.147 s per micro-batch** (target building, encoding and sync; 16 micro-batches per step → ≈ 2.35 s, ≈ 16 % of the step, as ADR-055 §4 estimated from FLOPs) |
| peak memory per GPU | **5.72 GiB** (limit 10 GiB; +0.41 GiB over L8 for the fp32 encoder and its activations) |
| $\mathcal{L}_q$ | 6.930 → 6.655 over 6 steps (ln 1024 = 6.931 at the start: the query starts uniform over each half's sub-keys) |
| total − main − router aux | 0.693 at step 1 = 0.1 × $\mathcal{L}_q$ (the loss carries $\lambda_q \mathcal{L}_q$ as specified) |
| valid targets | 83 % of positions on held-out windows (the agreement probe on `173`, 32 windows: the first chunk and EOS-touching chunks are masked) |
| wall-clock | 163 s including startup (table and encoder load) and one full eval (74 s) |

**Baseline of the ADR-055 §5 agreement probe** (`probe_l8_lookup.teacher_agreement` on `173`, 32 held-out windows, 54,400 valid positions per iteration): top-1 row = teacher row in 0–1.8 × 10⁻⁵ of positions, column in 0.94–1.32 × 10⁻³, against chance 9.8 × 10⁻⁴. L8 without the loss is at or below chance, so `174` has a clean zero to compare with. (A side check, not the probe.)

**Estimate for `174`:** 1907 steps × 14.55 s ≈ 7.7 h, plus ≈ 0.3 h of evals and probes → **≈ 8.0 h**.
