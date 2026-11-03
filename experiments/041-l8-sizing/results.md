# 041-l8-sizing — L8 under ADR-053 fits (5.31 GiB per GPU) and runs at 41.1 k tok/s, **0.85× L5** → `173-l8-screen` ≈ 7.2 h; the wiki-lead eval and probes run end to end

2026-11-03 22:31–22:35 UTC+1 · idle rig, by hand, before `queue_screen_15` · tree `ba89ed2` (ruff and the test suite passed on it) · driver 615.71.09 ·
`python -m scripts.train.train --rung L8 --id 041-l8-sizing --tokens 1e9 --micro-batch 4 --seed 0 --smoke-steps 6 --out-dir /mnt/nvme/ckpt/041-l8-sizing`.
Six optimiser steps of 524,288 tokens, DDP on 4 × RTX 3060, then one held-out eval. `config.yaml`, `log.jsonl` and `summary.json` are copied here. The checkpoint stays off-repo.
Then `python -m scripts.analysis.eval_wiki_lead --id 041-l8-sizing --probes --max-tokens 200000` on GPU 0, as a smoke test of ADR-053 §7/§8's eval (`wiki_lead.json`).

**Bears on:** nothing directly. This is rig sizing for ADR-053 §9b (H17's arm). Rig series id `041`.

| | |
|---|---|
| config (derived) | expert width **704** (exact 712.6, snapped to 64; L5 896), memory 2²⁰ cells, $k_m$ 32, half top-k 32, table `040` |
| params | 74,686,849 total, 50,110,849 non-embedding: **−2,793,472 vs `110`** (−3,538,944 from the narrower experts, +745,472 memory: $W_m$ 106,496, $h^{1,2}$ 49,152, $W_v$ 589,824). The table is data, not parameters |
| FLOPs | +1.70 MFLOP per token-iteration of memory, matched to L5's 0.6 GFLOP training budget by the width (−0.3 % against budget) |
| step time | 12.76 s at step 6 |
| throughput | **41.1 k tok/s** aggregate (`110` L5 median 48,553 → **0.85×**) |
| peak memory per GPU | **5.31 GiB** (limit 10 GiB): the 1.5 GiB bf16 table is resident on every card |
| wall-clock | 151 s including startup (table load) and one full eval (74 s) |
| eval smoke test | 73 s for 4,000 articles plus 97 held-out windows, twice (with and without random cells). At step 6: lookup accuracy 0 at every iteration, 0.46 % of cells used, α entropy 3.4655 of a maximum 3.4657 (uniform), random cells change nothing. That is the untrained, zero-init read-out, as expected |

**Estimate for `173`:** 1907 steps × 12.76 s ≈ 6.8 h, plus ≈ 0.4 h of evals → **≈ 7.2 h**, shorter than ADR-053's 9–11 h. The extra 15 % per step over L5 is the memory lookup at every iteration, recomputed in backward under iteration checkpointing. The narrower experts give back part of it.
