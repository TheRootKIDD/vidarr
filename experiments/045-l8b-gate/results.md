# 045-l8b-gate — L8b's retrieval is **precise**: 98.3 % of covered in-table first-sentence tokens have the article's own sentence among their 8 neighbours, and 78.2 % of those tokens are covered. L8b fits (5.72 GiB per GPU) and runs at 33.8 k tok/s (0.93× `174`), with retrieval taking ≈ 32 % of the step → `175-l8b-screen` ≈ 9 h

2026-11-04 23:05–23:22 UTC+1 · rig otherwise idle · driver 615.71.09 · tree `6f6ed73` (ruff, and the full suite with 170 passed) · commands in `config.yaml`.
- `wiki_lead_110_covered.json`: `110` re-scored with the pipeline's masks, 82 s on GPU 0.
- `sizing/`: six DDP steps of L8b on 4 GPUs, 4.4 min. The checkpoint stays off-repo.
- One false start was not a run: launching the trainer under `torchrun` doubled its own `mp.spawn` and failed at process-group init. Rerun as `043` was.

**Bears on:** ADR-057 §7's pipeline reading and §9c's sizing. It bears on no hypothesis: `110` is only re-scored, and its numbers reproduce its committed `wiki_lead.json` exactly (every split, every per-article sum, and the held-out 3.3284). Rig series id `045`.

## Pipeline reading (2,000 in-table articles; the masks depend on the pipeline alone, not on any model)

| | value |
|---|---|
| first-sentence tokens | 70,952 |
| covered (a boundary with ≥ 8 query tokens precedes them) | 55,516 (**78.2 %**) |
| covered, with the own sentence among the 8 neighbours | 54,597 (**98.3 %** of covered) |
| `110` on the covered tokens · on the own-retrieved tokens | 3.1963 · 3.1856 (all first-sentence tokens 3.1307) |

**Reading (ADR-057 §7): precise** (≥ 50 %). For comparison, `044` showed that the product grid returns the own cell 16.8–22.8 % of the time even for a perfect query.

The covered tokens cost `110` more than the first sentence as a whole (3.196 against 3.131). The uncovered first 16 positions are mostly the predictable opening ("X is a …").

## Sizing (6 steps, 524,288 tokens each, DDP)

| | `174` L8q (`043`) | **L8b** |
|---|---|---|
| peak memory per GPU | 5.72 GiB | **5.72 GiB** |
| throughput (steps 3–6) | 36.1 k tok/s | **33.8 k tok/s** (0.93×) |
| step time | 14.4 s | 15.5 s |
| retrieval per micro-batch | — (teacher 0.148 s) | **0.31 s** (≈ 5.0 s of the step, 32 %) |
| one full held-out eval (5,606 windows) | ≈ 16 s | 159 s (retrieval on every window) |
| loss, steps 1 → 6 | | 10.52 → 8.52, val 8.43 |

Estimate for `175-l8b-screen`: 1,907 steps × 15.5 s ≈ 8.2 h, plus ≈ 20 in-training evals × 159 s ≈ 0.9 h, ≈ **9.1 h**, then ≈ 10 min of post-run evals. This is within ADR-057's ≈ 9 h. Width is 704 and memory FLOPs 1.505 MFLOP per token-iteration, as the ADR states.
