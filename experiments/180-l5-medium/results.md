# 180-l5-medium — L5 at `medium`

`python -m scripts.train.train --rung L5-medium --id 180-l5-medium --tokens 7e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · gate `c8f3ce0` (`queue_medium_1`) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** H17, as its matched arm at `medium` (ADR-061 §5). No hypothesis is scored on this run alone. The H17 verdict is `181` L8b-pv-medium against this run, by ADR-059's rule (ADR-061 §6–§7).

This is also the programme's first `medium` run, and the first measurement of DDP at this size over a full run (ADR-061 §3).

## Numbers

| | |
|---|---|
| shape | `06 §3` `medium`: $d$ = 1024, 16/4 heads, $E$ = $F$ = 2, skeleton $d_{ff}$ 4096, fixed $r$ = 16, $N_e$ = 8, $k$ = 2, expert width **1920**, 2.0 GFLOP per token |
| params | 143.7 M total, 110.9 M non-embedding (`06 §3` anchor ≈ 144 M) |
| tokens | 7.000 B in 13,351 steps |
| **final val loss** (full held-out, 5606 windows) | **2.7250** nats (L5 `small` pair 3.0748 → −0.3498; `132` at the same seed 3.0778) |
| final train loss (last step, main head) | 2.7132 |
| throughput | median **18,874 tok/s** aggregate (`049` smoke 18,534; L5 `small` 49,133 → 0.38× for 3.3× the FLOPs per token) |
| peak memory per GPU | 5.93 GiB (as `049`), under DDP |
| wall-clock | 374,409 s = **104.0 h** (`049` projection ≈ 109 h), of which 3,552 s are 134 in-training evals (≈ 27 s each, not the ≈ 100 s `049` assumed). The post-run wiki-lead eval took 3.0 min |
| thermal | `n_thermal` = 0 in 1,335 records; max 80 °C; SM ≥ 1882 MHz |
| routing (last step) | 0 dead experts, minimum effective experts 7.998 of 8, load 0.119–0.130 |
| seed | 0 |

### Held-out trajectory

| step | 500 | 1000 | 2000 | 3000 | 4800 | 6000 | 8000 | 10000 | 12000 | 13351 |
|---|---|---|---|---|---|---|---|---|---|---|
| tokens (B) | 0.26 | 0.52 | 1.05 | 1.57 | 2.52 | 3.15 | 4.19 | 5.24 | 6.29 | 7.00 |
| val loss | 3.8452 | 3.3740 | 3.1273 | 3.0199 | 2.9118 | 2.8649 | 2.8041 | 2.7599 | 2.7309 | **2.7250** |

At 2.5 B tokens (step 4800), `180` is at 2.912 against `132`'s final 3.078: −0.166 at equal tokens. That part of the gain is the model size; the rest of the −0.350 comes from the longer schedule.

### wiki-lead (`wiki_lead.json`; covered splits, no probes: L5 has no memory)

| split | tokens | `180` L5-medium | `132` L5 `small` | Δ |
|---|---|---|---|---|
| in-table first sentence, covered | 55,516 | 2.4346 | 2.8950 | −0.460 |
| in-table first sentence, full | 70,952 | 2.3761 | 2.8422 | −0.466 |
| in-table rest of lead | 77,911 | 2.7335 | 3.1383 | −0.405 |
| out-of-table lead (control) | 148,298 | 2.5571 | 2.9866 | −0.430 |
| FineWeb-Edu subset (976 windows) | | 2.7211 | 3.0732 | −0.352 |

Coverage is 78.2 % and accuracy 98.3 %: the retrieval masks are the same as in every L8b eval.

## Interpretation

**DDP at `medium` works as `049` measured.** The run had 5.93 GiB flat, 18.9 k tok/s from start to end, no thermal events, stable gradients (norm ≈ 0.19 late) and healthy routing. It took 104 h, 5 h under the projection, because the in-training evals cost a quarter of what was assumed. This confirms ADR-061 §3's amendment for this configuration.

**The `medium` L5 improves more on Wikipedia leads than on general text.** Against `132`, it gains −0.41 to −0.47 on the wiki-lead splits but −0.35 on FineWeb-Edu. A larger model stores more of the encyclopaedic facts that the L8b-pv memory supplies, so the covered first sentence starts 0.46 nats lower. That leaves `181`'s memory less to add.

At `small`, L8b-pv's covered gain over L5 was −0.054, and its rest-of-lead gain −0.015. Whether the memory keeps that margin against a base this much better is what `181` measures.

**What comes next.** `181-l8b-pv-medium` started at 20:13 on 2026-11-13. It is projected at ≈ 123 h (`049`'s step time scaled by `180`'s measured-to-smoke ratio, plus ≈ 1 h of evals), so it should finish ≈ 23:30 on 2026-11-18. Its wiki-lead eval with probes and `--compare 181 180` follow.
