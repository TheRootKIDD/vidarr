# 178-l8b-pv-small-s0 — L8b-pv at `small`

`python -m scripts.train.train --rung L8b-pv --id 178-l8b-pv-small-s0 --tokens 2.5e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · gate `3a4e219` (`queue_small_12`) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** H17. This is seed 0 of ADR-060's pair, so the verdict is **provisional** until `179` completes the pair against `132`/`133` (ADR-060 §6, read by ADR-058 §3 and ADR-059). Against `132`, the per-seed reading is **supports** under ADR-059's rule:
- in-table first sentence on covered tokens: **−0.0563**, beats its 0.0151 floor;
- rest of the lead: **−0.0249**, beats its 0.0121 floor;
- FineWeb-Edu: **−0.0072** on the full held-out set and −0.0067 on the eval's subset. Both are below +2σ (0.0059).

The full-split `verdict_h17` and the 0.5 % literal reading also come out *supports*.

This is the first H17 *supports* reading in the L8 family. It holds at one seed only: the rest-of-lead clause is the one I42 flags (L5's own seed spread there is 0.013, at the floor), and `132` is L5's worse seed on that split.

## Numbers

| | |
|---|---|
| params | 77.1 M total, 52.5 M non-embedding (`176` L8b 74.7 / 50.2 M; `132` L5 77.5 / 52.9 M) |
| expert width | 832 (`176` 704; L5 896), ADR-060 |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0706** nats (`132` 3.0778 → **−0.0072**, −2.4σ; `176` 3.0904 → **−0.0198**, 6.6σ; L5 pair 3.0748 → −0.0042) |
| final train loss (last step, main head) | 3.0733 |
| throughput | median **34,576 tok/s** aggregate (`176` 34,109 → 1.01×; `132` 49,133 → 0.70×) |
| retrieval | median 0.320 s per micro-batch, ≈ 5.1 s of the 15.16 s median step (≈ 34 %, as `176`) |
| peak memory per GPU | 5.78 GiB (`048` 5.78) |
| wall-clock | 73,904 s = **20.53 h** (`048` estimate ≈ 20.8 h), of which 1,432 s are in-training evals. The post-run eval and compare took 5.0 min |
| thermal | `n_thermal` = 0 in 476 records; max 78 °C; SM ≥ 1890 MHz |
| seed | 0 |

σ = 0.0030 (twelve `small` pairs; 2σ 0.0059, 4σ 0.0119).

### Held-out trajectory against `132` and `176` (same seed, same steps)

| step | 100 | 700 | 1300 | 1900 | 2500 | 3100 | 3700 | 4300 | 4768 |
|---|---|---|---|---|---|---|---|---|---|
| `178` val loss | 5.7033 | 3.6288 | 3.3816 | 3.2660 | 3.1911 | 3.1382 | 3.1005 | 3.0766 | 3.0706 |
| Δ vs `132` L5 | +0.0047 | −0.0229 | −0.0157 | −0.0101 | −0.0098 | −0.0077 | −0.0077 | −0.0071 | **−0.0072** |
| Δ vs `176` L8b | −0.0002 | −0.0174 | −0.0195 | −0.0190 | −0.0193 | −0.0189 | −0.0192 | −0.0186 | **−0.0198** |

`178` never crosses `132`. Its lead narrows from −0.023 at 0.37 B tokens to −0.007 by 1.6 B tokens, and then stays flat at −0.007 to −0.008 through 2.5 B. `176` crossed `132` at 0.4–0.7 B tokens and settled at +0.012. The gap to `176` opens by step 700 and then holds at −0.019 to −0.020 from step 1300 to the end.

### wiki-lead against `132` (`wiki_lead_compare_vs_132.json`; base = `046`'s covered re-score of `132`)

| split | tokens | `132` L5 | `178` L8b-pv | Δ (floor) | beats? | `176` vs `132` |
|---|---|---|---|---|---|---|
| in-table first sentence, **covered** (ADR-059's clause) | 55,516 | 2.8950 | 2.8386 | **−0.0563** (0.0151) | **yes** | −0.0276 |
| in-table first sentence, own retrieved | 54,597 | 2.8833 | 2.8270 | −0.0563 (0.0151) | yes | −0.0285 |
| in-table first sentence, full | 70,952 | 2.8422 | 2.7404 | −0.1018 (0.0138) | yes | −0.0446 |
| in-table rest of lead | 77,911 | 3.1383 | 3.1134 | **−0.0249** (0.0121) | **yes** | −0.0057 |
| out-of-table lead (control) | 148,298 | 2.9866 | 2.9294 | −0.0572 (0.012) | yes | −0.0234 |
| first sentence, not covered (derived; branch zero) | 15,436 | 2.6527 | 2.3871 | −0.2656 | | −0.1061 |
| FineWeb-Edu subset (976 windows) | | 3.0732 | 3.0664 | −0.0067 | within 2σ | +0.0121 |

Retrieval coverage is 78.2 % and accuracy on covered tokens 98.3 %, the same as `045` and `176`: the retriever is fixed and so is the table.

Against `176` (same seed, means from the two files; no bootstrap): covered −0.0288, rest −0.0193, control −0.0338, FineWeb subset −0.0189. The width gain is **of one size on every split**, including FineWeb and the out-of-table control, where the branch has nothing specific to supply. The covered-token gain over `132` doubled (−0.028 → −0.056), and all of that doubling is this general shift, not the memory. In neither arm does the covered gain exceed the control's: `176` −0.028 against −0.023, `178` −0.056 against −0.057.

The uncovered first-sentence tokens deserve a warning. There the branch is zero, and `178`'s 2.387 is **0.27 below `132`**. That sits inside the L5 seeds' own spread on those tokens (2.653 and 2.359, `046`), so it is not evidence for the memory. It is I42's pre-boundary seed noise again, and it is why the full in_first row (−0.102) is not the clause.

### Usage probes (random table rows in place of the neighbours)

Under hoist the random rows are drawn once per segment, not once per iteration (`048`, recorded). The probe is therefore a slightly different intervention from `176`'s.

| | `176` L8b | `178` L8b-pv |
|---|---|---|
| reliance, own retrieved first-sentence tokens (floor 0.0151) | +0.0514 | **+0.0525** (reads) |
| reliance, covered / full first sentence | +0.0511 / +0.0400 | +0.0523 / +0.0409 |
| reliance, rest of lead / control | +0.0297 / +0.0305 | +0.0275 / +0.0295 |
| reliance, FineWeb-Edu held-out | +0.0205 | +0.0206 |
| α entropy (of ln 8 = 2.079) | 1.894 (91.1 %) | 1.901 (91.4 %) |
| table rows retrieved at least once (976 held-out windows) | 13.7 % | 13.7 % |

The two arms use the neighbours in the same way and to the same degree. Every reliance figure matches `176` to within 0.003.

## Interpretation

ADR-060 hoisted L8b's two projections out of the iteration loop. The function is the same (`048`: 2.4 × 10⁻⁷ nats), so the only effect on the trained model is the 0.6 GFLOP budget spent on width 832 instead of 704.

That width buys −0.020 on FineWeb-Edu against `176` (6.6σ) and an equal shift on every wiki-lead split. It is enough to move the L8b arm from +0.0126 above same-seed L5 to **−0.0072 below** it. So `176`/`177`'s general-text cost was the width the branch took, as ADR-060's hypothesis said, and not the retrieval itself.

What the memory adds over that shift stays the same. Reliance is +0.052 on own-retrieved tokens in both arms and α is equally sharp. In neither arm is the covered first-sentence gain over L5 larger than the out-of-table control's. So the memory's specific contribution shows in the reliance probe, not in a covered-minus-control margin. At this seed, H17's three clauses all pass against `132` for the first time:
- the stored sentence: −0.056, 3.7× its floor;
- the rest of the lead: −0.025, 2.1× its floor;
- FineWeb-Edu: below L5, not merely within 2σ.

Two things limit the reading.
1. **The rest-of-lead clause sits on I42's limit.** `132` is L5's worse seed there: `133` is 0.013 below it, which is at the floor. Against the L5 pair mean, `178`'s rest-of-lead margin is roughly halved. `179` against `133` decides this clause.
2. **Some of the general gain may be seed luck.** The out-of-table control and FineWeb move by the same amount as the in-table splits, so the per-split wins are mostly one shift.

The verdict is the pair's (ADR-060 §6): `179` against `133`, then `--compare-pair` against `132`/`133`, with ADR-060 §7's readings already set. A pair *supports* turns `medium` into a question for a new ADR. The pair's held-out Δ against the L5 pair (3.0748) stays within 2σ if `179` finishes at or below 3.0908, and passes 4σ only above 3.1028. The recorded limits stand:
- vector values;
- the corpus mismatch (a Wikipedia table against FineWeb-Edu training);
- retrieval outside "matched active FLOPs": ≈ 34 % of the step, 0.70× L5's throughput.
