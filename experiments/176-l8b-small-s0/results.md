# 176-l8b-small-s0 — L8b at `small`

`python -m scripts.train.train --rung L8b --id 176-l8b-small-s0 --tokens 2.5e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** H17. This is seed 0 of ADR-058's pair, so the verdict is **provisional** until `177` completes the pair against `132`/`133`. The per-seed reading against `132` is **weakens** under ADR-059's rule, which ADR-055 §6 also gives on the full splits:
- in-table first sentence on covered tokens: −0.0276, beats its 0.0146 floor;
- rest of the lead: −0.0057, misses its 0.0125 floor;
- FineWeb-Edu: **+0.0126** (4.2σ) on the full held-out set and +0.0121 on the eval's subset, both past 4σ (0.0119).

`175`'s general-loss parity at `screen` (−0.0007 vs `110`) does not survive 2.5 B tokens.

## Numbers

| | |
|---|---|
| params | 74.7 M total, 50.2 M non-embedding (`132` L5: 77.5 / 52.9 M) |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0904** nats (`132` 3.0778 → **+0.0126**, 4.2σ; L5 pair 3.0748 → +0.0156) |
| final train loss (last step, main head) | 3.0916 |
| throughput | median **34,109 tok/s** aggregate (`175` 34,140; `132` 49,133 → **0.69×**) |
| retrieval | median 0.321 s per micro-batch, ≈ 5.1 s of the 15.37 s step (≈ 33 %, as `175`) |
| peak memory per GPU | 5.72 GiB |
| wall-clock | 74,909 s = **20.81 h** (estimate ≈ 21 h; `132` 14.25 h → 1.46×), of which 1,445 s are in-training evals. Post-run eval and compare took 5.0 min |
| thermal | `n_thermal` = 1 sample in 476 records (step 3500, 75 °C, SM 1897 MHz, step 15.44 s: no slowdown); max 79 °C; SM ≥ 1890 MHz |
| seed | 0 |

σ = 0.0030 (twelve `small` pairs; 2σ 0.0059, 4σ 0.0119).

### Held-out trajectory against `132` (same seed, same steps)

| step | 100 | 700 | 1300 | 1900 | 2500 | 3100 | 3700 | 4300 | 4768 |
|---|---|---|---|---|---|---|---|---|---|
| `176` val loss | 5.7036 | 3.6462 | 3.4012 | 3.2850 | 3.2104 | 3.1571 | 3.1197 | 3.0952 | 3.0904 |
| Δ vs `132` | +0.0050 | **−0.0055** | +0.0039 | +0.0089 | +0.0096 | +0.0112 | +0.0115 | +0.0115 | **+0.0126** |

L8b leads early and crosses `132` between steps 700 and 1300 (0.37–0.68 B tokens), and the gap widens to +0.012 by 1.5 B tokens, where it stays. `175` ended at 1 B tokens, just after the crossing. That is why it matched `110`.

### wiki-lead against `132` (`wiki_lead_compare_vs_132.json`; base = `046`'s covered re-score of `132`)

| split | tokens | `132` L5 | `176` L8b | Δ (floor) | beats? | `175` vs `110` at `screen` |
|---|---|---|---|---|---|---|
| in-table first sentence, **covered** (ADR-059's clause) | 55,516 | 2.8950 | 2.8674 | **−0.0276** (0.0146) | **yes** | −0.0369 |
| in-table first sentence, own retrieved | 54,597 | 2.8833 | 2.8549 | −0.0285 (0.0154) | yes | −0.0383 |
| in-table first sentence, full | 70,952 | 2.8422 | 2.7976 | −0.0446 (0.0135) | yes | −0.0541 |
| in-table rest of lead | 77,911 | 3.1383 | 3.1327 | **−0.0057** (0.0125) | **no** | −0.0105 |
| out-of-table lead (control) | 148,298 | 2.9866 | 2.9632 | −0.0234 (0.012) | yes | −0.0276 |
| first sentence, not covered (derived; branch zero) | 15,436 | 2.6527 | 2.5466 | −0.1061 | | −0.1160 |

The compare ran before `adr059` was merged, so its file has no `verdict_h17_covered` field. The field's value follows from the file's own covered row under ADR-059 §2: *weakens*, because the held-out Δ is past 4σ. The full-split `verdict_h17` in the file is also *weakens*. Coverage is 78.2 % and accuracy 98.3 %, as in `045`.

On the uncovered tokens, `176`'s 2.547 sits between the two L5 seeds (2.653 and 2.359, `046`). So the ≈ 0.1 advantage there is within L5's own seed spread (I41).

### Usage probes (random table rows in place of the neighbours)

| | `175` (`screen`) | `176` (`small`) |
|---|---|---|
| reliance, own retrieved first-sentence tokens (floor 0.0154) | +0.0621 | **+0.0514** (reads) |
| reliance, covered / full first sentence | +0.0618 / +0.0483 | +0.0511 / +0.0400 |
| reliance, rest of lead / control | +0.0412 / +0.0395 | +0.0297 / +0.0305 |
| reliance, FineWeb-Edu held-out | +0.0339 | +0.0205 |
| α entropy (of ln 8 = 2.079) | 1.982 (95.3 %) | 1.894 (**91.1 %**) |
| table rows retrieved at least once (976 held-out windows) | 13.7 % | 13.7 % |

## Interpretation

At `small` the retrieval still does what `175` showed.
- **The model reads the neighbours.** Random rows cost +0.051 on own-retrieved first-sentence tokens, 3.3× the floor, and α is sharper than at `screen` (91 % of uniform against 95 %).
- **Loss on the stored sentence falls.** Covered first-sentence tokens are 0.028 below `132`, about twice the floor and 14× `046`'s covered seed spread.

The gain is smaller than at `screen` (−0.037 against `110`), and the model leans on the neighbours less everywhere: +0.020 on FineWeb-Edu against +0.034.

The cost on general text grows with tokens. The held-out gap to the same-seed L5 run opens after 0.4–0.7 B tokens and settles at +0.012 to +0.013 (4.2σ). The 0.6 GFLOP budget keeps the L8b experts at width 704, against L5's wider ones. At `screen` the neighbours paid that back; at 2.5 B tokens they do not. This is the same pattern as L5's tie with dense at `screen`, which did not survive `small` either (`132`/`133`).

The rest of the lead is −0.006, under half its floor and under half `046`'s seed spread there. So the knowledge the memory supplies is confined to the sentence it holds, and it does not spread to the rest of the article.

Per seed, H17 is **weakened**, by the held-out clause. The verdict is the pair's (ADR-058 §3, read under ADR-059): `177` against `133`, then `--compare-pair`. `176` alone is +0.0156 above the L5 pair mean (3.0748). For the pair's Δ to fall back inside 4σ, `177` would have to finish below 3.0830, and below 3.0710 to reach the 2σ that *supports* needs. The recorded limits stand: vector values, corpus mismatch, and the retrieval cost outside "matched active FLOPs" (≈ 33 % of the step, 0.69× L5's throughput).
