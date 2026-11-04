# 174-l8q-screen — L8q (ADR-055: L8 plus a query loss distilled from the frozen BGE teacher on the preceding chunk) = **3.3457**, +0.0129 vs L5 `110` (4.3σ). The query now passes ADR-055's mechanism bar, but no wiki-lead split beats `110` → **H17 weakened**. By ADR-055 §6 this time the verdict bears on H17 itself, not only on the lookup, though the margin by which the lookup "works" is thin

`python -m scripts.train.train --rung L8q --id 174-l8q-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · `queue_screen_16`, gate `8e26bd4` (ruff + unit tests), 2026-11-04 11:25:34 → 19:07:15 UTC+1 · table `040` (hashes in `wiki_lead.json`), teacher BGE-base-en-v1.5 at `040`'s revision · config in `config.yaml`, per-step log in `log.jsonl` (`train.log` is gitignored and stays on disk).
Then, on GPU 0 with the rig otherwise idle (19:07–19:10):
- `python -m scripts.analysis.eval_wiki_lead --id 174-l8q-screen --probes` → `wiki_lead.json`;
- `--compare 174-l8q-screen 110-l5-screen` → `wiki_lead_compare.json`;
- `--compare 174-l8q-screen 173-l8-screen` → `wiki_lead_compare_vs_173.json`;
- `python -m scripts.analysis.probe_l8_lookup --id 174-l8q-screen --out 174-l8q-screen --teacher-windows 32` → `diag.json` (`042`'s diagnostic plus ADR-055 §5's teacher agreement; 23 s, peak 3.03 GiB).

**Bears on:** H17 ("per-iteration knowledge memory from a clean pipeline lowers loss on knowledge-heavy evals at matched active FLOPs with ≤ 0.5 % regression elsewhere"), and on I40. All readings are ADR-055 §6's, set before the run, at one seed. The floor per wiki-lead split is 4 × max(0.0030, bootstrap SE over articles). The general loss is read under ADR-013 with σ = 0.0030 over twelve pairs (2σ 0.0059, 4σ 0.0119).

**Verdicts:**
- **Mechanism (I40): the query carries the article,** by ADR-055's reading. At iterations 6, 7 and 8 the median own-row and own-column ranks are both ≤ 256 (251/251, 227/230, 232/237), and lookup accuracy is 1.6–1.9 × 10⁻³, 52–60× chance. In `173` both were at chance.
- **H17 against `110`: weakens, on both of ADR-055's grounds.** Neither in-table split beats the floor (−0.0071 against 0.0133, and +0.0259). The FineWeb-Edu Δ is +0.0129, which is above 4σ.
- **Combined (ADR-055 §6):** the mechanism works, but H17 is not supported. So this is evidence on H17 itself: at `screen`, a lookup that finds the article more often than chance does not lower loss on that article.
- **The limit, stated rather than read into the rule:** "works" here means the own cell is among the 32 retrieved for 0.19 % of first-sentence tokens. The key-side ceiling is 6.45 % (below). The run rules out "a query that cannot find anything". It does not rule out a lookup that hits often.

## Numbers

| | |
|---|---|
| config (derived) | as `173`: expert width **704**, memory 2²⁰ cells on 1024 × 1024, $d_m$ 128, $k_m$ 32; plus `query_aux` $\lambda_q$ 0.1, chunk 64, query inputs detached |
| params | 74,686,849 total, 50,110,849 non-embedding (as `173`; the teacher is training-only and not counted) |
| tokens | 999,817,216 in 1907 steps |
| **final val loss** (full held-out, 5606 windows) | **3.3457** nats |
| final train loss (last step, main head) | 3.3132 |
| $\mathcal{L}_q$ (logged per step, mean over halves, iterations and valid tokens) | 6.93 at step 1 (= ln 1024, chance) → 4.58 at step 200 → 3.64 at the end (min 3.38) |
| router (last step) | aux 0.0079, load 0.120–0.129, 0 dead experts |
| throughput | median **36,495 tok/s** aggregate (`173` 42,322 → **0.86×**; `110` 48,553 → 0.75×; rig `043` 36.1 k) |
| teacher | median 0.148 s per micro-batch, ≈ 16 % of the 14.37 s step (as `043`) |
| peak memory per GPU | 5.72 GiB (as `043`) |
| wall-clock | 27,692 s = **7.69 h** (`043`'s estimate 8.0 h; `173` 6.67 h → 1.15×), of which 315 s are in-training evals. Post-run evals and the probe took 3.3 min |
| thermal | `n_thermal` = 0 in all 190 records; max 79 °C; SM clock ≥ 1897 MHz throughout |
| seed | 0 |

### General held-out loss (H17's "elsewhere")

| run | rung | val loss | Δ (174 − run) | reading |
|---|---|---|---|---|
| `110` | L5 (matched arm) | 3.3328 | **+0.0129** (+0.39 %, 4.3σ) | past 4σ: regression |
| `173` | L8 (same width) | 3.3438 | +0.0019 (0.6σ) | the query loss costs nothing measurable |
| `100` | L0 dense ($L_{ref}$) | 3.336 | +0.0097 (3.2σ) | |

The eval script's own 2 M-token held-out loss gives 3.3421 vs `110`'s 3.3284 (+0.0138, past 4σ) and vs `173`'s 3.3400 (+0.0021). ADR-053's literal 0.5 % rule would call +0.39 % "within", which is the slip recorded at `173`. The verdict is *weakens* under both rules.

### wiki-lead (ADR-053 §7; 2,000 in-table and 2,000 out-of-table articles, ≤ 256 tokens, lead tokens only)

| split | tokens | `110` L5 | `173` L8 | `174` L8q | Δ vs `110` (floor) | Δ vs `173` (floor) |
|---|---|---|---|---|---|---|
| in-table, first sentence (in the table) | 70,952 | 3.1307 | 3.1328 | 3.1236 | **−0.0071** (0.0133) no | −0.0093 (0.0130) no |
| in-table, rest of lead | 77,911 | 3.4080 | 3.4320 | 3.4339 | **+0.0259** (0.012) no, worse | +0.0019 (0.012) no |
| out-of-table lead (control) | 148,298 | 3.2613 | 3.2748 | 3.2726 | +0.0112 (0.012) no | −0.0022 (0.012) no |

Bootstrap SEs vs `110`: 0.0033, 0.0029 and 0.0022.

### Mechanism: the `042` diagnostic on `174` (70,952 in-table first-sentence tokens per iteration)

| iteration | own-cell rank, median (chance 524,288) | own row / column rank, median (chance 512) | lookup accuracy (chance 3.1 × 10⁻⁵) | ≤ 256 and ≥ 10× chance |
|---|---|---|---|---|
| 1 | 445,679 | 460 / 441 | 1.3 × 10⁻⁴ | no |
| 2 | 320,793 | 375 / 360 | 5.4 × 10⁻⁴ | no |
| 3 | 251,017 | 319 / 315 | 1.07 × 10⁻³ | no |
| 4 | 225,851 | 296 / 295 | 1.42 × 10⁻³ | no |
| 5 | 268,306 | 331 / 325 | 8.7 × 10⁻⁴ | no |
| 6 | 170,949 | **251 / 251** | **1.62 × 10⁻³** | **yes** |
| 7 | 147,790 | **227 / 230** | **1.86 × 10⁻³** | **yes** |
| 8 | 153,539 | **232 / 237** | **1.79 × 10⁻³** | **yes** |

Lookup accuracy comes from `wiki_lead.json`. The probe's own top-32 share agrees within 7 % (1.7–2.0 × 10⁻³ at iterations 6–8). `173` had 501–525 / 503–515 for row and column, and 1.4–8.5 × 10⁻⁵ for accuracy. At iteration 7, 1.86 × 10⁻³ is 132 of 70,952 tokens.

### Teacher agreement (ADR-055 §5; 32 held-out FineWeb-Edu windows, 54,400 valid positions per iteration)

| iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| top-1 row = teacher's | 12.7 % | 19.6 % | 23.1 % | 25.2 % | 26.5 % | 27.7 % | 28.6 % | 30.0 % |
| top-1 column = teacher's | 10.3 % | 16.1 % | 19.6 % | 22.0 % | 23.0 % | 24.9 % | 26.1 % | 27.2 % |
| both | 5.2 % | 8.0 % | 9.9 % | 11.2 % | 11.9 % | 12.7 % | 13.4 % | 14.3 % |

Chance is 0.098 % per half. `173`'s baseline (rig `043`) was ≤ 1.8 × 10⁻⁵ for rows and 0.94–1.32 × 10⁻³ for columns. The query learned its target. It predicts the previous chunk's PQ code 100–300× above chance, and it does better with depth.

### Keys, maps and gradient (`042`'s readings 2–4 on `174`)

| | `173` (`042`) | `174` |
|---|---|---|
| own-key oracle in top 32 (2,000 in-table cells) | 7.8 % | **6.45 %** |
| own-key rank, median | 945 | 3,976 |
| own-key row / column rank per half, median | 4 / 25 | 49 / 18 |
| sub-key norm $\lVert h^{1,2}(\mu)\rVert$, median | 0.17 / 0.17 | 0.67 / 0.54 |
| $W_m$ singular values (top … min), effective rank of 128 | 2.18 … 0.35, 121 | 8.41 … 0.24, **72.7** |
| $h^1$, $h^2$ effective rank of 64 | 61 | 49 |
| $W_v$ top three singular values, effective rank of 768 | 3.09, 1.50, 1.33; 173 | 2.86, 1.57, 1.28; 176 |
| top-1 − top-32 score, mean over iterations 1–8 | 0.58 → 1.22 | 1.05 → 1.72 (peak 1.83 at iteration 7) |
| $\lVert g\rVert / \lVert w\rVert$ from the main loss: $W_m$ / $h^{1,2}$ / $W_v$ / experts | 7.4e-4 / 2.9e-4 / 3.0e-2 / 1.5e-2 | 1.7e-4 / 3.1e-4 / 3.0e-2 / 1.5e-2 |
| ‖Mem‖ per iteration | 1.28–1.35 | 1.35–1.36 |

The gradient row is the main loss only; the probe passes no teacher target, so $\mathcal{L}_q$'s gradient into $W_m$ and $h^{1,2}$ is not in it. Cross-batch coherence is again at the 1/√8 noise ratio for every group (0.36–0.39), so reading 4 stays unresolved.

### Usage probes (ADR-053 §8)

| probe | `173` | `174` |
|---|---|---|
| **reliance**: random cells in place of the retrieved ones, Δ loss | in_first +0.0103, in_rest +0.0063, out_lead +0.0078, FineWeb +0.0025 | in_first **−0.0030**, in_rest +0.0062, out_lead +0.0018, FineWeb **+0.0127** |
| α entropy (of ln 32 = 3.466) | 3.451 (99.6 %) | 3.176 (**91.6 %**) |
| table use (cells retrieved at least once, 976 held-out windows, all iterations) | 6.1 % | 7.0 % |

## Interpretation

The query loss did what ADR-055 built it to do. $\mathcal{L}_q$ fell from chance (6.93) to 3.64. On held-out FineWeb-Edu the query's top-1 row matches the teacher's code of the preceding chunk 13–30 % of the time, against 0.1 % chance. On in-table first sentences, the article's own row and column move from chance (≈ 512) to a median of about 230 by iteration 7, and its own cell is among the 32 retrieved 52–60× more often than chance. That passes ADR-055's mechanism bar at iterations 6–8. The softmax sharpens (α 91.6 % of uniform, against 99.6 %).

It did not buy any loss. Against `173`, the same architecture without the query loss, every wiki-lead split is within its floor: −0.009 on the first sentence, +0.002 on the rest, −0.002 on the control. FineWeb-Edu is +0.0019 (0.6σ). Against `110`, the in-table first sentence is −0.007 (half the floor), the rest of the lead +0.026, and FineWeb-Edu +0.0129, past 4σ. So H17 is **weakened** on both of ADR-055's grounds.

The reliance probe says what the retrieved content is used for. On general text the model now depends on it about five times more than `173` did: random cells cost +0.013 on FineWeb, against +0.003. On the in-table first sentence, random cells are no worse (−0.003, within one SE). So the retrieval feeds the model a topic signal that it uses everywhere, roughly RETRO's "related neighbours", which is the risk ADR-055 recorded. It does not feed article facts where they would help. The −0.009 first-sentence gain over `173` is therefore not the retrieved sentence's content, because swapping the cells does not remove it. It sits below the floor and stays unexplained.

By ADR-055 §6, this is evidence on H17 itself: a lookup that finds the article above chance does not help at `screen`. Its weight is limited by how rarely the lookup actually hits, 0.19 % of tokens. The keys cap even a perfect query at 6.45 %, which is worse than `173`'s 7.8 %. The key maps grew (sub-key norms ×3–4) and lost rank (49 of 64) while serving the teacher's topical target. So what is ruled out at `screen` is a query-side fix with these keys. A lookup that retrieves the right sentence for a large share of tokens has not been tested.

The general-loss regression vs `110` (+0.013) is as in `173` (+0.011). It remains ADR-053 §6's confound of the narrower experts and a branch that adds little, and only an L5 control at width 704 would separate the two.

No `small` pair follows, since §6 triggers one only on *supports*. The next step for L8 (the key side, I40's reading 2, or closing L8 at `screen`) is the user's call.
