# 173-l8-screen — L8 (ADR-053: a frozen Wikipedia lead-sentence table, 2²⁰ cells, $k_m$ = 32, expert width 704) = **3.3438**, +0.0110 vs L5 `110` (3.7σ). On wiki-lead it is no better than L5 on any split, and the lookup finds the article's own cell at chance → **H17 weakened by ADR-053 §10; by its last clause the cause is the lookup, not H17**

`python -m scripts.train.train --rung L8 --id 173-l8-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · `queue_screen_15`, gate `ba89ed2` (ruff + unit tests), 2026-11-03 22:36:07 → 2026-11-04 05:16:36 UTC+1 · table `040` (hashes in `wiki_lead.json`) · config in `config.yaml`, per-step log in `log.jsonl` (`train.log` is gitignored and stays on disk).
Then, on GPU 0 with the rig otherwise idle (05:16–05:21): `python -m scripts.analysis.eval_wiki_lead --id 110-l5-screen` (→ `../110-l5-screen/wiki_lead.json`), `--id 173-l8-screen --probes` (→ `wiki_lead.json`), `--compare 173-l8-screen 110-l5-screen` (→ `wiki_lead_compare.json`).

**Bears on:** H17 ("per-iteration knowledge memory from a clean pipeline lowers loss on knowledge-heavy evals at matched active FLOPs with ≤ 0.5 % regression elsewhere"). The reading is ADR-053 §10, set before the run, at one seed. The floor per wiki-lead split is 4 × max(0.0030, bootstrap SE over articles). The general loss is read under ADR-013 with σ = 0.0030 over twelve pairs (2σ 0.0059, 4σ 0.0119).

**Verdict:** **weakens.** Neither in-table loss beats `110` by the floor: both are *worse*, by +0.0021 and +0.0240. §10's last clause applies: reliance is low, lookup accuracy is at chance and α is near-uniform. So the cause sits with the lookup (or the table), not with H17. The run shows that this L8 does not retrieve. It does not show that a memory that retrieves would not help. That question is I40.

## Numbers

| | |
|---|---|
| config (derived) | expert width **704** (L5 896; ADR-053 §6's matched-FLOPs re-solve), memory 2²⁰ cells on 1024 × 1024, $d_m$ 128, top-32 per half → $k_m$ 32 |
| params | 74,686,849 total, 50,110,849 non-embedding (−2,793,472 vs `110`; the 1.5 GiB table is data) |
| tokens | 999,817,216 in 1907 steps |
| **final val loss** (full held-out, 5606 windows) | **3.3438** nats |
| final train loss (last step, main head) | 3.3112 |
| router (last step) | aux 0.0079, load 0.123–0.127, 0 dead experts |
| throughput | median **42,322 tok/s** aggregate (`110` 48,553 → **0.87×**; rig `041` predicted 0.85×) |
| peak memory per GPU | 5.31 GiB (as `041`) |
| wall-clock | 24,021 s = **6.67 h** (`041`'s estimate was 6.8 h plus evals; `110` 5.79 h → 1.15×), of which 315 s are in-training evals. Post-run evals took 4 min |
| thermal | `n_thermal` = 0 in all 190 records; max 77 °C; SM clock ≥ 1897 MHz throughout |
| seed | 0 |

### General held-out loss (H17's "elsewhere")

| run | rung | val loss | Δ (173 − run) | |
|---|---|---|---|---|
| `110` | L5 (matched arm) | 3.3328 | **+0.0110** (+0.33 %, 3.7σ) | between 2σ and 4σ: no detectable regression is *not* shown, nor is one past 4σ |
| `100` | L0 dense ($L_{ref}$) | 3.336 | +0.0078 (2.6σ) | |

The eval script's own 2 M-token held-out loss gives 3.3400 vs 3.3284 (+0.35 %).

**A recorded correction to the verdict code, not a silent fix.** ADR-053 §10 says "regresses by ≤ 0.5 %", and `eval_wiki_lead.py` tests 0.5 % of $L$ literally (`"within": true`). ADR-013 reads every "0.5 %" threshold as Δ ≤ 2σ, which is the same slip the `137`–`143` H8 entries made (corrected 2026-10-03). Under ADR-013, +0.0110 is above 2σ but below 4σ. So the regression clause is neither met nor failed at one seed. The verdict does not change, because §10's "neither in-table loss beats the floor" already gives *weakens*.

### wiki-lead (ADR-053 §7; 2,000 in-table + 2,000 out-of-table articles, ≤ 256 tokens, lead tokens only)

| split | tokens | `110` L5 | `173` L8 | Δ | bootstrap SE | floor | beats floor |
|---|---|---|---|---|---|---|---|
| in-table, first sentence (in the table) | 70,952 | 3.1307 | 3.1328 | **+0.0021** | 0.0033 | 0.0134 | no |
| in-table, rest of lead | 77,911 | 3.4080 | 3.4320 | **+0.0240** | 0.0030 | 0.0121 | no (worse by 2× the floor) |
| out-of-table lead (control) | 148,298 | 3.2613 | 3.2748 | **+0.0135** | 0.0023 | 0.0120 | no (worse by 1.1× the floor) |

### Usage probes (ADR-053 §8)

| probe | value | reading |
|---|---|---|
| **reliance**: random cells in place of the retrieved ones, Δ loss | in_first **+0.0103**, in_rest +0.0063, out_lead +0.0078, FineWeb held-out +0.0025 | the retrieved content is worth ≤ 0.01 nats over noise cells. That is low, and nearly as large on the control as on the in-table first sentence |
| **lookup accuracy** (in-table first-sentence tokens whose own article's cell is among the 32 retrieved), iterations 1–8 | 7.0, 7.0, 5.6, 1.4, 2.8, 2.8, 1.4, 8.5 × 10⁻⁵ (5, 5, 4, 1, 2, 2, 1, 6 of 70,952 tokens) | mean 4.6 × 10⁻⁵ against chance 32 / 2²⁰ = 3.1 × 10⁻⁵: **at chance at every iteration** |
| **α entropy** | 3.451 of a maximum ln 32 = 3.466 (99.6 %) | the softmax over the 32 candidates stays almost uniform |
| **table use** (cells retrieved at least once on 976 held-out windows, all iterations) | **6.1 %** of 2²⁰ | ≈ 5 × 10⁸ draws land on ≈ 64 k cells, so the queries concentrate on a small region of the grid |

## Interpretation

L8 as ADR-053 built it trains without trouble. It fits in 5.3 GiB, runs at 0.87× L5's speed with no throttling, and routing stays healthy. But it does not do what H17 needs: it never learns to look anything up. At every middle iteration, the share of first-sentence tokens whose own article is among the 32 retrieved cells is at chance (4.6 × 10⁻⁵ against 3.1 × 10⁻⁵). The weights over those 32 stay 99.6 % uniform, and the queries keep to about 6 % of the table. What the branch delivers is a near-uniform average of 32 sentences from a narrow region, which acts as a context-dependent bias. Swapping it for random cells costs 0.003 nats on FineWeb-Edu and 0.006–0.010 on wiki-lead, with the in-table first sentence (+0.010) barely ahead of the out-of-table control (+0.008). On the evals, L8 is behind L5 on every split: +0.002 on the in-table first sentence, +0.024 on the rest of the lead, +0.013 on the control, and +0.0110 (3.7σ) on the general held-out loss. So by ADR-053 §10 the result **weakens** H17. By the same rule's last clause, low reliance places the cause on the lookup, not on per-iteration memory as such. ADR-053 §6's confound cannot be separated here. The +0.011 general gap is some mix of the narrower experts (704 vs 896, −3.5 M expert params) and a branch that contributes almost nothing. Only an L5 control at width 704 would split the two, and it bears on nothing if the lookup stays at chance. One post-hoc observation, not a verdict input: the in-table first-sentence gap (+0.002) is smaller than the control's (+0.013), ≈ 2.8 unpaired SEs apart. Reliance does not explain it, and the rest of the same articles' leads is the worst split (+0.024), so it is not the articles being easier. It stays unexplained. Why the query never aligns with the product keys is the open question, I40. No `small` pair follows, since §10 triggers one only on *supports*. The next step for L8 is the user's call.
