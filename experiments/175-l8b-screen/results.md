# 175-l8b-screen — L8b at `screen`

`python -m scripts.train.train --rung L8b --id 175-l8b-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** H17 (P17, per-iteration knowledge memory). **Silent** at one seed by ADR-057 §7 (ADR-055 §6's rule against `110`). The in-table first sentence beats its floor, −0.0541 against 0.0138. The rest of the lead misses its floor, −0.0105 against 0.0121. FineWeb-Edu is within 2σ, Δ −0.0007. ADR-057's readings: the pipeline is **precise** (`045`: 98.3 % of covered tokens retrieve their own sentence) and the model **reads** the neighbours (reliance on own-retrieved first-sentence tokens is +0.0621 against a floor of 0.0148). No `small` pair follows, since ADR-031 triggers one only on *supports*. L8b's first screen is the first L8-family run to beat `110` anywhere.

## Numbers

| | |
|---|---|
| params | 74.7 M total, 50.2 M non-embedding (`110` 77.5 / 52.9 M; `174` 74.7 / 50.1 M) |
| tokens | 1.000 B in 1907 steps |
| **final val loss** (full held-out, 5606 windows) | **3.3321** nats |
| final train loss (last step, main head) | 3.3009 |
| router (last step) | aux 0.0079, load 0.122–0.129, 0 dead experts |
| throughput | median **34,140 tok/s** aggregate (`174` 36,495 → **0.94×**; `110` 48,553 → 0.70×; rig `045` 33.8 k) |
| retrieval (encoder + exact top-8 search) | median 0.321 s per micro-batch, 16 per step → ≈ 5.1 s of the 15.36 s step (**≈ 33 %**, as `045`) |
| peak memory per GPU | 5.72 GiB (as `045`) |
| wall-clock | 30,036 s = **8.34 h** (`045`'s estimate 9.1 h; `174` 7.69 h → 1.08×), of which 678 s are in-training evals. Post-run evals took 5.0 min, and the `174` covered re-score below 1.7 min |
| thermal | `n_thermal` = 0 in all 190 records; max 79 °C; SM clock ≥ 1897 MHz throughout |
| seed | 0 |

σ = 0.0030 (twelve `small` pairs; 2σ 0.0059, 4σ 0.0119).

### General held-out loss (H17's "elsewhere")

| run | rung | val loss | Δ (175 − run) | reading |
|---|---|---|---|---|
| `110` | L5 (matched arm) | 3.3328 | **−0.0007** (0.2σ) | within 2σ: no regression |
| `174` | L8q (same width 704) | 3.3457 | −0.0136 (4.5σ) | |
| `173` | L8 (same width 704) | 3.3438 | −0.0117 (3.9σ) | |
| `100` | L0 dense ($L_{ref}$) | 3.336 | −0.0039 (1.3σ) | |

The eval script's own 2 M-token held-out loss gives 3.3289 vs `110`'s 3.3284 (+0.0005) and vs `174`'s 3.3421 (−0.0133). The regression that `173` and `174` carried against `110` (+0.011 / +0.013) is gone. Swapping in random table rows costs +0.0339 on this held-out set (below), so the branch is doing work on FineWeb-Edu too.

### wiki-lead (ADR-053 §7; 2,000 in-table and 2,000 out-of-table articles, ≤ 256 tokens, lead tokens only)

| split | tokens | `110` L5 | `174` L8q | `175` L8b | Δ vs `110` (floor) | Δ vs `174` (floor) |
|---|---|---|---|---|---|---|
| in-table, first sentence (in the table) | 70,952 | 3.1307 | 3.1236 | **3.0766** | **−0.0541** (0.0138) **yes** | −0.0469 (0.0137) yes |
| in-table, rest of lead | 77,911 | 3.4080 | 3.4339 | **3.3974** | **−0.0105** (0.0121) **no** | −0.0365 (0.012) yes |
| out-of-table lead (control) | 148,298 | 3.2613 | 3.2726 | 3.2337 | −0.0276 (0.012) yes | −0.0388 (0.012) yes |

Bootstrap SEs vs `110`: 0.0035, 0.0030, 0.0023. The compare files are `wiki_lead_compare.json` (base: `045`'s covered re-score of `110`, which reproduces `110`'s own file) and `wiki_lead_compare_vs_174.json`. The script's verdict vs `110` is `silent` under both the 2σ and the literal-0.5 % rules.

### Covered tokens (ADR-057 §6; the pipeline's masks, identical for every run)

The pipeline reproduces `045` exactly: coverage 78.2 % (55,516 of 70,952 first-sentence tokens have neighbours) and accuracy 98.3 % on covered tokens (54,597 retrieve their own sentence; 0.7695 of all first-sentence tokens, the same at all eight iterations, since every iteration reads the same neighbours). `174` was re-scored on the same masks after the queue (`wiki_lead_174_covered.json`, written here because `174`'s directory is closed).

| subset of in-table first sentence | tokens | `110` | `174` | `175` | 175 − 110 (floor) | 175 − 174 |
|---|---|---|---|---|---|---|
| covered | 55,516 | 3.1963 | 3.2150 | **3.1594** | **−0.0369** (0.0140) yes | **−0.0556** |
| own sentence retrieved | 54,597 | 3.1856 | 3.2030 | **3.1473** | **−0.0383** (0.0148) yes | **−0.0557** |
| **not covered** (before the first boundary, branch exactly zero) | 15,436 | 2.8949 | 2.7948 | 2.7789 | −0.1160 | −0.0159 |

Uncovered values are derived from the full and covered splits.

**The split's gain is in two parts.**
- **Covered tokens:** they carry −0.0288 of the −0.0541 against `110`. That is the retrieval's contribution.
- **Uncovered tokens:** they carry −0.0252. On these tokens `175`'s memory branch is exactly zero: random rows leave them at 2.7790 against 2.7789. `174` has the same advantage there over `110` (−0.100). So this part belongs to the L8-family models, not to retrieval. Its cause is not measured (I41).

The covered tokens alone still beat the floor against `110` (2.6×). So the first-sentence half of the verdict does not depend on the uncovered part.

### Usage probes (ADR-057 §6)

| probe | `174` | `175` |
|---|---|---|
| **reliance**: random table rows in place of the retrieved neighbours, Δ loss | in_first −0.0030, in_rest +0.0062, out_lead +0.0018, FineWeb +0.0127 | in_first **+0.0483**, in_rest **+0.0412**, out_lead **+0.0395**, FineWeb **+0.0339**; covered **+0.0618**, own retrieved **+0.0621** (floor 0.0148 → **reads**) |
| α entropy | 3.176 of ln 32 (91.6 %) | 1.982 of ln 8 = 2.079 (**95.3 %**) |
| table use (rows retrieved at least once, 976 held-out windows) | 7.0 % of cells | 13.7 % of rows |

## Interpretation

The pipeline ADR-057 built works as `045` measured it. 78 % of in-table first-sentence tokens have neighbours, and 98 % of those have the article's own sentence among their eight. The model learned to read them. Swapping in random table rows costs +0.062 on own-retrieved first-sentence tokens, four times the floor, and +0.034 even on FineWeb-Edu. With the true neighbours it gives up nothing on general text: −0.0007 against `110` on the full held-out set. That removes the +0.011 / +0.013 regression that `173` and `174` carried at the same width. So the narrower experts' cost is paid back by what the neighbours supply.

**Against `110`, H17 is silent by ADR-057 §7.**
- **In-table first sentence:** beats its floor, −0.0541. Half of that sits on tokens before the first retrieval boundary, which `174` gains as well. The covered half alone (−0.037 on 55 k tokens) still beats the floor.
- **Rest of the lead:** misses its floor, −0.0105 against 0.0121, by 0.0016. This at one seed, where the bootstrap SE is 0.0030.
- **Out-of-table control:** gains as well, −0.0276.

Reliance is similar across the three wiki-lead splits: +0.040 to +0.048 on the full splits, and +0.062 where the own sentence is retrieved. So most of what the neighbours carry is not the stored fact itself. It is that they are Wikipedia sentences on a related subject. The own sentence adds roughly +0.02 of reliance on top of the control's level. That is a cross-split comparison, indicative only.

ADR-057 §7 set a combination for "precise, read, and H17 not supported": that a memory handed the right sentence as a vector does not lower loss. Its wording assumed a null result. Here, loss on the stored sentence falls by 0.037–0.056 on covered tokens, so that sentence does not describe this run. Silent here means one split short, not no effect. So this is not recorded as evidence against H17; it is recorded as silent, leaning supports.

The recorded limits stand:
- **Vector values:** the gain is what a 768-d CLS vector can carry.
- **Corpus mismatch:** the model trained on FineWeb-Edu, and the store is Wikipedia.
- **Retrieval cost outside "matched active FLOPs":** the encoder and search, ≈ 0.65 GFLOP per token by ADR-057's estimate, against the model's 0.6. Measured: ≈ 33 % of the step, 0.70× `110`'s throughput.

The next step is the user's call. ADR-031's `small` pair triggers only on *supports*. A second `screen` seed or a `small` pair would decide whether the rest-of-lead split clears its floor. Either needs an ADR.
