# 181-l8b-pv-medium — L8b-pv at `medium`

`python -m scripts.train.train --rung L8b-pv-medium --id 181-l8b-pv-medium --tokens 7e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · gate `c8f3ce0` (`queue_medium_1`) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** H17, by ADR-059's rule on `181` against `180` L5-medium (ADR-061 §6–§7), at one seed, σ = 0.0030 borrowed from `small`. **Verdict: H17 *silent* at `medium`.**
- The covered first sentence beats its floor, but only just: −0.0173 against 0.0145.
- The rest of the lead misses its floor: −0.0040 against 0.0120.
- Held-out is +0.0026, within 2σ.

By ADR-061 §7 ("otherwise silent"), the `small` pair `178`/`179` (*supported*) stays the H17 result. Nothing follows automatically, and any further H17 arm needs a new ADR.

## Numbers

| | |
|---|---|
| shape | `06 §3` `medium`: $d$ = 1024, 16/4 heads, $E$ = $F$ = 2, skeleton $d_{ff}$ 4096, fixed $r$ = 16, $N_e$ = 8, $k$ = 2, expert width **1856** (`180`: 1920), 2.0 GFLOP per token. The memory is as in `178`/`179`: table 2²⁰, retrieval every 16 tokens from ≤ 32 tokens, $K$ = 8, $d_m$ 128, maps per boundary |
| params | 143.1 M total, 110.4 M non-embedding |
| tokens | 7.000 B in 13,351 steps |
| **final val loss** (full held-out, 5606 windows) | **2.7276** nats (`180` 2.7250, so **Δ = +0.0026**, 0.9σ) |
| final train loss (last step, main head) | 2.7176 (`180` 2.7132) |
| throughput | median **15,979 tok/s** aggregate, **0.847×** `180`'s 18,874. `049`'s smoke ratio was 15.6 k / 18.5 k = 0.84. `retrieval_s` is 0.37 s of a 33.0 s step (1.1 %), so most of the cost lies outside that timer |
| peak memory per GPU | 7.85 GiB (as `049`) |
| wall-clock | 444,707 s = **123.5 h** (projection ≈ 123 h), of which 6,000 s are in-training evals. The post-run wiki-lead eval with probes took 8.0 min |
| thermal | `n_thermal` = 2 in 1,335 records, one sample each at steps 3370 and 13060. SM min 1890 MHz, max 81 °C. Throughput is unaffected |
| routing (last step) | 0 dead experts, minimum effective experts 7.998 of 8, load 0.122–0.130 |
| seed | 0 |

### Held-out trajectory against `180` (in-training evals, same windows)

| step | 100 | 900 | 1700 | 2500 | 3300 | 4100 | 5700 | 7300 | 8900 | 10500 | 12100 | 12900 | 13351 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tokens (B) | 0.05 | 0.47 | 0.89 | 1.31 | 1.73 | 2.15 | 2.99 | 3.83 | 4.67 | 5.51 | 6.34 | 6.76 | 7.00 |
| `181` − `180` | −0.018 | −0.0097 | −0.0059 | −0.0022 | −0.0016 | +0.0004 | +0.0014 | +0.0023 | +0.0028 | +0.0031 | +0.0029 | +0.0037 | **+0.0025** |

The early lead closes by ≈ 2 B tokens. That is later than at `small`, where the crossing was at 0.4–0.7 B for L8b and the lead was still −0.002 at 2.5 B for L8b-pv. After the crossing, the gap grows slowly to +0.003 and holds there from ≈ 5 B tokens. It never comes near 4σ (0.0119).

### wiki-lead (`wiki_lead.json`, `wiki_lead_compare_vs_180.json`; floor = 4 × max(σ, bootstrap SE))

| split | tokens | `181` | `180` | Δ | floor | beats | `small` pair Δ (`178`/`179` vs `132`/`133`) |
|---|---|---|---|---|---|---|---|
| **in-table first sentence, covered** (clause 1) | 55,516 | 2.4173 | 2.4346 | **−0.0173** | 0.0145 | yes | −0.054 |
| in-table first sentence, own retrieved | 54,597 | 2.4054 | 2.4230 | −0.0175 | 0.0145 | yes | |
| in-table first sentence, full | 70,952 | 2.3293 | 2.3761 | −0.0468 | 0.0129 | yes | |
| **in-table rest of lead** (clause 2) | 77,911 | 2.7296 | 2.7335 | **−0.0040** | 0.0120 | **no** | −0.015 |
| out-of-table lead (control) | 148,298 | 2.5344 | 2.5571 | −0.0227 | 0.0120 | yes | −0.038 |
| FineWeb-Edu subset (976 windows) | | 2.7242 | 2.7211 | +0.0032 | | | |

Against the control:
- covered minus control = **+0.005** (`small` −0.016);
- rest minus control = +0.019 (`small` +0.023).

Retrieval is unchanged: coverage 78.2 %, accuracy on covered tokens 98.3 %. The retrieval masks are the same as in every L8b eval.

### Reliance (random cells − real cells; `random_cells` in `wiki_lead.json`)

| split | `181` | L8b-pv `small` (`178`/`179`) |
|---|---|---|
| own retrieved first-sentence tokens | **+0.0235** | +0.051 / +0.053 |
| covered / full first sentence | +0.0236 / +0.0184 | +0.051 / +0.040 |
| rest of lead / control | +0.0103 / +0.0101 | +0.028–0.030 / +0.029–0.031 |
| FineWeb-Edu held-out | +0.0052 | +0.020 |

α entropy is 1.867 nats of a 2.079 maximum (`small` 1.90–1.91), and 13.7 % of cells are used on held-out text, the same share as at `small`.

## Interpretation

**H17 is silent at `medium`.** The memory still helps on the one sentence it holds: the covered first sentence beats its floor at −0.017. That gain is a third of the `small` pair's −0.054, and smaller than the general gain on Wikipedia-lead text: the out-of-table control improves by −0.023. Covered minus control is +0.005. At `small` it was −0.016, so the sentence-specific margin the `small` pair had is gone at this size.

The rest-of-lead clause fails outright (−0.004 against 0.012). I42 already noted that this clause sits at its seed-noise floor, and a second seed would not move a −0.004 reading past 0.012.

On general text, L8b-pv pays a small cost that does not grow: held-out +0.0026 (0.9σ), steady at about +0.003 since 5 B tokens. ADR-061 §7's weakening branch (held-out past 4σ) is not reached, so the L8b-pv design does not close. By §7's silent branch, the `small` pair's *supported* stands as the H17 result, with this run recorded as its limit.

**The larger model reads the memory less.** Retrieval is the same as at `small`: 78 % coverage, 98 % accurate on covered tokens. α is about as sharp. But swapping in random cells now costs half as much as it did at `small`:
- +0.024 instead of +0.053 on the retrieved sentence;
- +0.010 instead of +0.03 on the control;
- +0.005 instead of +0.020 on FineWeb.

`180` shows why: the `medium` L5 already knows much of the table. Its covered first sentence starts 0.46 nats below `132`'s, more than its −0.35 on general text. A memory that stores what a model of this size already learns from the corpus has less to add. The −0.047 on the full first sentence (versus −0.017 on covered tokens) is mostly the uncovered tokens before the first retrieval boundary. At `screen`, that part was shared by `174` L8q, whose lookup does not retrieve the sentence (I41), so it is not credited to retrieval here.

**Cost.** L8b-pv runs at 0.85× L5's throughput at `medium` (`049`: 0.84×), at 7.85 GiB against 5.93. The measured retrieval call is 1 % of the step, so most of the 15 % falls outside that timer and is not attributed here. Converting this into an ADR-013 cost-per-quality statement needs the S-scenario model, which is not calibrated (I1).

**What this leaves open.** At two scales, the memory's specific value shrinks as the base model knows more (I44, opened with this write-up). Three things could change that reading, and all are the user's call by ADR:
- a table of facts the base does *not* learn from the corpus;
- a second `medium` seed (≈ 5 days, and it cannot rescue the rest clause);
- stopping H17 here, with the `small` pair as its result.

None is scheduled. The Embla chain (`queue_screen_18`–`25`) started at 23:59 on 2026-11-18 (182 L5d-nope).
