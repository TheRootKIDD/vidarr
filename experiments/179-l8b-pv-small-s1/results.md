# 179-l8b-pv-small-s1 — L8b-pv at `small`, seed 1, and the L8b-pv pair

`python -m scripts.train.train --rung L8b-pv --id 179-l8b-pv-small-s1 --tokens 2.5e+09 --micro-batch 4 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · gate `3a4e219` (`queue_small_12`) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** H17. This run completes ADR-060's pair. **The L8b-pv pair `178`/`179` *supports* H17 at two seeds** against the L5 pair `132`/`133`, under ADR-059's rule on pair means (ADR-058 §3). The rule's readings:
- in-table first sentence on covered tokens: **−0.0543**, beats its 0.0120 floor;
- rest of the lead: **−0.0148**, beats its 0.0120 floor by 0.0028;
- FineWeb-Edu held-out: **−0.0017**, within 2σ.

The full-split `verdict_h17` and the 0.5 % literal reading agree.

This is the first H17 *supports* reading in the L8 family. By ADR-060 §7, it makes `medium` for H17 **a question for the user, by ADR**; nothing follows automatically.

The support is narrow on the rest-of-lead clause, as set out below. Seed 1 alone against `133` is **silent**: covered −0.0522 beats, rest −0.0046 misses, held-out +0.0037.

## Numbers

| | |
|---|---|
| params | 77.1 M total, 52.5 M non-embedding (`177` L8b 74.7 / 50.2 M; `133` L5 77.5 / 52.9 M) |
| expert width | 832 (L8b 704; L5 896), ADR-060 |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0754** nats (`133` 3.0717 → **+0.0037**, 1.2σ; `177` 3.0836 → **−0.0082**, 2.7σ) |
| final train loss (last step, main head) | 3.0578 |
| throughput | median **34,556 tok/s** aggregate (`178` 34,576; `133` 0.70×) |
| retrieval | median 0.320 s per micro-batch, ≈ 5.1 s of the 15.17 s median step (≈ 34 %) |
| peak memory per GPU | 5.78 GiB |
| wall-clock | 73,949 s = **20.54 h**, of which 1,431 s are in-training evals. The post-run eval and compares took 5.0 min |
| thermal | `n_thermal` = 1 sample in 476 records (step 4000, 78 °C, SM 1897 MHz, step 14.98 s: no slowdown); SM ≥ 1897 MHz |
| seed | 1 |

σ = 0.0030 (twelve `small` pairs; 2σ 0.0059, 4σ 0.0119).

### Held-out trajectory (same steps)

| step | 700 | 1300 | 1900 | 2500 | 3100 | 3700 | 4300 | 4768 |
|---|---|---|---|---|---|---|---|---|
| `179` val loss | 3.6445 | 3.3909 | 3.2729 | 3.1962 | 3.1430 | 3.1050 | 3.0807 | 3.0754 |
| Δ `179` vs `133` | −0.0189 | −0.0003 | +0.0018 | +0.0014 | +0.0025 | +0.0025 | +0.0033 | **+0.0037** |
| Δ `179` vs `177` | −0.0205 | −0.0066 | −0.0073 | −0.0085 | −0.0081 | −0.0082 | −0.0083 | **−0.0082** |
| Δ pair vs L5 pair | −0.0209 | −0.0080 | −0.0042 | −0.0042 | −0.0026 | −0.0026 | −0.0019 | **−0.0017** |

`179` crosses `133` at ≈ 0.68 B tokens and drifts to +0.004. The pair's lead shrinks from −0.021 at 0.37 B tokens to −0.002 at 2.5 B and is still closing slowly: about +0.001 per 0.5 B tokens over the last 1.5 B.

The gain over L8b differs by seed: −0.020 at seed 0 but −0.008 at seed 1. The pair's width gain is −0.0140 (below).

### The pair against the L5 pair (`wiki_lead_compare_pair.json`; bases `046`'s covered re-scores of `132`/`133`)

| split | tokens | L5 pair | L8b-pv pair | Δ (floor) | beats? | L8b pair (`177`) |
|---|---|---|---|---|---|---|
| in-table first sentence, **covered** (ADR-059's clause) | 111,032 | 2.8940 | 2.8397 | **−0.0543** (0.0120) | **yes** | −0.0303 |
| in-table first sentence, own retrieved | 109,194 | 2.8825 | 2.8277 | −0.0548 (0.0120) | yes | |
| in-table first sentence, full | 141,904 | 2.8095 | 2.7347 | −0.0748 (0.0120) | yes | |
| in-table rest of lead | 155,822 | 3.1317 | 3.1170 | **−0.0148** (0.0120) | **yes** | +0.0030 |
| out-of-table lead (control) | 296,596 | 2.9654 | 2.9276 | −0.0379 (0.0120) | yes | −0.0074 |
| first sentence, not covered (derived; branch zero) | 30,872 | ≈ 2.506 | ≈ 2.357 | ≈ −0.149 | | |
| **FineWeb-Edu, full held-out (seed means)** | 5606 windows | 3.0748 | 3.0730 | **−0.0017** | **within 2σ** | +0.0122 |

Per seed:

| | covered first | rest | control | held-out | reading |
|---|---|---|---|---|---|
| `178` vs `132` | −0.0563 | −0.0249 | −0.0572 | −0.0072 | supports |
| `179` vs `133` | −0.0522 | **−0.0046** (misses) | −0.0185 | +0.0037 | silent |

### The width effect: L8b-pv pair against the L8b pair (`wiki_lead_compare_pair_vs_l8b.json`)

The two arms compute the same function (`048`) and differ only in expert width (832 against 704).

| split | Δ L8b-pv − L8b (floor 0.012) |
|---|---|
| covered first sentence | −0.0240 |
| rest of lead | −0.0177 |
| out-of-table control | −0.0305 |
| full first sentence | −0.0510 |
| FineWeb-Edu held-out | **−0.0140** (4.7σ) |

### Usage probes (random table rows; drawn once per segment under hoist, `048`)

| | `177` L8b | `179` L8b-pv | `178` L8b-pv |
|---|---|---|---|
| reliance, own retrieved first-sentence tokens | +0.0511 | **+0.0534** | +0.0525 |
| reliance, covered / full first sentence | +0.0509 / +0.0398 | +0.0532 / +0.0417 | +0.0523 / +0.0409 |
| reliance, rest of lead / control | +0.0279 / +0.0290 | +0.0300 / +0.0307 | +0.0275 / +0.0295 |
| reliance, FineWeb-Edu held-out | | +0.0195 | +0.0206 |
| α entropy (of ln 8 = 2.079) | | 1.907 (91.7 %) | 1.901 (91.4 %) |

Coverage is 78.2 % and accuracy on covered tokens 98.3 %, as in every L8b run.

## Interpretation

**The width was the cost.** L8b-pv computes L8b's function and spends the saved FLOPs on width. That takes the held-out set from +0.0122 above the L5 pair to −0.0017 (a gain of 0.014, 4.7σ), and every wiki-lead split moves by a similar amount. ADR-060's premise holds: L8b's general-text regression at `small` came from the expert width the branch displaced, not from retrieval interfering with the residual.

**What the memory itself does has not changed.** Reliance on the neighbours is +0.053 for both L8b-pv seeds and +0.051 for L8b. α is equally sharp, and retrieval is fixed. So the per-split gains of L8b-pv are L8b's gains plus the width shift, and with the general-text cost gone, ADR-059's rule now reads *supports*.

**How firm the reading is: the first-sentence clause is solid, the rest-of-lead clause is not.**
1. **Covered first sentence (−0.054).** This is 4.5× its floor and 27× `046`'s covered seed spread (−0.002). Both seeds agree (−0.056 / −0.052).
2. **Rest of the lead (−0.0148 against 0.0120).** It passes by 0.003, about one bootstrap SE. Seed 1 alone misses (−0.005), and L5's own seeds differ by 0.013 on this split. That is I42's recorded limit, and ADR-059 kept the clause anyway.
3. **The control.** The out-of-table control (Wikipedia leads that are *not* in the table) improves by −0.038, more than the rest-of-lead clause (−0.015) and not far short of the covered first sentence (−0.054). The rule has no control-difference term, so the verdict stands. But the specific margins are covered minus control ≈ −0.016 and rest minus control ≈ +0.023. Read that way, the memory's knowledge stays confined to the sentence it holds, as `176`/`177` found. The rest-of-lead pass is part of a general improvement on Wikipedia-lead text: the reliance probe shows the model reading related neighbours on every split (+0.03 on the control).
4. **The uncovered first-sentence tokens, where the branch is zero.** They improve by ≈ 0.15, inside L5's own 0.29 seed spread there. This is I42's pre-boundary noise; the clause excludes these tokens.

**General text.** The pair matches L5 (−0.0017) and does not beat it. It costs 0.70× L5's throughput: retrieval is ≈ 34 % of the step and sits outside "matched active FLOPs". The pair's lead over L5 is still shrinking slowly at 2.5 B tokens, which matters for any `medium` reading.

**Readings under ADR-060 §7.**
- The pair is *supports*, so H17 at `medium` (7 B tokens, one seed, FSDP) is the user's call, by ADR.
- The limits are recorded:
  - the rest-of-lead pass is marginal and not specific beyond the control;
  - the held-out lead is closing;
  - vector values;
  - the corpus mismatch (a Wikipedia table against FineWeb-Edu training);
  - retrieval outside matched FLOPs.
- A further H17 arm (token-level neighbours, Wikipedia in the training mix) still needs its own ADR.
