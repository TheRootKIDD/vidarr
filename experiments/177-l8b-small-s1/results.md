# 177-l8b-small-s1 — L8b at `small`

`python -m scripts.train.train --rung L8b --id 177-l8b-small-s1 --tokens 2.5e+09 --micro-batch 4 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** H17. This completes ADR-058's pair. The pair verdict, under ADR-059's rule against the L5 pair `132`/`133`, is **weakens**, on the held-out clause:
- FineWeb-Edu +0.0122 (4.1σ, past the 0.0119 line by 0.0003);
- in-table first sentence on covered tokens −0.0303, beats its 0.012 floor;
- rest of the lead +0.0030, misses.

ADR-055 §6's full-split reading agrees. By ADR-058 §3, two seeds close H17's L8b arm at `small`. Per seed, `177` against `133` is *silent* (held-out +0.0119 on the eval subset, just under 4σ; covered −0.0330 beats; rest +0.0116 misses).

## Numbers

| | `177` (seed 1) | `176` (seed 0) |
|---|---|---|
| **final val loss** (full held-out, 5606 windows) | **3.0836** (`133` 3.0718 → +0.0118, 3.9σ) | 3.0904 (`132` 3.0778 → +0.0126) |
| **pair mean** | **3.0870** against the L5 pair's **3.0748**: **+0.0122 (4.1σ)** | |
| throughput, median | 34,093 tok/s (`133` 48,958 → 0.70×) | 34,109 |
| retrieval, median per micro-batch | 0.321 s (≈ 33 % of the 15.38 s step) | 0.321 s |
| peak memory per GPU | 5.72 GiB | 5.72 GiB |
| wall-clock | 74,944 s = **20.82 h**, of which 1,445 s are in-training evals; post-run eval 5.0 min, pair compare < 1 s | 20.81 h |
| thermal | `n_thermal` 1 sample in 476 records; max 79 °C; SM ≥ 1890 MHz | 1 in 476 |
| params | 74.7 M total, 50.2 M non-embedding | |

σ = 0.0030 (2σ 0.0059, 4σ 0.0119). The config is as `175`/`176`, seed 1.

### Held-out trajectory (Δ vs the same-seed L5 run, and pair mean vs pair mean)

| step | 100 | 700 | 1300 | 1900 | 2500 | 3100 | 3700 | 4300 | 4768 |
|---|---|---|---|---|---|---|---|---|---|
| `177` − `133` | −0.0620 | +0.0016 | +0.0063 | +0.0090 | +0.0099 | +0.0106 | +0.0107 | +0.0116 | +0.0119 |
| pair − pair | −0.0285 | −0.0019 | +0.0051 | +0.0090 | +0.0097 | +0.0109 | +0.0111 | +0.0116 | **+0.0122** |

The pair gap opens between 0.4 and 0.7 B tokens and still widens at the end. So the 0.0003 margin over the 4σ line is not a final-step fluctuation. On the trend, the gap would only grow with more tokens.

### wiki-lead, pair against pair (`wiki_lead_compare_pair.json`; per-article sums of both seeds, bootstrap over articles)

| split | tokens per seed | L5 pair | L8b pair | Δ (floor) | beats? | `177` vs `133` | `176` vs `132` |
|---|---|---|---|---|---|---|---|
| in-table first sentence, **covered** (ADR-059's clause) | 55,516 | 2.8940 | 2.8637 | **−0.0303** (0.012) | **yes** | −0.0330 | −0.0276 |
| in-table first sentence, own retrieved | 54,597 | 2.8825 | 2.8516 | −0.0309 (0.012) | yes | −0.0333 | −0.0285 |
| in-table first sentence, full | 70,952 | 2.8095 | 2.7857 | −0.0238 (0.012) | yes | −0.0030 | −0.0446 |
| in-table rest of lead | 77,911 | 3.1317 | 3.1347 | **+0.0030** (0.012) | **no** | +0.0116 | −0.0057 |
| out-of-table lead (control) | 148,298 | 2.9654 | 2.9581 | −0.0074 (0.012) | no | +0.0087 | −0.0234 |
| first sentence, not covered (derived; branch zero) | 15,436 | 2.5056 | 2.5051 | −0.0005 | | +0.1050 | −0.1061 |

The `tokens` field in the pair file counts both seeds. Each bootstrap SE is below 0.0030, so every floor is 4σ = 0.012. Coverage is 78.2 % and accuracy 98.3 %, as in `045`.

**The pre-boundary tokens confirm I41's likely answer:** the pair means are equal (−0.0005), while the per-seed differences are ±0.1 with opposite signs. The L8-family "advantage" there was seed noise.

The full first-sentence split swings from −0.045 to −0.003 between the two seeds. The covered split stays at −0.028 / −0.033. That is ADR-059's reason, seen directly.

### Usage probes (`177`; `176` in brackets)

- Reliance (random table rows in place of the neighbours):
  - own-retrieved first-sentence tokens **+0.0511** (+0.0514), floor 0.0157 → reads;
  - covered +0.0509 (+0.0511), full first sentence +0.0398 (+0.0400);
  - rest of lead +0.0279 (+0.0297), control +0.0290 (+0.0305);
  - FineWeb-Edu +0.0179 (+0.0205).
- α entropy 1.910 of ln 8 = 2.079, 91.9 % (91.1 %). 13.7 % of rows retrieved (13.7 %).

## Interpretation

Two seeds agree on what L8b does at `small`. Retrieval is precise, and the model reads it: random rows cost +0.051 on own-retrieved tokens in both seeds. Loss on the sentence the memory holds falls by 0.030 against the L5 pair, 2.5× the floor and about 15× `046`'s covered seed spread.

The memory's value is confined to that sentence.
- The rest of the same article's lead is +0.003 (no gain).
- Out-of-table leads are −0.007 (within floor).
- General text is **+0.0122, 4.1σ worse**. The gap opens after ≈ 0.5 B tokens and is still widening at 2.5 B.

At matched active FLOPs, L8b's experts are narrower (width 704) to pay for the branch. At `small` the neighbours do not pay that back on general text, as they did at `screen` (`175`: −0.0007 vs `110`), where the tie came from the 1 B budget.

**H17 is weakened at two seeds by ADR-059's rule, on the held-out clause.** The margin over the 4σ line is small (0.0003). But the trajectory is monotone, and the first-sentence gain does not carry to the rest of the lead, so the in-table clause would not have produced *supports* even inside 4σ.

By ADR-058 §3 this closes H17's L8b arm at this budget. The verdict is on the design as specified, with the limits recorded in ADR-057 §8:
- **values are 768-d sentence vectors**, so the memory can say what a sentence is about, not reproduce it;
- **corpus mismatch**: the model trains on FineWeb-Edu, and the store is Wikipedia;
- **retrieval cost outside "matched active FLOPs"**: ≈ 33 % of the step, 0.70× L5's throughput.

Any further H17 arm (RETRO-style token neighbours, Wikipedia text in the training mix, or a memory that does not take width from the experts) is a new ADR. That is the user's call.
