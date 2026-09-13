# 113-l3-full-screen — L3 with every MTP head trained on every position (I27 #3) at `screen`

`python -m scripts.train.train --rung L3 --id 113-l3-full-screen --tokens 1e+09 --micro-batch 1 --seed 0 --set mtp.subsample=1.0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (64 × 1 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Second run of queue 4.

**Bears on:** **H13**, as I27's third experiment — *not* a ladder rung and not a verdict (ADR-030 decides
H13 at `small`). The question it was built to answer: is `108`'s acceptance miss caused by ADR-012's
subsampling, which trains each position on exactly one of heads 2–4 so head 2 sees ≈ 1/3 of the token
budget? This run is `108` with subsampling off and nothing else changed (same 272.9 M parameters, same
seed, same data). Micro-batch 1 because three full logits passes at micro-batch 2 would exceed the
10 GB cap; peak came in at **9.54 GiB**.

## Numbers

| | |
|---|---|
| params | 272.9 M total, 248.3 M non-embedding (identical to `108`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.2342** nats |
| final train loss (last step, main head) | 3.2042 |
| throughput (median step) | 29,735 tokens/s aggregate |
| peak memory per GPU | 9.54 GiB |
| wall-clock (this session) | 9.48 h |
| seed | 0 |
| val acceptance vs head 1, heads 2 / 3 / 4 | **39.7 / 23.0 / 16.1 %** |
| val top-1 vs truth, heads 2 / 3 / 4 | 21.8 / 12.9 / 9.0 % |

First rung with the I27 metrics logged in-trainer (`mtp_accept_h*` beside `mtp_top1_h*`); `108`'s
acceptance is from the offline probe on its checkpoint.

## Against `108`, and the answer to I27 #3

| | `108` L3 (subsampled) | **`113` L3-full** | Δ |
|---|---|---|---|
| held-out main loss | 3.2486 | **3.2342** | **−0.0144 nats, −0.44 %** |
| main-loss regression vs L2 (3.1495) | +3.15 % | **+2.69 %** | |
| acceptance h2 (H13: ≥ 70 %) | 38.5 % | **39.7 %** | +1.2 pt |
| acceptance h3 (≥ 55 %) | 21.6 % | **23.0 %** | +1.4 pt |
| acceptance h4 (≥ 45 %) | 15.0 % | **16.1 %** | +1.1 pt |
| top-1 h2 vs truth | 21.5 % | 21.8 % | +0.3 pt |
| step cost vs L2 | +25 % (6.51 h) | **+46 % (9.48 h)** | |

**Subsampling was not the cause.** Training every head on three times the positions moves acceptance
by about one point per head and closes the main-loss gap to L2 by 0.014 nats — real, consistent across
heads, and an order of magnitude short of the 30-point shortfall against H13's targets. I27 asked
whether the heads were "starved by our own optimisation" or whether "the task is too hard at this
scale"; **this run says the second**, and it says so at the cost of 3 extra GPU-hours rather than the
17 h of the `small` L3 pair. ADR-012's subsampling is thereby confirmed as what it was meant to be: a
speed choice worth 1.8 h per `screen` run for ≈ 1 point of acceptance.

**The main head did not get worse — it got slightly better.** Three full auxiliary losses on the same
final hidden state might have been expected to pull the representation further from head 1's needs;
instead the regression against L2 shrank from 3.15 % to 2.69 %. At this scale the extra multi-token
signal is mildly helpful to next-token prediction, not harmful, and the remaining 2.7 % is the price
of the heads sharing the final state at all. That is the number a curriculum (I20's L3b reading) or
sequential modules (ADR-012's) would have to beat.

**Acceptance trajectory** (held-out, head 2): 35.1 % at step 100, flat to 34.4 % at 400, then 35.4 →
37.8 → 38.8 → 39.5 → 39.8 % at steps 700 / 1000 / 1300 / 1600 / 1900. The same decelerating shape as
`108`, from a higher floor that appears in the first 100 steps and never widens — full-position
training gives the heads an earlier start, not a different destination.

## Routing and run health

Routing is `108`'s transient again: 34 dead of 96 at step 16, one router at 2.06 of 8 effective at
step 19, recovered to 8.00 of 8 by the end. Not a confound.

| | |
|---|---|
| wall-clock | 9.48 h; summed `step_s` 9.336 h + evals 0.126 h + checkpoints 0.013 h → **0.05 % unaccounted** |
| throughput | 29.7 k tok/s median at micro-batch 1; `108` ran 43.3 k at micro-batch 2 |
| thermal | 7 of 190 records flagged (single samples, mostly eval steps), max 83 °C, min clock 1807 MHz on one such sample |

## Interpretation

Turning off ADR-012's subsampling buys one point of acceptance per head and 0.014 nats of main loss for
46 % more step time. I27's experiment 3 is closed: the H13 miss at `screen` is not an artefact of our
optimisation, and the acceptance ceiling at this budget is the task. **No verdict on H13** (ADR-030),
but the question `small` must answer narrows to budget alone, and the L3b decision tilts further toward
a method change over a schedule change: neither more positions (this run) nor, on `108`'s curve shape,
more tokens looks likely to reach 70 % with independent heads.

## Addendum (same day) — like-for-like against `108`, the gain is smaller still

The table above compares `113`'s **in-trainer, full held-out** acceptance (5606 windows) with `108`'s
**offline probe on 32 sequences**, because `108` predates the in-trainer metric. Same probe, same 32
sequences, on `113`'s checkpoint (`mtp_speculative.json`):

| 32-sequence probe | `108` | `113` | Δ |
|---|---|---|---|
| acceptance h2 / h3 / h4 | 38.5 / 21.6 / 15.0 % | 38.6 / 21.9 / 15.2 % | **+0.1 / +0.3 / +0.2 pt** |
| expected run length (joint) | 0.511 | 0.518 | +0.007 |
| tokens per verify pass | 1.511 | 1.518 | |
| acceptance h2 where head 1 is near-certain (≤ 0.5 nats, 34 % of positions) | 56.4 % | 57.2 % | +0.8 pt |

And on the full held-out set the one metric both runs logged, head-2 top-1 vs truth, is 21.49 % vs
21.83 %: **+0.3 pt**. The "+1.2 pt" above is mostly the difference between a 32-sequence subset and
the full set, not between the runs. **The conclusion sharpens rather than changes**: full-position
training moves acceptance by a few tenths of a point, the run length by 0.007 tokens, and the
confidence profile not at all. Subsampling was not the cause, and the 0.014-nat main-loss gain is the
only thing the extra 46 % of step time bought.
