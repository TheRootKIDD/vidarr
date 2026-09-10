# 018-train-step-rungs-recheck — `014` re-measured on the spaced layout: every rung was heat

2026-09-10 23:41–00:00 UTC+2 · `bench_train_step --ddp --steps 150 --warmup 10 --rung L0:4 L1:2 L2:2
L3:2 L5:4 L5d:4 L5-ne128:4 L5-ne1:4 L7b:4` — **the same nine rungs, in the same order, as `014`** ·
cards cold at 48/50/50/43 °C · candidate-C layout, all four cards spaced on the board · thermals
sampled every 10 s alongside, which `014` did not do.

**Bears on:** no hypothesis. Replaces `014-train-step-rungs` as the per-rung throughput table and
supplies `docs/06 §4`'s variant column. **Settles the DDP half of I21.**

## Every rung, then and now

| rung | `014` | **`018`** | `014` / `018` | `018` p90/median | peak GiB |
|---|---|---|---|---|---|
| L0 dense | 694 ms | **519** | 1.34 | 1.003 | 5.91 |
| L1 layered MoE | 1013 | **565** | 1.79 | 1.007 | 7.85 |
| L2 + parallel form | 1363 | **570** | 2.39 | 1.006 | 7.71 |
| L3 + MTP (m = 4) | 1938 | **628** | 3.09 | 1.006 | 7.88 |
| L5 recurrent, $N_e$ = 8 | 2526 | **737** | 3.43 | 1.005 | 3.59 |
| L5d final-vector global | 3779 | **1026** | 3.68 | 1.122 | 3.63 |
| L5 $N_e$ = 128, $k$ = 4 | 5862 | **2263** | 2.59 | 1.010 | 5.33 |
| L5 $N_e$ = 1 (dense recurrent) | 2652 | **636** | 4.17 | 1.003 | 3.40 |
| L7b ($L_e$ = 2) | 2678 | **815** | 3.28 | 1.015 | 3.59 |

Aggregate tokens/s and the `screen` hours that follow, at micro-batch as listed, no accumulation:

| rung | tok/s aggregate | `screen` (1 B tokens) |
|---|---|---|
| L0 | 63,176 | 4.4 h |
| L1 | 29,007 | 9.6 h |
| L2 | 28,727 | 9.7 h |
| L3 | 26,094 | 10.6 h |
| L5 | 44,472 | 6.2 h |
| L5d | 31,946 | 8.7 h |
| L5-ne128 | 14,480 | 19.2 h |
| L5-ne1 | 51,557 | 5.4 h |
| L7b | 40,187 | 6.9 h |

## The contamination is visible in the ratio column

`014` / `018` rises **monotonically across the first six rungs in run order** — 1.34, 1.79, 2.39,
3.09, 3.43, 3.68. That is not a property of the rungs; it is a property of *when in the sequence they
ran*. Cards that heat progressively through a back-to-back sequence produce exactly this, and `015`
measured the mechanism on that layout the same day `014` ran: 328 MHz steady, 93 °C. The last three
rungs break the monotone (2.59, 4.17, 3.28) because they are different workloads at different
utilisation, not because the pattern fails.

**Thermals here, which `014` never recorded**: 130 samples per card over **21.7 minutes of continuous
load across all nine rungs**, max 79 / 69 / 71 / 67 °C, **zero thermal-slowdown samples**, one
power-cap sample on GPU 0. Every rung's p90 sits within 1.5 % of its median except L5d at 12 %.

So the back-to-back sequence itself is not the fault — on this layout it is fine. **The fault was the
old layout**, and `018` is a valid replacement for `014` rather than a differently-run experiment.
L5 lands at 737 ms against `024`'s 737.5 ms from an isolated 1100-step run, so the two agree to 0.1 %
and the sequence costs nothing.

## What this does to the `screen` budget

`06 §4` carried ≈ 6 h for L5 and ≈ 8.4 h for L5d from `014`'s *accumulated* rates. Those happen to sit
close to `018`'s **no-accumulation** 6.2 h and 8.7 h, so the headline `screen` figure barely moves —
but it now means something different, and every other rung does move. The five queued ladder rungs
106–110 (L1, L2, L3, L5, L5d) cost **44.8 h** at these no-accumulation rates, and less with the
trainer's gradient accumulation. **L5-ne128 at 19.2 h is the expensive one** — it is the
parameter-matched $N_e$ = 128 comparator H6 needs, and it is worth its own scheduling decision rather
than being dropped into a queue.

## I21: the DDP half is settled, the single-GPU half is not

`014` read the gap between its single-GPU L5 (651 ms) and its DDP L5 (2526 ms) as a ≈ 1.9 s
per-micro-step all-reduce, concluded the recurrent rungs pay ≈ 8× the dense DDP overhead for a smaller
gradient, and opened I21. `018` puts DDP L5 at 737 ms. Against `014`'s own single-GPU figure that is
**86 ms**, below the dense rung's.

But `014`'s single-GPU numbers came from the same back-to-back session and are themselves suspect —
`018`'s L3 (628 ms, DDP) is *faster* than `014`'s single-GPU L3 (673 ms), which is impossible unless
the single-GPU figure was also degraded. So the DDP overhead cannot be settled against `014` at all.
`026-train-step-rungs-1gpu` re-measures the single-GPU column on this layout; **I21's verdict waits
for it.** What `018` does establish is that the ≈ 1.9–2.9 s figure I21 was opened on does not exist.

## Interpretation

Silent on H1–H19. This is the throughput table Phase 1 actually runs on, and the first one on this rig
taken with thermal telemetry proving the cards held clock throughout. `014` is superseded for every
rung; its erratum points here.
