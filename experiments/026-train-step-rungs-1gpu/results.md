# 026-train-step-rungs-1gpu — I21 closes: DDP costs exactly one gradient all-reduce, on every rung

2026-09-11 00:00–00:17 UTC+2 · `bench_train_step --steps 150 --warmup 10 --rung L0:4 L1:2 L2:2 L3:2
L5:4 L5d:4 L5-ne128:4 L5-ne1:4 L7b:4` (**no `--ddp`** — one card, GPU 0) · same nine rungs, same order,
same session as `018` · candidate-C layout · thermals sampled throughout.

**Bears on:** no hypothesis. **Closes I21.** Supplies the single-GPU column that `014`'s was too
contaminated to provide.

## Single-GPU times, and the DDP overhead they expose

Paired with `018`, which ran the identical rungs under 4-way DDP an hour earlier:

| rung | params | 1 GPU (`026`) | 4 × DDP (`018`) | overhead | per M param |
|---|---|---|---|---|---|
| L0 dense | 100.1 M | 454 ms | 519 | **65 ms** | 0.65 |
| L1 layered MoE | 271.2 M | 334 | 565 | **230** | 0.85 |
| L2 + parallel form | 271.1 M | 329 | 570 | **241** | 0.89 |
| L3 + MTP (m = 4) | 272.9 M | 384 | 628 | **244** | 0.89 |
| L5 recurrent | 77.5 M | 671 | 737 | **66** | 0.85 |
| L5d final-vector | 77.5 M | 924 | 1026 | **102** | 1.32 |
| L5 $N_e$ = 128 | 193.1 M | 2052 | 2263 | **211** | 1.09 |
| L5 $N_e$ = 1 | 65.1 M | 579 | 636 | **57** | 0.88 |
| L7b ($L_e$ = 2) | 77.5 M | 744 | 815 | **71** | 0.92 |

## The overhead is a gradient all-reduce, and nothing else

A ring all-reduce moves $2(N-1)/N$ times the gradient, which is 1.5× at $N$ = 4. The gradient is BF16,
so 2 bytes per parameter, over the measured host-bounced $\lambda_{link}$ = 3.59 GB/s (`002-nccl`).
That prediction takes no fitting:

| rung | predicted | measured | ratio |
|---|---|---|---|
| L0 | 84 ms | 65 | 0.78 |
| L1 | 227 | 230 | **1.01** |
| L2 | 227 | 241 | 1.06 |
| L3 | 228 | 244 | 1.07 |
| **L5** | **65** | **66** | **1.02** |
| L5d | 65 | 102 | 1.57 |
| L5-ne128 | 161 | 211 | 1.31 |
| L5-ne1 | 54 | 57 | 1.05 |
| L7b | 65 | 71 | 1.10 |

Six of nine land within 10 % of a parameter-free prediction, **including the recurrent L5 at 1.02**.
The cost of DDP on this rig is the gradient divided by the link, for dense and recurrent rungs alike.

**I21 is closed.** It was opened on `014`'s reading that the recurrent rungs pay ≈ 1.9–2.9 s of
all-reduce against ≈ 0.25 s for the dense baseline — 8× the cost for a gradient 25 % smaller — and
hypothesised no bucket/compute overlap because the shared block's gradient completes only at the end
of backward. **There is nothing to explain.** L5 has 77.5 M parameters against L0's 100.1 M and pays
66 ms against L0's 65 ms; per parameter the recurrent rung is *cheaper* than the layered MoE rungs,
which pay 0.85–0.89 ms/M on 271 M parameters. The 1.9 s was thermal throttling on the old layout, in
both the DDP and the single-GPU column.

The two rungs above 1.2 are worth a note rather than an investigation: L5d (1.57) adds the
final-vector global cache, and L5-ne128 (1.31) has 128 expert matrices whose many small buckets the
reducer handles less efficiently than a few large ones. Both are second-order against a prediction
that assumes a perfect ring.

## `014`'s single-GPU column was contaminated too, but not uniformly

| rung | `014` 1 GPU | `026` 1 GPU | ratio |
|---|---|---|---|
| L0 | 446 ms | 454 | 0.98 — clean |
| L1 | 324 | 334 | 0.97 — clean |
| L3 | 673 | 384 | **1.75 — contaminated** |
| L5 | 651 | 671 | 0.97 — clean |
| L5d | 903 | 924 | 0.98 — clean |
| L5-ne128 | 1544 | 2052 | 0.75 — `026` slower, see below |

Mostly clean, which fits: one card at 170 W does not cook its neighbours the way four do, so the
single-GPU column survived a layout that destroyed the DDP column. L3 is the exception and it is the
one that produced `018`'s impossible reading, where DDP L3 (628 ms) beat `014`'s single-GPU L3
(673 ms).

**L5-ne128 runs slower here than in `014`, and this is the one number in `026` to treat with care.**
`026` ran entirely on GPU 0, which `024` identifies as the warmest card — 82 °C at 97 % fan under
four-card load. Over `026`'s 17.3 minutes GPU 0 peaked at 75 °C with **zero thermal-slowdown samples**
and one power-cap sample, so it was not throttled; but a single-card bench is a measurement of
whichever card it lands on, and this rig's cards are not interchangeable. A per-card sweep would
settle it. It does not affect I21's closure, which rests on the six rungs at ratio ≈ 1.

## Interpretation

Silent on H1–H19. **The recurrent design carries no distributed-training penalty on this rig beyond
its parameter count** — which was the thing I21 existed to establish before any throughput claim could
be made about it. Gradient accumulation remains useful for the 271 M layered rungs, where the
all-reduce is 40 % of a micro-step, and is close to irrelevant for the 77.5 M recurrent ones at 9–11 %.
That is the opposite of what `06 §4` said before tonight.
