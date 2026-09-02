# 007-faiss-1m — CPU IVF-PQ at 1 M × 768-d

2026-09-02 · `scripts/bench/bench_faiss.py --n 1000000` · 24 OpenMP threads on
the 12-core Threadripper · numbers in `result.json`. **008-faiss-10m** is the
10 M run; read both together.

**Bears on:** the arithmetic of `docs/06 §5.1`, which rules per-iteration
external retrieval out of the training loop. After ADR-009 this bench no longer
gates L8 (the table is GPU-resident); it sizes the **offline** L8b neighbour
precompute and the cold-tier ANN over $\mathcal{G}$ (`docs/01 §6.2`, `02 §7`).

Index `IVF4000,PQ96x8` (nlist = 4√N, 96 subquantisers × 8 dims), trained on
200 k vectors, faiss-cpu 1.15.0.

## Measured

| | |
|---|---|
| build (train + add) | **53.2 s** (36.4 + 16.8) |
| exact search, `IndexFlatL2` | 209 QPS |
| single-query latency, nprobe = 8 | **1.35 ms** → 741 QPS if serialised |
| single-query latency, nprobe = 32 | 3.94 ms → 254 QPS |

Batched throughput (10 000 queries):

| nprobe | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| QPS | 27 490 | 12 294 | **7 224** | 5 796 | 2 772 | 1 554 |
| recall@10 | 0.004 | 0.011 | 0.018 | 0.026 | 0.033 | 0.043 |

## Interpretation

**`docs/06 §5.1` is confirmed, and by a wide margin.** It estimates a `small`
step needs ≈ 3 × 10⁵ QPS for per-iteration retrieval at $r$ = 8 and expects
IVF-PQ on this CPU to land in the 10⁴–10⁵ class. Measured: 7.2 k QPS at
nprobe = 8, 27 k at nprobe = 1 — the *bottom* of the expected band at a useless
nprobe, and **40× short** of the requirement at a usable one. And that is the
batched figure; per-iteration retrieval is a latency problem, and one query
alone costs 1.35 ms, which at $r$ = 8 would add ~11 ms per token to a step that
processes 0.5 M tokens. The conclusion that in-loop knowledge must be a memory
*layer* (GPU product-key, ADR-009) rather than a database round-trip is not
close.

**The recall column is degenerate and must not be read as a recall result.**
Random Gaussian vectors in 768 dimensions are all nearly equidistant
(concentration of measure), so "nearest neighbour" is not a well-defined target
and PQ has nothing to compress. Recall@10 of 0.04 at nprobe = 64 says nothing
about how this index would behave on sentence embeddings, which cluster and for
which IVF-PQ routinely reaches 0.8–0.95 at the same nprobe. **The QPS numbers
are valid** — the work per query is set by nlist, nprobe and m, not by the data
distribution — but the accuracy they buy is unknown until the real corpus
exists. Re-run against actual sentence embeddings once I8/I9 (encoder, corpus
slice) are settled; that run supersedes this one for recall.

**Sizing the offline L8b precompute.** A 1 B-token corpus slice at one chunk per
64 tokens is 15.6 M chunk queries. At 7.2 k QPS that is ≈ 36 minutes batched on
the CPU — comfortably a one-off preprocessing job, as `docs/06 §5.1` assumes. At
nprobe = 32 it is ≈ 94 minutes. Either fits in the "offline" budget; the choice
is recall, not time.

**Build cost is dominated by training, not insertion**: 36 s to train on 200 k
vectors versus 17 s to add 1 M. For the 10 M run, expect insertion to scale
linearly and training with the larger sample; see 008.
