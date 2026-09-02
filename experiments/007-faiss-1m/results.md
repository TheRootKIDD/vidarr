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

## Decision, 2026-09-02: 1 M accepted as the Phase-0 measurement; 10 M dropped

The 10 M run (`008-faiss-10m`) exceeded the 15-minute session rule and was cut
with no result. It is not rerun. The 1 M figures above are the Phase-0
measurement, for three reasons: the conclusion they support (`docs/06 §5.1`) is
settled by a 40× margin that a 10× larger index only widens; the accuracy half
of the measurement is void on random vectors anyway (I14) and must be redone on
real embeddings, at which point 10 M can be run in the same job; and ADR-009
already removed the in-loop dependency on this index, so nothing in the ladder
waits on it.

**What 10 M would have shown, as an expectation and not a measurement.** Insert
time scales linearly with $N$ (≈ 170 s); k-means training grows with nlist and
its sample (≈ 632 k vectors at nlist = 12 649) and is the part that overran.
QPS at a fixed nprobe falls with the codes scanned per probe, $N \cdot$
nprobe / nlist — 6.3 k codes at 10 M against 2 k at 1 M — so roughly a third of
the 1 M throughput, ≈ 2–3 k QPS at nprobe = 8. That makes the per-iteration
verdict *stronger* at scale, not weaker.

### Is this bound by the current hardware, and what the "correct" hardware changes

**The QPS number is a property of this CPU; the conclusion is not.**

*What is hardware-bound.* IVF-PQ throughput is set by core count, SIMD width and
memory bandwidth. This is 12 Zen 2 cores with AVX2 and no AVX-512, and the
number should be read as "one mid-range 2020 workstation CPU". Two hardware
changes are within reach of this project and would move it:

- **`faiss-gpu` on one RTX 3060.** Not installed and not measured. GPU IVF-PQ is
  typically 10–50× a CPU of this class at batch, so 1–3 × 10⁵ QPS is a
  reasonable expectation — *at* the 3 × 10⁵ the `small` loop would need, but
  only batched, with per-query latency in the hundreds of microseconds and one
  of four cards given over to it. Worth measuring when the real embeddings
  exist (I14), because it is the honest local ceiling; it does not change the
  design decision, since a lookup that costs a card and a batch round-trip is
  still a database call rather than a memory layer.
- **More or newer cores.** Linear in cores, roughly 1.5–2× per generation of
  SIMD. Neither reaches four orders of magnitude.

*What is not hardware-bound.* The note's P17 asks for retrieval **at every
iteration of every token**. At note scale ($d$ = 8192, $r_{max}$ = 128) a
datacenter serving 10⁶ tokens/s issues ≈ 10⁸ queries/s. That is four to five
orders above this CPU, two to three above a GPU index, and the per-query
*latency* — 1.35 ms here, ~100 µs on a GPU — is larger than an entire middle
iteration's compute time at unit scale (≈ 13 GFLOP per token per iteration
across a 64-chip unit). No index technology on the horizon closes a gap of that
shape, and `docs/06 §5.1`'s conclusion — a per-iteration lookup must be a
memory *layer* resident with the compute, not a round-trip to a database — is
therefore structural and transfers to the "correct" hardware unchanged. This is
exactly what ADR-009's parametric product-key memory is, and what the author's
own Q10 answer (learned entailment-to-vector map, refreshable table) describes.

*What does transfer as a number.*

- **Offline L8b precompute** is throughput-only and scales linearly: ≈ 36 min
  per 1 B tokens here, minutes on a GPU index, hours on a laptop. Any hardware
  is fine; it is a preprocessing job.
- **The cold-tier ANN over $\mathcal{G}$** (`docs/01 §6.2`, `02 §7`) is queried
  at the cold-*hit* rate, not per iteration. 7.2 k QPS on this CPU is the number
  S6 uses as the local cold tier's lookup ceiling — and in the simulator it is a
  scenario input, so `note64` and `gpu_today` carry their own, sourced
  separately (I4). P15's bandwidth claim rests on cold hits being rare; this
  measurement says how rare they must be *here*: fewer than ≈ 7 k per second
  across all sequences the rig serves.
