# 005-sparse24 — 2:4 structured sparsity vs dense at inference shapes

2026-09-02 · `scripts/bench/bench_sparse24.py` · gpu0 · numbers in `result.json`
Extended to larger batches in **006-sparse24-large**; read both together.

**Bears on: H3 (note P3), speed half only.** `docs/06 §2` assigns H3's quality
half to L10a training runs; nothing here speaks to quality. ADR-019 fixes what
is measured: weights only, magnitude 2:4, mask fixed after 1 % of tokens.

BF16, expert weight $(d_{ff} \times d)$ against activations $(d \times b)$, all
widths multiples of 64 per 004-gemm-alignment. Sparse TFLOPS are counted at
**dense** FLOPs — the useful work is identical, half the multiplies are skipped —
so 2.0× is the ceiling and 1.0× is break-even.

## Measured

Speedup vs batch, selected shapes:

| $d$ × $d_{ff}$ | b=256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| 768 × 832 | 0.09 | 0.09 | 0.18 | 0.32 | 0.64 | 1.27 |
| 768 × 1664 | 0.09 | 0.20 | 0.31 | 0.61 | 1.10 | 1.36 |
| 768 × 2816 | 0.16 | 0.29 | 0.48 | 1.02 | 1.24 | 1.24 |
| 1024 × 1664 | 0.11 | 0.23 | 0.37 | 0.72 | 1.30 | **1.44** |
| 1024 × 4096 | 0.26 | 0.46 | 0.89 | 1.26 | 1.13 | 1.13 |

Median across the whole grid **0.61×**; best **1.44×**.

## Interpretation

**2:4 loses to dense at every batch below ~2048, badly.** At b = 256 the sparse
path is 4–11× *slower*. Sparse throughput scales exactly linearly with $b$ over
the small-batch range (0.89 → 1.78 → 3.53 → 6.96 TFLOPS as $b$ doubles), which
means sparse *time* is constant there: a fixed per-call cost dominates, and it is
large. The dense path is already near $\phi$ at b = 512, so the crossover is set
entirely by the sparse side's overhead.

**The ceiling is ~1.6×, not 2×** (006-sparse24-large): 1.58× at
$d$=1024, $d_{ff}$=1664, b=32768, plateauing there and slightly declining at
65536. In effective terms the best sustained figure is ≈ 40 TFLOPS against
`docs/06 §1`'s nominal "≈ 51 with 2:4 sparsity" — about 78 % of it, and 1.5×
rather than 2× over the measured dense 26–27.

**Consequence for H3.** The speed premise of P3 is materially weaker on this
hardware than the note assumes. 2:4 pays only when (a) the batch per expert is
≥ 4096, which is 8× the empirical batch floor of 512 from 003-gemm and far above
the ≥ 128 tokens/expert that keeps grouped GEMM efficient, and (b) even then it
returns ~1.3–1.6×, not 2×. For the ladder this means L10a's quality result must
clear a bar of ~1.4×, not 2×, when the trade is priced — and at decode-style
batches 2:4 is a straight loss. Recorded as an input to ADR-013's threshold for
H3 rather than as a verdict on H3.

**Caveat, and it is a real one.** This measures PyTorch's
`to_sparse_semi_structured` path (cuSPARSELt underneath), not the hardware
ceiling. Compression happens once, outside the timed region, and 5 warmup calls
precede timing, so plan caching should be warm — but a hand-written cuSPARSELt
call with a chosen layout could plausibly do better, and the large fixed
per-call cost smells like a framework-level rather than silicon-level limit.
The claim made here is about the path a PyTorch model would actually take.
Whether the gap is closable is a question for Phase 3, when kernels are in
scope; it is **not** a reason to treat the nominal 51 TFLOPS as achievable.
