# 006-sparse24-large — where the 2:4 speedup plateaus

2026-09-02 · `scripts/bench/bench_sparse24.py --batches 4096 … 65536` · gpu0

Extends **005-sparse24**, which found 2:4 still climbing at its largest batch
(8192) and could not say whether it approaches the theoretical 2×.

**Bears on: H3, speed half.**

## Measured

| $d$ × $d_{ff}$ | b=4096 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|
| 1024 × 1664 | 1.29 | 1.46 | 1.48 | **1.58** | 1.50 |
| 768 × 2816 | 1.28 | 1.26 | 1.27 | 1.33 | 1.30 |
| 1024 × 4096 | 1.14 | 1.13 | 1.14 | 1.17 | 1.17 |

## Interpretation

**It plateaus at 1.3–1.6× and does not approach 2×.** The best shape peaks at
1.58× by b = 32768 and turns over at 65536; the other two are flat from b = 4096
onward. An eightfold further increase in batch buys nothing, so 005's 1.44× was
already near the asymptote rather than partway up a curve.

Effective sparse throughput saturates at ≈ 40 TFLOPS (1024 × 1664), ≈ 34
(768 × 2816) and ≈ 31 (1024 × 4096) — i.e. the *wider* the expert, the worse the
2:4 advantage, which is the opposite of the direction that would help the note's
design, where P4 makes experts as wide as a unit.

Taken with 005: on this rig 2:4 is a loss below b ≈ 2048–4096, and worth
1.15–1.6× above it, against a nominal 2×. `docs/06 §1`'s "≈ 51 TFLOPS with 2:4"
should be read as ≈ 40 at best, and only for the narrower expert shapes.
