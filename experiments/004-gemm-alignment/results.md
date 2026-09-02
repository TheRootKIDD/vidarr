# 004-gemm-alignment — odd expert widths cost 12–27 % of throughput

2026-09-02 · `scripts/bench/bench_gemm.py --alignment-probe` · one card (gpu0),
$b$ = 8192, $d$ = 768, median of 30 iterations.

**Bears on:** ADR-025 (proposed, `costmodel/README.md §A`). Feeds the rounding
rule that `costmodel.solve.solve_d_ff`'s `multiple_of` argument exists for.

## Why this was run

003-gemm showed a throughput trough at exactly the two widths `costmodel/`
derives from the 0.6 GFLOP budget — $d_{ff}$ = 833 and 1665 — while their
neighbours 512 and 2816 were fast. Both derived widths are odd. This walks each
against nearby widths to separate alignment from noise.

## Measured

| $d_{ff}$ | TFLOPS | | $d_{ff}$ | TFLOPS |
|---|---|---|---|---|
| 832 (×64) | 20.91 | | 1660 (even) | 21.43 |
| **833 (odd)** | **16.46** | | 1664 (×64) | 21.80 |
| 834 (even) | 21.21 | | **1665 (odd)** | **19.52** |
| 840 (×8) | 20.98 | | 1666 (even) | 23.45 |
| 896 (×64) | 22.52 | | 1672 (×8) | 23.64 |
| | | | 1728 (×64) | 24.66 |

## Interpretation

**The penalty is for *odd* widths specifically, not for non-multiples of 8.**
834 and 1666 are even but not multiples of 8, and both run at full speed; 840
and 1672 (multiples of 8) are no better than their even neighbours. Only the two
odd widths are slow: 833 is 21 % below 832/834, and 1665 is 11 % below 1664 and
17 % below 1666. The natural explanation is load vectorisation — at 2 bytes per
BF16 element an odd row length puts the row stride on a 2-byte boundary, which
defeats the wider vector loads — but the mechanism is inferred, and only the
measurement is claimed.

**Consequence for the cost model.** A width derived from a FLOPs budget is an
arbitrary integer and will be odd about half the time, so `solve_d_ff` must
round. Rounding to the nearest multiple of 64 is close to free in FLOPs and
recovers the loss:

| derived | rounded | FLOPs change | throughput change |
|---|---|---|---|
| 1665 | 1664 | −0.06 % | +11.7 % |
| 1665 | 1728 | +3.8 % | +26.3 % |
| 833 | 832 | −0.12 % | +27.0 % |
| 833 | 896 | +7.6 % | +36.8 % |

Rounding *down* to a multiple of 64 costs essentially nothing in budget and buys
12–27 %, so it should be the default; rounding up buys more but spends budget and
should be an explicit choice. Recommendation for ADR-025: `solve_d_ff` is called
with `multiple_of=64` and the resulting width is what `docs/06 §3` quotes.

Two caveats. 1728 beating 1664 by 13 % is not explained by alignment — both are
multiples of 64 — and is more likely tile quantisation against 28 SMs; it rests
on single measurements and would need repeats before being treated as real. And
this was run on one card at one $(b, d)$ point; the odd-width penalty is large
and consistent enough to act on, the finer structure is not.
