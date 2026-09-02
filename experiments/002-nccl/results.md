# 002-nccl — host-bounced collective bandwidth and latency, 4 × RTX 3060

2026-09-02 · `scripts/bench/bench_nccl.py`, `--max-log2 30` · numbers in `result.json`
Supersedes nothing; `001-nccl-probe` is the same measurement capped at 256 MiB
and agrees with this one everywhere the two overlap.

**Bears on:** open question **I5** (are S0/S10 viable on four cards?). Feeds
`lambda_link` and the latency floor of `sim/scenarios/local_3060.yaml`. No
hypothesis H1–H19 directly; it sets the fabric inputs several of them are
scored against.

NCCL 2.29.7, `NCCL_P2P_DISABLE=1`, 4 ranks, one process per card, FP32 buffers,
1 KiB → 1 GiB per rank, median of 8–200 timed iterations depending on size.

## Measured

| | latency @ 1 KiB | plateau bus bw | message size for ½ plateau |
|---|---|---|---|
| all-reduce | 46.0 µs | 3.59 GB/s | 512 KiB |
| all-gather | 39.7 µs | 0.89 GB/s | 128 KiB |
| all-to-all | 43.9 µs | 3.60 GB/s | 256 KiB |

Bus bandwidth uses the nccl-tests correction factors, so the three columns are
not directly comparable. Converted to **bytes actually crossing the link per
rank per second**, the three agree closely:

| from | λ_link |
|---|---|
| all-reduce | 3.59 GB/s |
| all-gather | 3.54 GB/s |
| all-to-all | 3.60 GB/s |

**λ_link ≈ 3.6 GB/s per rank, latency floor ≈ 44 µs.** Three independent
collectives landing within 2 % of each other is the main reason to trust the
number. The curve saturates by 8 MiB and is flat to 1 GiB.

## Interpretation

The link delivers **11 % of nominal PCIe 4.0 x16 (32 GB/s), and 22 % of the
16 GB/s ceiling** that a device→host→device bounce implies. The cards are
confirmed x16 gen 4 under load (`000-env`), so this is not a slot problem: it is
the cost of staging every transfer through host memory with four cards sharing
one root complex, which is exactly the configuration `docs/06 §1` predicts and
the reason it says "measure".

**S10 is viable and well-posed — arguably better than intended.** `docs/06 §6`
asks S10 to find "the link-bandwidth ratio at which TP loses" with "a real,
deliberately poor λ_C". Evaluating the tensor-parallel condition of `docs/02 §2`,
λ_C ≫ φ·b_act / (3 (d_ff/U) L_e), at the `small` recurrent shape
(d_ff = 512, U = 4, L_e = 1, BF16):

| φ per card | λ_C required | shortfall vs measured 3.6 GB/s |
|---|---|---|
| 25.5 TFLOPS (nominal) | 133 GB/s | 37× |
| 15 TFLOPS (plausible measured) | 78 GB/s | 22× |
| 10 TFLOPS | 52 GB/s | 15× |

TP across these four cards is **15–37× bandwidth-starved**. S10 will therefore
show TP losing decisively rather than marginally; the useful output is the
crossover, which has to be reached by sweeping message size and rank count
rather than by finding it at the operating point. The φ column is provisional
until `bench_gemm`.

**S0 is measurable but is a fabric-dominated calibration point, and should be
labelled as one.** At the `small` recurrent config the fabric carries
2·k·d·b_act = 12 KiB per token per iteration, 96 KiB per token over r = 8:

- bandwidth ceiling: **36.5 k tokens/s**
- compute ceiling at 10–15 TFLOPS × 4 cards: 200–300 k tokens/s
- → **fabric-bound by 5–8×**
- collective latency is not the constraint: r × 2 × 44 µs = 704 µs per step, which
  at a 512-token batch is equivalent to 727 k tokens/s.

So S0 will validate how the simulator *composes* compute and communication, and
will validate the communication half well, but it will barely exercise the
compute-bound expert-queue behaviour that matters at DC scale — the regime
`docs/02 §2` is actually about, where b ≥ b_min makes the expert path
compute-bound by construction. Recommendation, for `docs/04`: keep S0 as
designed and run it, but record it as a **fabric-dominated** point, take the
compute-side calibration separately from `bench_gemm` plus a single-card
throughput check, and rely on I1 (a published deployment) for a compute-bound
system-level point. A two-card S0 variant is *not* a fix — halving the rank
count does not move the 15–37× gap.

Two secondary numbers worth carrying into the scenario file. The ½-plateau
message size (256–512 KiB) is the practical "message big enough to amortise
latency" threshold, and is the right minimum granularity for the simulator's
dispatch batching. And all-gather's much lower bus bandwidth is a convention
artefact, not a slow path: per byte on the wire it matches the other two.

**No change to `docs/06 §1` is implied** — it predicted host-bounced NCCL and
declined to guess a number. This supplies the number.
