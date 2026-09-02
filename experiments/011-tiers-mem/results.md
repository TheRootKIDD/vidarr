# 011-tiers-mem — VRAM and host memory tiers

2026-09-02 · `scripts/bench/bench_tiers.py --tiers vram host` · gpu0 ·
numbers in `result.json`. The **storage** tier is deliberately absent: it waits
for the NVMe (I10) so that $\beta_{cold}$ is measured on the drive that will
hold the cold tier.

**Bears on:** the tier table of `docs/02 §7` and S6's inputs; supplies
$\beta_C$ to `local_3060`, and with it the roofline $b_{min}$ of `docs/02 §2`.

## Measured

| tier | | |
|---|---|---|
| VRAM, on-device copy, ≥ 256 MiB | **333 GB/s** (read + write traffic) | nominal 360 |
| VRAM, read-only sweep, 1 GiB | **342 GB/s** | |
| GPU → GPU through host, 256 MiB | 11.6 GB/s | one pair, one direction |

Host ↔ VRAM over PCIe 4.0 x16, GB/s:

| block | pinned H2D | pinned D2H | pageable H2D | pageable D2H |
|---|---|---|---|---|
| 4 KiB | 0.46 | 0.91 | 0.52 | 0.47 |
| 16 KiB | 7.6 | 3.6 | 1.7 | 1.6 |
| 64 KiB | 14.0 | 14.5 | 4.6 | 4.1 |
| 256 KiB | 17.6 | 25.0 | 8.1 | 7.0 |
| 1 MiB | 20.4 | 26.4 | 10.1 | 8.4 |
| ≥ 16 MiB | **24.3** | **27.1** | 14.8 | 9.0 |

## Interpretation

**$\beta_C$ ≈ 342 GB/s, 95 % of nominal.** With the measured $\phi$ of 27.1
TFLOPS the ridge point is 79 FLOP/byte and the roofline $b_{min}$ at BF16 is
**79 tokens** — against the 71 of `docs/06 §1` (nominal inputs) and the 512
empirical batch floor of 003-gemm. The 7× gap between roofline and occupancy
floors stands.

**The host link delivers 24–27 GB/s pinned, 75–85 % of nominal 32**, and
pageable copies halve that (D2H pageable is a flat 9 GB/s: the driver's staging
copy is the bottleneck, not the bus). For the warm tier this means: pinned
buffers are mandatory, and `docs/06 §2`'s "≈ 20–25 GB/s" was right.

**Small blocks are latency-bound.** A 4 KiB transfer runs at 0.5–0.9 GB/s,
i.e. 5–9 µs per call; the knee is at 64–256 KiB. A single $\mathcal{G}$ entry at
`small` is 1 KiB and a $B_{idx}$ = 16 block is 16 KiB, so **warm-tier fetches
must be batched to ≥ 64 KiB — several blocks per transfer — to see more than a
quarter of the link.** This is a direct input to the simulator's fetch model and
to the L9 indexer's block size: $B_{idx}$ = 16 on its own is too small a unit to
move.

**GPU→GPU through host at 11.6 GB/s reconciles with `002-nccl`'s 3.6 GB/s.**
One pair, one direction, no contention gets a third of the pinned H2D+D2H
budget; a 4-way ring collective shares the same root complex among four cards
and three transfers per step, and lands at 3.6 per rank — the ratio is what
contention on one PCIe root predicts. Both numbers are right; they answer
different questions, and the scenario file carries both.
