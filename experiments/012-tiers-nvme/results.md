# 012-tiers-nvme — storage (cold) tier on the NVMe

2026-09-02 · `scripts/bench/bench_tiers.py --tiers storage --storage-path
/mnt/nvme/bench/bench.bin` · 8 GiB file of random bytes, `O_DIRECT`, one
synchronous reader (queue depth 1) · numbers in `result.json`.

Hardware: WD_BLACK SN850X 4 TB (PCIe 4.0 x4), installed today with the 1 TB
WD10EZEX HDD; both wiped and formatted ext4 (`-m 0`, no lazy init) and mounted
at `/mnt/nvme` and `/mnt/hdd`. This closes I10 and I12 (`000-env`); the SATA
850 PRO was never measured, as I10 required.

**Bears on:** the cold row of `docs/02 §7`'s tier table and S6's $\beta_{cold}$
input; H15b (cold hits per token vs SSD bandwidth) and H16 through the tier
budgets. **Silent on both until L9 supplies the hit-locality model** — this run
fixes the denominator, not the hit rate.

## Measured

| block | random latency, median (QD 1) | random IOPS | random GB/s | sequential GB/s |
|---|---|---|---|---|
| 4 KiB | 46 µs | 21.5 k | 0.09 | 0.22 |
| 16 KiB | 70 µs | 14.3 k | 0.24 | 0.60 |
| 64 KiB | 92 µs | 10.9 k | 0.71 | 2.26 |
| 256 KiB | 116 µs | 8.3 k | 2.18 | 4.16 |
| 1 MiB | 240 µs | 4.1 k | 4.29 | 4.84 |
| 4 MiB | 668 µs | 1.4 k | 5.90 | 6.49 |
| 16 MiB | 2.4 ms | 353 | 5.92 | 7.10 |
| 64 MiB | 9.2 ms | 91 | 6.09 | **7.29** |

## Interpretation

**$\beta_{cold}$ = 7.3 GB/s sequential at bulk sizes, the drive's rated 7.3;
random reads reach 6 GB/s from 4 MiB up.** `docs/06 §2`'s "≈ 3–7 GB/s" for the
cold tier is met at the top of the range. The measured tier ladder is now
hot 342 : warm 24–27 : cold 7.3 GB/s (≈ 47 : 3.5 : 1), against `02 §7`'s
$\beta_C$ / host link / $\beta_{cold}$ — a clean order of magnitude per step
at bulk transfer sizes, which is what the simulator's tier model assumes.

**At KV-block sizes the cold tier is latency-bound, ten times worse than the
warm tier.** A $B_{idx}$ = 16 block at `small` is 16 KiB: 70 µs and 0.24 GB/s
from one reader, versus 5–9 µs and 3.6–7.6 GB/s for the same block over the
pinned host link (`011`). The knee where random reads reach more than half the
drive is 1 MiB, i.e. 64 blocks per fetch — four times the warm tier's 64 KiB
knee. For S6 this means a cold fetch is a batch of blocks or it is a 70 µs
stall, and P15's condition "cold hits per token × block bytes × throughput
$< \beta_{cold}$" must be evaluated with the block-size-dependent number, not
the 7.3 GB/s headline.

**Queue depth 1 is a lower bound.** The bench issues one read at a time; the
drive is rated for ~10⁶ random 4 KiB IOPS at high queue depth, so concurrent
fetches (several sequences' cold hits in flight) would lift the small-block
figures well above the table. How far, on this rig, is open (I16): before an
S6 scenario models concurrent cold fetches, `bench_tiers` needs a queue-depth
option and a rerun. Until then the scenario carries the QD 1 numbers and the
simulator's cold tier is pessimistic by construction.

**Sequential at 4–16 KiB (0.2–0.6 GB/s) is the corpus-memmap case** only if
the loader reads page by page; the `uint16` corpus memmap (`06 §7`) is read
in contiguous windows of at least a sequence (≥ 4 KiB at 2048 tokens) and the
page cache sits in front of it, so this is not a training-throughput concern.
