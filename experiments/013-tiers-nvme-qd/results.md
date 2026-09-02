# 013-tiers-nvme-qd — cold tier vs queue depth (I16)

2026-09-02 · `scripts/bench/bench_tiers.py --tiers storage --storage-path
/mnt/nvme/bench/bench.bin --queue-depth 1 4 16 64` · same 8 GiB file and drive
as `012-tiers-nvme` (WD_BLACK SN850X 4 TB, ext4, `O_DIRECT`) · 29 s wall ·
numbers in `result.json`.

Queue depth is `--queue-depth` Python threads each issuing synchronous
`preadv` (the GIL is released inside the syscall). The process ran at 177 %
CPU, i.e. the reader itself is a ceiling at small blocks — see below.

**Bears on:** $\beta_{cold}$ and the cold-fetch latency model of `docs/02 §7`
/ S6, and H15b's "cold hits per token low enough for SSD bandwidth". Closes
I16. Still **silent on H15b** until L9 supplies hit locality.

## Measured (random reads; sequential is within 5 % of random at QD ≥ 4)

| block | QD 1 | QD 4 | QD 16 | QD 64 |
|---|---|---|---|---|
| **16 KiB**, GB/s | 0.23 | 0.88 | **2.07** | 2.19 |
| 16 KiB, IOPS | 14.3 k | 53 k | 127 k | 133 k |
| 16 KiB, latency median / p99 | 70 / 75 µs | 72 / 98 | 114 / 227 | 418 / 1873 |
| 4 KiB, IOPS | 21 k | 74 k | 144 k | 131 k |
| 64 KiB, GB/s | 0.71 | 2.46 | 5.78 | **6.89** |
| 64 KiB, latency median / p99 | 92 / 96 µs | 96 / 164 | 164 / 408 | 382 / 2713 |
| 256 KiB, GB/s | 2.20 | 6.41 | **7.33** | 7.32 |
| ≥ 1 MiB, GB/s | 4.4–5.8 | 7.0–7.2 | 7.1–7.3 | 7.2–7.3 |

## Interpretation

**Concurrency lifts a 16 KiB cold fetch from 0.23 to 2.1 GB/s (9×) and it
saturates by queue depth 16.** Four in-flight requests already give 3.8×.
Latency stays near the QD 1 floor up to QD 4 (72 µs median) and reaches 114 µs
median / 227 µs p99 at QD 16; QD 64 buys 6 % more throughput for 4× the
latency — pure queueing, the useful operating point is QD 8–16.

**The drive delivers its full 7.3 GB/s from 256 KiB at QD 16, and 6.9 GB/s at
64 KiB at QD 64.** With concurrency the batching knee moves from 1 MiB (`012`,
QD 1) down to 64–256 KiB: 4–16 $B_{idx}$ = 16 blocks per fetch, the same
64 KiB knee as the warm tier (`011`). So the S6 fetch model can use one rule
for both tiers: aggregate to ≥ 64 KiB, keep ≥ 8 requests in flight.

**Below 64 KiB the ceiling is the reader, not the drive.** 4 KiB and 16 KiB
both stop at ≈ 130–145 k IOPS, and 4 KiB *drops* from QD 16 to 64: that is
one Python process saturating on syscall issue and GIL hand-off (177 % CPU),
against a drive rated ~1.2 M IOPS. **For the scenario file this is the honest
number**: a single-process cold-tier server on this rig gets ≈ 1.3 × 10⁵
blocks/s. Should an S6 scenario need more than that, remeasure with
`io_uring` or several processes under a new id; the 64 KiB and larger rows
are drive-bound and stand regardless.

**Bulk QD 1 came in at 5.6–5.8 GB/s against `012`'s 7.1–7.3** on the same file
an hour earlier, with a 3–11 ms p99 tail at ≥ 4 MiB. Run-to-run variance of
a consumer drive (SLC-cache and thermal state after the sweep's ~100 GiB of
reads); at QD ≥ 4 every run lands at 7.0–7.3. `local_3060` keeps 7.3 as
$\beta_{cold}$ with the QD condition attached.
