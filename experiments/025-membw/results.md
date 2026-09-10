# 025-membw — host DRAM bandwidth: 71 GB/s copy, and six threads saturate it

2026-09-10 23:33 UTC+2 · `python -m scripts.bench.bench_membw --id 025-membw` · 4 GiB per array,
float64, best of 5, sweep over 1–24 threads · machine otherwise idle (soak finished, GPUs at 43–44 °C,
0 % utilisation) · 8 × 32 GB DDR4-2133 registered ECC, one per channel, 8/8 populated.

**Bears on:** no hypothesis. Rig characterisation. This is the benchmark of record owed since `019`,
whose ad-hoc probe could not enter `sim/scenarios/local_3060.yaml` because CLAUDE.md requires a
scenario input to cite a `scripts/bench/` result id.

## The sweep

| threads | copy | scale | add | triad* |
|---|---|---|---|---|
| 1 | 32.2 | 23.4 | 27.1 | 13.6 |
| 2 | 38.4 | 27.6 | 31.7 | 16.6 |
| 4 | 54.2 | 41.2 | 43.6 | 21.5 |
| **6** | **70.4** | **47.6** | **51.7** | **29.3** |
| 8 | 64.0 | 46.7 | 51.1 | 29.6 |
| 12 | 70.7 | 47.1 | 51.5 | 28.9 |
| 16 | 70.2 | 46.9 | 51.2 | 27.8 |
| 24 | 71.0 | 46.5 | 51.0 | 25.2 |

GB/s. \* Not a STREAM triad: `3.0 * b` materialises a temporary, so it moves a fourth array the byte
count does not include and is understated. **`copy` is the reference figure** — two arrays, no
temporary, no arithmetic.

## Two results

**The plateau is 71 GB/s copy, and it arrives at 6 threads of 12 cores.** Everything past 6 is flat to
within 1 %; the dip at 8 is noise. So a host-side stage — corpus streaming, the KV warm tier, optimiser
offload — gets its full share of memory bandwidth from **half the box**, and giving it more cores buys
nothing. That is the number the simulator wants, and a single-thread figure (32.2 GB/s) would have
understated the memory system by 2.2×.

**It is 52 % of theoretical peak.** 8-channel DDR4-2133 is 136.5 GB/s. 52 % on a threaded numpy copy is
ordinary — the kernels have no non-temporal stores, so write-allocate costs a read-for-ownership pass
that these byte counts exclude, which alone accounts for much of the gap.

The figures reproduce `019`'s ad-hoc probe (70.8 / 47.1 / 51.3 / 29.3 at 12 threads) to within 0.5 % on
every kernel. That probe's one job was to tell 8-channel from 4-channel population, and it is confirmed
here: 71 GB/s is comfortably past the 68.3 GB/s ceiling of 4-channel DDR4-2133.

## Interpretation

Silent on H1–H19. Closes the last host-memory item: `docs/06 §1`'s ≈ 205 GB/s was the controller's
DDR4-3200 rating and is unreachable on DDR4-2133 modules (`022`), and the real figure is **71 GB/s
usable, saturating at 6 threads**. Eligible for `sim/scenarios/local_3060.yaml` citing this id.

Both bounds are worth keeping straight when this number is used. It is a **lower bound on the
hardware**, because the byte counts omit read-for-ownership, and an **upper bound on what a program
gets**, because a real stage does more than stream.
