# 030-thermal-soak-driver-615 — driver 615.71 changes nothing measurable: pass, 738 ms, 75 / 65 / 64 / 70 °C

2026-09-19 11:11–11:25 UTC+2 · `scripts/bench/thermal_soak.sh 030-thermal-soak-driver-615 1100`
(= `bench_train_step --ddp --steps 1100 --warmup 10 --rung L5:4`) · cards cold at 44/42/42/39 °C,
2 min after boot · 170 W power limit on all four, unchanged · layout and software as captured in
`029` (driver 615.71.09, kernel 7.2.5; bus map identical to `027`).

**Bears on:** no hypothesis. Rig characterisation; step 3 of the post-driver-upgrade pick-up
(`docs/04` 2026-09-19, which named this soak "rig id 029" — `029` went to the env capture, the soak
is `030`).

## Verdict: pass on every clause

| criterion | required | measured |
|---|---|---|
| every card under load | < 85 °C | **75 / 65 / 64 / 70 °C** |
| `n_thermal` samples | 0 | **0 of 82 per card, whole 13.7 min load window** |
| `step_s_median` | ≈ 737 ms (`028`) | **738.4 ms** (+0.16 %) |

## Thermals over the load window, against `028` by UUID

| GPU | UUID | `028` max temp / fan | **`030` max temp / fan** | clock min / mean | thermal | power-cap | `0x400` |
|---|---|---|---|---|---|---|---|
| 0 | `GPU-4673cc7d…` | 76 °C / 91 % | **75 °C / 90 %** | 1905 / 1907 MHz | **0** | 0 | 82/82 |
| 1 | `GPU-88074816…` | 66 °C / 82 % | **65 °C / 80 %** | 1935 / 1937 MHz | **0** | 0 | 82/82 |
| 2 | `GPU-8cb4f7cc…` | 64 °C / 80 % | **64 °C / 79 %** | 1957 / 1958 MHz | **0** | 0 | 82/82 |
| 3 | `GPU-56c7aa62…` | 71 °C / 87 % | **70 °C / 85 %** | 1920 / 1930 MHz | **0** | 0 | 82/82 |

Temperatures, fans and clocks are within 1 °C / 2 points / 10 MHz of `028` on every card.

## Step time

| | median | min | p90 | p90/median | aggregate |
|---|---|---|---|---|---|
| **`030`** (615.71) | **738.4 ms** | 725.0 | 741.1 | 1.004 | 44.38 k tok/s |
| `028` (610.57) | 737.2 | 726.2 | 741.7 | 1.006 | 44.45 k tok/s |

+1.2 ms (+0.16 %) at the median, −1.2 ms at the minimum, identical on all four ranks; peak memory
3.588 GiB on both. One soak each side, so the 0.16 % is not resolvable as a driver effect. The
`018`/`026` per-rung table stands under the new driver.

## New under 615.71: the throttle bitmask carries `0x400` on every loaded sample

Every under-load row of `thermals.csv` reads `0x0000000000000400` on all four cards; under 610.57
the same workload read `0x0` (or `0x4` at the power cap). `nvidia-smi -q` on this driver names the
bit: **"Clocks Event Reasons → Reliability: Active"**, a reason the 610 series did not list (it also
adds "Board Limit"). It is not a slowdown: clocks equal `028`'s, the step time is unchanged, and the
thermal (`0x20 | 0x40`) and power-cap (`0x4`) bits are all zero. The reading is that the card sits
at its voltage-reliability ceiling — the top boost bin — which is what an un-throttled card does,
and that the old driver simply did not report it. **`n_thermal` is unaffected** (the soak summary and
the trainer both mask on `0x60`), but anything that treats "bitmask ≠ 0" as throttled will now flag
every row: the rule in `CLAUDE.md` — read `n_thermal`, never `n_throttled` — matters more, not less,
from `131` on. Zero power-cap samples on GPU 0 (2 in `028`) is within noise at 82 samples.

## Interpretation

Silent on H1–H19. The rig is accepted on driver 615.71.09: four cards at full clock for 13.7 min
with zero thermal-slowdown samples and the L5 DDP step unchanged to 0.2 %. Throughput figures from
`131` on are comparable with `116`–`122`. The queue `queue_small_3.sh` (`131`–`137`) is cleared to
launch; next rig id **031**, next ladder id after the queue **138**.
