# 024-thermal-soak-bracket-move — I23 closes: 1100 steps, zero thermal throttling, 737 ms

2026-09-10 23:16–23:30 UTC+2 · `scripts/bench/thermal_soak.sh 024-thermal-soak-bracket-move 1100`
(= `bench_train_step --ddp --steps 1100 --warmup 10 --rung L5:4`) · cards cold at 42/45/44/36 °C ·
170 W power limit on all four, unchanged · layout as captured in `023`, after the user's bracket and
GPU work.

**Bears on:** no hypothesis. Rig characterisation; **closes I23** and supplies the first valid
un-throttled DDP step time this rig has produced.

## Verdict: pass on every clause

| criterion (`docs/04`, 2026-09-04 checklist step 2) | required | measured |
|---|---|---|
| every card under load | < 85 °C | **82 / 70 / 71 / 67 °C** |
| thermal-slowdown samples | 0 | **0 of 82 per card, over the whole 13.7 min load window** |
| `step_s_median` vs the ≈ 760 ms cold floor | within a few percent | **737.5 ms — below the floor** |

## Thermals over the 13.7-minute load window

| GPU | UUID | max temp | clock min / mean | fan max | thermal | power-cap |
|---|---|---|---|---|---|---|
| 0 | `GPU-4673cc7d…` | 82 °C | 1875 / 1894 MHz | 97 % | **0** | 6 |
| 1 | `GPU-88074816…` | 70 °C | 1927 / 1931 MHz | 85 % | **0** | 0 |
| 2 | `GPU-56c7aa62…` | 71 °C | 1920 / 1925 MHz | 87 % | **0** | 0 |
| 3 | `GPU-8cb4f7cc…` | 67 °C | 1950 / 1952 MHz | 82 % | **0** | 0 |

Every card's clock minimum is within 20 MHz of its mean across 82 samples, i.e. flat for the whole
run. `GPU-88074816` — starved at 93 °C and 1499 MHz mean in `022`, and the card `019` reported with a
dead fan — runs at 70 °C and 1931 MHz here. **The starvation is gone.**

## The step time, and how flat it is

| | median | min | p90 | p90/median |
|---|---|---|---|---|
| **`024`** | **737.5 ms** | 723.0 | 742.4 | **1.007** |
| `022` | 968.1 | 838.0 | 999.2 | 1.032 |
| `016` | 825.8 | 758.6 | 1086.0 | 1.315 |
| `015` | 1277.2 | 760.6 | 2546.0 | 1.993 |
| `014` | 2526.4 | 2495.7 | 3808.8 | 1.507 |

All five rows are the same rung at the same micro-batch with the same 3.59 GiB peak, so this is one
workload measured five times. **The p90/median ratio is the honest thermal signal**: a card that
degrades under sustained load has a long tail, and `024`'s tail is 0.7 % over 1100 steps. `015` and
`016` reached the same ≈ 760 ms *minimum* on their first cold steps and then lost it; `024` holds it
for fourteen minutes and lands slightly under it.

Against `022`, the last measurement of the previous arrangement, the step falls **968 → 737.5 ms, a
23.8 % gain** — more than the 13.4 % `022` priced from its own un-throttled minimum, because that
minimum was itself taken with the other three cards already warm. Per `screen` rung this is 8.2 h →
**6.3 h**, and ≈ 9.5 h over the five queued rungs 106–110.

## The pass criterion needed fixing before it could be read

The script's own summary printed `GPU0: throttled 3/30 samples`, which reads as a fail. All three are
`0x4`, the **software power cap** — GPU 0 drawing its full 170 W at 1890 MHz. NVML packs every reason
into one bitmask, and the summary counted anything that was not idle. That is backwards for this
criterion: these cards are thermally, not power, limited (`015`), so **better cooling produces more
power-cap samples, not fewer**. `thermal_soak.sh` and `scripts/train/train.py` now count
`SwThermalSlowdown | HwThermalSlowdown` separately from `SwPowerCap`; `n_throttled` is kept in the
trainer's `log.jsonl` so records before `024` stay comparable. Checklist step 6's abort rule for the
ladder reads `n_thermal`.

## What this says about `014`, and about I21

`014` measured single-GPU L5 at 651 ms and DDP L5 at 2526 ms, read the 1.9 s gap as the cost of the
per-micro-step all-reduce, concluded the recurrent rungs pay ≈ 8× the dense DDP overhead for a smaller
gradient, and opened **I21** on it. `024` measures the same DDP configuration at **737.5 ms**. Against
`014`'s own 651 ms single-GPU figure that is an **86 ms** DDP overhead, not 1.9 s — smaller than the
≈ 250 ms the dense rung pays.

`014`'s min (2496 ms) sits 1 % under its median, so it never ran a single fast step; `015` and `016`,
starting cold, both reach 760 ms before degrading. That is the signature of a run whose cards were
**already saturated when the rung began** — `014` ran nine rungs back to back on the pre-rebuild
layout, and L5 was the fifth. `014` recorded no thermal telemetry (it predates the instrumentation),
so this is inference from the step-time distribution rather than a direct reading, but the mechanism
is the one `015` measured the same day: 328 MHz steady on that layout.

**Consequences:** `014`'s L0 row ran first and from cold and is credible; every later row is suspect,
and with them the variant column of `06 §4` and the per-rung micro-batch choices. I21's premise is
probably an artefact. Neither is settled here — `024` measures one rung. The clean test is `018`, the
per-rung re-bench, run **one rung at a time from cold with a cooldown between rungs**, which is a
better reason to run it than the warm-vs-cold comparison it was reserved for.

## Interpretation

Silent on H1–H19. I23 closes: the rig sustains four cards at full clock for fourteen minutes with no
thermal slowdown, which it has never done before, and the ladder is cleared to restart at L1 with id
106. The one card to watch is GPU 0 at 82 °C with its fan at 97 % — it passes, but it has almost no
margin left, where the other three hold 15–20 points of fan in reserve. A `screen` rung runs for
hours, not fourteen minutes, so checklist step 6's watch on the first eval of `106` still matters.

**Still owed and not answerable from a command:** which slot or bracket physically holds which card.
Open since `017`.
