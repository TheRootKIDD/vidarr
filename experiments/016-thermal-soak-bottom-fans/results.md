# 016-thermal-soak-bottom-fans — L5 DDP soak after the first cooling step (three bottom 140 mm intakes)

2026-09-04 13:22–13:29 · `scripts/bench/thermal_soak.sh 016-thermal-soak-bottom-fans 350` ·
same rung, micro-batch, power limit (170 W) and sampler as `015-thermal-soak-170w` ·
cards started **cold** (33–42 °C, machine idle since yesterday; 015 started at 50–59 °C) ·
case change since 015: **three 140 mm intake fans added in the bottom of the Lian Li
O11D EVO XL**, blowing up into the card stack. Everything else unchanged: four dual-slot
cards back to back in slots 1/3/5/7 (bus IDs 01/02/41/42 as before), 3 side intakes,
1 rear exhaust, no top exhaust. The upright bracket, vertical kit and 900 mm riser have
not arrived, so no card has been moved yet.

**Bears on:** no hypothesis; interim reading of the I23 remedy. This is *not* the
acceptance soak of `04 §I23` — see the note on duration below.

## Measured

Steady-state window = the last 30 samples per card (the run's last 5 min). Throttle
flag 0x20 = `sw_thermal_slowdown`; GPU 0 also shows one 0x4 (`sw_power_cap`) sample.

| GPU | temp start → max | clock, first 2 min | clock, last 5 min mean (min) | mean power | first throttle | GPU fan |
|---|---|---|---|---|---|---|
| 0 | 40 → 93 °C | 1948 MHz | 1815 (1320) | 132 W | 120 s | 100 % |
| 1 | 42 → 93 °C | 1903 MHz | 1518 (990) | 120 W | 110 s | 100 % |
| 2 | 33 → 62 °C | 1954 MHz | 1946 (1942) | 121 W | never | 77 % |
| 3 | 40 → 93 °C | 1930 MHz | 1688 (1320) | 120 W | 130 s | 100 % |

Same columns for 015 (no bottom fans): clocks 1600 (1320) / 328 (225) / 1950 / 1269 (825)
MHz, mean power 104 / 84 / 89 / 85 W, first throttle 70 / 50 / – / 60 s.

DDP step, slowest card (`result.json`):

| | 015 (no bottom fans) | 016 (bottom fans) |
|---|---|---|
| step median | 1277 ms | 826 ms |
| step min (cold) | 761 ms | 759 ms |
| step p90 | 2546 ms | 1086 ms |
| aggregate | 25.7 k tok/s | 39.7 k tok/s |

Throttled samples in the last 5 min: GPU 0 20/30, GPU 1 22/30, GPU 2 0/30, GPU 3 20/30.

**Duration note.** The script runs a fixed 350 steps, which at 826 ms is 4.8 min of
load, not the 15 min the acceptance test asks for; 015's slower steps gave 9 min. The
three hot cards nevertheless reached 93 °C and their steady-state clock band within
2.5 min, so the verdict below would not change with a longer run. For the acceptance
soak after the rebuild use ≈ 1100 steps (`thermal_soak.sh 017-… 1100`).

## Interpretation

**The bottom intakes help but do not fix it: the three sandwiched cards still reach
93 °C and `sw_thermal_slowdown` within about two minutes.** The extra airflow roughly
doubles the time to first throttle (50–70 s → 110–130 s), lifts the steady-state clock
floor of the worst card from 225 to 990 MHz and its mean from 328 to 1518 MHz, and lets
the cards draw 120–132 W instead of 84–104 W — so the DDP step drops from 1277 to
826 ms (1.55×), against a cold-card floor of ≈ 760 ms in both runs. GPU 2, the card
with a free slot beside it, is unchanged at 62 °C and full clocks, which again points
at spacing rather than case airflow as the remaining constraint: the three back-to-back
cards recirculate their own exhaust no matter how much air enters the bottom. The
result is consistent with 015's conclusion and does not warrant any training run yet:
throughput at this state is still set by whichever card throttles hardest and is not
stable across a multi-hour run (this soak started cold). Next: install the upright
bracket and vertical kit when they arrive, so that every board-mounted card has a free
slot on each side, and rerun the soak as `017-thermal-soak-<config>` for ≥ 15 min.
Optional in the meantime, if a top exhaust fan can be fitted without the brackets: it
would remove the hot air the bottom fans now push into the top of the case.
