# 015-thermal-soak-170w — 15-minute L5 DDP soak at the default 170 W limit (I23)

2026-09-03 16:51–17:00 · `scripts/bench/thermal_soak.sh 015-thermal-soak-170w 350` ·
L5 rung, micro-batch 4, 4 × DDP, an all-reduce every micro-step · `nvidia-smi`
sampled every 10 s into `thermals.csv` · cards started warm (50–59 °C, ten
minutes after the aborted `101` run) · case: Lian Li O11D EVO XL, 3 side
intakes, 1 rear exhaust, no top or bottom fans; board ASRock WRX80 Creator
R2.0, four dual-slot cards in slots 1/3/5/7, i.e. back to back.

**Bears on:** no hypothesis; closes the *cause* half of I23 and invalidates
the power-cap remedy.

## Measured

| GPU | temp start → max | clock, first 2 min | clock, last 5 min (min) | mean power | first throttle |
|---|---|---|---|---|---|
| 0 | 55 → 94 °C | 1747 MHz | 1600 (1320) | 104 W | 70 s |
| 1 | 59 → 93 °C | 1590 MHz | **328 (225)** | 84 W | 50 s |
| 2 | 50 → 67 °C | 1797 MHz | 1950 (1950) | 89 W | never |
| 3 | 57 → 93 °C | 1695 MHz | 1269 (825) | 85 W | 60 s |

DDP step (slowest card): see `result.json`; the pace is GPU 1's.

## Interpretation

**The cards are thermally limited at 85–105 W of draw, far below the 170 W
cap — a power cap cannot fix this.** Three cards hit `sw_thermal_slowdown`
within a minute of load and settle at 93–94 °C with clocks between 225 and
1600 MHz while drawing half their limit; the fourth (GPU 2, the GA106 card,
the one with free air) holds 1950 MHz at 67 °C on the same workload. Heat
removal, not heat production, is the constraint: the three sandwiched cards
recirculate their own exhaust. Lowering the limit to 120 W would change
almost nothing (they already draw less); locking the clock would make the
step time predictable but not faster. The fix is airflow and spacing (bottom
intake fans blowing into the stack, top exhaust, one card on a riser so the
other three get a free slot each), then this soak is rerun under the same
id pattern (`016-thermal-soak-<config>`) and must show all four cards
un-throttled for 15 minutes before any training run is launched.
