# 028-thermal-soak-slot-swap — clearing GPU 0's intake takes it from 82 to 76 °C; pass, 737 ms

2026-09-15 10:41–10:55 UTC+2 · `scripts/bench/thermal_soak.sh 028-thermal-soak-slot-swap 1100`
(= `bench_train_step --ddp --steps 1100 --warmup 10 --rung L5:4`) · cards cold at 42/37/36/38 °C ·
170 W power limit on all four, unchanged · layout as captured in `027`.

**Bears on:** no hypothesis. Rig characterisation; the acceptance soak for the post-shutdown layout
(pick-up checklist step 3, `docs/04` 2026-09-14) and an extension of I23's caveat on GPU 0.

**The id's name is a misnomer, kept because ids are append-only.** The planned GPU 0 ↔ GPU 3 slot
swap did not happen: both cards came out on their risers and, by chance, `GPU-4673cc7d` went back
into the same slot at the same physical position (user's account, `027`). What changed for that card
is that the riser cable that ran under its slot and part-blocked its fan intake was moved. **This
soak therefore tests the cleared intake, not slot-vs-card.**

## Verdict: pass on every clause

| criterion (`docs/04` 2026-09-14, step 3) | required | measured |
|---|---|---|
| every card under load | < 85 °C | **76 / 66 / 64 / 71 °C** |
| `n_thermal` samples | 0 | **0 of 82 per card, whole 13.7 min load window** |
| `step_s_median` | ≈ 737 ms (`024`) | **737.2 ms** |

## Thermals over the 13.7-minute load window, against `024` by UUID

| GPU (`027` index) | UUID | root port | `024` max temp / fan | **`028` max temp / fan** | clock min / mean | thermal | power-cap |
|---|---|---|---|---|---|---|---|
| 0 | `GPU-4673cc7d…` | 00:03.1 | 82 °C / 97 % | **76 °C / 91 %** | 1897 / 1903 MHz | **0** | 2 |
| 1 | `GPU-88074816…` | 20:03.1 (x8) | 70 °C / 85 % | **66 °C / 82 %** | 1935 / 1943 MHz | **0** | 0 |
| 2 | `GPU-8cb4f7cc…` | 20:03.3 (x8) | 67 °C / 82 % | **64 °C / 80 %** | 1950 / 1953 MHz | **0** | 0 |
| 3 | `GPU-56c7aa62…` | 40:01.1 | 71 °C / 87 % | **71 °C / 87 %** | 1927 / 1930 MHz | **0** | 0 |

Every clock minimum is within 10 MHz of its mean. **`GPU-4673cc7d` drops 6 °C and 6 points of fan
at the same 1.9 GHz and the same 170 W cap**, with the same ambient class (cold start, same room):
the blocked intake was a real part of that card's problem. It is still the hottest card by 5 °C and
the only one that touches the power cap, so it remains the card that sets the DDP step — but it now
has margin where `024` gave it almost none, and the 84–86 °C / 100 % fan / 4–5 % trim that `110`,
`114` and `115` logged over hours had no equivalent here. Whether the margin survives an 8-hour
`screen` rung is the thing to read off the first eval of the next run (`n_thermal` on GPU 0).

`GPU-8cb4f7cc`, the card that moved slot (`40:03.1` → `20:03.3`, x16 → x8), is 3 °C cooler than in
`024` and its step time is identical: **x8 costs nothing measurable, as `027` argued from the
host-bounce numbers.**

## Step time

| | median | min | p90 | p90/median |
|---|---|---|---|---|
| **`028`** | **737.2 ms** | 726.2 | 741.7 | **1.006** |
| `024` | 737.5 | 723.0 | 742.4 | 1.007 |

Identical to `024` to 0.3 ms on all four ranks; 44.4 k tok/s aggregate, 6.2 h per `screen` rung at
this micro-batch without accumulation. The soak was never the workload that trimmed GPU 0 — the
trim showed up hours into the ladder runs — so an unchanged 15-minute step is expected and is not
evidence that the hours-long trim is gone; the 6 °C is.

## Interpretation

Silent on H1–H19. The rig is accepted on the `027` layout: four cards at full clock for 13.7 minutes
with zero thermal-slowdown samples, GPU 0 six degrees cooler than at the last acceptance. The
slot-vs-card question the shutdown was meant to answer is **still open**; it only matters if GPU 0
throttles again over a full rung, in which case the deferred slot swap (or a repaste) is the next
lever. The ladder is cleared to queue; next ladder id 116, next rig id 029.
