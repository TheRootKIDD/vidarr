# 049-medium-sizing — L5-medium and L8b-pv-medium under DDP (ADR-061 §4)

**Bears on:** H17 at `medium`, indirectly. This is a rig record, so it gives no hypothesis verdict. It decides ADR-061 §3–§4: **DDP fits both arms at micro-batch 4, and the two runs project to ≈ 10 days, under the 16-day stop. `queue_medium_1` launches** with micro-batch 4 for both.

## Numbers (6 smoke steps each, last step; tree `5149a15`)

| arm | width | params (total / non-emb.) | micro-batch | peak GiB / GPU | step s | tok/s | retrieval s / micro-batch |
|---|---|---|---|---|---|---|---|
| L5-medium | 1920 | 143.7 / 110.9 M | **4** | **5.93** | 28.29 | **18,534** | — |
| L5-medium | 1920 | | 2 | 4.80 | 31.18 | 16,815 | — |
| L8b-pv-medium | 1856 | 143.1 / 110.4 M | **4** | **7.85** | 33.55 | **15,626** | 0.369 (≈ 18 % of the step) |
| L8b-pv-medium | 1856 | | 2 | 6.71 | 36.27 | 14,455 | 0.183 |

- **Parameters:** 143.7 M for L5-medium against `06 §3`'s anchor of ≈ 144 M.
- **Memory:** both arms are under the 10 GB cap at micro-batch 4. Micro-batch 2 is 7–9 % slower, so ADR-061 §4's rule ("the largest that fits") picks 4.
- **Thermals:** each smoke ran 6 steps, and the trainer samples the cards every 10, so no sample was taken. The cards are watched in the runs themselves.

## Projection (7 B tokens = 13,351 steps)

| run | training | in-training evals (134 × ≈ 100 s, scaled from `small` by FLOPs) | total |
|---|---|---|---|
| `180-l5-medium` | 13,351 × 28.29 s = 104.9 h | ≈ 3.7 h | **≈ 109 h ≈ 4.5 d** |
| `181-l8b-pv-medium` | 13,351 × 33.55 s = 124.4 h | ≈ 3.7 h | **≈ 128 h ≈ 5.3 d** |

The total is **≈ 9.9 days** plus ≈ 15 min of wiki-lead evals. ADR-061 §9's projection was ≈ 12 d, and `06 §4`'s ≈ 16 d.

L8b-pv-medium runs at 0.84× L5-medium's throughput, against 0.70× at `small`. The retrieval takes the same ≈ 0.37 s per micro-batch at both sizes, but at `medium` that is a smaller share of a longer step.

## Interpretation

**DDP at `medium` is a measured fit for these two configurations.** At micro-batch 4, L8b-pv-medium has 2.1 GiB of headroom under the cap, and L5-medium has 4.1 GiB. So `06 §3`'s FSDP requirement is not needed for them (ADR-061 §3). It stays the specification for the larger `medium` configurations, which this probe does not cover.

**The FLOPs-scaled projection was pessimistic:** per token, L5-medium is 2.65× slower than L5 at `small` (49.1 k → 18.5 k tok/s) for 3.3× the FLOPs. The larger matrices run more efficiently.

Smoke-step rates at `small` came within 3 % of the full runs (`048` 33.6 k against `178`'s median 34.6 k), so the projection should hold to a few percent.
