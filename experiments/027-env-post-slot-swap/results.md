# 027-env-post-slot-swap — the hot card is where it was, with its intake cleared; two cards now share one x16 as x8/x8

2026-09-15 · `python -m scripts.bench.bench_env --id 027-env-post-slot-swap` · first capture after the
user's GPU 0 work of 2026-09-14/15: the GPU 0 card was swapped and, in the same session, the riser
cable that sat directly under the GPU 0 slot and part-blocked that card's fan intake was moved. The
user notes the move leaves two cards at PCIe 4.0 x8.

**Bears on:** no hypothesis. Rig characterisation; pick-up checklist steps 1–2 of the order of work in
`docs/04` (2026-09-14 shutdown entry).

## Bus map, re-keyed by UUID

| GPU | bus | root port | UUID | chip | width | gen under load |
|---|---|---|---|---|---|---|
| 0 | 01:00.0 | 00:03.1 | `GPU-4673cc7d…` | GA104 | x16 | 4 |
| 1 | 21:00.0 | **20:03.1** | `GPU-88074816…` | GA104 | **x8** | 4 |
| 2 | 22:00.0 | **20:03.3** | `GPU-8cb4f7cc…` | GA104 | **x8** | 4 |
| 3 | 41:00.0 | 40:01.1 | `GPU-56c7aa62…` | GA106 | x16 | 4 |

`all_x16` = **false** (two cards), `all_gen4_under_load` = true, `bf16_autocast.all_ok` = true
(relative error vs FP64 3.10e-3 on every card, identical to `021` and `023`, under the 5e-3
tolerance). Gen-1 at idle is ASPM down-training, as on every capture. No PCIe, AER or NVRM errors in
the boot journal.

## What moved, against `023`

| root port | in `023` | in `027` | link |
|---|---|---|---|
| 00:03.1 | `GPU-4673cc7d` (index 0) | `GPU-4673cc7d` (index 0) | x16 |
| 20:03.1 | empty | **`GPU-88074816` (index 1)** | **x8** |
| 20:03.3 | `GPU-88074816` (index 1) | **`GPU-8cb4f7cc` (index 2)** | **x8** |
| 40:01.1 | `GPU-56c7aa62` (index 2) | `GPU-56c7aa62` (index 3) | x16 |
| 40:03.1 | `GPU-8cb4f7cc` (index 3) | empty | — |

Two things the capture settles and one it cannot:

1. **`GPU-4673cc7d` did not move, electrically or physically.** It is on root port `00:03.1` at
   x16, exactly as in `023`, and keeps index 0. The 2026-09-14 plan was to exchange its slot with
   `GPU-8cb4f7cc`'s so that a soak would separate *card* from *position*. The user's account
   (2026-09-15): both cards were taken out on their risers and refitted, and by chance the same card
   went back into the GPU 0 slot at the same physical position; what changed is the riser cable that
   ran under that slot and part-blocked the card's fan intake, which was moved. **So `028` tests the
   intake-blockage hypothesis, not slot-vs-card**; the slot-vs-card question stays open for a later
   swap if `028` does not settle the heat.
2. **The x8 pair is one bifurcated x16.** `20:03.1` and `20:03.3` are two ports of the same root
   complex, each at x8 with x16 cards behind them; in `023` `20:03.1` was unused and `20:03.3` was
   already x8. The riser move put a second card on that complex. `GPU-8cb4f7cc`, x16 in every
   earlier capture, is x8 now; `GPU-88074816` stays x8, as it was in `017`, `020` and `023`.
3. Index ↔ UUID changed for indices 2 and 3 (`GPU-56c7aa62` is index 3 now; `GPU-8cb4f7cc` index 2).
   Everything in `docs/04` that names GPU 0 as the hot card still refers to `GPU-4673cc7d`.

## Why x8 on two cards costs nothing measurable

Every collective on this rig is host-bounced with `NCCL_P2P_DISABLE=1` and runs at
$\lambda_{link}$ = 3.59 GB/s per rank (`002-nccl`, `local_3060.yaml`). PCIe 4.0 x8 is ≈ 16 GB/s per
direction nominal and the pinned host link measured 24–27 GB/s at x16 (`011`), so the host bounce,
not the lane count, sets the collective rate by a factor of 4–7. The DDP step is 86 ms of all-reduce
on a 651 ms compute step (`024`), and that 86 ms is the 3.59 GB/s figure. Nothing in `06 §3`'s
budgets or in `018`/`026` depends on lane width. Where it *could* show: S0/S10, which stripe an
expert across cards and are scored against $\lambda_c$ = $\lambda_{link}$ — still host-bounced, still
3.59 GB/s. `06 §1` is corrected to the new map; the x8 sentence there already said "immaterial for
throughput — re-read it, never assume it", and this capture is that re-read.

## Interpretation

Silent on H1–H19. The rig is back with four cards at gen 4 under load and BF16 intact. The
acceptance soak on this layout is `028`, whose decisive reading is which UUID is hot: if
`GPU-4673cc7d` still runs 84–86 °C at 100 % fan in the same position, the riser-cable intake blockage
was not the cause, and the slot swap that was planned (or a repaste) is the next lever.
