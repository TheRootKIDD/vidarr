# 023-env-post-bracket-move — the layout changed again; one card is back on the x8 link

2026-09-10 23:13 UTC+2 · `python -m scripts.bench.bench_env --id 023-env-post-bracket-move` · machine
booted 22:27 after the user's hardware work on RAM and GPUs · first capture on the post-bracket-move
layout.

**Bears on:** no hypothesis. Rig characterisation; checklist step 1 and step 2 of the order of work in
`docs/04` (2026-09-10 late night).

## Bus map, re-keyed by UUID

`021` recommended adding `index,pci.bus_id,uuid` to `bench_env`'s capture so that tracking a card
across a re-slotting stops being a hand reconstruction. That is done: `pcie_state()` now carries the
UUID and the upstream root port (read from the sysfs device chain) on every row.

| GPU | bus | root port | UUID | chip | width | gen under load |
|---|---|---|---|---|---|---|
| 0 | 01:00.0 | 00:03.1 | `GPU-4673cc7d…` | GA104 | x16 | 4 |
| 1 | 21:00.0 | **20:03.3** | `GPU-88074816…` | GA104 | **x8** | 4 |
| 2 | 41:00.0 | 40:01.1 | `GPU-56c7aa62…` | GA106 | x16 | 4 |
| 3 | 42:00.0 | 40:03.1 | `GPU-8cb4f7cc…` | GA104 | x16 | 4 |

`all_x16` = **false**, `all_gen4_under_load` = true, `bf16_autocast.all_ok` = true (relative error vs
FP64 3.10e-3 on every card, the same value `021` measured, under the 5e-3 tolerance). The gen-1 idle
reading is ASPM down-training, as before.

## Two cards swapped positions, and the hot card went back into the x8 slot

Against `021`, keyed by UUID rather than by index:

| root port | in `021` | in `023` | link |
|---|---|---|---|
| 00:01.1 | `GPU-8cb4f7cc` (index 0) | empty | — |
| 00:03.1 | `GPU-88074816` (index 1) | `GPU-4673cc7d` (index 0) | x16 |
| **20:03.3** | empty | **`GPU-88074816` (index 1)** | **x8** |
| 40:01.1 | `GPU-56c7aa62` (index 2) | `GPU-56c7aa62` (index 2) | x16 |
| 40:03.1 | `GPU-4673cc7d` (index 3) | `GPU-8cb4f7cc` (index 3) | x16 |

Three of the four cards moved. `GPU-88074816` — the card that was thermally starved in `022`, and the
one whose fan read 0 % through 93 °C in `019` — now sits on root port `20:03.3`, which is the x8 port
that `017` and `020` both flagged and that `021` had emptied.

**This is a regression against `021` on the link, and it does not matter for throughput.** Collectives
on this rig are host-bounced at 3.59 GB/s (`002-nccl`), an order of magnitude under gen4 x8, so no
measured number moves. It matters only because `06 §1` was corrected two hours ago to cite `021` for
x16-on-all-four as measured fact, and that sentence is false again. `06 §1` is corrected below.

No AER or PCIe errors in the boot journal; the platform still does not export AER
(`_OSC: platform does not support [AER LTR DPC]`). Host RAM is unchanged and healthy: `MemTotal`
251.5 GiB, 8/8 channels.

## Fan and clock check before committing to a soak

Checklist step 3 asks for a fan check on any card the move touched, before a 15-minute soak. A 60 s
BF16 GEMM burn on all four cards at once (scratch script, not a benchmark, no id):

| GPU | UUID | idle | fan at +60 s | temp at +60 s | clock at +60 s | power |
|---|---|---|---|---|---|---|
| 0 | `GPU-4673cc7d…` | 42 °C, 0 % | 84 % | 71 °C | 1890 MHz | 169 W |
| 1 | `GPU-88074816…` | 45 °C, 0 % | 81 % | **68 °C** | 1935 MHz | 164 W |
| 2 | `GPU-56c7aa62…` | 44 °C, 0 % | 81 % | 67 °C | 1927 MHz | 160 W |
| 3 | `GPU-8cb4f7cc…` | 36 °C, 0 % | 78 % | 65 °C | 1942 MHz | 169 W |

All four fans drive, so no fan fault survives the move. Every card holds 1890–1942 MHz at 160–169 W of
its 170 W limit, i.e. at the power cap and not throttling, on a workload heavier than the L5 training
step. **`GPU-88074816` reaches 68 °C where `022` had it pinned at 93 °C with its fan at 100 %.** A 60 s
burn is not a soak and this is not the I23 verdict — but it is the first evidence that the starved
card's position has actually changed, and it clears the run to proceed.

## Interpretation

Rig characterisation only; silent on H1–H19. Two facts for the record: the card that has set the DDP
step since `022` is no longer heat-limited at one minute of full load, and it is back on the x8 link,
which costs nothing measurable here. **Still owed and not answerable from a command: which slot or
bracket physically holds which card.** That half of checklist step 1 has been open since `017`.

The acceptance soak on this layout is `024`. `018` stays reserved for the warm re-bench of checklist
step 3.
