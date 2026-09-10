# 021-env-gpu-remap — all four cards reach x16 gen4 for the first time

2026-09-10 21:33 UTC+2 · `python -m scripts.bench.bench_env --id 021-env-gpu-remap` · machine booted
21:26 after the user re-slotted the GPUs "to match the PCIe configurations" · no cooling hardware
added or removed since `017`.

**Bears on:** no hypothesis. Rig characterisation; closes the x8-link half of the `04` three-item
shutdown list and supplies checklist step 1's bus map.

## Result: the link problem is fixed

| GPU | bus | root port | chip | width | gen idle | gen under load |
|---|---|---|---|---|---|---|
| 0 | 01:00.0 | 00:01.1 | GA104 | **x16** | 1 | **4** |
| 1 | 02:00.0 | 00:03.1 | GA104 | **x16** | 1 | **4** |
| 2 | 41:00.0 | 40:01.1 | GA106 | **x16** | 1 | **4** |
| 3 | 42:00.0 | 40:03.1 | GA104 | **x16** | 1 | **4** |

`all_x16` = true, `all_gen4_under_load` = true, `bf16_autocast.all_ok` = true (relative error vs FP64
3.10e-3 on every card, under the 5e-3 tolerance). The gen-1 idle reading is ASPM down-training, not a
fault. **This is the first time on this rig that all four cards have negotiated x16.** `000-env`
claimed it, `017` measured two cards at x8 and `020` measured one.

No PCIe or AER errors in the boot journal, and no `NVRM`/Xid entries. The platform still does not
export AER at all (`_OSC: platform does not support [AER LTR DPC]`), so absence of AER logging is not
evidence either way; the x8 warnings that do appear in the journal are for the chipset uplink at
`60:03.1`, not for any GPU. `amd64_edac` now reports 262 144 MB over `mc0` with `ce_count` = 0 and
`ue_count` = 0, which agrees with SMBIOS and `MemTotal` for the first time — the 192 GiB / 6-channel
misreport noted in `019` is gone.

## Stable card identity — the record `000-env` never had

`04`'s 20:04 update warns that the GPU index moves under re-slotting and must be re-read. The index
and the bus id are both properties of *position*, so neither identifies a card across a move. The
GPU UUID does. Recorded here so the next re-slotting is traceable:

| GPU | UUID | chip | VBIOS |
|---|---|---|---|
| 0 | `GPU-8cb4f7cc-d2a2-fb38-c9ee-14dc7f1515ab` | GA104 | 94.04.71.40.2F |
| 1 | `GPU-88074816-a56d-dc95-d744-6096efc59bfc` | GA104 | 94.04.71.40.2F |
| 2 | `GPU-56c7aa62-b07a-0d35-3792-44b0a78ef58d` | GA106 | 94.06.2F.40.95 |
| 3 | `GPU-4673cc7d-9e44-4578-a9ca-48215bc5ddc8` | GA104 | 94.04.71.40.2F |

GeForce cards report no serial (`[N/A]`), so the UUID is the only stable handle. **Add
`index,pci.bus_id,uuid` to `bench_env`'s capture** so this is never lost again.

## What moved, and what that implies for `019`

The user confirms **only the card that was in the x8 slot moved**; the other three stayed. Pairing
that with the root ports gives an unambiguous before/after:

| root port | in `020` | in `021` |
|---|---|---|
| 00:01.1 | empty | **GPU 0** (the card that moved out of the x8 slot) |
| 00:03.1 | GPU 0 — the card whose fan read 0 % | **GPU 1** — same card, did not move |
| 20:03.3 | GPU 1 at **x8** | empty |
| 40:01.1 | GPU 2 (GA106) | GPU 2 (GA106) |
| 40:03.1 | GPU 3 | GPU 3 |

**So `019`'s "dead fan" card is `021`'s GPU 1, and its fan is not dead.** It ran to 100 % duty under
load in `022` an hour later, having read 0 % through 93 °C in `019`. Nothing was done to the card
between the two runs except the re-slotting the user performed in the same downtime. That retires the
"fan lead unseated at the PCB" candidate and leaves the mechanical one `019` listed first: the riser
cable, or the bracket the user was fitting, fouled the fan or its header, and freeing the bracket
freed the fan. **No card in this rig has a hardware fan fault.**

## Interpretation

Two of the three items on `04`'s "next shutdown closes three things at once" list are settled by this
capture and by `022`: the x8 link is gone and the fan was never broken. The third, DRAM frequency,
is untouched and is now known to have been the wrong instruction — see `022` §Host memory and the
session log. The link result is not expected to change any throughput number, because collectives are
host-bounced at 3.59 GB/s (`002-nccl`), an order of magnitude under even gen4 x8; it matters because
`06 §1` asserted x16-on-all-four as a measured fact and until tonight that was false.
