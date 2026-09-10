# 017-thermal-soak-riser-respaced — L5 DDP soak after the re-slotting (riser + upright bracket)

2026-09-10 19:31:50–19:43:36 · `scripts/bench/thermal_soak.sh 017-thermal-soak-riser-respaced 1100` ·
same rung (L5:4), micro-batch, power limit (170 W) and sampler as `015-thermal-soak-170w` and
`016-thermal-soak-bottom-fans` · cards started **cold** (40–41 °C, machine booted 19:16 after the
rebuild) · case change since 016: the **cards have been re-slotted** using the upright bracket, the
vertical kit and the 900 mm riser, so the four cards are no longer back to back. Bottom intakes,
top exhaust, 3 side intakes and 1 rear exhaust unchanged from 016.

**Bears on:** no hypothesis. Interim reading of the I23 remedy after the spacing fix.
**This is not a completed acceptance soak of `04 §I23`** — the run was stopped by the user at
11.8 min of the requested 1100 steps to take the machine down for a BIOS memory investigation.
See "Status" below.

## New physical layout (checklist step 1, partially done)

Bus-ID → index map changed with the re-slotting. Old (015/016): bus 01/02/41/42 = GPU 0/1/2/3,
GA106 at bus 41. New:

| GPU | bus | chip | root complex | PCIe width | PCIe max |
|---|---|---|---|---|---|
| 0 | 01:00.0 | GA104 | pci0000:00 / 00:03.1 | **x16** | x16 gen4 |
| 1 | 21:00.0 | **GA106** | pci0000:20 / 20:03.1 | **x8** | x16 gen4 |
| 2 | 22:00.0 | GA104 | pci0000:20 / 20:03.3 | **x8** | x16 gen4 |
| 3 | 41:00.0 | GA104 | pci0000:40 / 40:01.1 | **x16** | x16 gen4 |

Link speed reads 2.5 GT/s on all four at idle (ASPM downclock, not a fault). **Two cards negotiate
x8, not x16** — GPUs 1 and 2 sit on two root ports under the same host bridge, the signature of one
x16 slot bifurcated x8/x8. Per `04 §I23` checklist step 1 this needs a BIOS check. It is not
expected to bind: collectives are host-bounced at 3.59 GB/s (`002-nccl`), far under even gen4 x8.
`nvidia-smi topo -m` reports PHB between GPU 1 and 2, NODE for every other pair. No PCIe or AER
errors in the boot journal. Which slot or bracket physically holds which card is **not yet
recorded** — that needs the user's eyes on the case.

## Measured

Load window = the 70 samples per card above 1000 MHz (19:32:00–19:43:36, 11.6 min of load).
Throttle flag 0x20 = `sw_thermal_slowdown`; 0x1 = idle clocks, not a throttle.

| GPU | temp start → max | clock mean (min) | mean power | max fan | first throttle | throttled samples |
|---|---|---|---|---|---|---|
| 0 | 40 → **78 °C** | 1918 (1912) MHz | 141 W | 93 % | never | **0 / 70** |
| 1 | 40 → **70 °C** | 1932 (1927) MHz | 135 W | 85 % | never | **0 / 70** |
| 2 | 41 → **70 °C** | 1916 (1905) MHz | 142 W | 85 % | never | **0 / 70** |
| 3 | 41 → **66 °C** | 1952 (1950) MHz | 139 W | 82 % | never | **0 / 70** |

Last-5-min window (last 30 samples), the same columns the script prints: max 78 / 70 / 70 / 66 °C,
clock mean 1912 / 1927 / 1905 / 1950 MHz with **min = mean to the MHz on every card**, mean power
144 / 136 / 144 / 140 W, 0/30 throttled everywhere.

Against the two earlier soaks (worst card first):

| | 015 (no bottom fans) | 016 (bottom fans + top exhaust) | 017 (re-slotted) |
|---|---|---|---|
| max temp, hot cards | 93 °C | 93 °C | **78 °C** |
| cards reaching 93 °C | 3 of 4 | 3 of 4 | **0 of 4** |
| time to first throttle | 50–70 s | 110–130 s | **never (11.6 min)** |
| worst-card steady clock | 328 MHz | 1518 MHz | **1912 MHz** |
| worst-card clock floor | 225 MHz | 990 MHz | **1905 MHz** |
| mean power per card | 84–104 W | 120–132 W | **135–142 W** |
| DDP step median | 1277 ms | 826 ms | not measured (see Status) |

## Status — why this is not the acceptance pass

The user stopped the run at 11.8 min to shut down for a BIOS investigation of the host RAM (below).
Consequences:

- **No `result.json` and no step timing.** `bench_train_step` was killed before it wrote its
  summary, so 017 has **no `step_s_median`** to compare against the ≈ 760 ms cold floor. The third
  half of the acceptance criterion is untested.
- **11.6 min of load, not 15.** The `04 §I23` criterion asks for ≥ 15 min. The thermal half of the
  criterion (< 85 °C, 0 throttled samples) is met across every sample taken, and all four cards were
  flat in temperature and clock over the last 5 min, but the duration is short of the bar.
- The DDP workers survived the first `SIGTERM` to the launcher and were killed individually; the
  GPUs were confirmed released and idle before shutdown.

**I23 is therefore not closed.** Rerun as `019-thermal-soak-…` for the full 1100 steps after the
BIOS work, and only then record a verdict. `018` stays reserved for the warm re-bench of checklist
step 3.

## Host RAM — the second upgrade did not take effect

Checked the same session. The board is an ASRock WRX80 Creator R2.0, BIOS 10.03 (12/19/2025).

- `MemTotal` = 131 728 696 kB ≈ 125.6 GiB, i.e. **unchanged from the 128 GB in `06 §1`**.
- Kernel at boot: `DMI: Memory slots populated: 4/8`.
- `dmidecode -t 17`: channels A–D hold 32 GiB `HMA84GR7MFR4N-TF` (SK Hynix 2Rx4 registered ECC);
  channels **E–H report `No Module Installed`**, i.e. the BIOS could not read their SPD at all.
- The user reports **all eight sticks are physically inserted**.
- **Configured memory speed is 2133 MT/s**, the JEDEC fallback. The installed part is a 2933 MT/s
  module and `06 §1` claims DDR4-3200, so that line is wrong twice over and needs correcting once
  the population is fixed.
- With 4 DIMMs on an 8-channel controller the machine runs **4-channel**, roughly half the
  ≈ 205 GB/s `06 §1` assumes. No host-memory bandwidth claim in `06` has been measured.
- `amd64_edac` reports 12 ranks × 16 GiB = 192 GiB over 6 channels. That contradicts both SMBIOS and
  `MemTotal` and is a **driver misreport**; dmidecode is authoritative. Recorded so the next reader
  does not chase it.

Leading hypothesis is stale cached memory training (Memory Context Restore / Fast Boot) surviving
the DIMM change, which fits both the unenumerated channels and the fallback speed. The user is
checking BIOS before reseating. Decisive hardware test if BIOS fails: move a known-good stick from
channel A to channel E and reboot — E reads 32 GiB ⇒ the new sticks are at fault; E still empty ⇒
the slot or the CPU socket is. **Caution recorded for the BIOS session:** a CMOS clear will also
discard whatever PCIe settings produced the x8 links above, so note them first.

## Interpretation

**Spacing was the constraint, as `015` and `016` argued, and the re-slotting removes it.** With the
cards no longer back to back, every card holds 1905–1950 MHz against a 2115 MHz maximum for the
whole 11.6-minute load window, no card ever raises `sw_thermal_slowdown`, and the hottest card peaks
at 78 °C against 93 °C in both earlier soaks. The clock floor equals the clock mean to the MHz on
all four cards, which is the signature of a genuinely un-throttled run rather than a run that
happened to avoid the threshold; in 016 the worst card's floor was 990 MHz against a 1518 MHz mean.
Power draw rising to 135–142 W per card, from 84–104 W in 015 and 120–132 W in 016, says the cards
are finally free to spend their thermal headroom. GPU 0 at 78 °C is 8–12 °C above the other three
and is the card to watch in the full soak; it is also the only one whose fan reached 93 %.

The caveat is duration and the missing step time: 11.6 min is not the 15 min the criterion asks for,
and without `result.json` there is no throughput number to put beside the ≈ 760 ms cold floor. The
thermal picture is nevertheless unambiguous and there is no sign of a slow drift — the last 5 min
are flat. Expect the full soak to pass, but **no training run may start until `019` records it**,
and the host-RAM and PCIe-width questions above should be settled in the same shutdown so that one
reboot closes all three.
