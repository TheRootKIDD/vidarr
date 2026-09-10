# 019-thermal-soak-riser-respaced-full — acceptance soak aborted: GPU 0's fan does not run

2026-09-10 20:43:29–20:48:41 · `scripts/bench/thermal_soak.sh 019-thermal-soak-riser-respaced-full 1100` ·
same rung (L5:4), micro-batch, power limit (170 W) and sampler as `015`/`016`/`017` · cards started
**cold** (44 / 44 / 45 / 37 °C, machine booted 20:39 after the RAM reseat) · physical layout as
`017` plus whatever the 20:04 riser re-slotting changed; **no cooling hardware added since 017**.

**Bears on:** no hypothesis. Second attempt at the `04 §I23` acceptance soak.
**Result: fail, and not for a thermal-design reason.** GPU 0's fan reported **0 % for every sample
of the run**, from 44 °C at idle to 93 °C under load, while the other three ramped to 77–81 %. The
run was stopped at 5.2 min. **I23 stays open.**

## Measured

Load window = the 31 samples per card above 400 MHz (20:43:39–20:48:41, 5.0 min of load).
Throttle flag 0x20 = `sw_thermal_slowdown`; 0x1 = idle clocks, not a throttle.

| GPU | temp start → max | clock mean (min) | mean power | max fan | first throttle | throttled samples |
|---|---|---|---|---|---|---|
| 0 | 44 → **93 °C** | 1249 (525) MHz | 100 W | **0 %** | **91 s** | **23 / 31** |
| 1 | 44 → 66 °C | 1921 (1920) MHz | 119 W | 81 % | never | 0 / 31 |
| 2 | 45 → 67 °C | 1944 (1942) MHz | 114 W | 81 % | never | 0 / 31 |
| 3 | 37 → 63 °C | 1961 (1950) MHz | 115 W | 77 % | never | 0 / 31 |

GPU 0's trace is a clean uncooled-card curve: 44 °C at idle, 55 °C ten seconds into load, then a
steady ≈ 3.5 °C per 10 s climb through 82 °C at 80 s, `sw_thermal_slowdown` raised at 91 s, 93 °C
by 130 s, and pinned at 92–93 °C for the remaining 3 min with the clock sawing between 525 and
1492 MHz. Peak power 148 W early, falling to ≈ 70 W once throttled — the card sheds power because it
cannot shed heat.

**After the run ended and all four GPUs were confirmed idle at 210 MHz**, GPU 0 sat at 90 °C and
took more than 4 min to reach 88 °C, with the fan still at 0 %, while GPUs 1–3 fell to 45–50 °C with
their fans at 71–76 %. That is passive cooling only.

GPUs 1, 2 and 3 reproduce `017`: flat clocks at 1920–1961 MHz against 2115 max, min = mean to within
2 MHz, peaks 63–67 °C, no throttling. **The spacing fix still holds for three of four cards.**

## Why this is a fan fault, not a thermal-design failure

`nvidia-smi` reports fan speed as a percentage of maximum, i.e. what the card's controller is
commanding. A card at 93 °C commanding 0 % is not a card whose cooling is inadequate; it is a card
whose fan is not being driven at all.

The decisive comparison is `017`, 70 min earlier: **GPU 0 was the card whose fan reached 93 %**, the
highest of the four, and it peaked at 78 °C. Same bus (01:00.0), same GA104, same root port
(00:03.1), so it is the same physical card. Between the two soaks the machine was opened twice — the
20:04 PCIe riser re-slotting and the RAM reseat before the 20:39 boot. **The fan worked before that
work and does not after it**, which puts the cause in the case, not in the card's design margin.

Candidates, in order: the 900 mm riser cable fouling the fan blades or the fan header (the user's
own first guess); the fan power lead unseated from the PCB at the connector; a cable resting against
the shroud so the controller reads a stall and gives up. All three are settled by looking, and the
first two are settled by hand. **No further soak is worth running until this is fixed** — 019 would
otherwise measure GPU 0's fan, not the cooling layout.

## Status

- **No `result.json` and no step timing**, for the second time: the soak was stopped inside its
  warmup-plus-run window, so `bench_train_step` never wrote its summary. The ≈ 760 ms cold-floor
  comparison is still untested.
- 5.0 min of load against the 15 min the `04 §I23` criterion asks for.
- The DDP launcher, its `torchrun`-spawned children and the `multiprocessing` workers each needed a
  separate `SIGTERM` — killing the launcher alone left four workers training on all four GPUs at
  full power for a further ≈ 3 min. Same trap as `017`; worth a `pkill -f bench_train_step` in the
  script's `trap` line.
- **I23 not closed.** The next acceptance soak takes a new id (`020` is taken by the env re-check,
  so `021`); `018` stays reserved for the warm re-bench of checklist step 3.

## Rig state recorded the same session (`020-env-post-rebuild`)

**Host RAM: the upgrade has taken.** Kernel at boot: `DMI: Memory slots populated: 8/8`.
`MemTotal` = 263 750 232 kB ≈ 251.5 GiB, up from 125.6 GiB. `dmidecode -t 17` shows all eight
channels A–H populated with 32 GiB `HMA84GR7MFR4N-TF` (SK Hynix 2Rx4 registered ECC, DDR4-2933).
The machine now runs **8-channel**. Serials: `277BCC7C` / `277BCD8D` / `914DD9A6` / `914DD9D6` /
`277BCBA1` / `277BCD67` / `277BCD6E` / `277BCD81` in A / B / C / D / E / F / G / H. Note this kills
the batch reading in `04`'s 20:04 update: the split is **six `277B…` and two `914D…`**, not four and
four, so neither batch maps onto "the original four sticks".

**But the speed is still wrong.** Configured memory speed is **2133 MT/s on all eight**, the JEDEC
fallback, against the part's rated 2933 MT/s — unchanged from every previous boot, and now clearly
*not* a symptom of the missing DIMMs. `06 §1`'s DDR4-3200 is wrong; the ceiling here is 2933.
Recovering 2933 is a **37 % memory-bandwidth gain for free** and is a BIOS-side question
(DRAM frequency currently Auto).

**Host memory bandwidth, first measurement ever taken on this box.** Ad-hoc threaded STREAM-style
probe, 12 threads, 4 GiB arrays, best of 5:

| kernel | GB/s |
|---|---|
| copy | 70.8 |
| scale | 47.1 |
| add | 51.3 |
| triad | 29.3 |

This is a crude probe (numpy over threads, temporaries not eliminated), so read every figure as a
**lower bound**. It still settles the channel question: 8-channel DDR4-2133 has a theoretical peak
of 136.5 GB/s and 4-channel has 68.3 GB/s, and the copy figure of 70.8 GB/s **exceeds the 4-channel
peak**, so the controller is genuinely running more than four channels. It also shows `06 §1`'s
≈ 205 GB/s assumption is unreachable at 2133 MT/s under any interleaving. Before any of this becomes
a `sim/scenarios/local_3060.yaml` input it needs a real `scripts/bench/bench_membw.py` under its own
id, per the CLAUDE.md rule that every scenario number carries a source line.

**PCIe.** Bus map held across the 20:39 reboot: bus 01 / 21 / 41 / 42 = GPU 0 / 1 / 2 / 3, GA106 at
bus 41 = GPU 2. Root ports 00:03.1, 20:03.3, 40:01.1, 40:03.1.

| GPU | bus | chip | width idle | width under load | gen idle | gen under load |
|---|---|---|---|---|---|---|
| 0 | 01:00.0 | GA104 | x16 | **x16** | 1 | **4** |
| 1 | 21:00.0 | GA104 | x8 | **x8** | 1 | **4** |
| 2 | 41:00.0 | GA106 | x16 | **x16** | 1 | **4** |
| 3 | 42:00.0 | GA104 | x16 | **x16** | 1 | **4** |

**One card still at x8, down from two in `017`** — the riser re-slotting fixed one of them. All four
reach **gen 4 under load**; the gen-1 idle reading is ASPM downclocking, not a fault. No AER or PCIe
errors in the boot journal (the platform does not export AER at all: `_OSC: platform does not
support [AER LTR DPC]`). BF16 autocast passes on all four cards (relative error vs FP64 under the
5e-3 tolerance). Not expected to bind — collectives are host-bounced at 3.59 GB/s (`002-nccl`), far
under even gen4 x8 — but checklist step 1 still wants the BIOS check, and it is worth doing in the
same downtime as the fan.

## Interpretation

The cooling rebuild's verdict is unchanged and still un-recorded. Three of the four cards behave
exactly as `017` said they would: flat clocks near maximum, peaks in the 60s, no throttling, and no
drift over the load window. The fourth card tells us nothing about cooling because its fan never
turned, and the comparison with `017` — where that same card ran its fan hardest of the four — dates
the fault to the two case openings between the runs rather than to anything the soak was testing.

The useful outcome of this attempt is therefore not thermal. It is that the RAM upgrade finally took
(8/8, 251.5 GiB, 8-channel, confirmed by a bandwidth figure above the 4-channel ceiling), that the
2133 MT/s fallback survives a correct population and so is a BIOS setting rather than a symptom, and
that one of the two x8 links cleared. **Next shutdown should close three things at once**: the GPU 0
fan or riser cable, DRAM frequency to 2933 MT/s, and the x8 link on bus 21. Then run the full soak
as `021` and let it write `result.json`.
