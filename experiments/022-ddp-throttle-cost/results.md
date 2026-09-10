# 022-ddp-throttle-cost — three cards are fixed, one is starved, and it sets the DDP step

2026-09-10 21:38:14–21:45:13 · `scripts/bench/thermal_soak.sh 022-ddp-throttle-cost 420` ·
rung `L5:4`, micro-batch 4 × 4 ranks, 170 W power limit, same sampler as `015`–`019` ·
physical layout as the user left it at the 21:26 boot (GPUs re-slotted for x16; **no cooling hardware
added since `017`**).

**Bears on:** no hypothesis. Rig characterisation.

**This is deliberately not the `04 §I23` acceptance soak.** I23 asks for 1100 steps (≈ 15 min) with
the cards **cold**, on the **final** layout. Neither held: 420 steps, and the cards were warm from an
ad-hoc fan check ten minutes earlier (GPU 1 entered the load window at 73 °C, the others at 48–50 °C).
The layout is also not final — the bracket relocation under discussion will change it, and `04`
already records that re-slotting invalidates a thermal result. The question this run *does* answer is
narrower and does not need a cold start: **what does one starved card cost the whole DDP job?**

## Measured

Load window = the 41 samples per card above 400 MHz. Throttle flag `0x20` = `sw_thermal_slowdown`.

| GPU | temp start → max | clock mean (min) | mean power | max fan | first throttle | throttled samples |
|---|---|---|---|---|---|---|
| 0 | 50 → 63 °C | 1958 (1950) MHz | 117 W | 78 % | never | 0 / 41 |
| 1 | 73 → **93 °C** | **1499 (1320)** MHz | 113 W | **100 %** | **40 s** | **37 / 41** |
| 2 | 48 → 66 °C | 1943 (1942) MHz | 115 W | 80 % | never | 0 / 41 |
| 3 | 49 → 66 °C | 1921 (1912) MHz | 122 W | 81 % | never | 0 / 41 |

DDP step, identical across all four ranks because the collective is lockstep:

| | seconds |
|---|---|
| `step_s_median` | 0.968 |
| `step_s_min` | 0.838 |
| `step_s_p90` | 0.999 |

33 847 tok/s aggregate, 8462 tok/s per card, peak 3.59 GiB, 77.5 M params, projected `screen` 8.2 h.

## The fan is not the fault this time — the air is

`019` failed because GPU 0's fan read 0 % at 93 °C. That card is this run's GPU 1 (`021`
§What moved), and here it runs its fan at **100 %** and still reaches 93 °C, throttling at 40 s and
holding `sw_thermal_slowdown` for 37 of 41 samples. A card at full fan duty that cannot stay under
93 °C is not a controller fault and not a power fault — mean draw is 113 W of a 170 W limit, which is
`015`'s finding again. It is a card with no air to move. The user reports it sits in an ordinary
motherboard slot, **side by side with another card**, because the bottom bracket consumes the space
that would otherwise separate them.

The other three are the best this rig has ever recorded: 63–66 °C peak, clocks flat within 8 MHz of
their means at 1921–1958 MHz against a 2115 max, zero throttled samples. **The cooling rebuild works.
It is being wasted by one slot position.**

## What the throttling costs

The un-throttled step is visible inside this run: `step_s_min` = 0.838 s is what the job does in the
first samples before GPU 1 crosses its limit, and the three healthy cards never leave full clock, so
that floor is set by GPU 1's brief cold window rather than by anything else.

| | step | vs this run's median |
|---|---|---|
| this run, median | 968 ms | — |
| this run, min (GPU 1 not yet throttled) | 838 ms | **−13.4 %** |
| `016`, three cards at 93 °C, bottom fans only | 826 ms | −14.7 % |
| cold floor quoted in `04` checklist step 2 | ≈ 760 ms | −21.5 % |

Read the `016` row with care — different layout, different session, and those cards were cold at the
start where GPU 1 here was not — but the direction is not in doubt and it is worth stating plainly:
**the current arrangement is no faster in DDP than the arrangement `016` measured before the bracket,
the riser and the re-slotting went in.** Spreading the throttling over three cards and concentrating
it in one produce about the same wall-clock, because DDP runs at the slowest rank either way. Three
healthy cards buy nothing while the fourth is at 1320 MHz.

At 13.4 %, a `screen` run goes from the 8.2 h this run projects to ≈ 7.1 h, so ≈ 1.1 h per rung and
≈ 5.5 h over the five queued `screen` rungs (106–110).

## Host memory — the `019` instruction was wrong

`019` and `04` both record the eight DIMMs as `HMA84GR7MFR4N-TF`, "SK Hynix 2Rx4 registered ECC,
**DDR4-2933**", and conclude that the 2133 MT/s configured speed is a JEDEC fallback worth ≈ 37 % of
host memory bandwidth "for free" if DRAM frequency is set to 2933 in BIOS.

**The part is a DDR4-2133 part.** In SK Hynix's DDR4 module part numbering the trailing speed code
gives both the grade and the timings: `TF` = DDR4-2133 15-15-15, `UH` = 2400 17-17-17, `VK` = 2666
19-19-19, `WM` = 2933 21-21-21, `XN` = 3200 22-22-22. `HMA84GR7MFR4N-**TF**` is therefore
PC4-17000 / DDR4-2133 CL15, which is what every vendor listing of that part number also says. A
2933 part would be `HMA84GR7MFR4N-WM`.

So 2133 MT/s is not a fallback and not a symptom. **It is the rated speed, correctly applied**, and
there is no free bandwidth to recover. Setting DRAM frequency to 2933 would be a 37 % overclock of
registered ECC modules, which is the one memory change on the shutdown list that could actually cost
something. **`04`'s instruction "(2) DRAM frequency to 2933 MT/s in BIOS" must not be carried out.**

This also re-reads the `019` bandwidth probe. 8-channel DDR4-2133 peaks at 136.5 GB/s, and the probe's
70.8 GB/s copy is 52 % of that — an ordinary figure for a threaded numpy probe, and no longer evidence
of anything being left on the table. It still exceeds the 68.3 GB/s 4-channel ceiling, so it still
confirms the 8-channel population.

**Confirmed by SMBIOS the same session.** `sudo dmidecode -t 17` on all eight modules, channels A
through H, reports `Part Number: HMA84GR7MFR4N-TF`, **`Speed: 2133 MT/s`**, `Configured Memory Speed:
2133 MT/s`, `Configured Voltage: 1.2 V`. The `Speed:` field is the SPD *maximum*, so the modules
themselves declare 2133 as their ceiling and the configured speed equals it. There is no gap between
rated and running, nothing to recover, and no BIOS change to make. `06 §1` corrected accordingly.

## Status

- `result.json` written, for the first time since `016` — the `pkill` in `thermal_soak.sh`'s trap
  held and no orphaned workers survived the run.
- **I23 stays open.** Not attempted here; it needs cold cards, 1100 steps and a final layout.
- Next acceptance soak takes id `023`. `018` stays reserved for the warm re-bench of checklist step 3.
