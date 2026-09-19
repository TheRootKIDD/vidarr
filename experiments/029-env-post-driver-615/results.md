# 029-env-post-driver-615 — same bus map, same links, same BF16; only the software moved (driver 610.57 → 615.71, kernel 7.2.4 → 7.2.5)

2026-09-19 · `python -m scripts.bench.bench_env --id 029-env-post-driver-615` · first capture after
the reboot that loaded the NVIDIA 615.71.09 kernel module. The package upgrade landed at 09:39 under
the running `queue_small_2` and burnt 123–129 at init (`nvmlInit_v2() failed: Driver/library version
mismatch`, `docs/04` 2026-09-19). Uptime at capture ≈ 1 min; nothing else on the GPUs.

**Bears on:** no hypothesis. Rig characterisation; step 2 of the post-driver-upgrade pick-up. The
`docs/04` 2026-09-19 entry said "rig id 029" for the soak; env takes 029 and the soak takes `030`.

## Bus map, keyed by UUID — unchanged against `027`

| GPU | bus | root port | UUID | width | gen under load |
|---|---|---|---|---|---|
| 0 | 01:00.0 | 00:03.1 | `GPU-4673cc7d…` | x16 | 4 |
| 1 | 21:00.0 | 20:03.1 | `GPU-88074816…` | x8 | 4 |
| 2 | 22:00.0 | 20:03.3 | `GPU-8cb4f7cc…` | x8 | 4 |
| 3 | 41:00.0 | 40:01.1 | `GPU-56c7aa62…` | x16 | 4 |

`all_x16` = false (the bifurcated x8/x8 pair of `027`, immaterial under the host bounce),
`all_gen4_under_load` = true, `bf16_autocast.all_ok` = true with the relative error identical to
`027`. Index ↔ UUID is as in `027`/`028`: index 0 is still `GPU-4673cc7d`.

## What differs from `027` (full field-by-field diff of `result.json`)

| field | `027` | `029` |
|---|---|---|
| driver | 610.57.04 | **615.71.09** (open kernel module) |
| kernel | 7.2.4-200.fc44 | **7.2.5-200.fc44** |
| `total_memory_bytes`, GPU 0 | 12 484 476 928 | **12 543 590 400** (+56.4 MiB) |
| `total_memory_bytes`, GPU 1–3 | 12 487 622 656 | **12 543 590 400** (+53.4 MiB) |
| git hash | `0f3c31e` | `1b54435` |

Nothing else differs bar date, free disk and a 272 kB change in host `mem_total`. The new driver
reports ≈ 55 MiB more usable memory per card and the same figure on all four; the 10 GB per-GPU cap
(`06`) is a project rule, not this number, so no budget moves. Idle: 13 MiB used on GPUs 0–2, 25 MiB
on GPU 3, no compute apps; power limit 170 W on every card, unchanged. Boot journal: NVRM load line
for 615.71.09 only — no Xid, no PCIe bus errors (the `_OSC: platform does not support [AER LTR DPC]`
lines are present on every boot of this board).

## Interpretation

Silent on H1–H19. The hardware side of the rig is bit-for-bit where `027` left it; the driver and
the kernel both moved, so **throughput numbers after this point are on a new software stack** and the
soak `030` doubles as the check that the L5 DDP step is still ≈ 737 ms (`028`). Loss numbers are not
expected to move with a driver, but 131–137 and 116–122 now sit on different drivers — recorded here
so that any seed-pair gap outside σ in the L3 pair (122 on 610.57, 131 on 615.71) is read with that
in mind.
