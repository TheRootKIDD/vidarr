# 123-l3-small-s1 — failed at initialisation (driver/library version mismatch); id burnt, reruns as `131`

Started by `queue_small_2.sh` on 2026-09-19 at ≈ 09:52 and exited 1 within 5 s, before the first step:
`torch.distributed.DistBackendError: NCCL error … nvmlInit_v2() failed: Driver/library version mismatch`.
The NVIDIA user-space packages had been upgraded to 615.71.09 at 09:39–09:40 while the kernel was still
running the 610.57.04 module; every CUDA process started after that fails until a reboot. `122`, already
running, finished cleanly. Full record in `docs/04` (2026-09-19).

**Bears on:** nothing. Zero steps, no numbers. Ids are append-only and consumed once started
(`101`, `118`), so this id is burnt; **L3 seed 1 at 2.5e9 tokens reruns as `131`** in
`queue_small_3.sh`.
