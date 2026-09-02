# 000-env — rig characterisation

2026-09-02 · `scripts/bench/bench_env.py` · numbers in `result.json`

**Bears on:** no hypothesis. This is the provenance record for every later bench
and the ground truth behind `docs/06 §1`, whose nominal column it partly
contradicts.

## Measured

| Item | Measured | `docs/06 §1` nominal | Verdict |
|---|---|---|---|
| GPUs visible to torch | 4 × RTX 3060, `sm_86`, 28 SMs, 11.63 GiB usable | 4 × RTX 3060, SM 8.6, 28 SMs, 12 GB | agrees |
| GPU die | 3 × **GA104** (bus 01, 02, 42), 1 × **GA106** (bus 41) | "GA106" for all four | **doc wrong** |
| PCIe link, idle | x16, **gen 1** | x16 | idle downtrain at P8 |
| PCIe link, under load | **x16 gen 4 on all four** | x16 gen 4 | agrees |
| BF16 autocast | works on all four; rel. err vs FP64 = 3.10e-3 at K=4096 | BF16 ✔ | agrees |
| Host RAM | 125 GiB total | 128 GB | agrees |
| CPU | Threadripper PRO 3945WX, 12C/24T | same | agrees |
| Storage | 1 × **Samsung SSD 850 PRO 256 GB (SATA III)**, 31 GiB free on `/home` | "NVMe assumed (unmeasured)" | **doc wrong** |

Driver 595.80, CUDA 13.2 runtime; torch 2.13.0+cu130 on CPython 3.12.13 in
`.venv`. System python (3.14.7) carries a **CPU-only** torch and is not used —
`faiss-cpu` has no cp314 wheel, which fixed the venv at 3.12.

## Interpretation

The rig is as `docs/06 §1` describes it on compute and on the link, and not on
storage. Three findings matter downstream.

**Storage is SATA, not NVMe.** A Samsung 850 PRO is a ~0.5 GB/s SATA-III drive
with no NVMe queue depth behaviour. `docs/06 §1–2` assumes 3–7 GB/s for the KV
cold tier and `docs/02 §7` builds a three-tier model on it; the real third tier
is roughly an order of magnitude slower and its random-read profile differs in
kind. `bench_tiers` measures what is actually there; the tier table and S6's
cold-tier bandwidth need the measured number, not the assumed one. Recorded as
open question **I10**.

**The four cards are not one part.** Same `sm_86` and same 28 SMs, so the FLOPs
budget is unaffected, but GA104 and GA106 differ in memory subsystem and boost
behaviour. Per-card throughput is therefore measured per card, not once and
multiplied by four, and in DDP the slowest card sets step time. Recorded as
**I11**; `bench_gemm` settles whether the difference is measurable.

**Disk headroom is 31 GiB.** `docs/06 §7` plans a 10 B-token `uint16` memmap at
≈ 20 GB plus FAISS indexes, which does not fit beside the venv and the OS.
Recorded as **I12**; a 1 TB HDD is available to install as bulk storage, which
solves capacity but not cold-tier bandwidth — an HDD is slower than the SSD
already present, so the SSD stays the measured cold tier.

The idle PCIe gen-1 reading is not a fault: Ampere downtrains the link at P8 and
retrains under traffic. Confirmed x16 gen 4 on all four cards while host→device
copies are in flight, so `bench_nccl` runs against a full-width link.

## Addendum, 2026-09-02 — question ids renumbered

The three findings above were first written against ids I7/I8/I9. `docs/04` has
since allocated those to different internal questions (I7 = v2 of the note
stored, closed; I8 = which pretrained sentence encoder; I9 = which corpus slice
to embed). Renumbered here to **I10** (SATA, not NVMe), **I11** (GA104/GA106
mix) and **I12** (31 GiB disk headroom); the commit message of `de13e02`
predates the renumbering and still names the old ids.

## Addendum, 2026-09-02 — I10 and I12 downgraded: NVMe and HDD are available

Hardware that was not present when this was measured is available to install: a
**4 TB NVMe** (moving from another workstation) and a **1 TB HDD**, both pending
a power-off.

This changes the standing of two of the three findings.

**I10 (SATA, not NVMe) — downgraded from a spec problem to a scheduling one.**
The three-tier model of `docs/06 §2` and `docs/02 §7` assumed a 3–7 GB/s cold
tier and got a ~0.5 GB/s SATA drive. With the 4 TB NVMe installed the assumption
is met by real hardware, so `docs/06 §1`'s "NVMe assumed (unmeasured)" becomes
correct rather than wrong, and `bench_tiers` measures the tier the design
actually specifies. **Consequence for sequencing: `bench_tiers` should not be run
against the SATA drive.** Measuring a cold tier that is about to be replaced
produces a `local_3060.yaml` value that is obsolete on arrival, and $\beta_{cold}$
is a scenario input other results are scored against. `bench_tiers` therefore
moves to after the install; every other bench is storage-independent and runs
before it.

**I12 (31 GiB headroom) — resolved by the same install.** 4 TB plus 1 TB against
a ≈ 20 GB tokenised corpus and FAISS indexes removes the constraint entirely,
including the `docs/06 §7` rule that the corpus lives as a `uint16` memmap on
NVMe, which becomes satisfiable as written. Role split: NVMe carries the corpus,
the KV cold tier and the indexes; the HDD is bulk storage (archives, checkpoints
kept for provenance) and is **not** a KV tier — spinning disk is slower than the
SATA SSD already present, which is in turn ~10× slower than the NVMe.

**I11 (GA104/GA106 mix) is unaffected** and still wants `bench_gemm` per card.

Note that ADR-021 has meanwhile shrunk the persistent cache by ~$r_{max}$×, and
`docs/06 §5.4` sets tier limits artificially small on purpose to exercise the
warm/cold paths. So the cold tier was already under less pressure than the v1
design implied; the NVMe removes the concern rather than merely easing it.

**Installed 2026-09-02**, same day: NVMe at `/mnt/nvme`, HDD at `/mnt/hdd`, both
ext4. The cold tier is measured in `012-tiers-nvme`; I10 and I12 are closed there.
