# 037-job-overhead-eta — the owner stages' efficiency against batch (ADR-051 §2–3): **η = 1 up to 48 sequences, then falls to 0.60–0.62 at 256** for attention and both skeleton halves; `post` stays at peak. The fixed costs repeat `035` within 2 %

2026-11-03 · idle rig, by hand · tree `0b48da1` (clean) · driver 615.71.09 ·
`python -m scripts.bench.bench_job_overhead --id 037-job-overhead-eta` (defaults: `132-l5-small-s0` in bf16, owner kinds at 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256 sequences, the expert also at 512 and 1024 rows, 1792 keys, collectives at $z$ 16–1024, 50 timed calls after 10 warm-up per point). Four processes, one per card, ran the same sweep at once. Wall-clock 11 s. `n_thermal` 0. Peak memory 2.36 GiB on rank 0. Everything is in `result.json`.

**Bears on:** no hypothesis. This supplies the whole `software` section of `sim/scenarios/local_3060.yaml` (ADR-051 §3) for S0's re-score (`038`). `035` stays as the record ADR-050 used.

## Fixed cost per job and `compose` (as ADR-050 §1–3)

| | `skel_in` | `attn` | `post` | `skel_out` | expert | $t_{coll}$ |
|---|---|---|---|---|---|---|
| $t_0$, `037` | 1.363 ms | 1.115 | 0.084 | 1.516 | 0.191 | 0.051 |
| $t_0$, `035` | 1.368 | 1.107 | 0.084 | 1.508 | 0.190 | 0.053 |

`compose` = **sum** again (worst relative error 0.31; max 0.59). At one row the four cards agree within 2 % (attention 1.108–1.126 ms), and at 256 sequences within 1.4 % (3.31–3.36 ms).

## η per owner kind (ADR-051 §2: min(1, roofline / ($t$ − $t_0$)))

| sequences | ≤ 48 | 64 | 96 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|
| `skel_in` measured (ms) | 1.36–1.83 | 2.218 | 2.844 | 3.472 | 4.850 | 6.072 |
| `skel_in` roofline (ms) | ≤ 0.62 | 0.788 | 1.131 | 1.475 | 2.162 | 2.848 |
| **η `skel_in`** | 1 | **0.921** | **0.764** | **0.699** | **0.620** | **0.605** |
| `attn` measured (ms) | 1.12–1.32 | 1.519 | 1.814 | 2.146 | 2.752 | 3.346 |
| `attn` roofline (ms) | ≤ 0.27 | 0.353 | 0.524 | 0.696 | 1.039 | 1.383 |
| **η `attn`** | 1 | **0.874** | **0.751** | **0.675** | **0.635** | **0.620** |
| `skel_out` measured (ms) | 1.52–1.95 | 2.353 | 3.093 | 3.726 | 5.295 | 6.541 |
| `skel_out` roofline (ms) | ≤ 0.76 | 0.932 | 1.275 | 1.618 | 2.305 | 2.992 |
| **η `skel_out`** | 1 | 1 | **0.809** | **0.732** | **0.610** | **0.595** |
| **η `post`** | see below | 1 | 1 | 1 | 1 | 1 |

Up to 48 sequences the measured work term is no larger than the roofline: the fixed cost hides it, so η is capped at 1 (ADR-051 §2). From 64 sequences on, the eager stack's KV reads and skeleton GEMMs run at 60–92 % of peak, and the efficiency flattens near 0.6 by 192–256.

`post` is flat at 0.084 ms over the whole sweep. Its work term is ≈ 0, so at 2–16 sequences the ratio is timing noise (0.12 / 0.20 / 0.78 / 0.66). These points are recorded as measured, not cleaned. In the sim they add 0.04–0.29 µs to an 84 µs job.

## Collectives

As `035`: the dispatch phase runs 19–23 % above $2 t_{coll}$ + bytes/λ (0.130 / 0.144 / 0.173 / 0.330 ms against 0.105 / 0.113 / 0.144 / 0.267), and the return phase within 3–20 %. ADR-051 leaves the dispatch's host read-back out.

## Interpretation

The owner stages reach peak only while the fixed cost dominates. Past 48–64 sequences they run at about 60 % of it, so ADR-043 decision 6's peak roofline under-charges them by up to 1.65×. That is the size of `036`'s miss at $z$ = 1024. The η values are a property of this stack (PyTorch 2.13 eager plus the reference decode) at `132`'s shapes and 1792 keys, like $t_0$, and go into `local_3060.yaml` as measured. Beyond 256 sequences the sim clamps at the 256 value, which is an extrapolation.
