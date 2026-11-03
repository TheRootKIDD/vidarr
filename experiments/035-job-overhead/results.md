# 035-job-overhead — S0's stack has a fixed cost per job (ADR-050 §3): **an expert call 0.19 ms, an attention stage 1.11 ms, the skeleton halves 1.37 / 1.51 ms, `post` 0.084 ms, one collective 0.053 ms**; `compose` = sum

2026-11-03 · idle rig, by hand · tree `48c3ed2` (clean) · driver 615.71.09 ·
`python -m scripts.bench.bench_job_overhead --id 035-job-overhead` (defaults: `132-l5-small-s0` in bf16, rows 1–1024, 1792 keys, owner kinds up to 256 sequences, z 16–1024 for the collectives, 50 timed calls after 10 warm-up per point). Four processes, one per card, ran the same sweep at once. Wall-clock 10 s. `n_thermal` 0. Peak memory 2.36 GiB on rank 0. Everything is in `result.json`.

**Bears on:** no hypothesis. This supplies the `software` section of `sim/scenarios/local_3060.yaml` for S0's re-score (`036`).

## Time per job vs batch (median over the four cards; ms)

| rows / sequences | 1 | 4 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|
| `skel_in` (embedding + 2 early blocks) | **1.368** | 1.399 | 1.535 | 1.662 | 2.214 | 3.468 | 6.067 | — | — |
| `attn` (DepthNorm, attention at 1792 keys, router, payload) | **1.107** | 1.126 | 1.118 | 1.181 | 1.515 | 2.145 | 3.339 | — | — |
| `post` (gate-weighted sum, residual) | **0.084** | 0.084 | 0.084 | 0.084 | 0.084 | 0.085 | 0.085 | — | — |
| `skel_out` (2 final blocks + head) | **1.508** | 1.552 | 1.661 | 1.796 | 2.353 | 3.725 | 6.546 | — | — |
| `expert` (one call, selection included) | **0.190** | 0.186 | 0.187 | 0.188 | 0.191 | 0.185 | 0.185 | 0.216 | 0.334 |

The four cards agree within 3 % at one row (expert 0.188–0.193 ms, attention 1.101–1.114 ms). **t₀** is the one-row column. An expert call is flat from 1 to 256 rows. The sim's roofline for that range is 0.012–0.039 ms, so the call is all fixed cost until ≈ 512 rows.

## `compose`

Worst relative error of $\mathrm{compose}(t_0, \mathrm{roofline})$ over this sweep (`sim/`'s own rooflines with $t_0$ = 0):

| rule | skel_in | attn | post | skel_out | expert | worst |
|---|---|---|---|---|---|---|
| max | 0.57 | 0.59 | 0.01 | 0.57 | 0.43 | 0.59 |
| **sum** | 0.31 | 0.25 | 0.04 | 0.31 | 0.24 | **0.31** |

**`sum`** is chosen by ADR-050 §1's rule. Sum's worst errors sit at the ends of the sweep:
- **Owner stages at 256 sequences** are 30–45 % *slower* than $t_0$ + roofline: attention 3.34 against 2.49 ms, `skel_in` 6.07 against 4.22, `skel_out` 6.55 against 4.50. At 64 sequences they agree within 4 %. The rooflines (ADR-043 decision 6) assume peak φ and β_C. At 256 sequences the attention's KV reads and the skeleton GEMMs reach less than that, so the work term is too small there.
- **The expert at 128–256 rows** is 20–23 % *faster* than $t_0$ + roofline. The fixed cost hides the small GEMM, so `sum` counts the work twice.

## Collectives (the harness's own dispatch and return code)

| $z$ | dispatch measured | model $2 t_{coll}$ + bytes/λ | return measured | model $t_{coll}$ + bytes/λ |
|---|---|---|---|---|
| 16 | 0.129 ms | 0.108 | 0.053 | 0.055 |
| 64 | 0.144 | 0.116 | 0.071 | 0.063 |
| 256 | 0.177 | 0.146 | 0.098 | 0.094 |
| 1024 | 0.329 | 0.270 | 0.256 | 0.217 |

$t_{coll}$ = 0.0526 ms is the return phase at $z$ = 16. The dispatch runs 17–22 % above its model: it also reads the counts back to the host before the rows collective. A bare `all_to_all_single` costs 0.039 ms at 32 bytes. That is close to `002-nccl`'s 44 µs floor.

## Interpretation

The costs S0's time split pointed at are real and measurable on their own. They are flat in the batch until the roofline overtakes them at 64–512 rows, depending on the kind. With $t_0$ and `sum` the model reproduces this bench within 4 % up to 64 sequences per owner, but not at 256. That limit is visible here before any S0 comparison. The values go into `local_3060.yaml` as measured, unfitted, and belong to this stack (PyTorch 2.13 eager plus the reference decode) as much as to the rig.
