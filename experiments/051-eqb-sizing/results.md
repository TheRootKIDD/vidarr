# 051-eqb-sizing — rig for ADR-069 §3: EQB at $N_e$ = 128 fits (6.26 GiB) and costs 3.2 % of the step → `192`/`193` run, ≈ 18.3 h each

Three 6-step DDP smokes (3.2 M tokens each) of the L5-ne128 preset under the three balancers, run
by `queue_screen_21` on 2026-11-22 07:52–08:09 before the EQB ladder runs. 4 × RTX 3060, DDP,
micro-batch 4, batch 524,288 tokens, seed 0, code at gate commit `36b7c2f`; commands in
`config.yaml`, per-arm logs in the sub-directories. Wall-clock 17 min in all (5.4–5.9 min per arm,
most of it the final full held-out eval).

**Bears on:** **H26** (EQB against ADR-004's balancer), as the gate ADR-069 §3 asked for: memory
and step time of `eqb_iteration` at $N_e$ = 128 against `112`. **No verdict; silent.** The gate
(the queue aborts above 10 GiB per GPU) **passed** for every arm.

## Numbers

| arm | `router.balance` | peak per GPU | tokens/s (median of steps 2–6) | s per step | vs the `aux_free` smoke |
|---|---|---|---|---|---|
| L5-ne128 | `aux_free` (ADR-004) | 6.01 GiB | 15,603 | 33.6 | — |
| L5-ne128-eqb-iter | `eqb_iteration` | **6.26 GiB** | 15,096 | 34.7 | +0.25 GiB, 0.968× |
| L5-ne128-eqb-step | `eqb_step` | **6.26 GiB** | 15,103 | 34.7 | +0.25 GiB, 0.968× |

`112-l5-ne128-screen` itself: 6.01 GiB, 16,529 tokens/s median over its 1907 steps, 16.95 h. The
`aux_free` smoke reproduces its memory exactly and runs 5.6 % below its median speed, as
six-step smokes at the start of a run do here; the comparison between balancers is therefore
made inside the rig, not against `112`'s median.

- **Memory.** EQB adds 0.25 GiB at $N_e$ = 128, the same for both pools. That is the router
  logits kept for the step's quantile search; 6.26 GiB is well inside the 10 GB rule.
- **Step time.** +3.2 % per step for either EQB form. Per-iteration and per-step pools cost the
  same.
- **Budget.** At the smoke's rate `192` and `193` take 1907 × 34.7 s ≈ **18.4 h each** (ADR-069
  planned ≈ 17 h from `112`); if the runs warm up to `112`'s rate less 3 %, ≈ 17.5 h.
- **Not a result, noted.** After six steps the effective expert count at the worst iteration is
  13.1 of 128 under `aux_free`, 21.2 under `eqb_step` and 48.1 under `eqb_iteration`; EQB's
  biases start at 0 and act from step 2. Six steps say nothing about the trained state.

ADR-069 §3 also names "the logits held per step". The trainer does not log that count; the 0.25
GiB of peak memory is what the rig measured for it.

## Interpretation

EQB is affordable at the large pool: a quarter of a GiB and about 3 % of the step, with no
difference between the two pooling choices, so the question H26 asks (does forcing balance per
iteration hurt a model that wants to partition experts by depth) can be put at $N_e$ = 128 without
changing the batch or the budget rules. The four runs go ahead as ADR-069 §4 lists them: `190` and
`191` at $N_e$ = 8 against `110` (≈ 5.7 h each; `190` started 08:09 at 49.0 k tokens/s, as `110`'s
48.6 k), then `192` and `193` against `112`, then ADR-074's control `203`.
