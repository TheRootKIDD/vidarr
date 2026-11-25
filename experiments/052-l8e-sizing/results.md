# 052-l8e-sizing — rig for ADR-071 §4: the host-resident engram table fits (4.62 GiB) but costs **1.60× L5's step against a 1.25× gate → gate failed, `197-l8e-screen` did not run**; three quarters of the overhead is the CPU-side sparse Adam update and scatter

Two 6-step DDP smokes (3.2 M tokens each) of the L5 and L8e presets, run by `queue_screen_23` on
2026-11-25 15:58–16:03 before the L8e ladder run. 4 × RTX 3060, DDP, micro-batch 4, batch 524,288
tokens, seed 0, code at gate commit `d5ac422`; commands in `config.yaml`, per-arm logs in the
sub-directories. Wall-clock 5.5 min in all (most of it the two final full held-out evals).

**Bears on:** **H24** (a hashed 2/3-gram table read once per token lowers held-out loss vs L5 by
more than 4σ at matched active FLOPs), as the gate ADR-071 §4 asked for: step time and per-GPU
memory of L8e against L5. **No verdict; H24 silent** (no loss number at six steps means
anything). The gate (≤ 25 % overhead on L5's step, ≤ 10 GB per GPU) **failed on step time**; the
queue aborted as written and `197` waits on a follow-up ADR (I45, the user's call).

## Numbers

Median over steps 2–6 (`step_s` from `log.jsonl`); step 1 carries warm-up.

| | L5 | L8e | ratio / Δ |
|---|---|---|---|
| step time | 10.87 s | **17.42 s** | **1.603×** (+6.55 s); gate 1.25× = 13.58 s |
| throughput | 48.3 k tok/s | 30.1 k tok/s | 0.62× |
| peak memory per GPU | 3.86 GiB | 4.62 GiB | +0.76 GiB; gate 10 GiB: passed |
| params on GPU | 77.48 M | 78.07 M | +0.59 M ($W_e$, $d \times d$); the 2²¹ × 768 table (1.61 B) is in host RAM and not in this count |
| table rows touched per step | — | 547 k of 2,097,152 (26 %) | ≈ 1.04 distinct rows per token, from 2 lookups per token |
| held-out loss after 6 steps | 8.4160 | 8.4281 | not a result |

**Where the 6.55 s go** (L8e's logged per-step timers, medians of steps 2–6):

| | seconds per step | share of the overhead | share of L8e's step |
|---|---|---|---|
| `engram_update_s`: lazy sparse Adam on the touched rows, one CPU process | 4.05 | 62 % | 23 % |
| `engram_scatter_s`: index-add of row gradients into the shared fp32 buffer | 0.92 | 14 % | 5 % |
| remainder: gathers on each rank, host → GPU copies, the injection ($W_e$) | 1.58 | 24 % | 9 % |

**Not measured, against ADR-071 §4's list:** the gather time has no timer of its own (it is inside
the remainder), and peak host RAM is not logged. The host had 185 GiB free of 251 after the run
and the checkpoint directory holds the table at 4.8 GB (bf16 values plus state); the ≈ 22.5 GB
figure of ADR-071's context is not confirmed by a measurement here.

## What it means for `197`

At this rate `197-l8e-screen` would take 1907 × 17.42 s = **9.2 h**, against 5.8 h for L5 (`110`
ran 5.79 h): 3.5 h more, on one `screen` run. Memory is not the problem. The routes ADR-071 §4
and the numbers leave open are in `docs/04` as **I45**:

- run `197` host-placed as built and accept the 1.60× (an ADR waiving the gate for this run);
- the fallback on record: a GPU-placed table of 2¹⁸ rows (8× fewer; a different test of H24);
- make the host update cheaper (code change, then this rig again);
- drop L8e.

## Interpretation

The engram table as built fits easily on the cards, since only the gathered rows and one
$d \times d$ map live there, but its optimiser does not fit the step: updating about 547 k rows of
768 in one CPU process takes 4 s, and with the scatter and the gathers the step grows from 10.9 s
to 17.4 s. That is 60 % over L5 where ADR-071 allowed 25 %, so the queue stopped before `197`, as
the ADR says it must ("not switched silently"). Nothing here bears on whether the table helps the
loss. The overhead is a cost of this placement on this host (12 cores, one updater), not of the
idea: ADR-071 puts the read at 1.2 MFLOP per token, 0.2 % of the budget. Thermal state was not
sampled in six steps (`n_thermal` is logged every tenth record); both arms ran on idle, cool
cards minutes apart, so the ratio is not a throttling artefact. One seed, six steps.
