# 188-l9-2b-screen — L9-2b, the two-branch softmax control for L9-relu2 (ADR-068; $o = o_{loc} + g_h\,o_{glob}$ on L5d): 3.3720 nats, **+0.0277 vs `162` (9.2σ worse than one softmax)** → ADR-008's single softmax stands; this run is H23's comparator and H23's verdict waits on `189`

`python -m scripts.train.train --rung L9-2b --id 188-l9-2b-screen --tokens 1e+09 --micro-batch 4 --seed 0 --set attention.rope_scope=all` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`attention.global_combine: two_branch`, `global_scoring: softmax`, `rope_scope: all`), per-step log in `log.jsonl`. Code at the `queue_screen_20` gate commit `9c0e5b7`.

**Base.** ADR-068 builds the L9 arms on L5d-nope unless ADR-066 §7 sends them to L5d. It did:
`204` (the rerun of `182`) = 3.3595 > `162` + 4σ, so `queue_screen_20` set `rope_scope: all` and
this arm differs from `162` only in how the two ranges are combined. **Where ADR-068 §4 says
$\Delta$(`188` − `182`), the like-for-like comparator is therefore `162`**, not `204`.

**Bears on:** **H23** (ReLU² scoring on the global range), as its control: ADR-068 §4 reads H23's
loss clause on $\Delta$(`189` − `188`) against $T$ = 1 % $L_{ref}$ = 0.0334. **This run alone
gives no H23 verdict (silent until `189`).** It also carries **ADR-068 §4's side reading on
ADR-008**: the two-branch form against one softmax over [$\mathcal{G}$ | local]. **Result:
$\Delta$(`188` − `162`) = +0.0277 = 9.2σ, 2.3× the 4σ line, on the wrong side.** The rule was
"if the two-branch form beats one softmax by > 4σ, ADR-008 gets a new ADR"; it loses by more than
that, so **ADR-008 stands** and the two-branch combination stays confined to `188`/`189`.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.91 M non-embedding (`162` + the gates $\gamma_h$) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3720** nats |
| final train loss (last step, main head) | 3.3393 |
| router aux (last step) | 0.0081 |
| throughput (median step) | 33,949 tokens/s aggregate (1.00× `162`) |
| peak memory per GPU | 3.89 GiB |
| wall-clock (this session) | 8.27 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `162` and the ladder

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `110-l5-screen` (L5) | 3.3328 | | |
| `162-l5d-segmean-screen` (L5d, one softmax over both ranges) | 3.3443 | — | comparator (same base) |
| `204-l5d-nope-screen` | 3.3595 | | not the base (ADR-066 §7) |
| **`188-l9-2b-screen` (two branches, gated sum, softmax global)** | **3.3720** | **+0.0277 vs `162`** (+0.83 %, 9.2σ) | **two-branch form worse than one softmax by > 4σ; ADR-008 stands** |
| | | +0.0125 vs `204` | for the record |
| | | +0.0362 vs $L_{ref}$ (+1.09 %) | |

**Curve** (976-window evals; last row full held-out):

| step | `162` | `188` | `188` − `162` |
|---|---|---|---|
| 100 | 5.5372 | 5.6214 | +0.0842 |
| 300 | 4.3094 | 4.3701 | +0.0607 |
| 500 | 3.8551 | 3.9074 | +0.0523 |
| 700 | 3.6599 | 3.7057 | +0.0458 |
| 1000 | 3.5084 | 3.5432 | +0.0348 |
| 1500 | 3.3835 | 3.4120 | +0.0285 |
| 1900 | 3.3436 | 3.3709 | +0.0273 |
| 1907 (full) | 3.3443 | 3.3720 | +0.0277 |

Behind from the first eval; the gap halves by step 1000 and is flat at ≈ 0.028 over the last
quarter. It is not closing at the end.

## Routing and run health

Worst point 3.72 of 8 effective experts at step 24, up to two dead experts (step 20); last dead
expert at step 27; 7.99 of 8 at the end, loads 0.115–0.132 — as `162`. Not a routing effect.

| | |
|---|---|
| wall-clock | 8.27 h; summed `step_s` 8.183 h + evals 0.082 h + checkpoints 0.004 h → **0.0 % unaccounted** |
| throughput | 33.9 k tok/s median, flat |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 75 °C, min clock 1905 MHz |

## Not in this write-up

The learned gates $g_h = \mathrm{sigmoid}(\gamma_h)$ are not in `log.jsonl`; reading them needs
the checkpoint and waits until the GPUs are free. ADR-068 §3's measures (zero fraction, V-skip
rate, eval-time cap) and ADR-066 §2's evals run for both arms after `189`, at the end of
`queue_screen_20`, and are written up with `189`.

## Interpretation

Summing a separately normalised global branch onto the local softmax, behind a per-head gate,
costs L5d 0.028 nats at 1 B tokens. In one softmax the two ranges compete for the same
probability mass, so a query with nothing useful far away gives the global range almost no
weight; in the two-branch form the global softmax always sums to one and only the per-head scalar
gate, the same for every token and iteration, can turn it down. That is the likely reason, and
the gate values will say whether the model shut the branch (gates near 0) or kept a noisy one,
but it is not shown here. For the programme the consequences are narrow. ADR-008's decision is
confirmed from the other side. And H23's loss clause compares `189` with this run, not with
`162`: ReLU² can produce exact zeros and an all-zero branch, which is the thing the softmax
control lacks, so `189` may land below `188`; whether an L9-relu2 that passes against `188` but
sits 0.03 above L5d is worth a `small` pair is a question for the `189` write-up and the user.
`189` started 19:00 at 22.8 k tok/s and 4.21 GiB (the ReLU² path is 0.67× this run's speed), so
it ends ≈ 07:15 on 2026-11-22, about 4 h later than planned.
