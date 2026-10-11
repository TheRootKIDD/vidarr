# 146-l5c-screen — L5c, one attention block per iteration with the expert pool shared: 3.2768 nats, **−1.68 % vs L5** (5.0× the 4σ line) — resolvable at one seed; ADR-031 sends it to a `small` pair

`python -m scripts.train.train --rung L5c --id 146-l5c-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. First run of `queue_screen_6.sh` (2026-10-10 14:58 → 20:40), after the unit suite passed on `dad83f8`. This is `110-l5-screen` with `MiddleCfg(shared_attention=False)` (ADR-032): each of the $r_{max}$ = 8 iterations has its own attention ($W_q, W_k, W_v, W_o$), while the depth-conditioned norms, router, $N_e$ = 8 expert pool ($k$ = 2, $d_{ff}$ pinned at 896) and halting head stay shared. Halted tokens' K/V would be recomputed per iteration (ADR-011), but this rung trains at fixed $r$ = 8, so nothing halts.

**Bears on:** **H6**, through `00 §P6` caveat 2: sharing attention across iterations stops the
block from simulating per-layer attention. L5c is ADR-032's upper bound on what untying attention
can recover. FLOPs are the same as L5's, while parameters are **+11.0 M (+20.8 % non-embedding, +14.2 % total)**.
Comparator: `110`, under ADR-013 at one seed, with σ = 0.0028 (2σ 0.0056, 4σ 0.0111).

## Numbers

| | |
|---|---|
| params | 88.49 M total, 63.91 M non-embedding (`110`: 77.48 / 52.90) |
| attention | 8 modules, one per iteration (`110`: 1, shared) |
| tokens | 1.000 B in 1907 steps |
| **final val loss** (full held-out, 5606 windows) | **3.2768** nats |
| final train loss (last step): main / total | 3.2467 / 3.2546 (router aux 0.0080) |
| throughput (median step) | 49,192 tokens/s aggregate (`step_s` median 10.658 s, p90/median 1.003), the same as L5's ≈ 49.3 k, as same-FLOPs predicts |
| peak memory per GPU | 4.07 GiB (`110`: 3.86) |
| wall-clock (this session) | 5.71 h |
| seed | 0 |
| routing at the last step | `load_max` 0.130 / `load_min` 0.120, `route_ent_mean` 0.9999, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 190 telemetry rows**; `temp_c_max` ≤ 80 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 17 rows |

Wall-clock accounting: summed `step_s` 5.631 h + evals 0.070 h + checkpoints 0.003 h = 5.704 h of
5.708 h, **0.07 % unaccounted**.

## Loss against the comparators

| comparator | loss | Δ (nats) | Δ % | ADR-013: matches ≤ 0.0056, resolvable > 0.0111 |
|---|---|---|---|---|
| **`110` L5, shared attention** | 3.3328 | **−0.0560** | **−1.68 %** | resolvable, 5.0× the 4σ line |
| `147` L5b, rank-16 per-iteration adapters | 3.3286 | −0.0518 | −1.56 % | resolvable |
| `100` L0, dense stacked | 3.336 | −0.059 | −1.77 % | |
| `106` L1, stacked MoE | 3.2048 | +0.0720 | +2.25 % | |
| `112` L5-ne128 | 3.2232 | +0.0536 | +1.66 % | |

The gap to `110` opens early and then holds. At 976-window evals it is −0.039 / −0.043 / −0.051 /
−0.055 / −0.057 / −0.056 at steps 100 / 500 / 800 / 1000 / 1600 / 1900, with one sign at every
eval. It has been flat since ≈ step 1100 (0.53 B tokens).

## What the parameters buy

The extra parameters cannot be separated from the untying in this run, because L5c adds 11.0 M
parameters and per-iteration attention at once. Per parameter, though, attention is far more
efficient than adding experts. On `screen` at matched FLOPs:

| added to L5 | Δ non-emb params | Δ loss | nats per M params |
|---|---|---|---|
| L5c: 7 more attention blocks | +11.0 M | −0.056 | 0.0051 |
| L5-ne128: 120 more experts, $k$ = 4 | +115.6 M | −0.110 | 0.0010 |
| L5b: rank-16 adapters on attention | +0.66 M | −0.004 (silent) | — |

So L5c gets half of L5-ne128's gain from a tenth of its added parameters. Part of that may still be
plain capacity. A parameter-matched control (L5 with ≈ 63.9 M non-embedding parameters at the same
FLOPs, e.g. $N_e$ ≈ 13 at $k$ = 2) would separate "more parameters" from "attention per
iteration". It is not in the plan and is raised as I31.

## Interpretation

**Resolvable at one seed: untying attention across iterations improves L5 by 0.056 nats (−1.68 %,
5.0× ADR-013's 4σ line) at the same FLOPs, for +20.8 % non-embedding parameters.** The `00 §P6`
caveat-2 cost therefore exists and is large at this scale. It is more than half of the 0.128 nats by
which L5 trails stacked MoE (`106`) at `screen`. L5b at rank 16 recovers none of it (`147`), which is
ADR-032's "L5c wins, L5b does not" outcome: the gap needs more than a rank-16 delta per iteration.
For H6, **this is silent on the clause and points at its mechanism**: a shared block falls short of
stacked layers partly because its attention is shared, and the expert pool alone does not make up
the difference. ADR-031's rule triggers, so **L5c goes to a `small` pair** (2.5 B tokens, seeds 0/1,
≈ 2 × 14.3 h at 49.2 k tok/s) against the L5 pair (3.0748). I31's parameter-matched control decides
how much of the gain is attention and how much is capacity. Rig: 5.7 h, zero `n_thermal`, ≤ 80 °C.
