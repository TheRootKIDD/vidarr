# 145-l5-ne1-screen — L5-ne1, one expert in the loop at matched FLOPs: 3.3788 nats, **+1.38 % vs L5** (4.1× the 4σ line) — eight experts beat one dense recurrent FF; no `small` pair (ADR-031)

`python -m scripts.train.train --rung L5-ne1 --id 145-l5-ne1-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Only run of `queue_screen_5.sh` (2026-10-10 09:55 → 14:56), the first Phase 1 leftover under ADR-031. `110-l5-screen` with `RouterCfg(k=1, n_experts=1)`: the budget solver sizes the single expert at **$d_{ff}$ = 1792** (= $k$ × 896), so FLOPs per token equal L5's and the middle block is a dense recurrent FF with $e_j$-conditioned norms. Parameters fall by 12.4 M (−23 % non-embedding).

**Bears on:** **H6**, through the $N_e$ ∈ {1, 8, 128} sweep in `03 §2`'s L5 row (the "dense recurrent
block" arm). It is not H6's own comparison: H6 sets the shared block against stacked MoE layers at
matched params *and* FLOPs, and here only FLOPs are matched. It measures what the expert pool adds
inside the loop. Comparator: `110` (L5, $N_e$ = 8), under ADR-013 at one seed, with σ = 0.0028
pooled over eight `small` pairs (2σ 0.0056, 4σ 0.0111).

## Numbers

| | |
|---|---|
| params | 65.09 M total, 40.52 M non-embedding (`110`: 77.48 / 52.90) |
| expert shape | **$N_e$ = 1, $k$ = 1, $d_{ff}$ = 1792** (`110`: $N_e$ = 8, $k$ = 2, $d_{ff}$ = 896) |
| tokens | 1.000 B in 1907 steps |
| **final val loss** (full held-out, 5606 windows) | **3.3788** nats |
| final train loss (last step): main / total | 3.3458 / 3.3538 (router aux 0.0080, constant with one expert) |
| throughput (median step) | 56,122 tokens/s aggregate (`step_s` median 9.342 s, p90/median 1.003). That is 14 % above L5's ≈ 49.3 k at equal FLOPs: one 1792-wide GEMM per token costs less than two routed 896-wide ones |
| peak memory per GPU | 3.63 GiB |
| wall-clock (this session) | 5.02 h |
| seed | 0 |
| routing | trivial (`load_max` = `load_min` = 1.0) |
| thermal (`n_thermal`, GPU-wide) | **0 of 190 telemetry rows**; `temp_c_max` ≤ 80 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 80 rows |

Wall-clock accounting: summed `step_s` 4.953 h + evals 0.065 h + checkpoints 0.002 h = 5.020 h of
5.024 h, **0.07 % unaccounted**.

**Rule slip during this run:** the full pytest suite ran once on the CPU at ≈ 10:03, beside the live
DDP job (`docs/04 §Session log` 2026-10-10 10:15). One logged step read 9.45 s against the 9.34 s
median (≈ +1 %). That affects throughput only, and the median is unaffected; loss is unaffected
(`06 §7`).

## Loss against the comparators

| comparator | loss | Δ (nats) | Δ % | ADR-013: matches ≤ 0.0056, resolvable > 0.0111 |
|---|---|---|---|---|
| **`110` L5, $N_e$ = 8, matched FLOPs** | 3.3328 | **+0.0460** | **+1.38 %** | resolvable, 4.1× the 4σ line |
| `100` L0, dense stacked, matched FLOPs | 3.336 | +0.043 | +1.28 % | resolvable |
| `106` L1, stacked MoE | 3.2048 | +0.1740 | +5.43 % | |
| `112` L5-ne128 | 3.2232 | +0.1556 | +4.83 % | |

Along the trajectory (976-window evals) the gap to `110` is +0.040 / +0.042 / +0.045 / +0.046 /
+0.047 at steps 800 / 1000 / 1200 / 1600 / 1900. It has had one sign since step 500 and is still
widening slowly at 1 B tokens, so a larger budget would not close it.

## The $N_e$ curve at `screen`, matched FLOPs

| $N_e$ ($k$, $d_{ff}$) | run | non-emb params | val loss | vs $N_e$ = 8 |
|---|---|---|---|---|
| 1 (1, 1792) | `145` | 40.5 M | 3.3788 | +1.38 % |
| 8 (2, 896) | `110` | 52.9 M | 3.3328 | — |
| 128 (4, 448) | `112` | 168 M | 3.2232 | −3.29 % |

Loss falls steadily with the size of the expert pool at fixed FLOPs: the usual MoE scaling, now
measured inside the shared loop. The step from 1 to 8 experts (+12.4 M params) is worth 0.046 nats,
and from 8 to 128 (+116 M) a further 0.110.

## Interpretation

**Resolvable at one seed: a dense recurrent FF is worse than the eight-expert L5 by 0.046 nats
(+1.38 %, 4.1× ADR-013's 4σ line) at matched FLOPs, and worse than the dense stacked L0 by 1.28 %.**
At `screen` budget, sharing one dense FF across 8 iterations costs about 1.3 % against 12 unshared
dense layers. Eight experts recover all of that, and L5 ties L0 at `screen` (`110`, although it
trails by 2.64 % at `small`). The pool is therefore what makes the shared block competitive. This
is the premise H6 builds on ($r_{max} n$ experts substituting for $r_{max}$ separate layers), but
it is **silent on H6's own clause**, because that clause compares against stacked MoE at matched
params, and L5-pm (I29) is the arm for it. ADR-031's `small` rule does not trigger (it is for arms
that beat L5), so L5-ne1 closes at `screen`. Rig: 5.0 h, zero `n_thermal`, ≤ 80 °C.
