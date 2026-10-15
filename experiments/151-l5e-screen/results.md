# 151-l5e-screen — L5e, one dense H block with its own weights after iteration 4: 3.2945 nats, **−1.15 % vs L5** (3.6× the 4σ line) at +6.3 M params and +7 % FLOPs → `small` pair (ADR-031); −0.0098 vs the extra-parameters control `148`, between 2σ and 4σ, so inconclusive at one seed

`python -m scripts.train.train --rung L5e --id 151-l5e-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. The first training run of `queue_screen_8.sh` (2026-10-15 00:02 → 05:58), after ruff and the unit suite passed on `36257dc`. Probe `031` ran ahead of it in the same queue and failed at config time (session log 2026-10-15). This is `110-l5-screen` with `MiddleCfg(h_every=8)` (ADR-034). The skeleton's dense parallel block (attention 12/4 heads + SwiGLU, $d_{ff}^H$ = 2048), with its own weights, runs once, after iteration 4 of 8. It writes active tokens only and has no $e_j$. Iterations 5–8 of the shared L block read its write. The expert pool is unchanged ($N_e$ = 8, $k$ = 2, $d_{ff}$ pinned at 896), with fixed $r$ = 8.

**Bears on:** **H6**, and the HRM-Text claim in `08 §4` that two weight sets beat one shared loop.
Comparator: `110`. Per ADR-034 §4, a gain is also read against `148` (L5-ne13, +10.3 M parameters
as experts at L5's FLOPs), which bounds what generic extra parameters buy. Under ADR-013 at one seed,
σ = 0.0026 re-pooled over nine pairs (`150`; 2σ 0.0053, 4σ 0.0105). FLOPs are **not** matched: H
adds +7.0 % training FLOPs per token (≈ +8.7 % counting attention scores), which ADR-031 says to
report, not trade.

## Numbers

| | |
|---|---|
| params | 83.77 M total, 59.20 M non-embedding (`110`: 77.48 / 52.90; +6.29 M in H) |
| H block | 1 dense parallel block, $d_{ff}^H$ = 2048, after iteration $j$ = 4 ($K$ = 8) |
| tokens | 1.000 B in 1907 steps |
| **final val loss** (full held-out, 5606 windows) | **3.2945** nats |
| final train loss (last step): main / total | 3.2632 / 3.2712 (router aux 0.0080) |
| throughput (median step) | 47,368 tokens/s aggregate (`step_s` median 11.068 s, p90/median 1.003). That is 3.9 % below L5's ≈ 49.3 k for +7 % FLOPs: H's dense 2048-wide GEMMs run faster per FLOP than the routed 896-wide ones |
| peak memory per GPU | 4.25 GiB (`110`: 3.86) |
| wall-clock (this session) | 5.93 h |
| seed | 0 |
| routing at the last step | `load_max` 0.129 / `load_min` 0.122, `route_ent_mean` 0.9999, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 190 telemetry rows**; `temp_c_max` ≤ 79 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 18 rows |

Wall-clock accounting: summed `step_s` 5.850 h + evals 0.073 h + checkpoints 0.003 h = 5.926 h of
5.930 h, **0.06 % unaccounted**.

## Loss against the comparators

| comparator | loss | Δ (nats) | Δ % | ADR-013: matches ≤ 0.0053, resolvable > 0.0105 |
|---|---|---|---|---|
| **`110` L5** | 3.3328 | **−0.0383** | **−1.15 %** | resolvable, 3.6× the 4σ line |
| **`148` L5-ne13, +10.3 M params as experts, L5's FLOPs** | 3.3043 | **−0.0098** | **−0.30 %** | inconclusive (3.7σ: above 2σ, under 4σ) |
| `146` L5c, +11.0 M params as per-iteration attention | 3.2768 | +0.0177 | +0.54 % | resolvable |
| `147` L5b | 3.3286 | −0.0341 | −1.02 % | |
| `112` L5-ne128 | 3.2232 | +0.0713 | +2.21 % | |
| `106` L1, stacked MoE | 3.2048 | +0.0897 | +2.80 % | |

Along the trajectory (976-window evals), the gap to `110` peaks at −0.082 at step 400 and settles at
−0.038 to −0.040 from step 800 to 1907. The gap to `148` has been negative since step 200. It shrinks
from −0.015 to −0.019 before step 600 to −0.010 to −0.011 from step 1400 on, still narrowing slowly.

## What the parameters buy at `screen`

| added to L5 | Δ non-emb params | Δ FLOPs | Δ loss vs `110` | nats per M params |
|---|---|---|---|---|
| L5e: one dense H block after iteration 4 | +6.29 M | +7.0 % | −0.0383 | 0.0061 |
| L5c: 7 more attention blocks | +11.0 M | 0 | −0.0560 | 0.0051 |
| L5-ne13: 5 more experts | +10.3 M | 0 | −0.0285 | 0.0028 |
| L5-ne128: 120 more experts, $k$ = 4 | +115.6 M | 0 | −0.1096 | 0.0010 |

Per parameter, H is the most effective addition so far, but it is also the only one that adds FLOPs.
Two things are mixed in its gain: the extra compute (one more block's worth on every token) and the
second weight set. Neither this run nor `148` separates them. A FLOP-matched control would be L5 with $r$ raised until
FLOPs match (≈ $r$ = 9), without H. By `115`, though, the depth curve is flat past 8 (+0.13 % at $r$ = 12), which makes
"more depth with the same weights" an unlikely source of 0.038 nats. That points the gain at the
second weight set. The ordering against L5c (+0.0177, resolvable) shows that a dense H block once is
worth less than attention untied at every iteration, at about half the parameters.

## Interpretation

**Resolvable at one seed: one dense block with its own weights, placed after iteration 4, improves
L5 by 0.0383 nats (−1.15 %, 3.6× ADR-013's 4σ line), for +6.29 M parameters (+11.9 % non-embedding)
and +7 % FLOPs.** Against the extra-parameters control `148`, which has more added parameters and no
added FLOPs, it is ahead by 0.0098. That is above 2σ but under 4σ, so it is **inconclusive at one
seed** whether L5e is more than generic capacity plus compute. For HRM-Text's claim this is the
first held-out-loss evidence that a second weight set helps a shared loop. It is suggestive and not
yet separated from parameters and compute. **H6: silent on the clause** (neither parameter- nor
FLOP-matched to L1). On the mechanism it agrees with `146`/`150`: the shared block gains most from
unsharing something. **ADR-031's rule triggers (> 4σ vs L5), so L5e goes to a `small` pair**
(2.5 B tokens, seeds 0/1, ≈ 2 × 14.7 h at 47.4 k tok/s) against the L5 pair. The `148` comparison is
re-read there only if an L5-ne13 pair exists, and ADR-033 does not schedule one. If H6's reading
needs the separation, a FLOP-matched control (≈ $r$ = 9) is the cheaper arm. That is raised as I32, not
scheduled. Rig: 5.9 h, zero `n_thermal`, ≤ 79 °C.
