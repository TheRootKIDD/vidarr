# 164-l3c-screen — L3c, sequential MTP modules: 3.1992 nats on the main head, **−0.0494 vs L3 (−1.52 %, 4.1× the 4σ line)** but **+0.0497 vs L2**; anchored-chain acceptance **34.5 / 16.1 / 8.4 %**, *below* L3's at every head (I34-corrected 38.6 / 22.0 / 15.2 %) → **H13 weakened at `screen` on both clauses**; ADR-039 §6's `small`-pair trigger fires on the main head, but the pair cannot reach H13's bounds (user's call)

`python -m scripts.train.train --rung L3c --id 164-l3c-screen --tokens 1e+09 --micro-batch 1 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (64 × 1 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Third and last run of `queue_screen_11.sh` (2026-10-30 10:58 → 20:05). It imported the tree at `537dc87`, which is docs and experiment folders on top of the gate commit `22d1cdd` (merge of `l3c-i33`; ruff and the unit suite passed). `git_dirty` is the untracked experiment directories. ADR-039: L2 (`107`) plus $m$ = 4 sequential modules, each $W_p[\mathrm{RMSNorm}(h^{(j-1)}); \mathrm{RMSNorm}(E x^{in})]$ → one dense parallel block ($d_{ff}$ 2048) → its own RMSNorm → tied head. Training is teacher-forced with ADR-012's loss subsampling. Evaluation uses the anchored greedy chain. Micro-batch 1 was chosen by `032` (9.73 GiB at micro-batch 2, over ADR-039's 9.5 GiB line).

**Bears on:** **H13**, both clauses: *greedy acceptance of heads 2/3/4 against the main head ≥ 70 /
55 / 45 % on held-out, with main-head Δ ≤ 2σ against L2*. H13 is decided at `small` (ADR-030), where
the L3 and L3b pairs weakened it. This is ADR-039's `screen` run of the last route. Comparators:
`108` L3, `152` L3b, `107` L2. Acceptance is compared against the **I34-corrected** numbers: the
verifier now includes the output RMSNorm, re-measured by `queue_screen_12` on 2026-10-30 (see below).
ADR-013 at one seed, σ = 0.0030 over twelve pairs (`161`; 2σ 0.0060, 4σ 0.0120).

## Numbers

| | |
|---|---|
| params | 293.6 M total, 269.0 M non-embedding (`108` L3: 272.9 / 248.3; `107` L2: 271.1 / 246.6). The modules add +20.7 M net over L3's heads (ADR-039 estimated ≈ +22.4 M gross) |
| tokens | 1.000 B in 1907 steps |
| **final val loss, main head** (full held-out, 5606 windows) | **3.1992** nats |
| **acceptance vs main head, modules 2 / 3 / 4** (anchored greedy chain, full held-out, corrected verifier) | **34.5 / 16.1 / 8.4 %** |
| module top-1 vs ground truth, 2 / 3 / 4 (same chain) | 19.0 / 9.3 / 5.0 % |
| teacher-forced acceptance at the last training step, 2 / 3 / 4 | 70.1 / 65.0 / 63.0 % (each module is fed the true token; not H13's quantity) |
| final train loss (last step): main / total | 3.1687 / 6.2275 (router aux 0.0124) |
| throughput (median step) | 31,180 tokens/s aggregate (`step_s` median 16.815 s, p90/median 1.002). `108` L3: 43.3 k; `107` L2: 50.9 k |
| peak memory per GPU | **9.96 GiB**: 7.97 GiB in training (steps 2–199); the jump to 9.96 is at the first eval (step 200), the anchored-chain path. The final full-set eval logged allocator OOM-retry warnings (`expandable_segments` mapping failed, ≤ 21 MB free) and completed. Under the 10 GB cap read as GiB, with 0.04 GiB to spare; `032`'s 9.73 GiB at micro-batch 2 was 3 training steps with no eval |
| wall-clock (this session) | 9.11 h |
| seed | 0 |
| routing at the last step | `load_max` 0.135 / `load_min` 0.116, `route_ent_mean` 0.9998, `route_eff_min` 7.99 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 190 telemetry rows**; `temp_c_max` ≤ 78 °C, `sm_mhz_min` ≥ 1882; `n_power_cap` 9 rows |

Wall-clock accounting: summed `step_s` 8.905 h + evals 0.183 h + checkpoints 0.015 h = 9.102 h of
9.108 h (queue log START → END), **0.07 % unaccounted**.

## H13's clauses at `screen`

| clause | H13 requires | **`164` L3c** | `108` L3 | `152` L3b |
|---|---|---|---|---|
| acceptance vs main head, head/module 2 | ≥ 70 % | **34.5 %** | 38.6 % | 38.2 % |
| acceptance, 3 | ≥ 55 % | **16.1 %** | 22.0 % | 20.8 % |
| acceptance, 4 | ≥ 45 % | **8.4 %** | 15.2 % | 14.0 % |
| top-1 vs ground truth, 2 / 3 / 4 | — | 19.0 / 9.3 / 5.0 % | 21.4 / 12.5 / 8.8 % | 21.6 / 12.1 / 8.2 % |
| **main-head Δ vs `107` L2** | ≤ 2σ = 0.0060 | **+0.0497 (+1.58 %)** | +0.0991 | +0.0599 |

L3 and L3b acceptance are the I34 probe's corrected verifier (`mtp_acceptance_i34.json`, 64 × 2
held-out batches, 261,632 positions). L3c's are from its own final eval over the full held-out set,
with the corrected verifier (the `l3c-i33` code). L3c misses H13 by 36 / 39 / 37 points, and its
acceptance is **lower than L3's at every head**: −4.1 / −5.9 / −6.8 points.

## Main loss against the comparators

| comparator | loss | Δ (nats) | Δ % | ADR-013: matches ≤ 0.0060, resolvable > 0.0120 |
|---|---|---|---|---|
| **`108` L3, independent heads** | 3.2486 | **−0.0494** | **−1.52 %** | **resolvable, 4.1× the 4σ line** |
| `152` L3b, L3 + curriculum | 3.2094 | −0.0101 | −0.32 % | silent (3.4σ) |
| **`107` L2, no MTP** | 3.1495 | **+0.0497** | **+1.58 %** | **H13's clause misses: Δ > T + 2σ** |
| `100` L0, dense, $L_{ref}$ | 3.3358 | −0.1366 | −4.10 % | |

Trajectory (976-window evals):

| step | tokens | L3c | − `108` L3 | − `152` L3b | − `107` L2 | acceptance 2 / 3 / 4 |
|---|---|---|---|---|---|---|
| 200 | 0.10 B | 4.6491 | −0.089 | +0.074 | +0.024 | 26.3 / 11.2 / 6.2 % |
| 600 | 0.31 B | 3.6412 | −0.090 | +0.037 | +0.052 | 30.8 / 13.6 / 7.0 % |
| 1000 | 0.52 B | 3.3828 | −0.056 | +0.010 | +0.052 | 33.0 / 15.0 / 7.7 % |
| 1400 | 0.73 B | 3.2616 | −0.051 | −0.001 | +0.050 | 33.9 / 15.6 / 8.1 % |
| 1800 | 0.94 B | 3.2039 | −0.051 | −0.009 | +0.050 | 34.5 / 16.1 / 8.3 % |
| 1900 | 1.00 B | 3.1963 | −0.051 | −0.011 | +0.050 | 34.7 / 16.2 / 8.4 % |

The cost against L2 is flat at ≈ +0.050 from 0.3 B tokens on. The lead over L3 levels at ≈ −0.051. L3c
starts behind L3b, whose phase 1 is plain L2, and passes it at ≈ 1400 steps.

## I34: the corrected verifier, re-measured (`queue_screen_12`, 2026-10-30 20:05)

The probe (`b96364b`, which fixed the device-mismatch crash that voided `queue_screen_11`'s attempt)
re-scored seven checkpoints with both verifiers: corrected = $\mathrm{RMSNorm}_{out}(h)E^\top$, raw =
$hE^\top$.

| run | corrected, 2 / 3 / 4 (%) | raw, 2 / 3 / 4 (%) | corrected − raw (points) |
|---|---|---|---|
| `108` L3 `screen` | 38.6 / 22.0 / 15.2 | 39.1 / 22.4 / 15.7 | −0.5 / −0.4 / −0.4 |
| `113` L3-full `screen` | 38.9 / 22.2 / 15.4 | 39.4 / 22.7 / 15.8 | −0.5 / −0.5 / −0.4 |
| `152` L3b `screen` | 38.2 / 20.8 / 14.0 | 38.6 / 21.1 / 14.4 | −0.4 / −0.4 / −0.4 |
| `122` L3 `small` s0 | 41.5 / 24.1 / 16.0 | 42.3 / 25.1 / 17.0 | −0.7 / −1.0 / −1.0 |
| `131` L3 `small` s1 | 41.7 / 24.1 / 16.0 | 42.4 / 25.5 / 17.4 | −0.7 / −1.4 / −1.4 |
| `156` L3b `small` s0 | 40.6 / 22.5 / 14.3 | 41.4 / 23.4 / 15.3 | −0.8 / −1.0 / −1.0 |
| `157` L3b `small` s1 | 40.7 / 22.4 / 14.5 | 41.4 / 23.5 / 15.5 | −0.7 / −1.1 / −1.0 |

The output norm's gain moves the verifier's argmax on 0.4–1.4 % of positions, always downward
for acceptance here, and more at `small` than at `screen`. Every H13 verdict stands: the corrected
pairs are L3 41.6 / 24.1 / 16.0 % and L3b 40.6 / 22.5 / 14.4 %, still 28–33 points short. **I34
closes.**

## Interpretation

**Weakened at `screen` on both clauses: sequential modules do not bring MTP within H13's bounds.**
The main head trails L2 by 0.0497 nats (+1.58 %, 8.3× the 2σ margin). Acceptance under the anchored
chain is 34.5 / 16.1 / 8.4 %, below L3's at every head and far below 70 / 55 / 45 %. **H13: weakened**
(it was already weakened at `small` by the L3 and L3b pairs; L3c was the last route). The two clauses
move in opposite directions. **The main head is resolvably better than L3's** (−0.0494, 4.1× 4σ),
the largest recovery of L3's main-head cost so far: about half of it, against L3b's 40 % at
`screen`. A reading not tested here is that the modules take the MTP gradient into their own blocks
instead of the trunk's last layer. **The drafts are worse**, and the gap between 70 % teacher-forced
and 34.5 % chained acceptance at module 2 shows why: in training each module is fed the true next
token, and at eval it is fed head 1's greedy guess, which is right ≈ 40 % of the time. Each step of the
chain compounds that (16 %, then 8 %), where L3's independent heads do not depend on one another.
Part of the gap may be the metric: the held-out proxy (I27) checks drafts against head 1 conditioned
on the *true* context, while real greedy decoding would condition the verifier on the drafted tokens
themselves. That matters more for a chain than for independent heads. It is raised as **I36**, not
assumed. **ADR-039 §6 says a `small` pair follows if `164` beats L3 by > 4σ on the main head *or* on
acceptance at every head.** The main-head condition is met (4.1×). The pair would cost ≈ 2 × 23 h,
and on this run's numbers it cannot reach H13's bounds: L3's main-head cost at `small` was 84 % of its
`screen` cost, and acceptance would need to double. Whether to run it or drop it by an ADR, as ADR-031
allows, is the user's call. Memory: training fits at 7.97 GiB, and the eval chain takes the peak to
9.96 GiB, at the cap. A `small` pair would need the eval path's memory looked at first. Rig: 9.1 h,
zero `n_thermal`, ≤ 78 °C.
