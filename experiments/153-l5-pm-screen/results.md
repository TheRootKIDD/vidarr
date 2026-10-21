# 153-l5-pm-screen — L5-pm, I29's parameter-matched H6 arm: 3.1942 nats, **−0.0106 vs L1** (−0.33 %, 3.9σ; just under the 4σ line, so silent at one seed) at L1's parameters and FLOPs; the gap to L1 is still closing at 1 B tokens; a fixed −0.029 vs L5-ne128 → `small` pair under ADR-035

`python -m scripts.train.train --rung L5-pm --id 153-l5-pm-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Third and last run of `queue_screen_8.sh` (2026-10-15 12:20 → 2026-10-16 13:12). It imported the tree at `117503f` (docs-only changes on top of the gate commit `36257dc`, where ruff and the unit suite passed). `git_dirty` is the untracked experiment directories. This is `112-l5-ne128-screen` with $N_e$ = 204 instead of 128 (ADR-035): $k$ = 4, $d_{ff}$ = 448, fixed $r$ = 8, per-iteration global range, so FLOPs per token are L5-ne128's, which are L1's budget (0.6 GFLOP).

**Bears on:** **H6** (I29). This is the first arm matched to L1 on both parameters (247.0 M vs
246.6 M non-embedding, +0.16 %) and FLOPs. Comparators: `106` L1 (H6's comparator) and `112`
L5-ne128 (the same arm with 128 experts). ADR-013 at one seed, with σ = 0.0027 re-pooled over ten
pairs (`155`; 0.00274, 2σ 0.0055, 4σ 0.0110). H6's threshold is "matches", |Δ| ≤ 2σ. At `screen` the
rule is silent unless |Δ| > 4σ.

## Numbers

| | |
|---|---|
| params | 271.56 M total, 246.98 M non-embedding (`106` L1: 271.15 / 246.58; `112`: 193.10 / 168.53) |
| experts | $N_e$ = 204, $k$ = 4, $d_{ff}$ = 448, one shared pool over $r$ = 8 iterations |
| tokens | 1.000 B in 1907 steps |
| **final val loss** (full held-out, 5606 windows) | **3.1942** nats |
| final train loss (last step): main / total | 3.1651 / 3.1722 (router aux 0.0071) |
| throughput (median step) | 11,257 tokens/s aggregate (`step_s` median 46.58 s, p90/median 1.004). That is 0.68× L5-ne128's 16.5 k at the same FLOPs; ADR-035 estimated 13–15 k |
| peak memory per GPU | 7.56 GiB (ADR-035 estimated ≈ 7.3; `112`: 6.01) |
| wall-clock (this session) | 24.86 h |
| seed | 0 |
| routing at the last step | `load_max` 0.0070 / `load_min` 0.0032 (uniform 0.0049), `route_ent_mean` 0.9994, `route_eff_min` 203.3 of 204, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 190 telemetry rows**; `temp_c_max` ≤ 66 °C, `sm_mhz_min` ≥ 1912; `n_power_cap` 0 rows |

Wall-clock accounting: summed `step_s` 24.679 h + evals 0.145 h + checkpoints 0.037 h = 24.861 h of
24.865 h, **0.02 % unaccounted**. The likely cause of the 0.68× throughput against `112` is per-expert
batch size (not profiled). 204 experts at $k$ = 4 leave ≈ 160 rows per expert per iteration per
micro-batch per GPU (8192 tokens × 4 / 204), against ≈ 256 at 128. Both are under the measured 512-row
batch floor (`06 §4`), and the dispatch loop is 1.6× longer.

## Loss against the comparators

| comparator | loss | Δ (nats) | Δ % | ADR-013: matches ≤ 0.0055, resolvable > 0.0110 |
|---|---|---|---|---|
| **`106` L1, stacked MoE, same params and FLOPs** | 3.2048 | **−0.0106** | **−0.33 %** | **neither: 3.9σ, above "matches", under 4σ → silent** |
| `112` L5-ne128, 68 % of the params, same FLOPs | 3.2232 | −0.0290 | −0.90 % | resolvable (2.6× 4σ) |
| `107` L2 | 3.1495 | +0.0447 | +1.42 % | |
| `110` L5 ($N_e$ = 8) | 3.3328 | −0.1386 | −4.16 % | |
| `100` L0, dense, $L_{ref}$ | 3.3358 | −0.1416 | −4.25 % | |

Under the nine-pair σ in force before `155` (0.0026, 4σ 0.0105) the same Δ sat at 4.0σ. Either way
it is on the line, and it does not decide H6.

## Trajectory (976-window evals)

| step | tokens | L5-pm | − `106` L1 | − `112` L5-ne128 | − `110` L5 |
|---|---|---|---|---|---|
| 200 | 0.10 B | 4.6386 | −0.666 | −0.036 | −0.116 |
| 600 | 0.31 B | 3.6117 | −0.148 | −0.031 | −0.138 |
| 1000 | 0.52 B | 3.3668 | −0.052 | −0.029 | −0.133 |
| 1400 | 0.73 B | 3.2533 | −0.024 | −0.029 | −0.135 |
| 1600 | 0.84 B | 3.2193 | −0.017 | −0.029 | −0.137 |
| 1800 | 0.94 B | 3.1981 | −0.012 | −0.029 | −0.138 |
| 1900 | 1.00 B | 3.1915 | −0.010 | −0.029 | −0.137 |

Two separate things show. **Against L5-ne128 the gain is a constant ≈ 0.029 nats from step 600 on.**
The 76 extra experts buy a fixed offset, not a different slope. **Against L1 the lead shrinks
monotonically** (−0.017 → −0.010 over the last 300 steps) and has not levelled at 1 B tokens. `112`
showed the same shape and crossed L1 at ≈ 1300 steps. L5-pm starts ≈ 0.03 lower, so it has not crossed
yet. If the constant offset carries to 2.5 B tokens, L5-pm at `small` lands near the L5-ne128 pair
minus 0.029 ≈ 2.930. That would be ≈ +0.046 vs the L1 pair (≈ 17σ), still *weakens*. This is an
inference from one seed's trajectory, not a result, and the pair is what tests it.

## Interpretation

**Silent at one seed: at L1's parameters and FLOPs, the shared middle block with 204 experts is
0.0106 nats (−0.33 %) ahead of stacked MoE at 1 B tokens.** That is 3.9σ, outside the "matches" band
on the favourable side and just under the 4σ line that would let one seed decide. **H6: silent** at
`screen`. The I29 caveat is answered in direction: giving the 128-expert arm L1's full parameter count
moves it by a fixed −0.029 nats (resolvable). That is the "more experts, same $k$" return `148` showed at
small $N_e$, and it puts the shared block ahead of L1 at this budget. But the lead is shrinking with
tokens exactly as `112`'s did before the `small` pair turned a `screen` tie into +2.60 %. So the
`screen` number cannot be read as *supports*. **ADR-035 §2 sends L5-pm to a `small` pair** against the
L1 pair (2.8839). That is H6's verdict at two seeds and closes I29. The cost is measured, not
estimated: ≈ 62 h per seed at 11.26 k tok/s, **≈ 5.2 days for the pair**, against ADR-035's 2 × 50 h.
Each seed is over one node-day, which CLAUDE.md allows now that this `screen` has finished. Memory
is 7.56 GiB, under the cap. A routing probe (depth partitioning, as for `138`/`142`) is owed on GPU
once the cards are idle (done: see the addendum). Rig: 24.9 h, zero `n_thermal`, ≤ 66 °C.

## Addendum 2026-10-21: routing probe (depth partitioning)

`probe_routing --id 153-l5-pm-screen --device cuda` ran on GPU 2 on 2026-10-20 at 11:25, after
`159`, on idle cards (`queue_screen_10`; log in `/mnt/nvme/post_queue10/`). It used 8 × 4 held-out
batches, as for every earlier `routing.json`. The measure is the mean correlation, across expert
loads, between each pair of iterations. Lower means the iterations use more distinct experts.

| run | $N_e$ | mean cross-iteration load correlation | effective experts, iterations 1 → 8 | near-dead experts (< 0.1 × uniform), iterations 1 → 8 |
|---|---|---|---|---|
| **`153` L5-pm** | 204 | **0.24** | 203 → 171 → 169 → 153 → 155 → 148 → 130 → 136 (64–99 %) | 0, 2, 1, 6, 1, 4, 8, 10 |
| `112` L5-ne128 `screen` | 128 | 0.34 | 128 → 112 → 114 → 102 → 111 → 104 → 91 → 90 (70–100 %) | 0, 0, 0, 1, 0, 1, 5, 8 |
| `138` / `142` L5-ne128 `small` | 128 | 0.35 / 0.32 | | |
| `110` L5 | 8 | 0.71 | | |

**The router partitions experts by depth more sharply as $N_e$ grows** (0.71 → 0.34 → 0.24). It
concentrates on a narrower set in the late iterations: 64 % of the pool is effectively used at
iteration 7, against 71 % for `112`. About 5 % of the pool is near-dead at iterations 7–8. On this
reading, the extra 76 experts are spent mostly as more depth-specific capacity, which is what
stacked layers have by construction. This is one probe at one seed, and it is descriptive. It does
not say whether the partitioning causes the −0.029 offset against `112`.
