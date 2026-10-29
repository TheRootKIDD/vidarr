# 161-l5-pm-small-s1 — L5-pm at `small`, seed 1: 2.9292 nats; **L5-pm pair 2.9321, +0.0482 vs the L1 pair (+1.67 %) at matched params and FLOPs — H6 weakened at two seeds**; I29 closes; σ re-pooled over twelve pairs = 0.0030

`python -m scripts.train.train --rung L5-pm --id 161-l5-pm-small-s1 --tokens 2.5e+09 --micro-batch 4 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Second and last run of `queue_small_10.sh` (2026-10-23 01:52 → 2026-10-25 15:04, then QUEUE DONE; the pause armed 2026-10-21 left the GPUs idle after it), on the tree at `a103586`–`4dc091e` (docs only on top of the gate commit `36257dc`). Seed 1 of `160`.

**Bears on:** **H6**: *one depth-conditioned shared middle block with $r_{max} n$ experts matches
$r_{max}$ stacked MoE layers with $n$ experts each, at matched params and FLOPs*. Under ADR-013
"matches" means $|\Delta| \le 2\sigma$ and the verdict is *weakens* above $T + 2\sigma$ = 4σ.
Comparator: the L1 pair (`130`/`119`) 2.8839, 246.6 M non-embedding; L5-pm has 247.0 M (+0.16 %) and
L5-ne128's FLOPs (ADR-035). **σ re-pooled over twelve pairs = 0.0030** (0.00296, this pair included;
was 0.00285 over eleven). 2σ = 0.0059, 4σ = 0.0119. This is the parameter-matched arm I29 asked for;
the L5-ne128 pair (`138`/`142`, 2.9587) is the same design at 68 % of the parameters.

## Numbers

| | |
|---|---|
| params | 271.6 M total, 247.0 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| final val loss (full held-out, 5606 windows) | **2.9292** nats |
| final train loss (last step, main head) | 2.9081 |
| throughput (median step) | 11,238 tokens/s aggregate (`step_s` median 46.654 s, p90/median 1.004); last step 11,209 |
| peak memory per GPU | 7.56 GiB |
| wall-clock (this session) | 62.21 h |
| seed | 1 |
| val_r_mean | 8.0000 |
| routing at the last step | `load_max` 0.0062 / `load_min` 0.0038 (uniform 0.0049), `route_ent_mean` 0.9994, `route_eff_min` 203.3 of 204, no dead experts (pooled over iterations) |
| routing probe, per iteration (`routing.json`, GPU, 8 × 4 held-out batches) | effective experts 203 → 175 → 164 → 170 → 138 → 138 → 144 → 121; near-dead 0, 1, 4, 1, 5, 6, 4, 23; mean cross-iteration load correlation **0.23** |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 64 °C, `sm_mhz_min` ≥ 1920; `n_power_cap` 0 rows |

Wall-clock accounting: summed `step_s` 61.793 h + evals 0.310 h + checkpoints 0.092 h = 62.195 h of
62.205 h, **0.02 % unaccounted**.

### The pair

| | seed 0 (`160`) | seed 1 (`161`) | pair | Δ vs L1 pair 2.8839 | Δ vs L5-ne128 pair 2.9587 | Δ vs $L_{ref}$ 2.9958 |
|---|---|---|---|---|---|---|
| final val loss | 2.9349 | 2.9292 | **2.9321** | **+0.0482 (+1.67 %), 8.1× 2σ, 4.1× 4σ** | −0.0266 (2.2× 4σ) | −0.0637 (−2.13 %) |

### Trajectory (pair means, full held-out evals)

| step | tokens | L5-pm | L1 pair | L5-pm − L1 | L5-pm − L5-ne128 |
|---|---|---|---|---|---|
| 400 | 0.21 B | 3.8935 | 4.1189 | −0.2254 | −0.0452 |
| 800 | 0.42 B | 3.4587 | 3.4893 | −0.0306 | −0.0273 |
| 1200 | 0.63 B | 3.2968 | 3.2893 | +0.0075 | −0.0246 |
| 2000 | 1.05 B | 3.1300 | 3.0987 | +0.0313 | −0.0245 |
| 2800 | 1.47 B | 3.0337 | 2.9926 | +0.0411 | −0.0257 |
| 3600 | 1.89 B | 2.9715 | 2.9259 | +0.0456 | −0.0264 |
| 4400 | 2.31 B | 2.9360 | 2.8879 | +0.0481 | −0.0269 |
| 4768 | 2.50 B | 2.9321 | 2.8839 | +0.0482 | −0.0266 |

### Routing (depth partitioning, `probe_routing`, same method as `153`'s addendum)

| run | $N_e$ | tokens | mean cross-iteration load correlation | worst-iteration effective experts | near-dead at iteration 8 |
|---|---|---|---|---|---|
| **`160` / `161` L5-pm** | 204 | 2.5 B | **0.20 / 0.23** | 133 / 121 (65 / 59 %) | 19 / 23 |
| `153` L5-pm | 204 | 1 B | 0.24 | 130 (64 %) | 10 |
| `138` / `142` L5-ne128 | 128 | 2.5 B | 0.35 / 0.32 | | |
| `132`–`133` L5 | 8 | 2.5 B | 0.75 | | |

## Interpretation

**Weakened at two seeds, at matched params and FLOPs: the shared block does not match stacked MoE
at `small`.** The L5-pm pair trails the L1 pair by 0.0482 nats (+1.67 %), 8.1× the 2σ "matches"
band and 4.1× the 4σ *weakens* line; the seeds are 0.0057 apart, both on the same side. **H6:
weakened, and I29 closes.** The L5-ne128 pair had already weakened H6 at 68 % of L1's parameters;
adding the missing 78 M parameters as 76 more experts recovers 0.0266 nats, 36 % of that pair's
+0.0748 deficit, and leaves the verdict unchanged. The trajectory explains the `screen` reading:
L5-pm learns faster early (−0.23 at 0.2 B tokens), crosses L1 at ≈ 0.58 B tokens, and the gap then
grows monotonically to +0.048, flattening only in the last 0.2 B tokens. `153`'s −0.33 % at 1 B tokens
was the end of a shorter schedule caught near the crossing, the same 1 B-token effect that made L5
tie L0 at `screen`. The offset against L5-ne128 is constant (−0.025 to −0.027 from 0.4 B tokens on),
so the extra experts buy a fixed amount and do not change the slope. The router partitions the pool
by depth more sharply with tokens (cross-iteration correlation 0.24 → 0.20 / 0.23) and leaves 9–11 %
of it near-dead at the last iteration, so a shared pool behaves like a partially unshared stack with
less usable capacity at late iterations; this is descriptive and one probe per seed. What stays
open: the side arms that beat L5 resolvably at `small` (L5c −1.70 %, L5e −0.96 %) do so at $N_e$ = 8
and are not H6 arms; whether the gap shrinks with scale is the `03` recurrence caveat, which only
`medium` could test. Rig: 62.2 h, zero `n_thermal`, ≤ 64 °C, 7.56 GiB per GPU.
