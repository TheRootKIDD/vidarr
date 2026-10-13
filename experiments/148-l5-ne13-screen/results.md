# 148-l5-ne13-screen — L5-ne13, I31's control (L5c's parameters at L5's FLOPs, spent on experts): 3.3043 nats, −0.86 % vs L5 but **+0.0275 vs L5c** (2.5× the 4σ line) — about half of L5c's gain is untied attention; ADR-033 sends L5c to a `small` pair

`python -m scripts.train.train --rung L5-ne13 --id 148-l5-ne13-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Only run of `queue_screen_7.sh` (2026-10-11 10:19 → 16:23), with the unit suite (74 tests) passing before launch on idle GPUs. This is `110-l5-screen` with `RouterCfg(k=2, n_experts=13)` and `ExpertsCfg(d_ff=896)` (ADR-033): five more experts at the same $k$ and width, so FLOPs per token are L5's and L5c's while parameters are 87.80 M against L5c's 88.49 M (−0.8 %).

**Bears on:** **H6**, through I31: is L5c's gain (`146`, −0.056 nats vs `110`) attention untied across
iterations, or the 11 M parameters it adds? Comparators: `146` (L5c, same params and FLOPs, attention
untied) and `110` (L5, same FLOPs, fewer params), under ADR-013 at one seed with σ = 0.0028 (2σ
0.0056, 4σ 0.0111). ADR-033 §3 fixes the reading in advance: if L5-ne13 − L5c > 4σ, the L5c `small`
pair runs; within 2σ, it is dropped.

## Numbers

| | |
|---|---|
| params | 87.80 M total, 63.23 M non-embedding (L5c: 88.49 / 63.91; `110`: 77.48 / 52.90) |
| expert shape | **$N_e$ = 13**, $k$ = 2, $d_{ff}$ = 896 (`110`: $N_e$ = 8) |
| tokens | 1.000 B in 1907 steps |
| **final val loss** (full held-out, 5606 windows) | **3.3043** nats |
| final train loss (last step): main / total | 3.2731 / 3.2809 (router aux 0.0077) |
| throughput (median step) | 46,502 tokens/s aggregate (`step_s` median 11.274 s, p90/median 1.009). That is 5 % below L5's ≈ 49.3 k and L5c's 49.2 k at equal FLOPs: the 13-way dispatch has smaller per-expert batches |
| peak memory per GPU | 4.05 GiB |
| wall-clock (this session) | 6.07 h |
| seed | 0 |
| routing at the last step | `load_max` 0.080 / `load_min` 0.073 (uniform = 0.077), `route_ent_mean` 0.9998, `route_eff_min` 12.99 of 13, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 190 telemetry rows**; `temp_c_max` ≤ 79 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 17 rows |

Wall-clock accounting: summed `step_s` 5.988 h + evals 0.071 h + checkpoints 0.003 h = 6.062 h of
6.066 h, **0.06 % unaccounted**.

## Loss against the comparators

| comparator | loss | Δ (nats) | Δ % | ADR-013: matches ≤ 0.0056, resolvable > 0.0111 |
|---|---|---|---|---|
| **`146` L5c, same params and FLOPs, attention untied** | 3.2768 | **+0.0275** | **+0.84 %** | resolvable, 2.5× the 4σ line |
| `110` L5, same FLOPs, 8 experts | 3.3328 | −0.0285 | −0.86 % | resolvable, 2.6× |
| `112` L5-ne128 | 3.2232 | +0.0811 | +2.52 % | |
| `106` L1, stacked MoE | 3.2048 | +0.0995 | +3.10 % | |

The gap to L5c (976-window evals) moves around in the first 600 steps (−0.030 to +0.036). From step
800 it settles at +0.024, then holds at +0.028 to +0.029 from step 1200 to 1900. It has one sign
from step 600 on and is not closing at 1 B tokens.

## Splitting L5c's gain

| | Δ vs `110` (nats) | share of L5c's gain |
|---|---|---|
| L5c (`146`): +11.0 M params, attention per iteration | −0.0560 | 100 % |
| L5-ne13 (`148`): +10.3 M params, all in experts | −0.0285 | ≈ 51 % (capacity) |
| difference: untied attention beyond the same params as experts | −0.0275 | ≈ 49 % |

The two halves are each ≈ 2.5× the 4σ line, so both are resolvable at one seed. Spent on experts,
the extra parameters give 0.0028 nats per M at `screen`, more than L5-ne128's 0.0010 per M. That
matches diminishing returns along the $N_e$ curve (8 → 13 → 128: 3.3328 → 3.3043 → 3.2232). Spent on
per-iteration attention, the same parameters give 0.0051 per M, nearly twice as much.

## Interpretation

**I31 answered at one seed: about half of L5c's gain is capacity and half is untied attention.**
L5-ne13 puts L5c's extra parameters into experts at the same FLOPs. It improves on L5 by 0.0285 nats
but still trails L5c by **0.0275 nats (+0.84 %, 2.5× ADR-013's 4σ line)**. Per-iteration attention
is therefore worth more than the same parameters as experts, so `00 §P6` caveat 2 is a real cost of
the shared block and not just a parameter count. For H6 this is **silent on the clause** (neither arm
is matched to stacked MoE) and **sharpens its mechanism**: of the 0.128 nats by which L5 trails L1
at `screen`, about 0.028 is attention sharing that more experts do not buy back. Per ADR-033 §3,
**the L5c `small` pair runs** (`149`/`150`, 2.5 B tokens, seeds 0/1) against the L5 pair (3.0748).
An L5-ne13 pair beside it is not scheduled; ADR-033 adds one only if H6's verdict needs it. The
"more experts at the same $k$" construction behaves as expected, which informs L5-pm's design
(I29). Rig: 6.1 h, zero `n_thermal`, ≤ 79 °C.
