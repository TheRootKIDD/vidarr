# 137-l7b-screen — L7b, two-layer experts ($L_e$ = 2) at matched FLOPs: 3.3519 nats, **+0.57 % vs L5** — resolvable at one seed, **above H8's 0.5 % allowance by 0.07 points**; a `small` pair decides

`python -m scripts.train.train --rung L7b --id 137-l7b-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Seventh and last run of `queue_small_3.sh`; the rerun of the burnt `129`. `110-l5-screen` with `ExpertsCfg(layers=2)`: each expert is a two-layer SwiGLU stack (residual + norm between the layers, `01 §4.5`) and the cost model halves the width to hold FLOPs — **$d_{ff}$ = 448 per layer against L5's 896**, so the FLOPs per token, the parameters (77.49 M vs 77.48 M) and the bytes of expert weights are the same and only the shape of the expert changes. Driver 615.71.09.

**Bears on:** **H8** — *$L_e$ = 2 halves fabric bytes per FLOP at ≤ 0.5 % loss.* This run is the
quality half (arm A); the systems half (halved dispatch bytes per FLOP, because a token visits an
expert once for twice the compute) is S3's in `sim/` and is arithmetic, not a hypothesis, once the
quality cost is known. One seed at `screen`; the comparator is `110`.

## Numbers

| | |
|---|---|
| params | 77.49 M total, 52.91 M non-embedding (`110`: 77.48 / 52.90) |
| expert shape | $N_e$ = 8, $k$ = 2, **$L_e$ = 2, $d_{ff}$ = 448** (`110`: $L_e$ = 1, $d_{ff}$ = 896) |
| tokens | 1.000 B in 1907 steps |
| **final val loss** (full held-out, 5606 windows) | **3.3519** nats |
| final train loss (last step): main / total | 3.3203 / 3.3284 (router aux 0.0081) |
| throughput (median step) | 44,944 tokens/s aggregate (`step_s` median 11.665 s, p90/median 1.009) — 9 % below L5's 49.3 k at equal FLOPs: two half-width matmuls per expert instead of one, an efficiency loss on this GPU, not a FLOP difference |
| peak memory per GPU | 3.86 GiB |
| wall-clock (this session) | 6.28 h |
| seed | 0 |
| routing at the last step | `load_max` 0.128 / `load_min` 0.122, `route_ent_mean` 1.0000, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 190 telemetry rows**; `temp_c_max` ≤ 79 °C, `sm_mhz_min` ≥ 1890 |

Wall-clock accounting: summed `step_s` 6.199 h + evals 0.076 h + checkpoints 0.003 h = 6.278 h of
6.282 h, **0.06 % unaccounted**. (`n_throttled` = 4 on every row is the 615.71 `0x400` bit, `030`.)

## Loss against `110`

| comparator | loss | Δ (nats) | Δ % | H8's $T$ = 0.5 % | vs 4σ = 0.0123 |
|---|---|---|---|---|---|
| `110` L5, $L_e$ = 1 | 3.3328 | **+0.0191** | **+0.57 %** | 0.0167 | 1.5× → resolvable |

Along the trajectory (976-window evals) the gap is +0.023 / +0.026 / +0.023 / +0.021 / +0.019 at
steps 400 / 800 / 1200 / 1600 / 1907 — one sign throughout, shrinking slowly as training proceeds.

**Reading under ADR-013.** The difference is outside the silent band, so it is resolvable at one
seed. H8's clause is $Δ \le$ 0.5 % = 0.0167 nats: Δ = +0.0191 sits **0.0024 nats (0.07 points)
above the allowance** and inside $T$ + 2σ = 0.0229, i.e. in the band where the rule says neither
*supports* nor *weakens* — the run is **inconclusive on H8 at one seed**, leaning against. Two
things make this a `small`-pair question rather than a verdict: the margin is under one σ, and the
gap was still closing at 1 B tokens (+0.026 → +0.019 over the second half), so at 2.5 B it may
cross the 0.5 % line from above, as L2's `screen` reading changed sign at `small` (`120`). A two-seed
L7b pair at `small` against the L5 pair (3.0748) is ≈ 2 × 16 h and needs no new code.

## What the shape costs

At equal FLOPs, parameters and bytes, the two-layer expert with half the width is 0.57 % worse in
loss and 9 % slower per step on this GPU. The loss is the price of narrowing each layer to 448, which
`03 §2`'s granularity row (H4: thinner experts are worse at matched FLOPs) predicts in direction;
the note's argument for $L_e$ = 2 is not quality but that a token's dispatch bytes buy twice the
compute, which S3 scores. Whether 0.5 % is the right allowance for a 2× cut in fabric bytes per FLOP
is a `02`/S3 question; this run supplies the quality number to plug in: ≈ 0.6 % at `screen`.

## Interpretation

**Inconclusive on H8's quality clause at one seed, leaning against** (Δ = +0.0191 nats, +0.57 %,
against $T$ = 0.5 %; resolvable, 0.07 points over the line, within $T$ + 2σ). Recommendation: an L7b
pair at `small` (`--tokens 2.5e9`, seeds 0/1, ≈ 32 h) if H8 is to be decided; otherwise carry
"≈ 0.6 % at 1 B tokens" into S3 as the quality cost of halving fabric bytes per FLOP. Rig: 6.3 h,
zero `n_thermal`, GPU 0 ≤ 79 °C. **`queue_small_3.sh` logged QUEUE DONE at 14:05:15; `queue_small_4.sh`
started `138-l5-ne128-small-s0` at 14:05:18** (16.0 k tok/s, 6.01 GiB, ≈ 43 h).
