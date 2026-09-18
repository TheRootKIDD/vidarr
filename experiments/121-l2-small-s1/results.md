# 121-l2-small-s1 — L2 at `small`, seed 1: 2.8824 nats; the L2 pair = 2.8851, +0.0012 vs the L1 pair — **supports H2 (quality clause)**; σ re-pooled to 0.0035 nats

`python -m scripts.train.train --rung L2 --id 121-l2-small-s1 --tokens 2.5e+09 --micro-batch 2 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (32 × 2 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Fourth run of `queue_small_2.sh`; same command as `120` with `--seed 1`.

**Bears on:** **H2**, quality clause — *parallel attention + MoE + memory costs ≤ 1 % loss at matched
FLOPs* — now at two seeds against the two-seed L1 comparator, which is the form ADR-013's Δ is
defined in. Silent on H2's critical-path clause (Tier S, S1 in `sim/`). As `107`/`120` note, L2
applies the parallel form to the layered stack; the shared-block setting comes with L5 onward.

## Numbers

| | |
|---|---|
| params | 271.1 M total, 246.6 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| final val loss (full held-out, 5606 windows) | **2.8824** nats |
| final train loss (last step, main head) | 2.8627 |
| throughput (median step) | 50,910 tokens/s aggregate (`step_s` median 10.298 s, p90 10.310, p90/median 1.001) |
| peak memory per GPU | 8.58 GiB |
| wall-clock (this session) | 13.82 h |
| seed | 1 |
| routing at the last step | `load_max` 0.134 / `load_min` 0.112 (ideal 0.125), `route_ent_mean` 0.9997, `route_eff_min` 7.99 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 78 °C, `sm_mhz_min` ≥ 1897; `n_power_cap` 39 rows (GPU 0 at its 170 W cap) |

Wall-clock accounting: summed `step_s` 13.645 h + evals 0.147 h + checkpoints 0.020 h = 13.812 h of
13.82 h, **0.07 % unaccounted**.

## The L2 pair

| | seed 0 (`120`) | seed 1 (`121`) | pair |
|---|---|---|---|
| final val loss, full held-out | 2.8878 | 2.8824 | **mean 2.8851**, $|Δ_{seed}|$ = 0.0054 |
| val loss at the 1 B-token eval (step 1900, 2 M-token subset) | 3.1019 | 3.0959 | $|Δ_{seed}|$ = 0.0060 |
| wall-clock | 13.83 h | 13.82 h | 50.9 k tok/s both |

Seed 1 is the *better* seed here, the reverse of the L1 pair (where seed 0 led by 0.0061); the gap is
a stable offset along the trajectory (seed 0 − seed 1 over the last 20 paired subset evals
+0.0057 ± 0.0007). Seed-to-seed spread is therefore not a property of the seed but of the run, and
it is ≈ 0.005–0.006 nats for the 271 M MoE rungs against 0.0026 for the 100 M dense model.

### Against L1, pair against pair

| | val loss (pair mean) | Δ |
|---|---|---|
| L1 pair (`130`/`119`), the comparator | 2.8839 | — |
| **L2 pair (`120`/`121`)** | **2.8851** | **+0.0012 nats, +0.04 %** |
| L0 pair, for scale | 2.9958 | L2 is −0.1107 (−3.70 %) below dense |
| `107` vs `106` at `screen`, one seed | 3.1495 vs 3.2048 | −0.0553 nats, −1.7 % |

**Verdict rule (ADR-013), σ = 0.0035 (re-pooled below):** $T$ = 1 % of $L_{ref}$ = 0.030 nats;
*supports* if $Δ \le T$, *weakens* if $Δ > T + 2σ$ = 0.037. Δ = +0.0012 → **supports H2's quality
clause**, two seeds each side. The parallel form is 25× under the 1 % allowance and inside the 2σ =
0.0070 band, so it also meets the stricter "no detectable regression" reading. It does *not*
improve on the sequential form: the `screen` −1.7 % (`107`) was a 1 B-token schedule effect, as `120`
found; at this budget the two forms are indistinguishable to 0.04 %. Per seed: L2 − L1 is +0.0070
(seed 0) and −0.0045 (seed 1), opposite signs, which is what "no difference" looks like at this σ.

## σ re-pooled over the L0, L1 and L2 pairs

| pair | $|Δ_{seed}|$ | $s = |Δ_{seed}|/\sqrt{2}$ |
|---|---|---|
| L0 (`116`/`117`) | 0.0026 | 0.00184 |
| L1 (`130`/`119`) | 0.0061 | 0.00431 |
| L2 (`120`/`121`) | 0.0054 | 0.00382 |
| **pooled σ** (rms over pairs) | | **0.0035 nats** (was 0.0033 after L1, 0.0018 after L0) |

Still under ADR-013's 0.01 third-seed trigger; **two seeds stand**. Bands at `small` (fractions of
$L_{ref}$ = 2.9958 unchanged): 2σ = **0.0070**, 4σ = **0.0140**, 1 % = 0.030 (weakens above 0.037),
2 % = 0.060 (weakens above 0.067). The `screen` re-scoring (`docs/04` 2026-09-17) recomputed against
4σ = 0.0140: no flip; `112` (L5-ne128, +0.0184) still *weakens* H6 by 0.0044 over the 2σ + 2σ line,
and `111` (L5d, +0.0149) is now at 1.07× the silent band — effectively at the edge of resolvable.

## Thermals

Fourth consecutive ≈ 14 h load at the 271 M rungs: zero thermal-slowdown rows, GPU 0
(`GPU-4673cc7d`) at 75–78 °C and ≥ 1897 MHz, step time flat to 0.1 %. Throughput identical to `120`.

## Interpretation

**Supports H2's quality clause, two seeds each side, Δ = +0.0012 nats against $T$ = 0.030 and
2σ = 0.0070.** The parallel form is a free rearrangement at this scale and budget: no cost, no
gain, and a 1.2 % faster step. H2's second clause (≥ 30 % shorter per-iteration critical path) is a
simulator result (S1) and is not touched here. The next `small` pair, L3 (`122`/`123`), is the one
ADR-030 deferred H13's verdict to; it is scored against this L2 pair (2.8851) with the 2σ = 0.0070
margin on the main head.
