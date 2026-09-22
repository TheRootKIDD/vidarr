# 134-l5d-small-s0 — L5d at `small`, seed 0: 3.0896 nats, **+0.48 % vs the L5 pair** against H15a's 2 % allowance — **supports H15a** (one seed; `135` completes the pair)

`python -m scripts.train.train --rung L5d --id 134-l5d-small-s0 --tokens 2.5e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Fourth run of `queue_small_3.sh`; the rerun of the burnt `126`. L5 ($r$ = 8, $N_e$ = 8, $k$ = 2) with the single change of `111`: global attention range moves from every iteration's own cache to the final-vector cache $\mathcal{G}$, per-iteration attention local ($W$ = 512), segment-recurrent training with stop-gradient memory (ADR-021/022/023). 77.5 M params, +768 over L5. Driver 615.71.09.

**Bears on:** **H15a** — *global attention over final vectors only, with per-iteration attention kept
local, costs ≤ 2 % loss vs global per-iteration attention at 2 k–8 k context.* The quality half of
v2's P7; the comparator is the L5 pair (`132`/`133`, 3.0748). Measured at the 2 k training context
only, as `111`; the 8 k end of the range is still an evaluation owed. **Verdict at one seed:
supports** — the second seed (`135`) makes it ADR-013's two-seed pair-against-pair Δ.

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0896** nats |
| final train loss (last step): main / total | 3.0901 / 3.1222 (router aux 0.0321) |
| throughput (median step) | 33,754 tokens/s aggregate (`step_s` median 15.532 s, p90 15.614, p90/median 1.005) — 31 % below L5's 49.2 k |
| peak memory per GPU | 3.89 GiB |
| wall-clock (this session) | 20.77 h |
| seed | 0 |
| depth | fixed $r$ = 8 (`r_mean` 8.0 train and eval) |
| routing at the last step | `load_max` 0.128 / `load_min` 0.119, `route_ent_mean` 0.9999, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 76 °C, `sm_mhz_min` ≥ 1897; `n_power_cap` 1 row |

Wall-clock accounting: summed `step_s` 20.570 h + evals 0.175 h + checkpoints 0.010 h = 20.755 h of
20.765 h, **0.05 % unaccounted**. (`n_throttled` = 4 on every row is the 615.71 `0x400` bit, `030`.)

## H15a at 2.5 B tokens

| comparator | loss | Δ (nats) | Δ % | $T$ = 2 % | 2σ |
|---|---|---|---|---|---|
| **L5 pair** (`132`/`133`) | 3.0748 | **+0.0148** | **+0.48 %** | 0.060 | 0.0067 |
| L5 seed 0 alone (`132`) | 3.0778 | +0.0118 | +0.38 % | | |
| at `screen`: `111` vs `110` | 3.3477 vs 3.3328 | +0.0149 | +0.45 % | 0.0667 | |

**Verdict rule (ADR-013), σ = 0.0034 (five pairs):** *supports* if $Δ \le T$ = 0.060, *weakens* if
$Δ > T + 2σ$ = 0.0667. Δ = +0.0148 → **supports H15a**, at a quarter of the allowance and 2.2× the
2σ band, so the cost is resolvable, small, and inside the claim. The `screen` and `small` readings
are the same to 0.0001 nats in absolute terms (+0.0149 vs +0.0148): **the cost of final-vector
global attention does not grow with the token budget** from 1 B to 2.5 B. Along the trajectory
against the L5 pair mean:

| step (tokens) | 500 (0.26 B) | 1000 (0.52 B) | 1900 (1.0 B) | 2900 (1.5 B) | 3800 (2.0 B) | 4700 (2.46 B) |
|---|---|---|---|---|---|---|
| L5d − L5 pair (nats) | −0.009 | +0.009 | +0.013 | +0.014 | +0.015 | +0.015 |

Flat at ≈ +0.015 from 1 B tokens on. Against dense, L5d sits +3.13 % above the L0 pair (L5: +2.64 %).

## Cost of the segment-recurrent form

31 % lower throughput than L5 (33.8 k vs 49.2 k tok/s) at the same memory: the segment loop and the
final-vector cache in the reference implementation, not a FLOP difference — the FLOPs per token are
those of L5 (`costmodel`). That is a property of `model/` (readability over speed, no custom
kernels) and is not a systems claim; the note's argument for P7 is the KV-cache and communication
saving at inference, which `sim/` scores, not this trainer. Power-cap rows fell from 29–40 (L5) to
1: the cards are lighter-loaded, consistent with the loop being host-side.

## Interpretation

**Supports H15a at one seed** (Δ = +0.0148 nats, +0.48 %, against $T$ = 2 %), with the same absolute
cost as at `screen`, so the allowance is met with a wide margin and the margin is stable in budget.
The 8 k-context end of H15a's range is still unmeasured (an eval on the checkpoint, owed). Rig:
20.8 h at 33.8 k tok/s, zero `n_thermal`, GPU 0 ≤ 76 °C. `135-l5d-small-s1` started 05:23 and
completes the pair (≈ 20.8 h).
