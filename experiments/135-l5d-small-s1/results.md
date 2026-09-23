# 135-l5d-small-s1 — L5d at `small`, seed 1: 3.0896 nats; **the L5d pair = 3.0896, +0.48 % vs the L5 pair against H15a's 2 % — supports H15a at two seeds**; σ re-pooled over six pairs to 0.0031

`python -m scripts.train.train --rung L5d --id 135-l5d-small-s1 --tokens 2.5e+09 --micro-batch 4 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Fifth run of `queue_small_3.sh`; the rerun of the burnt `127`. Same model as `134`: L5 with global attention over the final-vector cache $\mathcal{G}$ only, per-iteration attention local ($W$ = 512), segment-recurrent training with stop-gradient memory (ADR-021/022/023). Driver 615.71.09.

**Bears on:** **H15a** — *global attention over final vectors only, with per-iteration attention kept
local, costs ≤ 2 % loss vs global per-iteration attention at 2 k–8 k context.* Completes the L5d
pair; the comparator is the L5 pair (`132`/`133`, 3.0748). Measured at the 2 k training context
only; the 8 k end of the range is still owed. **Verdict: supports, two seeds each side.**

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0896** nats (3.08958) |
| final train loss (last step): main / total | 3.0736 / 3.1055 (router aux 0.0320) |
| throughput (median step) | 33,852 tokens/s aggregate (`step_s` median 15.488 s, p90 15.574, p90/median 1.006) — `134`: 33,754 |
| peak memory per GPU | 3.89 GiB |
| wall-clock (this session) | 20.71 h |
| seed | 1 |
| depth | fixed $r$ = 8 (`r_mean` 8.0 train and eval) |
| routing at the last step | `load_max` 0.130 / `load_min` 0.119, `route_ent_mean` 0.9998, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 77 °C, `sm_mhz_min` ≥ 1897; `n_power_cap` 1 row |

Wall-clock accounting: summed `step_s` 20.520 h + evals 0.175 h + checkpoints 0.010 h = 20.704 h of
20.714 h, **0.05 % unaccounted**. (`n_throttled` = 4 on every row is the 615.71 `0x400` bit, `030`.)

## The L5d pair

| | seed 0 (`134`) | seed 1 (`135`) | **pair** |
|---|---|---|---|
| full held-out loss | 3.08956 | 3.08958 | **3.0896** |

**The two seeds land 0.00002 nats apart.** That is a coincidence of the final evaluation, not a
property of the recipe: the runs differ along the whole trajectory (step-1 loss 10.5526 vs 10.5532,
last-batch train loss 3.0901 vs 3.0736, seed 0 − seed 1 over the last 20 paired evals
−0.0008 ± 0.0005, and the 976-window evals at 4700 read 3.0865 vs 3.0876) and only meet at the
5606-window final eval. The pair's sample sd, $s$ = 0.00001, enters the pool below as ADR-013
prescribes, but a spread of zero at one pair says nothing about σ; the honest reading of L5d's seed
spread is the trajectory's ≈ 0.001, the tightest of the six pairs (L0 0.0026, L3 0.0018 were the
previous tight ones).

## H15a at two seeds

| comparator | loss | Δ (nats) | Δ % | $T$ = 2 % | weakens above |
|---|---|---|---|---|---|
| **L5 pair** (`132`/`133`) | 3.0748 | **+0.0148** | **+0.48 %** | 0.060 | 0.0667 (σ = 0.0034) / 0.0661 (σ = 0.0031) |
| at `screen`: `111` vs `110` | 3.3477 vs 3.3328 | +0.0149 | +0.45 % | 0.0667 | |

**Verdict rule (ADR-013):** pair against pair, *supports* if $Δ \le T$, *weakens* if $Δ > T + 2σ$.
Δ = +0.0148 → **supports H15a**, two seeds each side, at a quarter of the allowance; 2.2× (σ =
0.0034) or 2.4× (σ = 0.0031) the 2σ band, so the cost is resolvable and small. The absolute cost is
the same as at `screen` to 0.0001 nats, and flat along the trajectory from 1 B tokens
(pair − pair: +0.014 at 1 B, +0.016 at 2.46 B):

| step (tokens) | 500 (0.26 B) | 1000 (0.52 B) | 1900 (1.0 B) | 2900 (1.5 B) | 3800 (2.0 B) | 4700 (2.46 B) |
|---|---|---|---|---|---|---|
| L5d pair − L5 pair (nats) | +0.008 | +0.010 | +0.014 | +0.015 | +0.016 | +0.016 |

Against dense the L5d pair sits +3.13 % above the L0 pair (L5: +2.64 %).

## σ re-pooled over six pairs

Pair sample sds: L0 0.00184, L1 0.00431, L2 0.00382, L3 0.00125, L5 0.00429, **L5d 0.00001** →
rms = **0.0031 nats** (was 0.0034 over five). Two seeds stand (well under the 0.01 trigger). Bands
at `small`: 2σ = **0.0061**, 4σ = **0.0123**, 1 % = 0.030 (weakens above 0.0361), 2 % = 0.060
(weakens above 0.0661). No verdict flips: H2 (+0.0012) inside 2σ; H13 (+0.0831) 6.7× the weakening
line; H15a as above; at `screen`, `112` (H6, +0.0184) weakens by 0.0061 over the 0.0123 line, `111`
(+0.0149) 1.2× the silent band. The pooled value is dragged down by the L5d coincidence; with L5d's
trajectory spread (≈ 0.001) in its place σ would be 0.0032, so nothing hangs on the choice.

## Interpretation

**Supports H15a at two seeds** (Δ = +0.0148 nats, +0.48 %, pair against pair, against $T$ = 2 %). The
quality half of v2's P7 holds at `screen` and `small` with the same absolute cost, so at 2 k context
final-vector global attention is a ≈ 0.015-nat tax that does not grow with the token budget. Still
owed for the 8 k end of H15a's range: an evaluation of an L5d checkpoint (`/mnt/nvme/ckpt/134`,
`135`) at 8 k context, which needs no training. The systems half of P7 — the KV-cache and
communication saving — is `sim/`'s, not this trainer's. Rig: 20.7 h, zero `n_thermal`, GPU 0 ≤ 77 °C.
`136-l5-noej-screen` started 02:06.
