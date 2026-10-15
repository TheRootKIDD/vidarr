# 150-l5c-small-s1 — L5c at `small`, seed 1: 3.0219 nats; **L5c pair 3.0225, −1.70 % vs the L5 pair** (5.0× the 4σ line) — untied per-iteration attention is a durable gain at two seeds; H6 silent on the clause; σ re-pooled over nine pairs = 0.0026

`python -m scripts.train.train --rung L5c --id 150-l5c-small-s1 --tokens 2.5e+09 --micro-batch 4 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Second and last run of `queue_small_8.sh` (2026-10-14 09:45 → 2026-10-15 00:02). Seed 1 of `149`: L5 with `MiddleCfg(shared_attention=False)`, $N_e$ = 8, $k$ = 2, $d_{ff}$ pinned at 896, fixed $r$ = 8. The run imported the tree at `36257dc`, which carries the not-yet-tested L5e/L3b/L5-pm code (`d355c9d`). On L5c's path that code is a no-op, and the unit suite passed on `36257dc` at 00:02, right after this run (`queue_screen_8.log`). `git_dirty` is the untracked experiment directories.

**Bears on:** **H6**, through `00 §P6` caveat 2. Comparator: the L5 pair (`132`/`133`) 3.0748, same
FLOPs, +11.0 M parameters (+20.8 % non-embedding). Under ADR-013 at two seeds, with **σ re-pooled
over nine pairs = 0.0026** (0.00264, this pair included; was 0.00278 over eight). 2σ = 0.0053, 4σ =
0.0105.

## Numbers

| | |
|---|---|
| params | 88.49 M total, 63.91 M non-embedding (L5: 77.48 / 52.90) |
| attention | 8 modules, one per iteration (L5: 1, shared) |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0219** nats |
| final train loss (last step): main / total | 3.0041 / 3.0117 (router aux 0.0076) |
| throughput (median step) | 49,104 tokens/s aggregate (`step_s` median 10.677 s, p90/median 1.003) |
| peak memory per GPU | 4.07 GiB |
| wall-clock (this session) | 14.27 h |
| seed | 1 |
| routing at the last step | `load_max` 0.130 / `load_min` 0.119, `route_ent_mean` 0.9999, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 80 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 43 rows |

Wall-clock accounting: summed `step_s` 14.108 h + evals 0.149 h + checkpoints 0.008 h = 14.265 h of
14.274 h, **0.07 % unaccounted**.

## The pair against the comparators

| comparator | loss | Δ (nats) | Δ % | ADR-013: matches ≤ 0.0053, resolvable > 0.0105 |
|---|---|---|---|---|
| **L5c pair (`149`/`150`) vs L5 pair (`132`/`133`)** | 3.0225 vs 3.0748 | **−0.0523** | **−1.70 %** | **resolvable, 5.0× the 4σ line** |
| `150` alone vs L5 pair | 3.0219 | −0.0529 | −1.72 % | resolvable |
| `149` alone vs L5 pair | 3.0230 | −0.0518 | −1.69 % | resolvable |
| L5c pair vs L0 pair (`116`/`117`), dense | 2.9958 | +0.0267 | +0.89 % | |
| L5c pair vs L5-ne128 pair (`138`/`142`) | 2.9587 | +0.0638 | +2.15 % | |
| L5c pair vs L1 pair (`130`/`119`), stacked MoE | 2.8839 | +0.1386 | +4.80 % | |

All four seed-wise differences against the L5 seeds are negative (−0.049 to −0.056). The seed gap is
0.0011, the third-smallest of the nine pairs. Against `132`, the 976-window trajectory is −0.055 to
−0.064 from step 600 on, and −0.056 at the end. As with `149`, it is flat from ≈ 1 B tokens.

**σ re-pooled.** Over nine pairs (L0, L1, L2, L3, L5, L5d, L5-ne128, L7b, L5c),
σ = √(mean of (a − b)²/2) = 0.00264. No earlier verdict moves under the slightly tighter bands. The
closest is L7b's +0.0178, which stays above 4σ = 0.0105.

## What the gain is, across budgets

| | `screen` (1 B) | `small` (2.5 B, pair) |
|---|---|---|
| L5c − L5 | −0.0560 (`146` − `110`) | **−0.0523** |
| L5 − L1 (the gap H6 is about) | +0.1280 | +0.1908 |
| share of that gap L5c recovers | 44 % | 27 % |
| L5c − dense L0 | −1.77 % | +0.89 % (L5: +2.64 %) |

The gain is roughly constant in nats from 1 B to 2.5 B tokens. The gap to stacked MoE grows with the
budget, so untied attention closes a shrinking share of it. At `screen`, `148` attributed about half
of the gain to untied attention and half to the 11 M extra parameters. Only a `small` L5-ne13 pair
would split it at this budget, and ADR-033 schedules one only if H6's verdict needs it. It does not:
neither arm is parameter-matched to L1.

## Interpretation

**Resolvable at two seeds: untying attention across iterations improves L5 by 0.0523 nats (−1.70 %,
5.0× ADR-013's 4σ line) at the same FLOPs and +20.8 % non-embedding parameters.** That is the
`screen` result (−0.0560) held at 2.5 B tokens. `00 §P6` caveat 2, the cost of sharing attention, is
real at both budgets. It is a fixed ≈ 0.05 nats, while L5's deficit to stacked MoE grows from 0.128 to
0.191 nats with tokens, so most of that deficit is not attention sharing. **H6: silent on the clause**
(L5c is matched to neither L1's parameters nor L5-pm's). The **mechanism reading sharpens**: a shared
block trails stacked layers by more than its shared attention explains, and the remainder, ≈ 0.14
nats at `small`, sits in the shared expert pool and depth-conditioned norms. That is what L5-pm (`153`,
ADR-035) and its `small` pair test at matched parameters. For designs built on L5, per-iteration
attention is the best-value addition measured so far: it gets 0.0047 nats per M parameters at
`small`, against L5-ne128's 0.0010 at `screen`. It costs no FLOPs and no throughput, and 0.2 GiB per
GPU. Rig: 14.3 h, zero `n_thermal`, ≤ 80 °C.
