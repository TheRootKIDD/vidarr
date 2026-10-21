# 157-l3b-small-s1 — L3b at `small`, seed 1: 2.9420 nats; **L3b pair 2.9447, +2.06 % vs the L2 pair, acceptance 41.7 / 23.9 / 15.7 % — H13 weakened on both clauses at two seeds**; the curriculum recovers 28 % of L3's main-head cost; L3c is the last route; σ re-pooled over eleven pairs = 0.0029

`python -m scripts.train.train --rung L3b --id 157-l3b-small-s1 --tokens 2.5e+09 --micro-batch 2 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (32 × 2 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Second and last run of `queue_small_9.sh` (2026-10-18 10:48 → 2026-10-19 02:40), on the tree at `ae08987` (docs-only on top of the gate commit `36257dc`). Seed 1 of `156`: L3 with the forward MTP curriculum (ADR-036), four phases of 1192 steps, boundaries at steps 1192 / 2384 / 3576.

**Bears on:** **H13**, both clauses: *greedy acceptance of heads 2/3/4 against the main head ≥ 70 /
55 / 45 % on held-out, with main-head Δ ≤ 2σ against L2*. Comparators: the L2 pair (`120`/`121`)
2.8851 and the L3 pair (`122`/`131`) 2.9682, whose acceptance was logged with the same metric
(`val_mtp_accept_h*`, acceptance against the main head). Under ADR-013 at two seeds, with **σ
re-pooled over eleven pairs = 0.0029** (0.00285, this pair included; was 0.00274 over ten). 2σ =
0.0057, 4σ = 0.0114. H13 is decided at `small` (ADR-030), and this is its last scheduled arm before
L3c.

## Numbers

| | |
|---|---|
| params | 272.92 M total, 248.34 M non-embedding (as L3) |
| tokens | 2.500 B in 4768 steps |
| **final val loss, main head** (full held-out, 5606 windows) | **2.9420** nats |
| acceptance vs main head, heads 2 / 3 / 4 (held-out) | **41.7 / 23.9 / 15.7 %** |
| head top-1 vs ground truth, heads 2 / 3 / 4 (held-out) | 24.2 / 13.9 / 9.0 % |
| final train loss (last step): main / total | 2.9248 / 7.4248 (router aux 0.0122) |
| throughput (median step) | 42,813 tokens/s aggregate (`step_s` median 12.246 s, p90/median 1.012) |
| peak memory per GPU | 9.35 GiB |
| wall-clock (this session) | 15.86 h |
| seed | 1 |
| routing at the last step | `load_max` 0.139 / `load_min` 0.114, `route_ent_mean` 0.9997, `route_eff_min` 7.99 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 79 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 31 rows |

Wall-clock accounting: summed `step_s` 15.627 h + evals 0.198 h + checkpoints 0.023 h = 15.848 h of
15.858 h, **0.06 % unaccounted**.

## H13's clauses, two seeds

| clause | H13 requires | **L3b pair** (`156` / `157`) | L3 pair (`122`/`131`) | L3b `screen` (`152`) |
|---|---|---|---|---|
| acceptance vs main head, head 2 | ≥ 70 % | **41.7 %** (41.8 / 41.7) | 42.5 % | 38.9 % |
| acceptance vs main head, head 3 | ≥ 55 % | **23.9 %** (23.9 / 23.9) | 25.6 % | 21.6 % |
| acceptance vs main head, head 4 | ≥ 45 % | **15.7 %** (15.6 / 15.7) | 17.6 % | 14.6 % |
| head top-1 vs ground truth, 2 / 3 / 4 | — | 24.2 / 14.0 / 9.0 % | 24.5 / 14.9 / 10.1 % | 21.6 / 12.1 / 8.2 % |
| **main-head Δ vs the L2 pair** | ≤ 2σ = 0.0057 | **+0.0596 (+2.06 %)** | +0.0832 (+2.88 %) | +0.0599 vs `107` |

Acceptance misses by 28 / 31 / 29 points. The main-head cost is 10.5× H13's 2σ margin. Each seed
misses each clause on its own.

## Main loss against the comparators

| comparator | loss | Δ (nats) | Δ % | ADR-013: matches ≤ 0.0057, resolvable > 0.0114 |
|---|---|---|---|---|
| **L3b pair vs L2 pair (`120`/`121`)** | 2.9447 vs 2.8851 | **+0.0596** | **+2.06 %** | **weakens: T = 2σ, Δ > T + 2σ** |
| **L3b pair vs L3 pair (`122`/`131`)** | 2.9682 | **−0.0236** | **−0.79 %** | resolvable, 2.1× the 4σ line |
| L3b pair vs L0 pair (`116`/`117`), dense | 2.9958 | −0.0511 | −1.71 % | |
| `157` alone vs L2 pair | 2.9420 | +0.0569 | +1.97 % | |

All four seed-wise differences against the L2 seeds are positive (+0.054 to +0.065). The seed gap is
0.0053.

**σ re-pooled.** Over eleven pairs (L0, L1, L2, L3, L5, L5d, L5-ne128, L7b, L5c, L5e, L3b),
σ = √(mean of (a − b)²/2) = 0.00285. No earlier verdict moves. The two readings closest to a line are
L7b's +0.0178 (weakens, still above 4σ = 0.0114) and `153`'s −0.0106 vs L1 at `screen` (silent,
now 3.7σ).

## The cost by phase (976-window evals, pair means)

| phase (steps) | active heads | L3b − L2 pair at the end of the phase | L3b − L3 pair |
|---|---|---|---|
| 1 (0–1191) | none | +0.002 (step 1100) | −0.100 |
| 2 (1192–2383) | {2} | +0.032 (step 2300) | −0.054 |
| 3 (2384–3575) | {2, 3} | +0.052 (step 3500) | −0.032 |
| 4 (3576–4767) | {2, 3, 4} | +0.061 (step 4700) | −0.023 |

This is `152`'s shape at 2.5× the tokens. In phase 1, L3b is L2 and ties it. Each head that switches
on adds a step (+0.071 a few steps after head 2 starts, at step 1200), and the gap recovers only
partly within the phase. In phase 4 it widens from +0.059 to +0.061 and is close to level. The
advantage over L3 shrinks phase by phase, because L3 pays for all heads from the start and L3b
catches up to paying for them.

## Interpretation

**Weakened at two seeds, on both clauses: the forward MTP curriculum does not bring L3 within H13's
bounds at `small`.** The main head trails L2 by 0.0596 nats (+2.06 %, 10.5× the 2σ margin), and
acceptance is 41.7 / 23.9 / 15.7 % against 70 / 55 / 45 %. The curriculum helps L3 resolvably
(−0.0236, 2.1× 4σ), but it recovers only **28 % of L3's main-head cost at `small`**, against 40 % at
`screen`. Acceptance drops 0.8 / 1.7 / 1.9 points below L3's, the trade ADR-036 stated in advance. The
phase table shows the mechanism: each active head costs the main head a roughly fixed amount, and a
curriculum only shortens the time it pays. Phase 4 at `small` is 625 M tokens, the budget I20 said
sub-1 B models need, and it did not settle the cost. **H13: weakened** (it was weakened by the L3
pair; this closes the recovery route ADR-036 opened). The remaining route is **L3c**, sequential MTP
modules (ADR-012's fallback, `01 §7`), which ADR-031 gates on "L3b fails, and I26 first". L3b has
failed. Whether L3c is worth an I26 design pass and a `screen` run is the user's call. The
alternative is to accept H13 as weakened at this scale and move on. Rig: 15.9 h, zero `n_thermal`,
≤ 79 °C, 9.35 GiB per GPU.
