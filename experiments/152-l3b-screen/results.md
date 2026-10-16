# 152-l3b-screen — L3b, the forward MTP curriculum: 3.2094 nats, **−1.21 % vs L3** (3.7× the 4σ line) but **+1.90 % vs L2**; acceptance 38.9 / 21.6 / 14.6 % — both H13 clauses still miss at `screen`; the `small` pair runs (ADR-036)

`python -m scripts.train.train --rung L3b --id 152-l3b-screen --tokens 1e+09 --micro-batch 2 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (32 × 2 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Second training run of `queue_screen_8.sh` (2026-10-15 05:58 → 12:20), after the gate passed on `36257dc`. This is `108-l3-screen` with `MTPCfg(curriculum="forward")` (ADR-036). The budget is split into four equal phases: next-token only, then head {2}, {2, 3}, {2, 3, 4}, with boundaries at steps 477 / 954 / 1430. Each position trains one head drawn among the active ones (ADR-012 subsampling). Inactive heads get a zero gradient. Evaluation always runs all heads. Everything else is L3's: L2's parallel form, $m$ = 4, zero-init adapters, loss weights.

**Bears on:** **H13**, both clauses: *greedy acceptance of heads 2/3/4 against the main head ≥ 70 /
55 / 45 % on held-out, with main-head Δ ≤ 2σ against L2*. Comparators: `107` (L2, 3.1495), `108`
(L3, 3.2486, same subsampling) and, for acceptance only, `113` (L3-full, 3.2342). Under ADR-013 at
one seed, σ = 0.0026 over nine pairs (2σ 0.0053, 4σ 0.0105). H13 is decided at `small` (ADR-030), so
this run reads direction and size.

## Numbers

| | |
|---|---|
| params | 272.92 M total, 248.34 M non-embedding (same as `108`) |
| curriculum | 4 phases × 477 steps (≈ 250 M tokens each); head 4 trains on the last 25 % |
| tokens | 1.000 B in 1907 steps |
| **final val loss, main head** (full held-out, 5606 windows) | **3.2094** nats |
| final train loss (last step): main / total | 3.1789 / 7.9469 (router aux 0.0122) |
| throughput (median step) | 42,748 tokens/s aggregate (`step_s` median 12.265 s, p90/median 1.011). Phase 1 ran at ≈ 50.9 k, phases 2–4 at ≈ 43.0 k: the auxiliary logits pass costs 16 % once any head is on |
| peak memory per GPU | **9.35 GiB** from phase 2 on (8.61 GiB in phase 1; `108`: 8.83 GiB). Flat through phases 3 and 4, since each position still trains one head |
| wall-clock (this session) | 6.36 h |
| seed | 0 |
| routing at the last step | `load_max` 0.143 / `load_min` 0.105, `route_ent_mean` 0.9997, `route_eff_min` 7.96 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 190 telemetry rows**; `temp_c_max` ≤ 80 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 12 rows |

Wall-clock accounting: summed `step_s` 6.252 h + evals 0.093 h + checkpoints 0.009 h = 6.354 h of
6.358 h, **0.06 % unaccounted**.

The 0.74 GiB rise over `108` is unexplained: in phases 2–4 the activations should match L3's. It
leaves 0.65 GiB of headroom under the cap, and the `small` pair runs at the same micro-batch and
sequence shape, so it should peak the same.

## H13's clauses

| clause | H13 requires | L3b `152` | L3 `108` | L3-full `113` |
|---|---|---|---|---|
| acceptance vs main head, head 2 | ≥ 70 % | **38.9 %** | — | 39.7 % |
| acceptance vs main head, head 3 | ≥ 55 % | **21.6 %** | — | 23.0 % |
| acceptance vs main head, head 4 | ≥ 45 % | **14.6 %** | — | 16.1 % |
| top-1 vs the true token, heads 2 / 3 / 4 | — | 21.6 / 12.1 / 8.2 % | 21.5 / 12.6 / 8.9 % | |
| main-head Δ vs L2 (`107`) | ≤ 2σ = 0.0053 | **+0.0599 (+1.90 %)** | +0.0991 (+3.15 %) | +0.0847 (+2.69 %) |

`108` predates the acceptance-vs-main-head metric (it logged top-1 vs the true token, which its
`results.md` reported as "acceptance"). Acceptance proper is compared against `113`, which trained
every head at every position and gained ≈ 1 point from it (I27 #3).

## Main loss against the comparators

| comparator | loss | Δ (nats) | Δ % | ADR-013: matches ≤ 0.0053, resolvable > 0.0105 |
|---|---|---|---|---|
| **`108` L3, same heads, no curriculum** | 3.2486 | **−0.0392** | **−1.21 %** | resolvable, 3.7× the 4σ line |
| **`107` L2, no MTP** | 3.1495 | **+0.0599** | **+1.90 %** | resolvable, 5.7× |
| `113` L3-full | 3.2342 | −0.0248 | −0.77 % | resolvable |

**The main-head cost tracks the number of active heads.** Against `107` along the trajectory
(976-window evals):

| phase (steps) | active heads | L3b − L2 at the end of the phase |
|---|---|---|
| 1 (0–476) | none | −0.071 (step 400) |
| 2 (477–953) | {2} | +0.029 (step 900) |
| 3 (954–1429) | {2, 3} | +0.050 (step 1400) |
| 4 (1430–1907) | {2, 3, 4} | +0.061 (step 1900), still widening |

In phase 1 L3b is L2 with three idle adapters, yet it leads `107` by 0.03–0.08. Early losses move
fast and the two runs draw different initial weights, so that lead says nothing about MTP. It is gone
within 30 steps of head 2 switching on. From then on, the gap grows with each added head and has not
levelled off at 1 B tokens. Acceptance rises within a phase (head 2: 20 % → 39 % over phases 2–4;
head 4: 12 % → 14.6 % in its 477 steps) and is still rising slowly at the end.

## Interpretation

**Resolvable at one seed: the forward curriculum improves L3's main head by 0.0392 nats (−1.21 %,
3.7× ADR-013's 4σ line), but L3b still trails L2 by 0.0599 (+1.90 %, 11× H13's 2σ margin).**
Acceptance is 38.9 / 21.6 / 14.6 % against H13's 70 / 55 / 45 %, about one point under L3-full's at
every head. That is ADR-036's stated trade, and it is smaller than feared. **H13: both clauses still
miss at `screen`**: weakened in direction, not decided, because H13 is read at `small`. The curriculum
recovers ≈ 40 % of L3's main-head cost (0.0392 of 0.0991). The trajectory says why it stops there:
the cost comes back with each head that switches on, and phase 4 ends before it settles. A curriculum
cannot remove a cost that grows with active heads; it only shortens the time the heads are on. **Per
ADR-036 §3 the `small` pair runs** (`156`/`157`, 2.5 B tokens, seeds 0/1, ≈ 2 × 16.3 h at 42.7 k
tok/s) against the L2 pair (2.8851) and the L3 pair (2.9682). There, phase 4 is 625 M tokens, the
budget I20 says MTP needs. If the pair still misses both clauses, L3c (sequential modules, `01 §7`)
is the last route left for H13. Rig: 6.4 h, zero `n_thermal`, ≤ 80 °C, 9.35 GiB per GPU.
