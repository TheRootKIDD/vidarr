# 142-l5-ne128-small-s1 — H6's matched arm at `small`, seed 1: 2.9586 nats; **the pair (2.9587) sits +2.60 % above the L1 pair — *weakens* H6** (13× the "matches" band), with I29's 68 %-params caveat

`python -m scripts.train.train --rung L5-ne128 --id 142-l5-ne128-small-s1 --tokens 2.5e+09 --micro-batch 4 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. First run of `queue_small_6.sh`; the rerun of `139` (killed at step 850 on 2026-09-25, burnt). Same preset as `138` — shared middle block at $r$ = 8, **$N_e$ = 128, $k$ = 4, $d_{ff}$ = 448** (ADR-025) — only the seed differs (config diff: id, seed, date, git hash). Driver 615.71.09, kernel 7.2.5 (`029`).

**Bears on:** **H6** — *one depth-conditioned shared middle block with $r_{max} n$ experts matches
$r_{max}$ stacked MoE layers with $n$ experts each, at matched params and FLOPs.* FLOPs match L1's;
parameters are **168.5 M non-embedding against L1's 246.6 M (68 %)** — I29's caveat applies to the
reading, as in `138`. ADR-013: "matches" = $|Δ| \le$ 2σ against the L1 pair, *weakens* above 4σ.
**This run completes the pair: verdict *weakens H6*.**

## Numbers

| | |
|---|---|
| params | 193.1 M total, **168.5 M non-embedding** (identical to `138`) |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **2.9586** nats (`138`: 2.9588) |
| final train loss (last step): main / total | 2.9398 / 2.9472 (router aux 0.0074) |
| throughput (median step) | 16,543 tokens/s aggregate (`step_s` median 31.693 s, p90/median 1.005) |
| peak memory per GPU | 6.01 GiB |
| wall-clock (this session) | 42.29 h |
| seed | 1 |
| depth | fixed $r$ = 8 |
| routing at the last step (trainer) | `load_max` 0.0090 / `load_min` 0.0064 (uniform 0.0078), `route_ent_mean` 0.9996, `route_eff_min` 127.8 of 128, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 67 °C, `sm_mhz_min` ≥ 1912; `n_power_cap` 0 rows |

Wall-clock accounting: summed `step_s` 41.999 h + evals 0.238 h + checkpoints 0.046 h = 42.283 h of
42.293 h, **0.02 % unaccounted**. Throughput is clean: the 3–12 % per-step cost of a concurrent CPU
probe measured on this run (`docs/04` 2026-09-30) touched steps 147–164 only (≈ 10 min, before the
probes were stopped); the median is unaffected.

## The pair against the comparators

**Seed spread.** The two seeds land **0.00014 nats apart** at the final eval, and the trajectories
agree as closely all the way: from step 1000 on, seed 1 − seed 0 averages +0.0003 with a largest
gap of 0.0022 (step 500: −0.008). Unlike the L5d pair's 0.00002 (a final-eval coincidence on
trajectories ≈ 0.001 apart), this is a genuinely tight pair.

**σ re-pooled over seven pairs** (per-pair variance $d^2/2$, pooled; the six-pair method reproduces
0.0031): **σ = 0.0028 nats**; bands 2σ = **0.0057**, 4σ = **0.0113**. Two of the seven pairs are now
near-zero, which pulls σ down by 0.0003; no verdict on record flips (the closest, L2 vs L1 at
+0.0012, stays inside 2σ).

| comparator | non-emb. params | loss | Δ (nats) | Δ % | vs 2σ = 0.0057 / 4σ = 0.0113 |
|---|---|---|---|---|---|
| **L1 pair** (`130`/`119`), H6's comparator | 246.6 M | 2.8839 | **+0.0748** | **+2.60 %** | **13× / 6.6×** — outside "matches", above the weakening line |
| L2 pair (`120`/`121`) | 246.5 M | 2.8851 | +0.0736 | +2.55 % | |
| L0 pair (`116`/`117`), dense | 75.5 M | 2.9958 | **−0.0371** | **−1.24 %** | 3.3× the 4σ band |
| L5 pair (`132`/`133`), $N_e$ = 8 | 52.9 M | 3.0748 | −0.1160 | −3.77 % | |

Trajectory of the pair (mean of `138`/`142`) — the same shape `138` showed alone:

| step (tokens) | 500 (0.26 B) | 1000 (0.52 B) | 1900 (1.0 B) | 2900 (1.5 B) | 3800 (2.0 B) | 4700 (2.46 B) |
|---|---|---|---|---|---|---|
| L5-ne128 pair − L1 pair (nats) | −0.099 | +0.020 | +0.055 | +0.068 | +0.073 | +0.076 |
| L5-ne128 pair − L0 pair (nats) | −0.232 | −0.088 | −0.050 | −0.040 | −0.038 | −0.036 |
| L5-ne128 pair − L5 pair (nats) | −0.124 | −0.100 | −0.103 | −0.110 | −0.113 | −0.115 |

## Routing

Not probed yet: a CPU probe beside a live DDP run costs it 3–12 % per step, so `probe_routing` on
this checkpoint runs after `queue_small_6` finishes (with the N_e = 8 `small` probes on `132`–`135`),
and the comparison with `138`'s depth partition (mean inter-iteration correlation 0.35, effective
experts 128 → 87 with depth) is added to the session log then. The trainer's last-step routing is
indistinguishable from `138`'s.

## Interpretation

**H6: *weakens*, two seeds.** The FLOPs-matched shared middle block with 128 experts sits at
2.9587 nats against the stacked-MoE L1 pair's 2.8839: **+0.0748 nats, +2.60 %**, 13× the 2σ
"matches" band and 6.6× the 4σ weakening line, with the two seeds 0.00014 apart — the gap is not
noise. The trajectory is the crossing-then-flattening shape seen three times now (lead to ≈ 0.4 B
tokens, then a gap that widens and settles near +0.075). I29's caveat stands with the verdict: the arm
has 68 % of L1's non-embedding parameters, and at this scale a 32 % parameter deficit could plausibly
account for a 2.6 % loss gap, so what is refuted is H6 at matched FLOPs as the ladder can build it;
an exactly parameter-matched shared block (≈ 1.45× L1's FLOPs) is the remaining test and needs its own
ADR. Independent of that caveat, the shared block **beats dense by 1.24 %** at both seeds, and the
sixteen-fold expert count is worth a stable −0.115 nats over $N_e$ = 8. Rig: 42.3 h, zero
`n_thermal`, ≤ 67 °C. `143-l7b-small-s0` started 08:03 (≈ 16 h).
