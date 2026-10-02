# 143-l7b-small-s0 — L7b ($L_e$ = 2) at `small`, seed 0: 3.0943 nats, **+0.64 % vs the L5 pair** — above ADR-013's H8 line (4σ) by 1.7×: provisional *weakens* (one seed; `144` was not run — paused)

`python -m scripts.train.train --rung L7b --id 143-l7b-small-s0 --tokens 2.5e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Second run of `queue_small_6.sh` (ex-`140`, never started under queue 5). L5 with `ExpertsCfg(layers=2)`: each of the $N_e$ = 8 experts ($k$ = 2) is a two-layer SwiGLU stack at $d_{ff}$ = 448 instead of one layer at 896 — same params and FLOPs; `137` is its `screen`. Driver 615.71.09, kernel 7.2.5.

**Bears on:** **H8** — *$L_e$ = 2 halves fabric bytes per FLOP at ≤ 0.5 % loss* (quality clause, arm
A; the bytes half is S3's arithmetic). Comparator: the L5 pair (`132`/`133`) 3.0748, same params and
FLOPs, $L_e$ = 1. **Threshold, per ADR-013** (binding): "0.5 %" is below the floor and becomes "no
detectable regression", **$T$ = 2σ**; with σ = 0.0028 over seven pairs: *supports* if Δ ≤ 0.0057,
*weakens* if Δ > $T$ + 2σ = **0.0113**, silent between. Earlier entries (`137`, the session log from
2026-09-23 on, the queue-6 notes) applied $T$ = 0.5 % of $L_{ref}$ instead (0.0154; weakens above
0.0210) — that is a misreading of ADR-013, recorded in `docs/04` 2026-10-03; both readings are
given below. **One seed: provisional.** `144` (seed 1) was not started — the queue was paused after
this run on the user's request.

## Numbers

| | |
|---|---|
| params | 77.49 M total, 52.91 M non-embedding (L5: 77.48 / 52.90) |
| expert shape | $N_e$ = 8, $k$ = 2, **$L_e$ = 2, $d_{ff}$ = 448** (L5: $L_e$ = 1, $d_{ff}$ = 896) |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **3.0943** nats |
| final train loss (last step): main / total | 3.0967 / 3.1047 (router aux 0.0080) |
| throughput (median step) | 45,059 tokens/s aggregate (`step_s` median 11.636 s, p90/median 1.008) — 8 % below L5's 49.2 k at equal FLOPs, as at `screen` (two half-width matmuls per expert: a GEMM-efficiency cost on this GPU, not FLOPs) |
| peak memory per GPU | 3.86 GiB |
| wall-clock (this session) | 15.62 h |
| seed | 0 |
| routing at the last step (trainer) | `load_max` 0.128 / `load_min` 0.122, `route_ent_mean` 1.000, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 80 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 29 rows |

Wall-clock accounting: summed `step_s` 15.444 h + evals 0.162 h + checkpoints 0.008 h = 15.614 h of
15.623 h, **0.06 % unaccounted**. The queue script was SIGSTOPped at 18:40 (step ≈ 3350) to stop
`144` starting; the trainer is a separate process and ran on unaffected (`step_s` unchanged).

## Against the comparator

| comparator | loss | Δ (nats) | Δ % | ADR-013 ($T$ = 2σ): supports ≤ 0.0057, weakens > 0.0113 | 0.5 %-of-$L$ reading: supports ≤ 0.0154, weakens > 0.0210 |
|---|---|---|---|---|---|
| **L5 pair** (`132`/`133`) | 3.0748 | **+0.0196** | **+0.64 %** | **weakens** (1.7× the line) | silent (0.0014 under the line) |
| L5 seed 0 (`132`) | 3.0778 | +0.0165 | +0.54 % | | |
| L5 seed 1 (`133`) | 3.0717 | +0.0226 | +0.74 % | | |
| L0 pair (`116`/`117`), dense | 2.9958 | +0.0985 | +3.29 % | | |

Trajectory against the L5 pair, and `137` against `110` at `screen`:

| step (tokens) | 500 (0.26 B) | 1000 (0.52 B) | 1900 (1.0 B) | 2900 (1.5 B) | 3800 (2.0 B) | 4700 (2.46 B) |
|---|---|---|---|---|---|---|
| `143` − L5 pair (nats) | +0.007 | +0.022 | +0.023 | +0.022 | +0.021 | +0.020 |
| `137` − `110` (`screen`) | +0.025 | +0.023 | +0.020 (final 0.0191) | | | |

The cost of two-layer experts is **flat in the token budget**: +0.019 at 1 B tokens (`screen`),
+0.023 at 1 B tokens here, +0.020 at 2.5 B, a slow narrowing of ≈ 0.003 nats over the last 1.5 B
tokens. Like L5d's tax and unlike L5-vs-dense, nothing in the trajectory suggests it closes with
budget at this scale.

## Interpretation

**Provisional *weakens* on H8's quality clause (one seed, ADR-013):** splitting each expert into two
half-width layers costs **+0.0196 nats, +0.64 %** against the L5 pair at equal params and FLOPs,
1.7× ADR-013's weakening line (4σ = 0.0113) and 3.4× its "no detectable regression" band. Under the
looser 0.5 %-of-$L$ reading used in earlier entries it sits in the silent band, 0.0014 nats under
that reading's weakening line — so *no reading supports H8*, and the readings differ only between
"weakens" and "silent". For a seed-1 run to bring the pair inside ADR-013's supports band it would
need Δ ≤ −0.008 (below L5's better seed); the largest seed gap at `small` is 0.006 nats, so the
verdict a pair would formalise is *weakens* under ADR-013. The cost is the same ≈ 0.02 nats measured
at `screen` and is flat in budget. What S3 should carry, if H8 is not pursued further: halving fabric
bytes per FLOP costs ≈ 0.6 % loss here, plus ≈ 8 % GEMM throughput on Ampere at these widths. Rig:
15.6 h, zero `n_thermal`, ≤ 80 °C. **Queue paused after this run** (user, 2026-10-02 18:40);
`144-l7b-small-s1` not started, id free.
