# 119-l1-small-s1 — L1 at `small`, seed 1: 2.8869 nats; the L1 pair gives 2.8839 (−3.74 % vs $L_{ref}$) and **σ re-pooled to 0.0033 nats**

`python -m scripts.train.train --rung L1 --id 119-l1-small-s1 --tokens 2.5e+09 --micro-batch 2 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (32 × 2 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Second run of `queue_small_2.sh`; same command as `130` with `--seed 1` (initialisation and data order only).

**Bears on:** H4 — as its *baseline arm*, not its test, as `130`/`106`. **On H4 itself silent.** What
the pair settles is (1) the L1 comparator at `small` for L2's H2 clause (`120`/`121`) and I29's H6
arm, (2) the size of the MoE-over-dense gap at 2.5 B tokens with both seeds in, and (3) the first
re-pooling of σ that `117/results.md` promised: ADR-013's σ is the pooled seed spread over the
`small` pairs, and the L1 pair spreads **2.3× wider** than the L0 pair.

## Numbers

| | |
|---|---|
| params | 271.2 M total, 246.6 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| final val loss (full held-out, 5606 windows) | **2.8869** nats |
| final train loss (last step, main head) | 2.8673 |
| throughput (median step) | 50,260 tokens/s aggregate (`step_s` median 10.432 s, p90 10.457, p90/median 1.002) |
| peak memory per GPU | 8.72 GiB |
| wall-clock (this session) | 14.00 h |
| seed | 1 |
| routing at the last step | `load_max` 0.134 / `load_min` 0.113 (ideal 0.125), `route_ent_mean` 0.9998, `route_eff_min` 7.99 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 78 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 28 rows (GPU 0 at its 170 W cap) |

Wall-clock accounting: summed `step_s` 13.823 h + evals 0.149 h + checkpoints 0.020 h = 13.992 h of
14.00 h, **0.06 % unaccounted**.

## The L1 pair

| | seed 0 (`130`) | seed 1 (`119`) | pair |
|---|---|---|---|
| final val loss, full held-out | 2.8808 | 2.8869 | **mean 2.8839**, $|Δ_{seed}|$ = 0.0061 |
| val loss at the 1 B-token eval (step 1900, 2 M-token subset) | 3.1084 | 3.1225 | $|Δ_{seed}|$ = 0.0141 |
| wall-clock | 14.00 h | 14.00 h | 50.3 k tok/s both |

Seed 0 leads seed 1 along the whole trajectory: over the last 20 paired subset evals the seed-0 −
seed-1 difference is **−0.0076 ± 0.0006 nats** (mean ± sd), a stable offset, not eval noise, as for
the L0 pair (+0.0023 ± 0.0006). The spread is a property of the seeds, and it is larger for the
271 M MoE than for the 100 M dense model — plausibly the router's early routing decisions, which
are seed-dependent and persist (the two runs' `load_min` differ at the end, 0.116 vs 0.113).

### Against the dense baseline, pair against pair

| | val loss (pair mean) | Δ vs $L_{ref}$(small) | params (total / non-emb) |
|---|---|---|---|
| L0 pair (`116`/`117`), $L_{ref}$(small) | 2.9958 | — | 100.1 M / 75.5 M |
| **L1 pair (`130`/`119`)** | **2.8839** | **−0.1119 nats, −3.74 %** | 271.2 M / 246.6 M |
| L1 vs L0 at `screen` (`106`/`100`), one seed | 3.2048 vs 3.3358 | −0.1310 nats, −3.93 % | 1 B tokens |

Matched FLOPs per token (top-2 of 8). The gap is 8.4× the new 4σ band and 34× σ; it is a settled
number at this scale. It narrows slightly from 1 B to 2.5 B tokens (−3.93 → −3.74 %), the opposite
direction from what `130` alone suggested, and inside what the two seeds can resolve — a
statement about direction, not a banded result. **Verdict rule:** no threshold names L1 against L0,
so no *supports*/*weakens* is recorded; the resolvable statement is *"a fine-grained MoE beats
matched-FLOPs dense by 3.7 % at `small`, two seeds each"*.

## σ re-pooled over the L0 and L1 pairs

ADR-013: σ is the pooled seed spread, measured at `small`; `117/results.md` fixed it from the L0 pair
alone (0.0018) and committed to re-pooling over every later pair. Pooled sample sd over two pairs of
two, $\sqrt{(s_{L0}^2 + s_{L1}^2)/2}$ with $s = |Δ_{seed}|/\sqrt{2}$:

| pair | $|Δ_{seed}|$ | $s$ |
|---|---|---|
| L0 (`116`/`117`) | 0.0026 | 0.00184 |
| L1 (`130`/`119`) | 0.0061 | 0.00431 |
| **pooled σ** | | **0.0033 nats** |

Still below ADR-013's 0.01 third-seed trigger, so **the `small` programme stays at two seeds**. The
bands move (fractions of $L_{ref}$ = 2.9958 unchanged):

| threshold | was (σ = 0.0018) | **now (σ = 0.0033)** | verdict rule (ADR-013) |
|---|---|---|---|
| 2σ ("matches", "no detectable regression") | 0.0037 | **0.0066** | supports if $|Δ| \le$ 2σ |
| 4σ (`screen` silent band) | 0.0073 | **0.0133** | a one-seed `screen` run is silent unless $|Δ|$ > 4σ |
| 1 % $L_{ref}$ | 0.0300 | 0.0300 | supports if $Δ \le$ 0.030; weakens if $Δ$ > 0.030 + 2σ = 0.0366 |
| 2 % $L_{ref}$ | 0.0599 | 0.0599 | supports if $Δ \le$ 0.060; weakens if $Δ$ > 0.0665 |

The 2026-09-16 `screen` re-scoring is recomputed against these bands in `docs/04` (2026-09-17); no
verdict flips, but the margins on `112` (L5-ne128, H6) and `111` (L5d, H15a) shrink to 1.4× and
1.1× the silent band. σ is re-pooled again when the L2 pair (`120`/`121`) lands.

## Thermals

Second consecutive 14 h load at the 271 M rungs' draw: zero thermal-slowdown rows on any card, GPU 0
(`GPU-4673cc7d`) at 75–78 °C and ≥ 1890 MHz, step time flat to 0.2 % between median and p90.
Throughput 50.3 k tok/s, identical to `130` to 0.1 %.

## Interpretation

Silent on H1–H19 as a test; decisive as a calibration in two ways. The L1 pair at `small` is now the
fine-grained MoE comparator every later `small` step is scored against (mean 2.8839, gap to dense
−0.1119 nats), and σ is re-pooled to 0.0033 nats — the L0 pair alone under-stated the seed spread
of the MoE rungs by 2.3×, and every remaining band widens accordingly. The `small` programme keeps
two seeds. L2's H2 clause (`120`/`121`) is scored against 2.8839 with $T$ = 0.030 and the 2σ margin
at 0.0066.
