# 131-l3-small-s1 — L3 at `small`, seed 1: 2.9691 nats; **the L3 pair = 2.9682, +2.88 % vs the L2 pair, acceptance 42.5 / 25.6 / 17.6 % — H13 weakened on both clauses at two seeds**; σ re-pooled over four pairs to 0.0031

`python -m scripts.train.train --rung L3 --id 131-l3-small-s1 --tokens 2.5e+09 --micro-batch 2 --seed 1` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (32 × 2 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. First run of `queue_small_3.sh`; the rerun of the burnt `123`. L2 plus four independent MTP heads (ADR-012; `mtp.subsample` at the default, as `108`/`122`), 272.9 M params vs L2's 271.1 M. **First ladder run on driver 615.71.09** (`029`/`030`); `122`, its pair partner, ran on 610.57.04.

**Bears on:** **H13**, both clauses — *greedy acceptance of heads 2/3/4 against the main head ≥ 70 / 55 /
45 % on held-out, with main-head $Δ \le 2σ$ against L2*. Completes the L3 pair with `122-l3-small-s0`,
which is the two-seed test ADR-030 deferred H13's verdict to. **Verdict: weakens, both clauses, two
seeds each side.**

## Numbers

| | |
|---|---|
| params | 272.9 M total, 248.3 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| **final val loss, main head** (full held-out, 5606 windows) | **2.9691** nats |
| final train loss (last step): main head / total incl. MTP aux | 2.9499 / 7.3904 (MTP aux 4.4283) |
| throughput (median step) | 42,353 tokens/s aggregate (`step_s` median 12.379 s, p90 12.394, p90/median 1.001) — `122`: 42,320, +0.08 % |
| peak memory per GPU | 8.83 GiB |
| wall-clock (this session) | 16.63 h |
| seed | 1 |
| routing at the last step | `load_max` 0.140 / `load_min` 0.110, `route_ent_mean` 0.9995, `route_eff_min` 7.98 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 77 °C, `sm_mhz_min` ≥ 1897; `n_power_cap` 35 rows |

Wall-clock accounting: summed `step_s` 16.401 h + evals 0.198 h + checkpoints 0.024 h = 16.623 h of
16.633 h, **0.06 % unaccounted**.

`n_throttled` reads 4.0 on all 476 rows. That is the 615.71 driver's new `0x400` "Reliability"
clocks-event bit on every loaded card (`030`), not throttling: the step time is 0.08 % *faster* than
`122`'s and the clock floor is the same 1897 MHz. The field to read is `n_thermal`.

## The L3 pair

| | seed 0 (`122`, driver 610.57) | seed 1 (`131`, driver 615.71) | **pair** |
|---|---|---|---|
| main head, full held-out | 2.9674 | 2.9691 | **2.9682** |
| acceptance vs main head, heads 2 / 3 / 4 (held-out) | 42.5 / 25.5 / 17.4 % | 42.5 / 25.8 / 17.8 % | **42.5 / 25.6 / 17.6 %** |
| head top-1 vs ground truth, heads 2 / 3 / 4 (held-out) | 24.5 / 14.9 / 10.1 % | 24.5 / 14.8 / 10.1 % | 24.5 / 14.9 / 10.1 % |

$|Δ_{seed}|$ = **0.0018 nats** ($s$ = 0.00125), the tightest `small` pair so far (L0 0.0026, L1 0.0061,
L2 0.0054); along the trajectory seed 0 − seed 1 over the last 20 paired evals is −0.0019 ± 0.0004
(sd), a stable offset. **The driver change is invisible in the loss**: a seed gap a third the size of
the L1/L2 pairs' leaves no room for a driver effect larger than seed noise.

**Correction to `122/results.md` (append-only, so recorded here).** Its acceptance row, 41.9 / 25.2 /
16.8 %, and top-1 row, 23.6 / 14.3 / 9.4 %, are the *last training batch's* `mtp_accept_h*` /
`mtp_top1_h*`, mislabelled as full held-out. `122`'s held-out figures (`val_mtp_*` at step 4768,
5606 windows) are the ones in the table above: 42.5 / 25.5 / 17.4 % and 24.5 / 14.9 / 10.1 %. The
difference is ≤ 0.6 points and changes no reading; the `screen` → `small` gain in acceptance is ≈ 3
points on head 2 (39.7 → 42.5), not 2.

## H13 at two seeds

| clause | H13 requires | `screen` (`113`, 1 B) | **`small` pair (`122`+`131`, 2.5 B)** |
|---|---|---|---|
| acceptance vs main head, head 2 | ≥ 70 % | 39.7 % | **42.5 %** |
| acceptance vs main head, head 3 | ≥ 55 % | 23.0 % | **25.6 %** |
| acceptance vs main head, head 4 | ≥ 45 % | 16.1 % | **17.6 %** |
| main-head regression vs L2 | $Δ \le 2σ$ | +0.0847 (+2.69 %, vs `107`) | **+0.0831 nats, +2.88 %**, pair against pair (2.9682 vs 2.8851) |

**Verdict rule (ADR-013).** With σ = 0.0035 as it stood when the run was queued: *supports* if
$Δ \le$ 0.0070, *weakens* if $Δ$ > 0.0140; Δ = +0.0831 → **weakens**, 5.9× the weakening line, 11.9×
the 2σ margin. With σ re-pooled below (0.0031): lines 0.0062 / 0.0123 → 6.7× and 13.5×. Acceptance:
head 2 at 42.5 % against 70 %, heads 3 and 4 at under half their targets → **weakens**. The two seeds
agree to 0.3 points on every head and 0.002 nats on the main head, so this is a property of the
recipe, not of a seed. **H13 is weakened at `small` on both clauses, two seeds each side; ADR-030's
deferral is fully discharged.**

Against the dense baseline the pair sits at **−0.0276 nats, −0.92 %** vs $L_{ref}$(small) = 2.9958,
where the L2 pair has −3.70 %: four MTP heads hand back three quarters of the MoE's gain over dense,
for 17 % more wall-clock per step.

## σ re-pooled over four pairs

Pair sample sds: L0 0.00184, L1 0.00431, L2 0.00382, **L3 0.00125** → rms = **0.0031 nats** (was
0.0035 over three). Under ADR-013's 0.01 third-seed trigger; two seeds stand. Bands at `small`:
2σ = **0.0062**, 4σ = **0.0123**, 1 % = 0.030 (weakens above 0.0362), 2 % = 0.060 (weakens above
0.0662). Re-scoring with the narrower bands: **no verdict flips.** H2 (L2 − L1 = +0.0012) stays
inside 2σ; at `screen`, `112` (H6, +0.0184) *weakens* by 0.0061 over the 0.0123 line instead of
0.0044, `111` (H15a, +0.0149) moves from 1.07× to 1.2× the silent band and still *supports*, `110`
(−0.0030) and `115` (+0.0042) stay silent, `114` (+0.0239) stays resolvable at +0.72 %.

## Interpretation

**Weakens H13 on both clauses at `small`, two seeds** (Δ_main = +0.0831 nats pair-against-pair
against $T$ = 2σ; head-2 acceptance 42.5 % against 70 %). More tokens are not the fix — 1 B → 2.5 B
moved head-2 acceptance 3 points and the main-head penalty not at all — so the only route left to
H13 at this scale is I20's recovery variant L3b (forward MTP curriculum), which needs its ADR, and
the four-head configuration should not be carried into L5 onward without it (as `122` concluded). The
run also serves as the first long load on driver 615.71: 16.6 h, zero `n_thermal`, GPU 0 ≤ 77 °C,
throughput within 0.1 % of `122`. `132-l5-small-s0` started 04:07.
