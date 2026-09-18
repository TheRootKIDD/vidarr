# 120-l2-small-s0 — L2 at `small`, seed 0: 2.8878 nats, +0.0039 vs the L1 pair; the `screen` advantage of the parallel form does not survive 2.5 B tokens

`python -m scripts.train.train --rung L2 --id 120-l2-small-s0 --tokens 2.5e+09 --micro-batch 2 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (32 × 2 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Third run of `queue_small_2.sh`. L1 with `block_form: parallel` (attention and the MoE FF read the same input and their outputs are summed), otherwise identical to `130`.

**Bears on:** **H2**, quality clause — *parallel attention + MoE + memory costs ≤ 1 % loss at matched
FLOPs*. Silent on H2's critical-path clause (Tier S, `sim/` S1). As `107` noted, L2 applies the
parallel form to the *layered* stack, so it tests the form, not yet the shared-block setting H2 is
finally about. One seed of the L2 pair; the banded verdict is provisional until `121`.

## Numbers

| | |
|---|---|
| params | 271.1 M total, 246.6 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| final val loss (full held-out, 5606 windows) | **2.8878** nats |
| final train loss (last step, main head) | 2.8935 |
| throughput (median step) | 50,890 tokens/s aggregate (`step_s` median 10.302 s, p90 10.316, p90/median 1.001) — 1.2 % above L1 |
| peak memory per GPU | 8.58 GiB |
| wall-clock (this session) | 13.83 h |
| seed | 0 |
| routing at the last step | `load_max` 0.141 / `load_min` 0.110 (ideal 0.125), `route_ent_mean` 0.9997, `route_eff_min` 7.98 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 79 °C, `sm_mhz_min` ≥ 1897; `n_power_cap` 43 rows (GPU 0 at its 170 W cap) |

Wall-clock accounting: summed `step_s` 13.650 h + evals 0.147 h + checkpoints 0.020 h = 13.817 h of
13.83 h, **0.07 % unaccounted**.

## Against L1, the sequential form

| | val loss | Δ vs L1 |
|---|---|---|
| L1 pair (`130`/`119`), the comparator | 2.8839 (seeds 2.8808 / 2.8869) | — |
| **`120` L2, seed 0** | **2.8878** | **+0.0039 nats, +0.13 %** vs the pair; +0.0070 vs L1 seed 0 |
| `107` L2 vs `106` L1 at `screen`, one seed | 3.1495 vs 3.2048 | −0.0553 nats, −1.7 % |
| L0 pair, for scale | 2.9958 | L2 is −0.1080 (−3.6 %) below dense |

**Verdict rule (ADR-013) with σ = 0.0033 (re-pooled, `119`):** $T$ = 1 % of $L_{ref}$ = 0.030 nats;
*supports* if $Δ \le$ 0.030, *weakens* if $Δ$ > 0.0366. Δ = +0.0039 → **supports H2's quality clause**,
provisionally at one seed: the parallel form costs 0.13 %, 7.7× under the 1 % allowance, and inside
the 2σ = 0.0066 band, i.e. *no detectable regression* against the pair. Against L1 seed 0 alone
(+0.0070) it sits just outside 2σ; the pair mean is the comparator ADR-013 names, and `121` settles it.

**What changed from `screen`.** At 1 B tokens L2 *beat* L1 by 1.7 % (`107`, 7.6× the old silent
band, "supports (improves)"). At 2.5 B tokens the advantage is gone: the L2 − L1 (seed 0) difference
over the last 20 paired subset evals is **+0.0054 ± 0.0017 nats**, a small, stable deficit along the
whole tail of training, and at this schedule's 1 B-token eval (step 1900) L2 is at 3.1019 against
L1's 3.1084, −0.0065 — a much smaller lead than `screen` showed at the same token count, so the
`screen` gap was mostly the shorter schedule (the cosine at 1 B ends at a learning rate where the
parallel form's faster early progress still shows), not a property of the form. The H2 claim is
*non-inferiority*, and non-inferiority is what `small` shows; the `screen` "improves" reading was a
budget artefact and should not be quoted.

## Thermals

Third consecutive ≈ 14 h load: zero thermal-slowdown rows, GPU 0 (`GPU-4673cc7d`) at 75–79 °C and
≥ 1897 MHz, step time flat to 0.1 %. The parallel form is 1.2 % faster per step than L1 at the same
memory minus 0.14 GiB, consistent with `018` (570 vs 565 ms is within noise there; here the
accumulation-amortised rate favours it slightly).

## Interpretation

**Supports H2 (quality clause), one seed, provisional.** The parallel form is within 0.13 % of the
sequential MoE at 2.5 B tokens, inside ADR-013's 1 % allowance and inside the 2σ no-regression
band against the L1 pair. It does *not* improve on the sequential form at this budget; `screen`'s
−1.7 % was a 1 B-token schedule effect. The pair verdict and the re-pooled σ come with
`121-l2-small-s1` (≈ 14 h). The critical-path half of H2 remains for S1 in `sim/`.
