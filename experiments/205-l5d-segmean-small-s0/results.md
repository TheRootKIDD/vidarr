# 205-l5d-segmean-small-s0 — L5d with `segment_aux: mean` at `small`, seed 0 (ADR-077 §1, the new L5d-family comparator): 3.0906 nats, **+0.0011 vs `134`** (the same seed with the summed aux, 0.4σ): per-seed and provisional, I33's confound looks negligible at `small` too; no hypothesis verdict until `206`

`python -m scripts.train.train --rung L5d --id 205-l5d-segmean-small-s0 --tokens 2.5e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`162`'s config at 2.5 B tokens), per-step log in `log.jsonl`. Code at the `queue_small_13` gate commit `d277bdc`. Run 2026-11-27 06:01 – 2026-11-28 02:51.

**Bears on:** no hypothesis by itself. It is the first seed of the comparator pair
(**ADR-077 §1**) that the L5h pair `207`/`208` (H20, ADR-077 §3) and the L5g pair `209`/`210`
(H21, ADR-078 §3) are read against, and it measures **I33's size at `small`** (ADR-038: the
router's auxiliary loss averaged over segments in place of summed). **H15a: silent** — the old
pair's verdict is not rewritten (ADR-077 §1), and this seed gives no reason to.

**Per-seed and provisional.** Pair readings wait for `206` (seed 1, running).

| | this seed | reference | Δ | in σ |
|---|---|---|---|---|
| vs `134` (L5d, summed aux, seed 0) — I33 at `small` | 3.0906 | 3.0896 | **+0.0011** | 0.4 |
| vs the old L5d pair `134`/`135` | 3.0906 | 3.0896 | +0.0011 | 0.4 |
| vs `132` (L5, seed 0) | 3.0906 | 3.0778 | +0.0129 | 4.3 |
| vs the L5 pair `132`/`133` | 3.0906 | 3.0748 | +0.0159 | 5.4 |

σ = 0.0030 (2σ 0.0059, 4σ 0.0119; twelve `small` pairs), σ multiples on the unrounded 0.00296.
At `screen` the same change was −0.0034 (`162` − `111`, 1.1σ); at `small` on seed 0 it is
+0.0011. Both are inside 2σ and of opposite sign.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.90 M non-embedding (as `134`) |
| tokens | 2.500 B in 4768 steps |
| final val loss (full held-out, 5606 windows) | **3.0906** nats |
| final train loss (last step, main head) | 3.0930 |
| throughput (median step) | 33,650 tokens/s aggregate |
| peak memory per GPU | 3.89 GiB |
| wall-clock (this session) | 20.83 h (`134`: 20.76 h) |
| seed | 0 |
| $r$ | fixed 8 |
| routing at the end | entropy 0.9996 of $\ln 8$, 7.99 effective experts, 0 dead |

**Curve** (976-window evals; last row full held-out).

| step | `132` (L5) | `134` (L5d, summed) | `205` | `205` − `134` |
|---|---|---|---|---|
| 500 | 3.8561 | 3.8676 | 3.8598 | −0.0078 |
| 1000 | 3.4914 | 3.5010 | 3.5050 | +0.0040 |
| 1500 | 3.3481 | 3.3571 | 3.3593 | +0.0022 |
| 2000 | 3.2627 | 3.2739 | 3.2749 | +0.0010 |
| 2500 | 3.2008 | 3.2117 | 3.2138 | +0.0021 |
| 3000 | 3.1541 | 3.1657 | 3.1681 | +0.0024 |
| 3500 | 3.1187 | 3.1304 | 3.1330 | +0.0026 |
| 4000 | 3.0940 | 3.1061 | 3.1080 | +0.0019 |
| 4500 | 3.0782 | 3.0907 | 3.0924 | +0.0017 |
| 4768 (full) | 3.0778 | 3.0896 | 3.0906 | +0.0011 |

**Train loss** (main head, same seed and data order, step-matched means): `205` − `134` =
+0.0020 over steps 500–1500, +0.0023 over 1500–3000, +0.0016 over 3000–4768. Against L5
`132`: +0.008, +0.012, +0.013.

## Run health

| | |
|---|---|
| wall-clock | 20.83 h; summed `step_s` 20.631 h + evals 0.176 h + checkpoints 0.010 h → **0.05 % unaccounted** |
| throughput | 33.6 k tok/s median, flat (first and last 300 steps agree to 0.1 %) |
| thermal | **0 of 476 records with `n_thermal` > 0**, max 73 °C, min clock 1897 MHz |
| pause | the queue script was SIGSTOPped 16:54–17:02 on 2026-11-27 (a pause requested and cancelled by the user); the trainer was never stopped and its step times show no gap |

## Interpretation

Seed 0 of the L5d-segmean comparator ends at 3.0906 nats, 0.0011 above `134`, the same seed
trained with the router's auxiliary loss summed over segments. That is 0.4σ, a step-matched
train-loss difference of +0.002 throughout, and opposite in sign to the −0.0034 seen at
`screen`, so on this seed I33's confound is not measurable at `small` either. Against L5 the
segment-recurrent form costs +0.0129 on the same seed, as the old pair's +0.0148 on pair means
did. No band of ADR-013 is applied: this run is a comparator, and ADR-077 §3 and ADR-078 §3
read pair means, which need `206`. If `206` lands as close to `135` as this does to `134`, the
new pair will sit within 0.001–0.002 of the old 3.0896 and the choice of comparator will not
move any L5d-family verdict.
