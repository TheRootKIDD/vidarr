# 162-l5d-segmean-screen — L5d with the router aux averaged over segments (ADR-038): 3.3443 nats, **−0.0034 vs `111` (1.1σ, inside the 2σ "negligible" band)** → I33's confound is negligible and every L5d verdict stands as written; I33 closes

`python -m scripts.train.train --rung L5d --id 162-l5d-segmean-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`router.segment_aux: mean`), per-step log in `log.jsonl`. Code at the `queue_screen_11` gate commit `22d1cdd` (merge of `l3c-i33`).

**Bears on:** **H15a** through **I33** and **ADR-038**. The only change against `111` is the router aux
reduction: `111` summed it over the four 512-token segments of a 2 k window, so it trained at 4× L5's
effective load-balancing weight; `162` averages it, as `01 §4.4` specifies. ADR-038 §3 reads the
pair: |`162` − `111`| ≤ 2σ → the confound is negligible, and every L5d verdict (`111`, `134`/`135`,
`159`) stands as written. **Result: negligible.** H15a's own test against `110` is **silent at one
seed** by the letter of ADR-013 (|Δ| = 0.97× 4σ), in the *supports* direction; H15a's 2 k verdict
is the `small` pair's (`134`/`135`, supported), and that is unaffected.

σ = 0.0030 over twelve pairs (`161`; 2σ 0.0059, 4σ 0.0119).

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (identical to `111`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3443** nats |
| final train loss (last step, main head) | 3.3104 |
| router aux (last step) | **0.0076** (`111` 0.0322, `110` 0.0079) |
| throughput (median step) | 33,913 tokens/s aggregate |
| peak memory per GPU | 3.89 GiB |
| wall-clock (this session) | 8.28 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

The logged aux confirms the fix: `162` sits at L5's value (0.0076 vs 0.0079) for the whole run, where
`111` sat at ≈ 4× (0.032).

## Against `111`, `110` and the ladder

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` (L0 dense) | 3.3360 | | |
| `110-l5-screen` (L5, global per-iteration attention) | 3.3328 | — | |
| `111-l5d-screen` (L5d, aux summed over segments) | 3.3477 | +0.0149 vs `110` (+0.45 %) | |
| **`162-l5d-segmean-screen` (L5d, aux averaged)** | **3.3443** | **−0.0034 vs `111`** (1.1σ) | **ADR-038: \|Δ\| ≤ 2σ = 0.0059 → confound negligible** |
| | | **+0.0115 vs `110` (+0.35 %)** | **H15a: Δ ≤ T = 0.02 × 3.3328 = 0.0667, but \|Δ\| = 0.97× 4σ → silent at one seed** |
| | | +0.0083 vs L0 (+0.25 %) | |

The direction is the one I33 predicted: the 4× balancing weight cost the main loss a little, so `111`
overstated L5d's cost. The size is 0.0034 nats, a quarter of `111`'s gap to `110`, and inside the
noise band. L5d's cost at 2 k, `screen`, becomes 0.35 % instead of 0.45 %, which is the same story.

**Curve.** The averaged aux helps most early and the advantage decays to noise:

| step | `110` (L5) | `111` (aux sum) | `162` (aux mean) | `162` − `111` |
|---|---|---|---|---|
| 100 | 5.6159 | 5.5506 | **5.5372** | −0.0135 |
| 300 | 4.3664 | 4.3252 | **4.3094** | −0.0158 |
| 500 | 3.8804 | 3.8741 | **3.8551** | −0.0189 |
| 700 | 3.6601 | 3.6708 | **3.6599** | −0.0109 |
| 1000 | 3.4995 | 3.5130 | 3.5084 | −0.0046 |
| 1500 | 3.3697 | 3.3865 | 3.3835 | −0.0031 |
| 1907 | 3.3328 | 3.3477 | 3.3443 | −0.0034 |

Strong balancing slows the router's early specialisation; once the loads settle (both runs end at
8.00 of 8 effective experts, loads 0.12–0.13), the extra weight has little left to push against. The
early gap is no guide to the final one, so nothing about the 8 k pair (`158`/`159`, 16× weight) can be
inferred from the 2 k gap beyond "small": ADR-038 §3 asks for a re-read of the 8 k pair only in the
> 4σ case, which this is not.

## Routing

Worst point 4.06 of 8 effective experts with one dead expert over steps 7–11 (`111`: 4.69 at step 19,
two dead at 12), **8.00 of 8 at the end with loads 0.121–0.130**, the same endpoint as `111` and `110`.
A lighter balancing weight lets the start be a little more lopsided and recovers just as fast.

## Run health

| | |
|---|---|
| wall-clock | 8.28 h; summed `step_s` 8.193 h + evals 0.082 h + checkpoints 0.004 h → **0.0 % unaccounted** |
| throughput | 33.9 k tok/s median (`111` 33.8 k) |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 73 °C, min clock 1897 MHz |

## Interpretation

Averaging the router aux over segments, as the spec always said, moves L5d's held-out loss by
−0.0034 nats at 1 B tokens and 2 k context: 1.1σ, inside the 2σ band ADR-038 set for "negligible".
The aux itself now matches L5's (0.0076 vs 0.0079), so the fix does what it should. The 4× balancing
weight slowed `111` early (up to 0.019 nats behind at step 500) but cost almost nothing by the end. **I33
closes: every L5d verdict stands as written** — H15a supported at 2 k on the `small` pair and at 8 k on
`158`/`159`, with the cost slightly overstated, not understated. This run is also the $r$ = 8 point of
ADR-041's L5d fixed-depth curve, which `167` ($r$ = 4) and `168` ($r$ = 6) complete for scoring L6 and
L6c.
