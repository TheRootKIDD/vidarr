# 159-l5d-8k-screen — L5d trained at context 8192: 3.3611 nats, **+0.0252 vs L5-8k (+0.76 %)** against H15a's 2 % allowance → **H15a's 8 k end supported at one seed** (2.2× the 4σ line, so the one-seed rule allows a reading); the cost levels off at ≈ +0.029 from 2 k to 8 k; I30 closes; I33 opened (router aux summed over segments)

`python -m scripts.train.train --rung L5d --id 159-l5d-8k-screen --tokens 1e+09 --micro-batch 1 --seed 0 --set context=8192 experts.d_ff=896` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 1 × 8192 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Second run of `queue_screen_10.sh` (2026-10-19 10:02 → 2026-10-20 11:25), on the tree at `ae08987` (docs-only on top of the gate commit `36257dc`). ADR-037 with its amendment: `111-l5d-screen` with context 8192 and $d_{ff}$ pinned at 896. Global attention reads the final-vector cache $\mathcal{G}$ (up to 7680 keys per 512-token segment), and per-iteration attention stays local ($W$ = 512). Everything else is `158`'s.

**Bears on:** **H15a** at 8 k: *final-vector global attention costs ≤ 2 % loss vs per-iteration
global attention*. ADR-037 §3 reads it as $\Delta$ = L5d-8k − L5-8k (`158`) against $T$ = 0.02 ×
3.3359 = 0.0667, at one seed, with the per-position split beside it. ADR-013 sets the 8 k test "after
the context-extension fine-tune"; ADR-037 replaced that with training at 8 k (I30), which is what ran.
σ = 0.0029 over eleven pairs (`157`; 2σ 0.0057, 4σ 0.0114). At one seed the rule is silent unless
|Δ| > 4σ.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.91 M non-embedding (as `111`) |
| tokens | 1.000 B in 1907 steps |
| **final val loss** (full held-out, 1401 windows of 8192) | **3.3611** nats (`eval_context` after training: 3.3610) |
| final train loss (last step): main / total | 3.3185 / 3.4465 (router aux **0.1280**; see I33 below) |
| throughput (median step) | 10,992 tokens/s aggregate (`step_s` median 47.70 s, p90/median 1.005); `031` measured ≈ 11.7 k; `111` at 2 k: 33.8 k |
| peak memory per GPU | 3.97 GiB (as `031`) |
| wall-clock (this session) | **25.38 h**, 1.4 h over a node-day (`031` projected ≈ 24 h). Allowed: this run is the `screen` |
| seed | 0 |
| routing at the last step | `load_max` 0.130 / `load_min` 0.118, `route_ent_mean` 0.9998, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 190 telemetry rows**; `temp_c_max` ≤ 68 °C, `sm_mhz_min` ≥ 1912; `n_power_cap` 12 rows |

Wall-clock accounting: summed `step_s` 25.258 h + evals 0.103 h + checkpoints 0.012 h = 25.373 h of
25.377 h, **0.02 % unaccounted**.

## H15a at 8 k

| | L5-8k (`158`) | **L5d-8k (`159`)** | Δ (nats) | Δ % | ADR-013 |
|---|---|---|---|---|---|
| full held-out, 8 k windows | 3.3359 | **3.3611** | **+0.0252** | **+0.76 %** | **supports: Δ ≤ T = 0.0667; \|Δ\| = 2.2× 4σ, so readable at one seed** |

Along the trajectory (244-window evals), L5d − L5 is +0.056 at step 500, +0.031 at 1000, +0.027 at
1500 and +0.027 at 1900. The gap narrows and then levels.

**By position** (`eval_context`, full held-out):

| positions | 0–512 | 512–1 k | 1–2 k | 2–4 k | 4–8 k |
|---|---|---|---|---|---|
| L5-8k (`158`) | 3.4827 | 3.3370 | 3.3272 | 3.3264 | 3.3240 |
| L5d-8k (`159`) | 3.4657 | 3.3573 | 3.3538 | 3.3549 | 3.3531 |
| **L5d − L5** | **−0.017 (−0.49 %)** | **+0.020 (+0.61 %)** | **+0.027 (+0.80 %)** | **+0.029 (+0.86 %)** | **+0.029 (+0.87 %)** |
| 2 k-trained `small` pairs, 8 k eval (2026-10-10) | −0.28 % | +0.62 % | +0.83 % | — | — |

**The open question from 2026-10-10 is answered: the cost does not keep growing past 2 k.** Inside
2 k it grows with position, as in the 2 k-trained pairs and to within 0.2 points of the same
percentages. From 2 k to 8 k it is flat at +0.029. In the first segment, L5d is *better*, as before.
There, $\mathcal{G}$ is empty, so both models attend locally, and L5d's segment-recurrent training
gives the first 512 positions a cleaner signal. That reading is from the design, not tested.

**Across contexts:** the L5d cost is +0.45 % at 2 k (`111`), +0.48 % for the 2 k `small` pair, and
+0.76 % at 8 k. The 8 k average is higher because 75 % of its positions sit in the plateau beyond
2 k, not because any position costs more.

## I33: L5d's router aux loss is summed over segments

L5d's total loss carries router aux **0.1280** against L5-8k's 0.0077, 16.6× as much, with balanced
loads in both. At 2 k, `111` carried 0.0322 against `110`'s 0.0079, 4.1×. The ratios are the segment
counts (8192 / 512 = 16, 2048 / 512 = 4). The reason is in `model/model.py`: the segment loop of the
shared-block forward adds each segment's router aux to the total (`aux = aux + a`) instead of
averaging it. So **L5d trains with an effective load-balancing weight 4× L5's at 2 k and 16× at
8 k.** That is a confound in every L5d comparison, including `111`, `134`/`135` and this pair. It is
not a typo-sized one at 8 k.

Direction: a stronger balancing term usually costs the main loss a little, so the confound most
likely *overstates* L5d's cost, and H15a's *supports* survives it. That is an expectation, not a
measurement. Per the spec-first rule, nothing is changed here. **I33** asks for the fix (averaging
over segments) through an ADR, and for one control to size the effect. The control is L5d at 2 k with
the aux averaged, ≈ 8.3 h at `screen`, against `111`.

## Interpretation

**Supported at one seed: at a trained 8 k context, final-vector global attention costs 0.0252 nats
(+0.76 %) against per-iteration global attention.** That is well inside H15a's 2 % allowance (T =
0.0667) and 2.2× the 4σ line, so the one-seed rule allows the reading. **H15a: supported at 8 k** (one
seed, `screen`), with the 2 k end already supported at two seeds (`134`/`135`). H15a's "2 k–8 k" range
is now measured end to end, and **I30 closes**. The per-position split adds the more useful fact: the
cost saturates at ≈ +0.029 nats (+0.87 %) beyond 2 k, so nothing here predicts a growing penalty at
longer contexts. That will matter for H15b's 8–32 k evals. Two caveats travel with the verdict. First,
**I33**: L5d trained with 16× L5's effective balancing weight, likely making its cost conservative.
Second, the time cost: in this reference implementation, an L5d step is 3.5× an L5 step at 8 k
(10.99 k vs 38.1 k tok/s). That is the explicit-mask attention over $\mathcal{G}$ off the SDPA fast path
(`031`), not a property of the design, and it bears on H15a's quality clause not at all. Rig: 25.4 h,
zero `n_thermal`, ≤ 68 °C.
