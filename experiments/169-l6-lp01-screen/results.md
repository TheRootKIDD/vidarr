# 169-l6-lp01-screen — L6 at $\lambda_p$ = 0.01 (ADR-045) at `screen`: 3.3685 nats at $\bar r$ = 3.81 → **below the curve's lowest point ($r$ = 4), so ADR-045 §3 fires and the verdict waits for `171` (L5d $r$ = 2)**; meets H7's numbers (−52 % middle-block FLOPs at +0.72 %), but is +0.0084 behind fixed $r$ = 4 with 0.19 fewer iterations

`python -m scripts.train.train --rung L6 --id 169-l6-lp01-screen --tokens 1e+09 --micro-batch 4 --seed 0 --set halting.ponder_cost=0.01` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (identical to `165` except `halting.ponder_cost: 0.01`; `halting.mode: act`, `gate: true`, $r_{min}$ 2, $r_{max}$ 8, `router.segment_aux: mean`), per-step log in `log.jsonl`. Code at the `queue_screen_13` gate commit `76d1965`.

**Bears on:** **H7** (learned $r_t$ saves ≥ 30 % middle-block FLOPs at ≤ 1 % loss), scored under
ADR-041 §4 against the I25 control and under ADR-013's own H7 clause.
- **Against the fixed-$r$ curve** (`167` / `168` / `162` = 3.3601 / 3.3492 / 3.3443 at $r$ = 4 / 6 /
  8): held-out $\bar r$ = 3.815 lies **below $r$ = 4, outside the measured curve.** ADR-045 §3
  fires: `171` L5d $r$ = 2 runs after `170`, and the control at 3.815 is interpolated on the 2–4
  segment once it exists. **No verdict until then; nothing is extrapolated.**
- **What `171` will decide.** The control at 3.815 is 3.3601 + 0.0926 × (L₁₇₁ − 3.3601), with
  (4 − 3.815) / 2 = 0.0926. Against `169`'s 3.3685, the bands fall at:
  - matches a fixed depth (|Δ| ≤ 2σ = 0.0059) if 3.3871 ≤ L₁₇₁ ≤ 3.5145;
  - above the curve by > 2σ (worse than a fixed depth) if L₁₇₁ < 3.3871;
  - adaptive depth works (> 4σ = 0.0119 below) only if L₁₇₁ > 3.5793, i.e. if $r$ = 2 costs
    > 0.22 nats against $r$ = 4. `167` → `162` is 0.0158 for four more iterations, so that would take
    a very steep bottom of the curve.
- **What is measured without `171`:** `169` is **+0.0084 (2.8σ, 0.7× 4σ) behind `167`**, which ran
  every token at $r$ = 4, while using 0.185 fewer iterations per token on average.
- **H7's own clause** (ADR-013): $\bar r$ = 3.815 ≤ 5.6 ✓ (−52.3 % middle-block FLOPs), and Δ vs
  `162` = +0.0242 (+0.72 %) ≤ 0.01 $L_{ref}$ = 0.0334 ✓. Both numbers are met. |Δ| = 2.0× 4σ, but
  as with `166`, I25 is why that is not yet credit to adaptive depth: `167` meets the same numbers at
  +0.47 %.

σ = 0.0030 over twelve pairs (`161`; 2σ 0.0059, 4σ 0.0119). $L_{ref}$(`screen`) = 3.336 (`100`).

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (= `165`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3685** nats |
| final train loss (last step, main head) | 3.3362 |
| held-out mean depth `val_r_mean` | **3.815** (train, last step: 3.798) |
| middle-block FLOPs vs `162` | $\bar r / 8$ = 47.7 % (**−52.3 %**). Notional: halted tokens are masked, not skipped (see throughput) |
| ponder loss (last step) | 0.0401 (`165` 0.0223 at $\lambda_p$ = 0.003) |
| router aux (last step) | 0.0081 (`162` 0.0076) |
| throughput (median step) | 34,915 tokens/s aggregate (`165` 33,135; `167` fixed $r$ = 4: 53,100) |
| peak memory per GPU | 3.93 GiB |
| wall-clock (this session) | 7.03 h (`165` 8.19 h, `167` 5.30 h) |
| seed | 0 |

## Scoring (ADR-041 §4, ADR-045 §3)

| | $\bar r$ | val loss | control at $\bar r$ | Δ vs control | reading |
|---|---|---|---|---|---|
| `171` L5d fixed | 2 | *pending* | — | | curve point (ADR-045 §3) |
| `167` L5d fixed | 4 | 3.3601 | — | | curve point |
| `168` L5d fixed | 6 | 3.3492 | — | | curve point |
| `162` L5d fixed | 8 | 3.3443 | — | | curve point, $r_{max}$ |
| **`169` L6, $\lambda_p$ 0.01** | **3.81** | **3.3685** | on the 2–4 segment, *pending `171`* | *pending* | **open** |
| `165` L6, $\lambda_p$ 0.003 | 7.38 | 3.3452 | 3.3458 | −0.0006 (0.2σ) | matches a fixed depth |
| `166` L6c | 5.52 | 3.3541 | 3.3518 | +0.0023 (0.8σ) | matches a fixed depth |

## Depth over training

| step | 10 | 50 | 100 | 300 | 500 | 700 | 1000 | 1500 | 1907 |
|---|---|---|---|---|---|---|---|---|---|
| train $\bar r$, `169` | 7.00 | 2.04 | 2.00 | 2.00 | 2.02 | 2.64 | 3.29 | 3.68 | 3.80 |
| train $\bar r$, `165` ($\lambda_p$ 0.003) | 7.00 | 2.32 | 2.09 | 2.19 | 3.84 | 5.32 (val) | 6.61 | 7.32 | 7.35 |
| val loss, `169` | | | 5.2665 | 4.2741 | 3.8756 | 3.6877 | 3.5300 | 3.4068 | 3.3685 |
| val loss, `165` | | | 5.2668 | 4.2701 | 3.8722 | 3.6708 | 3.5149 | 3.3854 | 3.3452 |
| val loss, `167` ($r$ = 4) | | | 5.3543 | 4.2735 | 3.8532 | 3.6681 | 3.5186 | 3.3985 | 3.3601 |

Held-out $\bar r$ by eval: 2.00 through step 400, 2.02 at 500, 2.18 at 600, then 2.67 / 3.02 / 3.27 /
3.54 / 3.67 / 3.78 / 3.84 at 700 / 800 / 1000 / 1200 / 1500 / 1700 / 1900, so it was still rising
slowly at the end of the budget.

The run follows `165`'s path with the climb delayed and flattened. The policy falls to $r_{min}$ = 2
within 50 steps, as at $\lambda_p$ = 0.003, but sits there almost exactly (2.00) for ≈ 550 steps
instead of ≈ 300. The climb starts at step ≈ 550 and is slower: +1.8 iterations over 1350 steps,
against `165`'s +5.2 over ≈ 1600. Until step ≈ 300 the two L6 runs have the same loss, since both are at
$r_{min}$. **From step 500 on, `169` trails `167` at every eval, by 0.022 at step 500 and 0.0084 at the
end, even while its mean depth is below 4.** The gap is closing as `169` deepens. A policy that held
most tokens at $r$ = 2 for half the run pays for it, and at 1 B tokens it has not caught up.

## Routing

Worst point 5.17 of 8 effective experts at step 21, up to two dead experts, none after step 22,
**7.99 of 8 at the end with loads 0.115–0.130**. Same endpoint as `162` and `165`.

## Run health

| | |
|---|---|
| wall-clock | 7.03 h; summed `step_s` 6.947 h + evals 0.075 h → **0.1 % unaccounted** |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 74 °C |
| context | first run after the `queue_screen_13` gate (ruff + unit tests passed 18:19, `76d1965`) |

**Throughput correction.** The ADR-045 session log said masked halting does not shorten the step. It
does, partly: `169` ran at 9.8 s per step while near $r$ = 2 and 15.8 s at the end ($\bar r$ ≈ 3.8),
against 15.5 s for `165` near $\bar r$ 7.4 and ≈ 9.9 s for `167` at fixed $r$ = 4. At equal mean depth,
masked ACT is still much slower than a fixed schedule (34.9 k against 53.1 k tok/s at $\bar r$ ≈ 4).
The likely reason is that a micro-batch keeps iterating while any token in it is still running.
That was not profiled.

## Interpretation

At $\lambda_p$ = 0.01, L6's gated ACT stays collapsed at $r_{min}$ for over a quarter of the run, then
climbs to $\bar r$ = 3.81 on held-out. It lands at 3.3685 nats, meets both of H7's stated numbers
(−52 % middle-block FLOPs at +0.72 % vs `162`), and is 0.0084 nats (2.8σ) behind fixed $r$ = 4 with
0.19 fewer iterations. Its mean depth is below the measured curve, so per ADR-045 §3 the I25 control
needs `171` (L5d $r$ = 2). **H7: verdict open until `171`.** Matching a fixed depth needs 3.3871 ≤ L₁₇₁
≤ 3.5145; beating it by 4σ needs L₁₇₁ > 3.5793. Raising the ponder price moved L6 down the depth axis (7.38 → 3.81). Whether it also moved it off
the fixed-depth curve is what `171` decides.
`170` ($\lambda_p$ = 0.03) is running.
