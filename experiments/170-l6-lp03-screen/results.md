# 170-l6-lp03-screen — L6 at $\lambda_p$ = 0.03 (ADR-045) at `screen`: 3.3987 nats at $\bar r$ = 2.06 → **the policy collapsed to $r_{min}$ and never left it; scored after `171`: +0.0174 vs the curve at 2.06 (5.8σ) — worse than a fixed depth**; fails H7's loss clause (+1.63 % vs `162`, over the 1 % bound)

`python -m scripts.train.train --rung L6 --id 170-l6-lp03-screen --tokens 1e+09 --micro-batch 4 --seed 0 --set halting.ponder_cost=0.03` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (identical to `165` / `169` except `halting.ponder_cost: 0.03`; `halting.mode: act`, `gate: true`, $r_{min}$ 2, $r_{max}$ 8, `router.segment_aux: mean`), per-step log in `log.jsonl`. Code at the `queue_screen_13` gate commit `76d1965`.

**Bears on:** **H7** (learned $r_t$ saves ≥ 30 % middle-block FLOPs at ≤ 1 % loss), scored under
ADR-041 §4 against the I25 control and under ADR-013's own H7 clause.
- **Against the fixed-$r$ curve** (`167` / `168` / `162` = 3.3601 / 3.3492 / 3.3443 at $r$ = 4 / 6 /
  8): held-out $\bar r$ = 2.060 lies **below $r$ = 4, outside the measured curve.** ADR-045 §3 fired
  at 06:17 and `171` L5d $r$ = 2 is running. **No verdict until it is done; nothing is extrapolated.**
- **Scored 2026-11-02 after `171` (3.3820 at $r$ = 2):** control at 2.060 = 3.3820 + 0.0301 ×
  (3.3601 − 3.3820) = **3.3814**, Δ = **+0.0174 (5.8σ, 1.46× 4σ)**. **Resolvably worse than running
  every token at a fixed $r$ ≈ 2.** H7: the learned policy at this price *weakens* H7's
  learned-depth arm at `screen`. The loss clause fails too.
- **What `171` will decide.** At $\bar r$ = 2.060 the control is 0.970 × L₁₇₁ + 0.030 × 3.3601, which
  is almost `171` itself. Against `170`'s 3.3987:
  - matches a fixed depth (|Δ| ≤ 2σ) if 3.3938 ≤ L₁₇₁ ≤ 3.4060;
  - worse than a fixed depth by > 2σ if L₁₇₁ < 3.3938;
  - adaptive depth works (> 4σ below) only if L₁₇₁ > 3.4122.
- **What is measured without `171`:** +0.0386 vs `167` ($r$ = 4; 3.2× 4σ), +0.0302 vs `169`.
- **H7's own clause** (ADR-013): $\bar r$ = 2.06 ≤ 5.6 ✓ (−74.2 % middle-block FLOPs), but Δ vs
  `162` = +0.0544 (**+1.63 %**) > 0.01 $L_{ref}$ = 0.0334 ✗. **The loss clause fails.** At this
  price L6 saves far more than H7 asks and pays more than H7 allows.

σ = 0.0030 over twelve pairs (`161`; 2σ 0.0059, 4σ 0.0119). $L_{ref}$(`screen`) = 3.336 (`100`).

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (= `165`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3987** nats |
| final train loss (last step, main head) | 3.3655 |
| held-out mean depth `val_r_mean` | **2.060** (train, last step: 2.057) |
| middle-block FLOPs vs `162` | $\bar r / 8$ = 25.8 % (**−74.2 %**). Notional: halted tokens are masked, not skipped |
| ponder loss (last step) | 0.0639 (`169` 0.0401, `165` 0.0223) |
| router aux (last step) | 0.0052 (`162` 0.0076) |
| throughput (median step) | 58,645 tokens/s aggregate (`169` 34,915; `167` fixed $r$ = 4: 53,100) |
| peak memory per GPU | 3.90 GiB |
| wall-clock (this session) | 4.93 h |
| seed | 0 |

## Scoring (ADR-041 §4, ADR-045 §3)

| | $\bar r$ | val loss | control at $\bar r$ | Δ vs control | reading |
|---|---|---|---|---|---|
| `171` L5d fixed | 2 | 3.3820 | — | | curve point (ADR-045 §3) |
| `167` L5d fixed | 4 | 3.3601 | — | | curve point |
| `168` L5d fixed | 6 | 3.3492 | — | | curve point |
| `162` L5d fixed | 8 | 3.3443 | — | | curve point, $r_{max}$ |
| **`170` L6, $\lambda_p$ 0.03** | **2.06** | **3.3987** | 3.3820 + 0.0301 × (3.3601 − 3.3820) = **3.3814** | **+0.0174 (5.8σ)** | **worse than a fixed depth** |
| `169` L6, $\lambda_p$ 0.01 | 3.81 | 3.3685 | 3.3621 | +0.0064 (2.1σ) | above the curve (2σ < Δ < 4σ) |
| `165` L6, $\lambda_p$ 0.003 | 7.38 | 3.3452 | 3.3458 | −0.0006 (0.2σ) | matches a fixed depth |
| `166` L6c | 5.52 | 3.3541 | 3.3518 | +0.0023 (0.8σ) | matches a fixed depth |

## Depth over training

| step | 10 | 50 | 100 | 300 | 500 | 700 | 1000 | 1500 | 1907 |
|---|---|---|---|---|---|---|---|---|---|
| train $\bar r$, `170` | 7.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.03 | 2.06 |
| train $\bar r$, `169` ($\lambda_p$ 0.01) | 7.00 | 2.04 | 2.00 | 2.00 | 2.02 | 2.64 | 3.29 | 3.68 | 3.80 |
| train $\bar r$, `165` ($\lambda_p$ 0.003) | 7.00 | 2.32 | 2.09 | 2.19 | 3.84 | 5.32 (val) | 6.61 | 7.32 | 7.35 |
| val loss, `170` | | | 5.2808 | 4.2803 | 3.8963 | 3.7057 | 3.5565 | 3.4355 | 3.3987 |
| val loss, `169` | | | 5.2665 | 4.2741 | 3.8756 | 3.6877 | 3.5300 | 3.4068 | 3.3685 |
| val loss, `167` ($r$ = 4) | | | 5.3543 | 4.2735 | 3.8532 | 3.6681 | 3.5186 | 3.3985 | 3.3601 |

Held-out $\bar r$ by eval: 2.000 through step 500, 2.001–2.002 to step 1000, then 2.004 / 2.011 /
2.027 / 2.040 / 2.061 at 1100 / 1300 / 1500 / 1700 / 1900. The policy falls to $r_{min}$ within 50
steps, like `165` and `169`, and essentially stays there. A slow drift upward starts at step
≈ 1100 and is still going at the end, but it is +0.06 iterations over 800 steps. At this price,
an iteration past the second has not been worth its ponder cost for any meaningful share of tokens
within 1 B tokens. **`170` is in effect an L5d $r$ = 2 run with a halting head**, which is why
`171` decides its reading so tightly.

## Routing

Worst point 5.02 of 8 effective experts at step 54, up to two dead experts, none after step 31
(the minimum came later, with all experts alive), **8.00 of 8 at the end with loads 0.118–0.130**.
Router aux ends lower than at any other depth (0.0052). With two iterations there are fewer
segments to balance.

## Run health

| | |
|---|---|
| wall-clock | 4.93 h; summed `step_s` 4.868 h + evals 0.059 h → **0.1 % unaccounted** |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 73 °C |
| context | second run after the `queue_screen_13` gate (`76d1965`), straight after `169` |

**Step time grows while depth barely moves.** It was 7.1 s at step 200, 9.0 s at 1000 and 11.7 s at
1907, while $\bar r$ only went from 2.00 to 2.06. That fits the explanation `169`'s results.md
offered: a micro-batch keeps iterating as long as *any* of its tokens is running, so a few deep
tokens cost the whole micro-batch's compute. The mean depth sets the notional FLOPs. The *maximum*
depth in each micro-batch sets the wall-clock. Not profiled; `171`'s step time at fixed $r$ = 2 will
show the no-halting floor.

## Interpretation

At $\lambda_p$ = 0.03 the ponder price is high enough that L6's gated ACT collapses to $r_{min}$ = 2 in
the first 50 steps and stays there. Held-out $\bar r$ ends at 2.06, the gate's gradient
(ADR-041 §1) moves it only slightly within 1 B tokens, and the loss is 3.3987 nats. That is +1.63 %
against `162`, so H7's loss clause fails while its FLOPs clause is met three times over (−74 %).
Its mean depth is below the measured curve, so per ADR-045 §3 the I25 control is `171`. With
$\bar r$ = 2.06 that control is essentially `171`'s own loss, 3.3814, and **Δ = +0.0174 (5.8σ):
L6 at this price is worse than a fixed $r$ = 2 at the same depth.** `170` trails `171` at every eval
from step 300 on, so the cost is the ACT machinery, not depth (`171`'s results.md). **H7 at
`screen`: weakened for L6 at $\lambda_p$ = 0.03 against the I25 control, and H7's own loss clause
fails.** Across the sweep, $\lambda_p$ = 0.003 / 0.01 / 0.03 lands L6 at $\bar r$ = 7.38 / 3.81 /
2.06 and at −0.0006 / +0.0064 / +0.0174 from the fixed-depth curve. The ponder price sets the depth
monotonically, and the further it pushes L6 down, the further above the curve L6 lands.
