# 165-l6-screen — L6 (ACT with the gated residual, ADR-041) at `screen`: 3.3452 nats at $\bar r$ = 7.38 → **−0.0006 vs the L5d fixed-$r$ curve at 7.38 (0.2σ): matches a fixed depth, so adaptivity bought nothing**; it saves 7.8 % of middle-block FLOPs, not H7's ≥ 30 %

`python -m scripts.train.train --rung L6 --id 165-l6-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`halting.mode: act`, `gate: true`, `ponder_cost: 0.003`, $r_{min}$ 2, $r_{max}$ 8, `router.segment_aux: mean`), per-step log in `log.jsonl`. Code at the `queue_screen_12` gate commit `5942ee1` (merge of `l6-adr041`).

**Bears on:** **H7** (learned $r_t$ saves ≥ 30 % middle-block FLOPs at ≤ 1 % loss), scored under
ADR-041 §4 against the I25 control and under ADR-013's own H7 clause ($\bar r \le 0.7\,r_{max}$ = 5.6
on held-out with $\Delta \le 0.01\,L_{ref}$).
- **Against the fixed-$r$ curve** (`167` / `168` / `162` = 3.3601 / 3.3492 / 3.3443 at $r$ = 4 / 6 /
  8), interpolated at $\bar r$ = 7.38: control **3.3458**, Δ **−0.0006** (0.2σ). Within 2σ, so **it
  matches a fixed depth: the adaptivity bought nothing.** "Adaptive depth works" needed > 4σ below.
- **H7's own clause:** $\bar r$ = 7.38 = 0.92 $r_{max}$, above the 5.6 bound, so the depth clause
  fails by measurement. The loss clause holds (+0.0009 vs `162`).
- **Verdict: H7 silent at `screen`** by ADR-013's letter (one seed, |Δ| < 4σ everywhere). L6 as
  configured gives H7 no support: it is barely shallower than $r_{max}$, and what it saves is what a
  fixed $r$ ≈ 7.4 would save.

σ = 0.0030 over twelve pairs (`161`; 2σ 0.0059, 4σ 0.0119). $L_{ref}$(`screen`) = 3.336 (`100`).

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (= L5d; the halting head is one $d \to 1$ linear layer) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3452** nats |
| final train loss (last step, main head) | 3.3133 |
| held-out mean depth `val_r_mean` | **7.380** (train, last step: 7.354) |
| middle-block FLOPs vs `162` | $\bar r / 8$ = 92.2 % (**−7.8 %**). Notional: the reference implementation computes halted tokens and masks them, so wall-clock does not fall |
| ponder loss (last step) | 0.0223 |
| router aux (last step) | 0.0081 (`162` 0.0076) |
| throughput (median step) | 33,135 tokens/s aggregate (`162` 33,913: −2.3 % for the halting head and masks) |
| peak memory per GPU | 3.93 GiB |
| wall-clock (this session) | 8.19 h |
| seed | 0 |

## Scoring (ADR-041 §4)

| | $\bar r$ | val loss | control at $\bar r$ (interpolated) | Δ vs control | reading |
|---|---|---|---|---|---|
| `167` L5d fixed | 4 | 3.3601 | — | | curve point |
| `168` L5d fixed | 6 | 3.3492 | — | | curve point |
| `162` L5d fixed | 8 | 3.3443 | — | | curve point, $r_{max}$ |
| **`165` L6** | **7.38** | **3.3452** | 3.3492 + 0.69 × (3.3443 − 3.3492) = **3.3458** | **−0.0006 (0.2σ)** | **matches a fixed depth** |
| `166` L6c | 5.52 | 3.3541 | 3.3518 | +0.0023 (0.8σ) | matches a fixed depth (`166`) |

The interpolated control carries the noise of two single-seed points (≈ σ), so nothing finer than
≈ 2σ separates L6 from the curve in either direction. **L6 vs L6c**, measured as each arm's distance
from the curve: −0.0006 vs +0.0023, so they are 0.0029 (1.0σ) apart. I24's prior, that expert-choice
should do better at equal $\bar r$, is not visible at this resolution.

## Depth over training — the collapse ADR-041 predicted, and the recovery the gate allowed

| step | 10 | 50 | 100 | 300 | 500 | 700 | 1000 | 1500 | 1907 |
|---|---|---|---|---|---|---|---|---|---|
| train $\bar r$ | 7.00 | 2.32 | 2.09 | 2.19 | 3.84 | 5.32 (val) | 6.61 | 7.32 | 7.35 |
| val loss, `165` | | | 5.2668 | 4.2701 | 3.8722 | 3.6708 | 3.5149 | 3.3854 | 3.3452 |
| val loss, `162` ($r$ = 8) | | | 5.5372 | 4.3094 | 3.8551 | 3.6599 | 3.5084 | 3.3835 | 3.3443 |

In the first ≈ 50 steps the policy falls to $r_{min}$ = 2. That is the ponder-cost-only collapse
ADR-041's context described. It stays there for ≈ 300 steps (0.16 B tokens), while extra iterations
are worth less than their price. From ≈ step 400 the depth climbs steadily to 7.4, as the gated
residual (ADR-041 §1) carries the main loss's gradient to the halting head and the deeper iterations
become useful. The ungated code has no gradient path for that climb (ADR-041's context; `gate = false` was not run). The early shallow phase gives L6 the
same head start that `167` ($r$ = 4) has over `162`. L6 leads `162` until step ≈ 300, falls behind
while its depth is between 4 and 6 (steps 500–1000), and converges onto `162` by the end.

ADR-041 §2 chose $\lambda_p$ = 3·10⁻³ to land $\bar r$ between 4 and 8. It landed at 7.38, near the top
of that range. The value per iteration that this budget measures in the end is lower than §2's
estimate from L5 (L5d's iterations 7–8 are worth 0.0049 / 2 ≈ 0.0025 nats per token per iteration,
`168`). The policy still went deep. ACT's ponder term counts $N + R$, not a per-iteration price, so
the two numbers are not on the same scale. That is the reason ADR-041 §2 said the result of one
$\lambda_p$ is just that, and a sweep is the user's call.

## Routing

Worst point 5.01 of 8 effective experts at step 19, up to two dead experts in the first 21 steps,
**8.00 of 8 at the end with loads 0.115–0.129**. Same endpoint as `162`.

## Run health

| | |
|---|---|
| wall-clock | 8.19 h; summed `step_s` 8.096 h + evals 0.082 h → **0.1 % unaccounted** |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 73 °C |
| context | ran first after the `queue_screen_12` gate (ruff + unit tests passed 20:05, `5942ee1`); I34 probes on GPU just before it |

## Interpretation

ACT halting with ADR-041's gated residual trains and learns a depth policy. It falls to $r_{min}$ in
the first 50 steps and climbs back to $\bar r$ = 7.38 as the gate's gradient takes over. By ADR-041's
reading the ungated code would have stayed collapsed; that was not run. The policy it learns, however, is no better
than a fixed depth: against the L5d fixed-$r$ curve at 7.38 it is −0.0006 nats (0.2σ). It saves 7.8 %
of middle-block FLOPs, where H7 asks for ≥ 30 %, and a fixed $r$ ≈ 7.4 would give the same saving at
the same loss. **H7: silent at `screen`** (one seed, |Δ| < 4σ); L6 at $\lambda_p$ = 3·10⁻³ gives it no
support. Whether a larger $\lambda_p$ pushes $\bar r$ into H7's range without falling below the
curve is the open question. That sweep, a `small` pair, or closing H7's L6 arm at `screen` is the
user's call (ADR-041 §2).
