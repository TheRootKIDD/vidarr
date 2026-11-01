# 166-l6c-screen — L6c (causal threshold expert-choice depth, ADR-042) at `screen`: 3.3541 nats at $\bar r$ = 5.52 → **+0.0023 vs the L5d fixed-$r$ curve at 5.52 (0.8σ): matches a fixed depth**; meets H7's numbers (−31 % middle-block FLOPs at +0.29 %), but so does a fixed schedule

`python -m scripts.train.train --rung L6c --id 166-l6c-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`halting.mode: expert`, survival `linear` to 0.30, `tau_momentum` 0.99, MoR score gate, no ponder cost, `router.segment_aux: mean`), per-step log in `log.jsonl`. Code at the `queue_screen_12` gate commit `5942ee1` (merge of `l6-adr041`).

**Bears on:** **H7** through **I24**'s quality half (ADR-042 §5). It is scored like L6 (ADR-041 §4)
at its measured $\bar r$, and against L6 by distance from the curve.
- **Against the fixed-$r$ curve** (`167` / `168` / `162` = 3.3601 / 3.3492 / 3.3443 at $r$ = 4 / 6 /
  8), interpolated at $\bar r$ = 5.515: control **3.3518**, Δ **+0.0023** (0.8σ). Within 2σ, so **it
  matches a fixed depth: the learned token selection bought nothing over stopping every token at the
  same mean depth.**
- **H7's own clause** (ADR-013): $\bar r$ = 5.515 ≤ 0.7 $r_{max}$ = 5.6 ✓ (−31.1 % middle-block
  FLOPs), and Δ vs `162` = +0.0098 (+0.29 %) ≤ 0.01 $L_{ref}$ = 0.0334 ✓. Both numbers are met.
  |Δ| = 0.8× 4σ, so the verdict is **silent at `screen`** by ADR-013's letter, in the *supports*
  direction. I25 is why that is not credit to adaptive depth: `167` meets the same numbers with no
  policy at all (−50 % at +0.47 %).
- **Against L6** (`165`, −0.0006 from the curve): 0.0029 apart (1.0σ). **I24's prior, that
  expert-choice beats token-choice at equal depth, is not visible at `screen`.**
- **I24's systems half stays open.** It asks what a per-iteration threshold rule costs P11's
  continuous scheduler, and that waits for `sim/`.

σ = 0.0030 over twelve pairs (`161`; 2σ 0.0059, 4σ 0.0119). $L_{ref}$(`screen`) = 3.336 (`100`).

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (= L6; the depth router reuses the halting linear layer) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3541** nats |
| final train loss (last step, main head) | 3.3217 |
| held-out mean depth `val_r_mean` | **5.515** (schedule target 5.55; train, last step: 5.486) |
| middle-block FLOPs vs `162` | $\bar r / 8$ = 68.9 % (**−31.1 %**). Notional: the reference implementation computes stopped tokens and masks them |
| router aux (last step) | 0.0077 (`162` 0.0076) |
| throughput (median step) | 32,083 tokens/s aggregate (`162` 33,913: −5.4 % for the thresholds' quantile, all-reduce and masks) |
| peak memory per GPU | 3.93 GiB |
| wall-clock (this session) | 8.74 h |
| seed | 0 |

## Scoring (ADR-041 §4, ADR-042 §5)

| | $\bar r$ | val loss | control at $\bar r$ (interpolated) | Δ vs control | reading |
|---|---|---|---|---|---|
| `167` L5d fixed | 4 | 3.3601 | — | | curve point |
| `168` L5d fixed | 6 | 3.3492 | — | | curve point |
| `162` L5d fixed | 8 | 3.3443 | — | | $r_{max}$ |
| **`166` L6c** | **5.52** | **3.3541** | 3.3601 + 0.7575 × (3.3492 − 3.3601) = **3.3518** | **+0.0023 (0.8σ)** | **matches a fixed depth** |
| `165` L6 | 7.38 | 3.3452 | 3.3458 | −0.0006 (0.2σ) | matches a fixed depth |

The control at 5.52 rests on two single-seed points, so it is good to ≈ σ. L6c is not resolvably
better or worse than stopping every token at $r$ ≈ 5.5.

## Depth over training

| step | 10 | 50 | 100 | 300 | 500 | 700 | 1000 | 1500 | 1907 |
|---|---|---|---|---|---|---|---|---|---|
| train $\bar r$ | 2.03 | 3.07 | 4.77 | 5.73 | 5.66 | 5.67 (val) | 5.50 | 5.66 | 5.49 |
| val loss, `166` | | | 5.2561 | 4.2495 | 3.8317 | 3.6535 | 3.5123 | 3.3924 | 3.3541 |
| val loss, `168` ($r$ = 6) | | | 5.5017 | 4.3169 | 3.8859 | 3.6737 | 3.5141 | 3.3871 | 3.3492 |

The thresholds start at the first step's quantiles, which stops nearly every token at $r_{min}$ (2.03 at
step 10). The EMA reaches the schedule's mean by step ≈ 300 and holds it within ± 0.1 after that
(ADR-042 §3: the schedule is met on average, not per batch). Held-out depth is 5.515 against the
designed 5.55, so the frozen thresholds carry over to evaluation as §2 intended.

**L6c leads every fixed-depth run through step 700** (−0.054 vs `168` at step 500) and has lost it
to `162` by step 1000. The early lead combines the shallow start, which `167` shows is
worth a lot early on, with a mean depth near 5.7. By 1 B tokens it is back on the curve.

## Routing

Worst point 4.51 of 8 effective experts at step 24, one dead expert at most, none after step 41,
**8.00 of 8 at the end with loads 0.120–0.132**. Same endpoint as `162`.

## Run health

| | |
|---|---|
| wall-clock | 8.74 h; summed `step_s` 8.645 h + evals 0.083 h → **0.1 % unaccounted** |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 73 °C |

## Interpretation

L6c's causal threshold router does what ADR-042 designed: it meets its survival schedule on
held-out (5.515 against 5.55), stays balanced, and leaks nothing by construction (the causality
tests passed at the gate). In loss, it lands on the L5d fixed-depth curve (+0.0023 nats at
$\bar r$ = 5.52, 0.8σ). It meets H7's stated numbers, −31 % middle-block FLOPs at +0.29 %, but a fixed
schedule meets them too, so the learned choice of which tokens go deeper buys nothing measurable at
`screen`. **H7: silent at `screen`** (one seed, |Δ| < 4σ, *supports* direction on H7's letter, no
gain over the I25 control). Against L6 (0.0029 apart, 1.0σ), I24's expert-choice advantage does not
show. Together, `165` and `166` say that at this scale and budget, neither learned depth policy beats
choosing a smaller fixed $r$. A `small` pair, a $\lambda_p$ sweep for L6, or closing H7's learned-depth
arms at `screen` is the user's call. I24's systems half waits for `sim/`.
