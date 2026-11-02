# 171-l5d-r2-screen — L5d at fixed $r$ = 2 (ADR-045 §3) at `screen`: 3.3820 nats → **completes the L5d fixed-$r$ curve down to $r_{min}$; scores `169` (+0.0064, 2.1σ) and `170` (+0.0174, 5.8σ) — both *above* the curve**

`python -m scripts.train.train --rung L5d --id 171-l5d-r2-screen --tokens 1e+09 --micro-batch 4 --seed 0 --set middle.r_min=2 middle.r_max=2 experts.d_ff=896` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the `167` construction at $r$ = 2: `experts.d_ff` 896 as in `167`, `router.segment_aux: mean`), per-step log in `log.jsonl`. Code at the `queue_screen_13` gate commit `76d1965`.

**Bears on:** **H7** through the **I25** control. It is a fixed-$r$ point for scoring learned depth,
not an arm of its own (ADR-045 §3). Against $L_{ref}$ it is +0.046 (+1.38 %), and against `162`
($r$ = 8) +0.0377 (+1.13 %) for −75 % middle-block FLOPs. A fixed $r$ = 2 therefore *misses* H7's
1 % loss allowance at `screen`, where $r$ = 4 meets it (+0.47 %).

**The two shallow L6 arms, scored on the 2–4 segment** (ADR-041 §4, ADR-045 §2–3):

| | $\bar r$ | val loss | control at $\bar r$ (interpolated) | Δ vs control | reading |
|---|---|---|---|---|---|
| `169` L6, $\lambda_p$ 0.01 | 3.815 | 3.3685 | 3.3820 + 0.9074 × (3.3601 − 3.3820) = **3.3621** | **+0.0064 (2.1σ)** | above the curve, just outside 2σ (worse than a fixed depth, not resolved at 4σ) |
| `170` L6, $\lambda_p$ 0.03 | 2.060 | 3.3987 | 3.3820 + 0.0301 × (3.3601 − 3.3820) = **3.3814** | **+0.0174 (5.8σ, 1.5× 4σ)** | **worse than a fixed depth** |

Neither is below the curve. `170` is resolvably worse than simply running every token at $r$ ≈ 2.

σ = 0.0030 over twelve pairs (`161`; 2σ 0.0059, 4σ 0.0119). $L_{ref}$(`screen`) = 3.336 (`100`).

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (= `167`; = `162` within 400 params, the $d_{ff}$ 896 rounding) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3820** nats |
| final train loss (last step, main head) | 3.3492 |
| middle-block FLOPs vs `162` | 2/8 = 25 % (**−75 %**), real: no masking |
| router aux (last step) | 0.0021 (`162` 0.0076, `170` 0.0052) |
| throughput (median step) | **74,185 tokens/s** aggregate, 7.07 s per step |
| peak memory per GPU | 3.75 GiB |
| wall-clock (this session) | 3.79 h |
| seed | 0 |

## The L5d fixed-$r$ curve, complete

| $r$ | 2 (`171`) | 4 (`167`) | 6 (`168`) | 8 (`162`) |
|---|---|---|---|---|
| val loss | 3.3820 | 3.3601 | 3.3492 | 3.3443 |
| gain from the previous point | | 0.0219 | 0.0109 | 0.0049 |
| vs `162` | +1.13 % | +0.47 % | +0.15 % | — |
| median tok/s | 74,185 | 53,100 | 41,300 | 33,913 |

Concave all the way down. Each pair of extra iterations is worth about half the previous pair. Step time
is 7.07 s, against the 1.4 s × 2 + 4.3 s = 7.1 s that `168`'s fit predicted, so the per-iteration
cost model holds at $r$ = 2.

## Depth over training (the L6 arms against fixed $r$ = 2 and 4)

| step | 100 | 300 | 500 | 700 | 1000 | 1500 | 1907 |
|---|---|---|---|---|---|---|---|
| val loss, `171` ($r$ = 2) | 5.2908 | 4.2745 | 3.8657 | 3.6759 | 3.5344 | 3.4173 | 3.3820 |
| val loss, `170` ($\bar r$ 2.00 → 2.06) | 5.2808 | 4.2803 | 3.8963 | 3.7057 | 3.5565 | 3.4355 | 3.3987 |
| val loss, `169` ($\bar r$ 2.00 → 3.81) | 5.2665 | 4.2741 | 3.8756 | 3.6877 | 3.5300 | 3.4068 | 3.3685 |
| val loss, `167` ($r$ = 4) | 5.3543 | 4.2735 | 3.8532 | 3.6681 | 3.5186 | 3.3985 | 3.3601 |

**`170` trails `171` at every eval from step 300 on (by 0.031 at step 500, 0.017 at the end) at almost the same
depth.** It is not a depth effect: both run nearly all tokens at $r$ = 2. The cost is the ACT
machinery itself. One candidate, not tested, is the ponder term's gradient, which the gated residual
(ADR-041 §1) carries into the shared block. That cost is paid every step while the halting head is
told that deeper is expensive. Not isolated. `169` sat at $r_{min}$ for the first ≈ 550 steps, and it
is above `171` at steps 500–700, below it from step 1000 as its depth climbs, and ends 0.0064 above
the 2–4 line.

## Routing

Worst point 4.68 of 8 effective experts at step 17, one dead expert at most, none after step 24,
**8.00 of 8 at the end with loads 0.118–0.132**.

## Run health

| | |
|---|---|
| wall-clock | 3.79 h; summed `step_s` 3.738 h + evals 0.045 h → **0.2 % unaccounted** |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 75 °C |
| context | third and last run of `queue_screen_13` (gate `76d1965`); ADR-045 §3 fired on `170`'s $\bar r$ = 2.06 |

**The masked-ACT wall-clock tax, now measurable.** At nearly the same mean depth, `170` ($\bar r$
2.06) ran at 58,645 tok/s and `171` (fixed 2) at 74,185: **−21 %**. `170`'s step time rose 7.1 →
11.7 s while its depth moved by 0.06. That points to the cost being set by the deepest token in a
micro-batch, not the mean, which is what `169`/`170`'s results.md suggested. The notional
$\bar r / 8$ FLOPs saving of L6 does not reach wall-clock in the reference implementation.

## Interpretation

Fixed $r$ = 2 gives 3.3820 nats and finishes the L5d depth curve, 2 / 4 / 6 / 8 → 3.3820 / 3.3601 /
3.3492 / 3.3443, which is concave with each step worth about half the last. On it, L6 at
$\lambda_p$ = 0.01 is +0.0064 (2.1σ) and at 0.03 is +0.0174 (5.8σ) above the curve, so neither
shallow arm beats a fixed depth, and the 0.03 arm is resolvably worse than one. **H7 at `screen`:
no arm supports it beyond 4σ, so H7 stays silent overall; the L6 arm at $\lambda_p$ = 0.03 is
weakened (5.8σ above the I25 control), and `169` is in between (2.1σ).** Across the four learned-depth runs
(`165` −0.0006, `166` +0.0023, `169` +0.0064, `170` +0.0174), nothing beats the fixed-$r$ curve,
and the higher the ponder price (the shallower the depth), the further above the curve L6 lands. Under
ADR-045 §4 this is the "in between" case (not all within 2σ, none below), so the next step is the
user's call.
