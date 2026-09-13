# 115-l5-r12-screen — L5 at fixed $r$ = 12, width pinned (the L5-r sweep, I25) at `screen`

`python -m scripts.train.train --rung L5 --id 115-l5-r12-screen --tokens 1e+09 --micro-batch 4 --seed 0 --set middle.r_min=12 middle.r_max=12 experts.d_ff=896` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Fourth and last run of queue 4.

**Bears on:** **H7**, as the third point of the fixed-depth control curve (I25; see `114`), and **H6**
as the measurement of how much depth the shared block wants at this budget. `110` with the iteration
count raised to 12 and $d_{ff}$ pinned at 896, so only depth changes. Forward FLOPs 233.8 MFLOP/token,
**+29 %** vs `110`; middle-block FLOPs +50 %. Not matched-FLOPs by design.

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (identical to `110`, `114`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3370** nats |
| final train loss (last step, main head) | 3.3062 |
| throughput (median step) | 37,008 tokens/s aggregate |
| peak memory per GPU | 3.95 GiB |
| wall-clock (this session) | 7.61 h |
| seed | 0 |
| $r$ | fixed 12 (val_r_mean 12.0000) |

## The depth curve, complete

| $r$ | id | val loss | Δ vs $r$ = 8 | fwd MFLOP/token | wall-clock |
|---|---|---|---|---|---|
| 4 | `114` | 3.3567 | +0.0239 nats, +0.72 % | 128.3 | 4.08 h |
| 8 | `110` | **3.3328** | — | 181.0 | 5.79 h |
| 12 | **`115`** | 3.3370 | **+0.0042 nats, +0.13 %** | 233.8 | 7.61 h |
| L0 dense | `100` | 3.3360 | +0.10 % | ≈ 200 | 4.4 h (`018`) |

**The curve bends flat at $r$ = 8.** Going from 4 to 8 iterations buys 0.024 nats; going from 8 to 12
buys nothing — 12 is 0.004 nats *worse*, which is inside the seed spread `06 §4` expects, for 29 %
more compute and 31 % more wall-clock. At 1 B tokens and $d$ = 768 with a 16.5 M-parameter expert
pool, **eight applications of the shared block is where this architecture saturates**, and the extra
iterations are FLOPs that buy nothing on held-out loss.

**What this does to H7.** H7 wants a learned $r_t$ to save ≥ 30 % of middle-block FLOPs at ≤ 1 % loss
against fixed $r_{max}$. The control curve now says: a fixed $r$ = 4 already saves 50 % at 0.72 %, and
$r_{max}$ = 8 is the *right* fixed depth here (12 gains nothing), so the fair comparison for L6 / L6c
is against fixed $r$ = 8 at a mean depth ≤ 8 — a learned policy that settles at a mean of 5–6 has to
beat the interpolated line between 3.3567 and 3.3328 at that mean, i.e. land below ≈ 3.345, to have
done anything a fixed schedule could not. And the systems reading cuts the other way from what the
note assumes: with the curve flat past 8, the win from variable depth at this scale is *all* on the
cheap side (tokens that can stop at 4), none on the expensive side (tokens that would benefit from 12).
Whether the top of the curve moves with scale is one of the specific things `medium` ($r$ = 16 by
`06 §3`) exists to find out.

**Curve shape.** $r$ = 12 trails `110` at every eval from step 100 (5.712 vs 5.616) to the end and
never crosses — deeper is slower to learn *and* no better converged here, the mirror image of `114`.
The three runs order by learning speed as 4 > 8 > 12 throughout the first half and by final loss as
8 ≈ 12 > 4.

## Routing and run health

Routing worst 2.88 of 8 at step 19, endpoint 8.00; twelve applications of the router balance as well
as four or eight. **Thermal: GPU 0 flagged on 188 of 190 records, max 85 °C, min clock 1807 MHz** —
the same sustained trim as `110` and `114`. Loss numbers stand; throughput is a lower bound. Wall-clock
7.61 h, 7.51 h of summed steps + 0.09 h of evals, 0.1 % unaccounted.

## Interpretation

Twelve iterations of the shared block cost 29 % more compute than eight and return 0.004 nats less,
inside noise: the depth curve is flat past $r$ = 8 at this scale. Together with `114` that gives L6 and
L6c their control — a learned policy is credited only for what it gains over the fixed-$r$ line at the
same mean depth — and it says where the value of adaptive depth lies here: in stopping easy tokens
early, not in running hard ones longer. One seed, no verdict; the shape of the curve, not the 0.004,
is the finding.
