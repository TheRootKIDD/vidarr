# 114-l5-r4-screen — L5 at fixed $r$ = 4, width pinned (the L5-r sweep, I25) at `screen`

`python -m scripts.train.train --rung L5 --id 114-l5-r4-screen --tokens 1e+09 --micro-batch 4 --seed 0 --set middle.r_min=4 middle.r_max=4 experts.d_ff=896` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Third run of queue 4.

**Bears on:** **H7**, as its missing *control* (I25), not as a test of it. H7 reads *"learned $r_t$ saves
≥ 30 % middle-block FLOPs at ≤ 1 % loss vs fixed $r_{max}$"*. Before a learned policy can be credited
with anything, the ladder needs the loss-vs-depth curve of the **same model at fixed depths**, so that
"adaptive depth works" can be told apart from "this model wanted to be shallower". This run is `110`
($r$ = 8) with the iteration count halved and **nothing else changed**: $d_{ff}$ is pinned at 896,
because the budget solver would otherwise widen the experts to 2496 at $r$ = 4 and the sweep would
measure width, not depth. It is therefore **not** matched-FLOPs to `110` — that is the point.
Also bears on **H6** as the first measurement of how much depth this architecture wants.

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (identical to `110`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3567** nats |
| final train loss (last step, main head) | 3.3250 |
| throughput (median step) | 69,188 tokens/s aggregate |
| peak memory per GPU | 3.77 GiB |
| wall-clock (this session) | 4.08 h |
| seed | 0 |
| $r$ | fixed 4 (val_r_mean 4.0000) |
| forward FLOPs / token (`costmodel/`) | 128.3 MFLOP — **−29 %** vs `110`'s 181.0; the middle block's share is **halved** (4 × 13.2 vs 8 × 13.2 MFLOP; skeleton 75.5 either way) |

## The depth curve so far

| $r$ | id | val loss | Δ vs $r$ = 8 | fwd MFLOP/token | middle-block FLOPs vs $r$ = 8 |
|---|---|---|---|---|---|
| 4 | **`114`** | **3.3567** | **+0.0239 nats, +0.72 %** | 128.3 | −50 % |
| 8 | `110` | 3.3328 | — | 181.0 | — |
| 12 | `115` | *(running)* | | 233.8 | +50 % |
| L0 dense, for scale | `100` | 3.3360 | +0.10 % | ≈ 200 (0.6 GFLOP training) | |

**Halving the depth costs 0.72 % of loss and saves half the middle-block FLOPs — which is H7's own
threshold, met by a fixed schedule with no learning in it at all.** That is exactly what I25 warned:
scored against fixed $r_{max}$, "≥ 30 % middle-block FLOPs at ≤ 1 % loss" is satisfied by the trivial
policy $r_t$ = 4. So H7 as written cannot be the test. The honest test for L6 (ACT) and L6c
(expert-choice) is: **at whatever mean depth the learned policy settles on, does it beat this curve
at the same mean?** With two points the curve is a line: 0.024 nats per halving between 4 and 8; `115`
adds the third point and says whether it bends.

**Two further readings.** First, $r$ = 4 lands within 0.6 % of the *dense* baseline at 64 % of its
training FLOPs and 70 % of its parameters — the shared block is cheap to run shallow. Second, the
curve crosses again: $r$ = 4 leads `110` for the first 700 steps (5.43 vs 5.62 at step 100, 3.738 vs
3.749 at 600) and is overtaken between steps 700 and 800; fewer iterations learn faster and plateau
lower. Same shape as `111` vs `110` and `112` vs `106`, and the same caveat: the size of the gap is a
function of the token budget, and `small` may move it.

## Routing and run health

Routing worst 3.95 of 8 effective experts at step 23, endpoint 8.00 — the shared router does not care
how many times it is applied. **Thermal: GPU 0 flagged on 185 of 190 records, max 86 °C, min clock
1807 MHz** — this rung's short step (69 k tok/s) keeps the card at its limit continuously, one degree
hotter than `110`. Aggregate throughput is unaffected in the sense that matters (loss numbers stand,
`06 §7`); the throughput figure is a lower bound. Wall-clock 4.08 h, 4.02 h of summed steps + 0.05 h
of evals, 0.2 % unaccounted.

## Interpretation

At fixed $r$ = 4 the shared block loses 0.72 % against $r$ = 8 for half the middle-block FLOPs, so
H7's threshold is reachable without any learned policy, and the ladder now has the control that lets
L6 and L6c be scored properly: against fixed depth at the matched mean, on this curve. The run also
shows the architecture is tolerant of shallow depth at this scale — 4 iterations sit within 0.6 % of
the dense baseline at two thirds of its compute — which is a systems-side point in the note's favour
(cheap tokens can be cheap) and a quality-side warning that at 1 B tokens deeper is only slightly
better. `115` at $r$ = 12 completes the curve.
