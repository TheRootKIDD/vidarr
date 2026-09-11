# 107-l2-screen — L2 at `screen`

`python -m scripts.train.train --rung L2 --id 107-l2-screen --tokens 1e+09 --micro-batch 2 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (32 × 2 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** **H2**, quality half. H2 has two clauses — *parallel attention + MoE + memory costs
≤ 1 % loss at matched FLOPs*, and *cuts the per-iteration critical path by ≥ 30 % in sim* (S1). This run
addresses the first and is **silent on the second**, which is Tier S and needs `sim/`. Note also that
H2 is stated for the **shared block**; L2 applies the parallel form to the *layered* stack, so it tests
the form, not yet the setting the hypothesis is finally about. That comes with L5 onward.

## Numbers

| | |
|---|---|
| params | 271.1 M total, 246.6 M non-embedding |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.1495** nats |
| final train loss (last step, main head) | 3.1209 |
| throughput (last step) | 50,859 tokens/s aggregate |
| peak memory per GPU | 8.58 GiB |
| wall-clock (this session) | 5.56 h |
| seed | 0 |

## Against L1, and against the dense baseline

L2 is L1 plus exactly one change: attention and the feed-forward run **in parallel** on the same input
rather than sequentially. Parameters are unchanged to within rounding (271.1 M vs 271.2 M — the parallel
form drops one norm), so this is a clean one-feature comparison at matched FLOPs.

| | val loss | Δ vs L1 | Δ vs L0 |
|---|---|---|---|
| `100-l0-screen` (L0 dense, $L_{ref}$) | 3.3360 | — | — |
| `106-l1-screen` (L1 MoE) | 3.2048 | — | −3.93 % |
| **`107-l2-screen` (L2 parallel)** | **3.1495** | **−0.0553 nats, −1.73 %** | **−5.59 %** |

**H2's quality clause predicts a cost of up to 1 %. The measured result is a 1.73 % *gain*.** The
hypothesis is written as a tolerance — "we can afford to pay ≤ 1 % for the latency win" — so the
direction is favourable and the *trade* H2 contemplates may not be a trade at all in this setting. The
latency half is untouched and still has to be earned in `sim/` (S1).

**No verdict is recorded**, for the same reason as `106`: ADR-013 states thresholds inside a measured-σ
band and takes σ from the L0 seed pair, **which exists only at `small`**. One seed at `screen` gives no
band. For scale, 1.73 % is ≈ 0.055 nats against the 0.005–0.01 nats of seed-to-seed spread `06 §4`
records at `small` — 5–11× the noise — so it is unlikely to be a seed artefact, but "unlikely" is not
the banded test ADR-013 asks for. **L2 is on the `small` list** (`03 §2`), so it gets the real verdict
at 2 seeds.

**Take the size of the gain with some caution.** A parallel block is not merely a reordering: with
attention and FF reading the same input, gradients reach both from the residual directly, and at
$d$ = 768 over 12 layers the optimisation effect can be worth more than the representational loss
costs. That is a plausible story, not a measured mechanism, and it is the kind of thing that can shrink
or invert with depth and width. `medium` is where it would show up.

## Run health

50.9 k tokens/s sustained, 5.56 h wall-clock, **0.1 % unaccounted** (5.473 h of summed `step_s` plus
evals and checkpoints) — the same clean accounting `106` established when I22 closed. Thermal flags
were confined to single samples, mostly on eval steps, with the minimum clock never below 1845 MHz and
no effect on throughput. Ambient rose a little through the morning and GPU 0 tracked it; the user
opened windows around 10:00 and it settled.

**No routing trajectory for this run.** The per-router entropy fields went into the trainer on
2026-09-11, after `107` had started, so only `106`-style `load_max`/`load_min` exist here. The endpoint
is recoverable from the checkpoint with `scripts/analysis/probe_routing.py` and is **owed** — deferred
to when the ladder is idle, because the probe saturates ~10 of 12 cores and would dent a running rung's
throughput.

## Interpretation

The parallel form is not a cost at `screen`; it is a 1.73 % improvement over the sequential MoE at
matched parameters and FLOPs, and takes the ladder to 5.59 % below the dense baseline. H2's quality
clause is satisfied with room to spare, its latency clause is untested and belongs to S1, and the
banded verdict belongs at `small`. Nothing here bears on whether the same holds in the *shared* block,
which is what H2 is ultimately about.
