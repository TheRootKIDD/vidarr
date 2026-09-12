# 110-l5-screen — L5 at `screen`

`python -m scripts.train.train --rung L5 --id 110-l5-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Rerun of `109`, which a power outage killed at step 170.

**Bears on:** **H6**, and is **silent on it as a verdict**, for two reasons that are stated up front. H6
compares the shared block *at matched params and FLOPs* against the stacked MoE; this run is the
author's local default $N_e$ = 8 (ADR-018), which matches L0's FLOPs but has **21 % of L1's
non-embedding parameters**. The parameter-matched arm is `L5-ne128` (`03 §2`, `06 §3`), not yet run.
And σ is unmeasured at `screen` (ADR-013). What this run does establish is the recurrent design's
position on the ladder, and it is the run every later recurrent rung (L5d, L6, L7, …) is measured
against.

**Where it sits in the ladder.** L5 branches from **L2**, not L3: the recurrent preset has the parallel
form (`model/middle.py`, `model/model.py` — the skeleton and middle block are parallel by construction,
so the `block_form: sequential` line in `config.yaml` is an unused default for `arch: recurrent`) and
**no MTP heads**, since ADR-030 parked H13 at `small`. The one change against L2 is therefore *twelve
layered MoE blocks → two early blocks, one shared MoE block applied $r$ = 8 times, two final blocks*.

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (16.5 M in experts, ADR-025) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3328** nats |
| final train loss (last step, main head) | 3.3016 |
| throughput (median step) | 48,553 tokens/s aggregate |
| peak memory per GPU | 3.86 GiB |
| wall-clock (this session) | 5.79 h |
| seed | 0 |
| $r$ | fixed 8 (val_r_mean 8.0000) |

## Against the ladder

| | val loss | Δ vs this run | non-emb params |
|---|---|---|---|
| `100-l0-screen` (L0 dense, $L_{ref}$) | 3.3360 | **−0.0032 nats, −0.10 %** | 75.5 M |
| `106-l1-screen` (L1 layered MoE) | 3.2048 | +0.1280 nats, +3.99 % | 246.6 M |
| `107-l2-screen` (L2 + parallel form) | 3.1495 | +0.1833 nats, +5.82 % | 246.6 M |
| **`110-l5-screen` (L5 shared block, $N_e$ = 8)** | **3.3328** | — | **52.9 M** |

**The shared block lands exactly on the dense baseline at 70 % of its non-embedding parameters and the
same FLOPs.** −0.10 % is well inside the seed spread `06 §4` records at `small` (0.005–0.01 nats), so
L5 and L0 are indistinguishable at one seed. Against the layered MoE it is 4 % behind — with a fifth
of the parameters. Both readings are the same fact: at matched FLOPs, recurrence trades parameters for
weight reuse, and at this scale eight iterations over one 16.5 M-parameter expert pool buy what 75 M
parameters of dense stack buy, no more.

**What that says about P6.** The note claims the repeated layer loses nothing "since you can simulate
the standard layer structure perfectly inside this structure". Against *dense* that holds here, and
at fewer parameters. Against the *layered MoE* it does not, at $N_e$ = 8 — but $N_e$ = 8 was never
the note's configuration; the note wants "128× as many experts" in the one layer, which is the
$N_e$ = 128 arm. The iso-depth prior in `05` (dense recurrence raises loss at matched compute; "the
expert pool must close the gap") is consistent with this: the pool of 8 closes the gap to dense, and
whether a pool of 128 closes the gap to L1 is precisely H6. **`L5-ne128` is now the most informative
run on the ladder** (19.2 h at `screen`, `018`), ahead of anything built on top of L5.

**Nothing to read into the depth.** $r$ is fixed at 8 and `val_r_mean` is 8.0 by construction; the
L5-r sweep (I25) is what gives a loss-vs-depth curve.

## Routing — one shared router, watched from step 1

The recurrent design has a single depth-conditioned router over 8 experts, used at every iteration.
Trajectory (`route_eff_min` is the effective expert count exp(H) at the worst point; 8.00 is uniform):

| step | eff. experts | load min / max | dead |
|---|---|---|---|
| 9 | — | — | **1** |
| 20 | **4.17** | — | 0 |
| 100 | 7.75 | 0.079 / 0.174 | 0 |
| 477 | 7.92 | 0.085 / 0.149 | 0 |
| 954 | 7.98 | 0.109 / 0.140 | 0 |
| 1907 | **8.00** | 0.118 / 0.129 | 0 |

The startup transient is milder than L3's (one dead expert at step 9, recovered by step 16, against
35 of 96 in `108`) and the endpoint is the most uniform of any rung so far. The aux-loss-free bias
controller holds a router that is shared across eight iterations as well as it holds twelve separate
ones. Not a confound.

## Run health — one card thermally trimmed for the whole run

| | |
|---|---|
| wall-clock | 5.79 h; summed `step_s` 5.715 h + evals 0.071 h + checkpoints 0.003 h → **0.05 % unaccounted** |
| throughput | 48.6 k tok/s median, against `018`'s 44.5 k without accumulation (+9 %, matching I21's 9–11 % all-reduce share for the recurrent rungs) |
| thermal | **GPU 0 flagged `n_thermal` > 0 on 180 of 190 records**, continuously from step 120 to the end; 83–85 °C, fan 100 %, SM clock 1807–1875 MHz against 1920–1942 on the other three cards |

This is the first rung with a *sustained* thermal flag rather than single eval-step samples. The
trim is ≈ 4 % of clock on one card and aggregate throughput did not move; opening a second window at
21:54 changed nothing in 45 minutes while the other cards held 68–74 °C, so it is GPU 0's own margin,
not ambient (`04` session log). Under `06 §7`, **the loss numbers stand and the throughput number is
a lower bound**. `109`, which ran the same config for 170 steps before the outage, held 1890 MHz on
the same card at the same tok/s.

**Reproducibility note.** `109` and `110` share seed 0 and config and differ by 0.04 nats at the
step-100 eval (5.6553 vs 5.6159). DDP over host-bounced NCCL with atomic kernels is not
bit-reproducible; one more reason ADR-013's verdicts wait for two seeds at `small`.

## Interpretation

At matched FLOPs and one seed, the shared recurrent block with the author's default pool of 8 experts
matches the dense baseline to within 0.1 % using 30 % fewer non-embedding parameters, and trails the
layered MoE by 4 % using 79 % fewer. **H6 is not decided here** — this is not its parameter-matched
arm and σ is unmeasured — but the run fixes the recurrent design's baseline for everything built on
it, and it sharpens H6 into one question the ladder can ask next: does a pool of 128 experts in the
shared block recover what the pool of 8 leaves on the table against L1. `L5-ne128` answers that and
should be queued ahead of the L5 extensions. Routing is healthy throughout; the only run-health caveat
is a sustained 4 % thermal trim on GPU 0, which touches throughput, not loss.
