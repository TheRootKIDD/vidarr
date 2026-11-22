# 189-l9-relu2-screen — L9-relu2, ReLU²-scored global range in the two-branch combination (ADR-068, on L5d): 3.3870 nats, **+0.0150 vs its control `188` (5.0σ, inside $T$ = 0.0334: silent, a cost inside the allowance)**; **exact zeros 53.7 % against the 90 % clause → H23's sparsity clause not met**; cap $K$ = 128 costs +0.0008 (met); +0.0427 vs `162`

`python -m scripts.train.train --rung L9-relu2 --id 189-l9-relu2-screen --tokens 1e+09 --micro-batch 4 --seed 0 --set attention.rope_scope=all` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`attention.global_combine: two_branch`, `global_scoring: relu2`, `rope_scope: all`), per-step log in `log.jsonl`. Code at the `queue_screen_20` gate commit `9c0e5b7`. Built on L5d, not L5d-nope, by ADR-066 §7 (see `188`).

This write-up also closes `queue_screen_20`: it carries ADR-068 §3's measures (`probe_relu2.json`),
the learned gates of both arms, and ADR-066 §2's evals for `188` and `189` (run 07:24–07:48 on
2026-11-22).

**Bears on:** **H23** (ReLU² scoring on the global range reaches ≥ 90 % exact zeros without a
sparsity loss at ≤ 1 % loss cost; a cap costs ≤ 1 %), by **ADR-068 §4**, clause by clause:

| clause | rule | measured | reading |
|---|---|---|---|
| loss | $\Delta$(`189` − `188`) vs $T$ = 1 % $L_{ref}$ = 0.0334, ADR-066 §1 | **+0.0150** (5.0σ) | 4σ < $\Delta \le T$ + 2σ: **silent** (a cost measured inside the allowance) |
| sparsity | final zero fraction ≥ 90 %, mean over heads and iterations | **53.7 %** (probe, 256 windows; 52.9 % in the last training step) | **not met** |
| cap | $\Delta_{cap}(K = 128) \le$ 1 % at 2 k | **+0.0008** (0.02 %) | **met** |

**H23 at `screen`, one seed: not supported.** Its defining claim, ≥ 90 % exact zeros with no
sparsity loss, misses by a wide margin (ADR-068 gives that clause a threshold and no band, so it
is recorded as not met, not as a banded *weakens*). The loss clause is silent and the cap clause
holds. The 8 k cap clause stays owed and is not worth running on this result. **What follows by
§4:** `189` lands within $T$ of its control, which is the stated condition for a `small` pair *by
its own ADR*. I do not think the condition's intent is met: the arm fails the clause the
hypothesis is about, and both two-branch arms sit far above L5d (`188` +0.0277, `189` +0.0427 vs
`162`). That is the user's call; nothing is queued.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.91 M non-embedding (as `188`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3870** nats |
| final train loss (last step, main head) | 3.3530 |
| router aux (last step) | 0.0080 |
| throughput (median step) | 22,667 tokens/s aggregate (**0.67× `188`'s 33,949**) |
| peak memory per GPU | 4.21 GiB (`188` 3.89) |
| wall-clock (this session) | 12.40 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `188`, `162` and the ladder

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `162-l5d-segmean-screen` (L5d, one softmax) | 3.3443 | | base |
| `188-l9-2b-screen` (two branches, softmax global) | 3.3720 | +0.0277 vs `162` | control |
| **`189-l9-relu2-screen` (two branches, ReLU² global)** | **3.3870** | **+0.0150 vs `188`** (+0.45 %, 5.0σ) | **inside $T$ = 0.0334: silent** |
| | | +0.0427 vs `162` (+1.28 %, 14σ) | |
| | | +0.0512 vs $L_{ref}$ (+1.53 %) | |

**Curve** (976-window evals; last row full held-out):

| step | `162` | `188` | `189` | `189` − `188` |
|---|---|---|---|---|
| 100 | 5.5372 | 5.6214 | 5.6013 | −0.0201 |
| 300 | 4.3094 | 4.3701 | 4.3637 | −0.0064 |
| 500 | 3.8551 | 3.9074 | 3.9033 | −0.0041 |
| 700 | 3.6599 | 3.7057 | 3.7045 | −0.0012 |
| 1000 | 3.5084 | 3.5432 | 3.5538 | +0.0106 |
| 1500 | 3.3835 | 3.4120 | 3.4257 | +0.0137 |
| 1900 | 3.3436 | 3.3709 | 3.3853 | +0.0144 |
| 1907 (full) | 3.3443 | 3.3720 | 3.3870 | +0.0150 |

ReLU² is slightly ahead of the softmax control until step 700, then falls behind; the gap is
still widening slowly at the end.

## ReLU² measures (ADR-068 §3; `probe_relu2.json`, 256 held-out windows, uncapped loss 3.3505)

| | |
|---|---|
| exact-zero fraction of (query, key) weights | **0.537** |
| per iteration 1–8 | 0.585, 0.423, 0.397, 0.538, 0.625, 0.589, 0.517, 0.620 |
| per query head 1–12 | 0.722, 0.739, 0.803, 0.338, 0.327, 0.389, 0.692, 0.552, 0.598, 0.407, 0.379, 0.498 |
| V-skip rate (keys zero for every head of their KV group) | **0.384** (per iteration 0.25–0.49) |
| cap $K$ = 64 / 128 / 256 (top-$K$ by pre-ReLU score) | **+0.0021 / +0.0008 / +0.0002** nats |
| cap to $|\mathcal{G}|$ = 1536 (the model's own set) | 0.0000 (exact) |

About half the scores are negative, which is what RMS-normalised $q$ and $k$ give with no
pressure either way; the model did not learn to push scores below zero. No head reaches 90 %
(best 80 %). The cap result is the useful one: keeping 128 of up to 1536 global keys costs
0.0008 nats and 64 keys cost 0.0021, so the weights are concentrated even though they are not
exactly zero. That is a statement about this arm; whether one-softmax L5d caps as cheaply is not
measured.

## The gates $g_h = \mathrm{sigmoid}(\gamma_h)$ (read from the checkpoints, 12 query heads)

| | $g_h$ |
|---|---|
| `188` | 0.42, 0.44, 0.39, 0.39, 0.39, 0.43, 0.60, 0.58, 0.53, 0.39, 0.46, 0.42 |
| `189` | 0.56, 0.54, 0.57, 0.39, 0.40, 0.43, 0.56, 0.47, 0.46, 0.38, 0.40, 0.39 |

Both arms keep every gate between 0.38 and 0.60 from an initial 0.50. Neither shut the global
branch. So `188`'s cost against one softmax is not a closed branch; it is a branch that is always
on at about half weight for every token.

## Loss by position, context 2048 (`eval_context`, 5606 windows)

| bucket | `162` | `188` | Δ vs `162` | `189` | Δ vs `162` | `189` − `188` |
|---|---|---|---|---|---|---|
| 0–512 (no global range) | 3.4106 | 3.4248 | +0.0143 | 3.4380 | +0.0274 | +0.0131 |
| 512–1024 | 3.3224 | 3.3536 | +0.0311 | 3.3695 | +0.0471 | +0.0160 |
| 1024–2048 | 3.3219 | 3.3546 | +0.0327 | 3.3700 | +0.0481 | +0.0154 |

The two-branch cost roughly doubles where $\mathcal{G}$ acts (+0.014 → +0.032 for `188`), so
about half of it is in the use of far context and half is a whole-model effect present already in
the first segment. ReLU²'s extra cost over `188` is nearly flat across buckets (+0.013 to +0.016).

## Copy probe (no threshold)

| | copy loss | control loss | gap (control − copy) |
|---|---|---|---|
| `162` | 3.9118 | 3.2506 | −0.661 |
| `188` | 3.9263 | 3.2800 | −0.646 |
| `189` | 3.9320 | 3.2912 | −0.641 |

No arm copies through $\mathcal{G}$.

## Routing and run health

Worst point 3.36 of 8 effective experts at step 22, up to three dead experts (step 19); last dead
expert at step 31; 7.99 of 8 at the end, loads 0.110–0.134.

| | |
|---|---|
| wall-clock | 12.40 h; summed `step_s` 12.253 h + evals 0.136 h + checkpoints 0.006 h → **0.0 % unaccounted** |
| throughput | 22.7 k tok/s median, 23.0 s per step, flat over the run |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 75 °C, min clock 1905 MHz |

**The ReLU² path costs a third of the throughput** (0.67× `188`) and 0.32 GiB. The rig is healthy
(`188` ran at 33.9 k immediately before), so it is this code path in the reference implementation;
recorded, not fixed. It moved the chain ≈ 4 h later.

## Interpretation

ReLU² scoring did not do what H23 hoped at this size. Trained with no sparsity loss, it zeroes
54 % of global weights, about what sign alone gives, against a 90 % target; a V-skip rate of 38 %
would not pay for an index. It costs 0.0150 nats against the matched softmax control, inside the
1 % allowance but a real, growing gap, and it trains a third slower here. The one positive is the
cap: the top 128 keys carry almost all of the global branch's effect (+0.0008), which says a
hard top-$K$ read is cheap for this arm at 2 k. Around that sits a larger fact from `188`: the
two-branch combination that ReLU² needs is itself 0.028 nats worse than ADR-008's single softmax,
with gates that stay near one half, so any L9-relu2 built this way starts 9σ behind L5d. A ReLU²
arm with a learned threshold, or a sparsity term, would be a different feature under a new ADR
(ADR-068 §2 excluded both). As measured: sparsity clause not met, loss clause silent, cap clause
met, one seed at `screen`.
