# 183-l5d-nope-all-screen — L5d with no RoPE on either range (ADR-066): 3.4891 nats, **+0.1448 vs `162` (48σ)** — the reference point, worse as expected; with the copy and context evals of `162`/`204`/`183`

`python -m scripts.train.train --rung L5d-nope-all --id 183-l5d-nope-all-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`attention.rope_scope: none`), per-step log in `log.jsonl`. Code at the `queue_screen_18b` gate commit `b64d4b6`.

**Bears on:** **H22**, as a reference point only: ADR-066 §7 makes `183` "expected worse", not a
verdict arm. It is worse by +0.1448 nats against `162` (48σ) and +0.1296 against `204`. **H22's
verdict is unchanged by this run**: first clause *weakened* at `screen` (`204`), second clause
*not met as written* (rig `050-g-recall-nope`: recall gain at 2 k, none at 8 k). The causal mask
and the 512-token window do not replace RoPE in the local range; almost all of the position signal
L5d needs is local.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (identical to `162`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.4891** nats |
| final train loss (last step, main head) | 3.4538 |
| router aux (last step) | 0.0082 |
| throughput (median step) | 35,873 tokens/s aggregate (1.06× `162`, 1.26× `204`) |
| peak memory per GPU | 3.88 GiB |
| wall-clock (this session) | 7.83 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `162`, `204` and the ladder

| | val loss | Δ vs `162` | |
|---|---|---|---|
| `100-l0-screen` ($L_{ref}$) | 3.3358 | | |
| `162-l5d-segmean-screen` (RoPE on both ranges) | 3.3443 | — | comparator |
| `204-l5d-nope-screen` (RoPE local, NoPE on $\mathcal{G}$) | 3.3595 | +0.0152 (5.1σ) | H22 first clause weakened |
| **`183-l5d-nope-all-screen` (no RoPE)** | **3.4891** | **+0.1448** (+4.33 %, 48σ) | reference point |

Removing RoPE from the global range costs 0.015; removing it from the local range as well costs a
further 0.130, nine times as much.

**Curve** (976-window evals; last row full held-out):

| step | `162` | `204` | `183` | `183` − `162` |
|---|---|---|---|---|
| 100 | 5.5372 | 5.5394 | 5.7386 | +0.2014 |
| 300 | 4.3094 | 4.3178 | 4.7856 | +0.4762 |
| 500 | 3.8551 | 3.8615 | 4.2540 | +0.3989 |
| 700 | 3.6599 | 3.6719 | 3.9610 | +0.3011 |
| 1000 | 3.5084 | 3.5210 | 3.7103 | +0.2019 |
| 1500 | 3.3835 | 3.3968 | 3.5387 | +0.1552 |
| 1900 | 3.3436 | 3.3578 | 3.4878 | +0.1442 |
| 1907 (full) | 3.3443 | 3.3595 | 3.4891 | +0.1448 |

The gap is largest early (0.48 at step 300) and closes to 0.14 by the end, still shrinking slowly:
the model learns some order from the causal mask, late. One seed at 1 B tokens says nothing about
where that ends.

## Post-run evals (ADR-066 §2; `queue_screen_18b`, GPUs idle, 2026-11-20 00:20–00:54)

**Loss by position, context 2048** (`eval_context`, 5606 windows; files `eval_ctx2048.json` in each
run's directory; the totals reproduce the training evals to 0.0003):

| bucket | `162` | `204` | Δ | `183` | Δ |
|---|---|---|---|---|---|
| 0–512 (no global range) | 3.4106 | 3.4246 | +0.0140 | 3.5443 | +0.1337 |
| 512–1024 | 3.3224 | 3.3374 | +0.0150 | 3.4682 | +0.1458 |
| 1024–2048 | 3.3219 | 3.3377 | +0.0158 | 3.4715 | +0.1496 |

**`204`'s cost is already +0.0140 in the first 512 tokens, where $\mathcal{G}$ is empty and
`rope_scope: local` computes the same function as `all`.** Only +0.001 to +0.002 is added where
the global range acts. So most of `204`'s loss gap is not worse use of far context at eval time; it
sits in the shared weights. ADR-066 §2 called 0–512 the in-distribution control, and the control
moved. This does not change the verdict (the rule is on the full held-out loss), but it changes
the story in `204`'s write-up: "position information on $\mathcal{G}$ is doing work" is not what
the buckets show.

**Copy probe** (`probe_copy`, 512 windows, positions 1792–1855 overwritten with a copy of 256–319,
loss on copied tokens 2–64; files `probe_copy.json`):

| | copy loss | control loss | gap (control − copy) |
|---|---|---|---|
| `162` | 3.9118 | 3.2506 | −0.661 |
| `204` | 3.4183 | 3.2573 | −0.161 |
| `183` | 3.5653 | 3.3824 | −0.183 |

No model copies: every gap is negative, i.e. a pasted span from 1,500 tokens back is harder to
predict than the text it replaced, for all three. The NoPE arms are hurt 0.50 nats less than L5d
by the paste. The probe has no SE and no arm passes it, so this is recorded as a difference in how
the models react to an out-of-place span, not as retrieval.

**Recall probe:** rig `050-g-recall-nope` (H22's second clause).

## Routing and run health

Worst point 4.32 of 8 effective experts with two dead at step 19; last dead expert at step 35;
7.99 of 8 at the end, loads 0.114–0.137.

| | |
|---|---|
| wall-clock | 7.83 h; summed `step_s` 7.747 h + evals 0.076 h + checkpoints 0.004 h → **0.0 % unaccounted** |
| throughput | 35.9 k tok/s median, flat |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 74 °C, min clock 1905 MHz |

## Interpretation

With no RoPE anywhere L5d ends at 3.4891, 0.145 nats (48σ) behind L5d and 0.130 behind L5d-nope:
the "expected worse" reference of ADR-066 §7, confirmed, and nine tenths of RoPE's value is in the
local range. The evals that came with it sharpen `204`'s result. Its +0.015 cost is present at full
size in the first 512 tokens, where the two models compute the same function, so it is a
whole-model effect of training with unrotated global keys and not a deficit confined to the global
range. No arm can copy a span through $\mathcal{G}$ at `screen`. H22 stands as `204` and rig `050`
left it: the loss clause weakened at one seed, the recall clause true at the trained length only.
Whether to spend a second seed on `204` (its Δ is 1.28× the 4σ line) or to drop NoPE on
$\mathcal{G}$ is the user's call; ADR-063 stays proposed.
