# 204-l5d-nope-screen — L5d-nope (RoPE on the local range only, NoPE on $\mathcal{G}$; ADR-066): 3.3595 nats, **+0.0152 vs `162` (5.1σ, above 4σ)** → H22's first clause *weakened* at `screen`; L9-relu2 builds on L5d (ADR-066 §7)

`python -m scripts.train.train --rung L5d-nope --id 204-l5d-nope-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`attention.rope_scope: local`), per-step log in `log.jsonl`. Code at the `queue_screen_18b` gate commit `b64d4b6` (`embla` merged as `c072e86`).

This is the rerun of `182-l5d-nope-screen`, which a power outage killed at step 117 (same preset,
seed and tokens; `docs/04 §Session log` 2026-11-19 06:39). **Wherever ADR-066–068 say `182`, read
`204`.**

**Bears on:** **H22** (first clause: NoPE on the global range is within noise of RoPE on both
ranges), by **ADR-066 §7** on $\Delta$(`204` − `162`): $|\Delta| \le 4σ$ silent, $\Delta > 4σ$
weakens at `screen`, $\Delta < -4σ$ beats L5d. The only change against `162` is `rope_scope`:
$\mathcal{G}$'s keys are stored unrotated and scored against the unrotated query; local scores are
unchanged. **Result: $\Delta$ = +0.0152 = 5.1σ = 1.28× 4σ → H22's first clause is *weakened* at
`screen`, one seed.** The second clause (ANN recall@32 over NoPE keys vs RoPE'd keys, rig `050`) is
not measured yet: the probes run after `183`, at the end of `queue_screen_18b`. No `small` pair
follows (ADR-066 §1: only an arm that beats its comparator gets one).

σ = 0.0030 over twelve `small` pairs (2σ 0.0059, 4σ 0.0119), as fixed in ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (identical to `162`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3595** nats |
| final train loss (last step, main head) | 3.3301 |
| router aux (last step) | 0.0081 (`162` 0.0076) |
| throughput (median step) | 28,458 tokens/s aggregate (**0.84× `162`'s 33,913**) |
| peak memory per GPU | 3.89 GiB |
| wall-clock (this session) | 9.86 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

## Against `162` and the ladder

| | val loss | Δ | reading |
|---|---|---|---|
| `100-l0-screen` (L0 dense, $L_{ref}$) | 3.3358 | | |
| `110-l5-screen` (L5) | 3.3328 | | |
| `162-l5d-segmean-screen` (L5d, RoPE on both ranges) | 3.3443 | — | comparator |
| **`204-l5d-nope-screen` (RoPE local, NoPE on $\mathcal{G}$)** | **3.3595** | **+0.0152 vs `162`** (+0.45 %, 5.1σ) | **ADR-066 §7: $\Delta$ > 4σ = 0.0119 → H22 first clause weakens at `screen`** |
| | | +0.0267 vs `110` (+0.80 %) | |
| | | +0.0237 vs $L_{ref}$ (+0.71 %) | |

**Curve** (976-window evals; the last row is the full held-out set):

| step | `162` (RoPE both) | `204` (NoPE on $\mathcal{G}$) | `204` − `162` |
|---|---|---|---|
| 100 | 5.5372 | 5.5394 | +0.0022 |
| 300 | 4.3094 | 4.3178 | +0.0084 |
| 500 | 3.8551 | 3.8615 | +0.0064 |
| 700 | 3.6599 | 3.6719 | +0.0120 |
| 1000 | 3.5084 | 3.5210 | +0.0126 |
| 1500 | 3.3835 | 3.3968 | +0.0133 |
| 1900 | 3.3436 | 3.3578 | +0.0142 |
| 1907 (full) | 3.3443 | 3.3595 | +0.0152 |

The gap opens by step 700 and does not close; it grows slowly to the end. Nothing in the curve
suggests a 1 B-token effect that more tokens would remove, but one seed at `screen` cannot say.

## Routing

Worst point 4.46 of 8 effective experts with three dead experts at step 16 (`162`: 4.06 with two
dead at step 14); the last step with a dead expert is 62. **8.00 of 8 at the end, loads
0.118–0.130** — the same endpoint as `162`. The loss gap is not a routing effect.

## Run health

| | |
|---|---|
| wall-clock | 9.86 h; summed `step_s` 9.760 h + evals 0.096 h + checkpoints 0.005 h → **0.0 % unaccounted** |
| throughput | 28.5 k tok/s median, 18.4 s per step, flat over the run |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 74 °C, min clock 1905 MHz |

**The `local` path costs 16 % of the step.** `162` ran at 33.9 k tok/s and `183` (`rope_scope:
none`, running now) started at 35.6 k, so the rig is healthy and the cost is this code path: it
scores the global range with an unrotated query and the local range with a rotated one. `182`
showed the same 18.4 s per step before the outage. `model/` is a reference implementation, so this
is recorded, not fixed; it is why the chain's rungs on this path take ≈ 9.9 h instead of 8.3 h.

## Post-run probes

Not run yet. `queue_screen_18b` runs them on idle GPUs after `183`: the copy probe and
`eval_context` at 2048 on `162`/`204`/`183` (ADR-066 §2) and the recall probe at 2 k and 8 k on
`162`/`204` (rig `050`, H22's second clause). Outputs go to `/mnt/nvme/post_queue18b/` and are
written up with `183`.

## Interpretation

Taking RoPE off the global range costs L5d 0.0152 nats at 1 B tokens: 5.1σ, past the 4σ line
ADR-066 §7 set, so H22's "within noise" clause is weakened at `screen` on one seed. The gap is
there from step 700 and widens slightly, with routing identical to `162`'s, so position
information on $\mathcal{G}$ is doing some work that the causal mask and the window do not replace.
The size is small (0.45 %), the same order as L5d's own cost against L5. Two things follow by the
ADRs as written: `queue_screen_20` reads this run's summary and builds the L9-relu2 arms
(`188`/`189`) on L5d with `rope_scope: all`, not on L5d-nope (ADR-066 §7; 3.3595 > 3.3443 +
0.0119); and ADR-063 (NoPE as the default) has no support from the loss side. Whether content-only
keys index better (H22's second clause, Q9's motivation) is a separate question that rig `050`
answers after `183`; a recall gain would now have to be weighed against this loss cost. `183`
(NoPE on both ranges) is the reference point for how much of the cost is the global range's.
