# 187-l5h-jb8-screen — L5h's untied control ($\mathcal{G}$ from the last iteration $j_b$ = 8 through its own projections; ADR-067): 3.3447 nats, **+0.0004 vs `162` (0.1σ)** → untying the global projections alone is worth nothing; L5h's gain at $j_b$ = 1 / 2 (−0.0154 / −0.0155 vs this run, 5.1σ) is the early source; H20's loss clause supported at `screen` with the sign reversed

`python -m scripts.train.train --rung L5h-jb8 --id 187-l5h-jb8-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`attention.global_source: bridged`, `bridge_iteration: 8`), per-step log in `log.jsonl`. Code at the `queue_screen_19` gate commit `15bd3b3`.

This write-up also closes `queue_screen_19`: it carries ADR-067 §5's table for `184`–`187` and
the copy and context evals of the four arms (ADR-066 §2), run 10:20–10:42 on 2026-11-21.

**Bears on:** **H20** (a global cache written from an early iteration costs little, so prefill can
stop early), by **ADR-067 §5**. This arm is the control: same parameters as `184`–`186`, but the
cache is read where L5d reads it. **$\Delta$(`187` − `162`) = +0.0004 (0.1σ): silent.** The
separate projections and their norm (+0.39 M parameters) do not change the loss. So the whole of
L5h's gain is the early source:

| arm | $j_b$ | val loss | Δ vs `162` ($T$ = 0.0667) | Δ vs `187` | prefill at 2 k (§3) |
|---|---|---|---|---|---|
| `184` | 1 | 3.3293 | **−0.0150** (5.0σ, beats) | **−0.0154** (5.1σ) | 0.34× |
| `185` | 2 | 3.3292 | **−0.0151** (5.0σ, beats) | **−0.0155** (5.2σ) | 0.44× |
| `186` | 4 | 3.3355 | −0.0088 (2.9σ, matches) | −0.0092 (3.1σ) | 0.63× |
| **`187`** | 8 (control) | **3.3447** | **+0.0004** (0.1σ, matches) | — | 1.00× |
| `162` | L5d, tied | 3.3443 | — | −0.0004 | 1.00× |

**H20's loss clause: supported at `screen`, one seed**, at every bridge point (all $\Delta \le T$),
and at $j_b$ = 1 and 2 the arm beats both L5d and the untied control by more than 4σ = 0.0119.
The loss is monotone in $j_b$: 3.3293 ≈ 3.3292 < 3.3355 < 3.3447. The prefill clause is §3's
arithmetic and is not measured here (TTFT is S5's). By §5 the smallest passing $j_b$, **$j_b$ = 1,
is the candidate for a `small` pair, by its own ADR**; ADR-066 §1 (an L5-family arm beating its
comparator by > 4σ) says the same. Nothing is queued for it; ADR-062 (the default change) waits on
that pair.

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1.

## Numbers

| | |
|---|---|
| params | 77.87 M total, 53.30 M non-embedding (as `184`; +0.39 M vs `162`) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3447** nats |
| final train loss (last step, main head) | 3.3133 |
| router aux (last step) | 0.0081 |
| throughput (median step) | 33,713 tokens/s aggregate (0.99× `162`) |
| peak memory per GPU | 3.90 GiB |
| wall-clock (this session) | 8.33 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 8 / 8.0000 |

Against the ladder: +0.0119 vs `110` (L5, 3.3328), +0.0089 vs $L_{ref}$ (3.3358), as `162`.

**Curve** (976-window evals; last row full held-out):

| step | `162` | `184` ($j_b$ 1) | `185` (2) | `186` (4) | `187` (8) | `187` − `162` |
|---|---|---|---|---|---|---|
| 100 | 5.5372 | 5.5346 | 5.5671 | 5.5768 | 5.5963 | +0.0591 |
| 300 | 4.3094 | 4.3274 | 4.3386 | 4.3583 | 4.3563 | +0.0469 |
| 500 | 3.8551 | 3.8690 | 3.8612 | 3.8668 | 3.8923 | +0.0372 |
| 700 | 3.6599 | 3.6565 | 3.6570 | 3.6593 | 3.6764 | +0.0165 |
| 1000 | 3.5084 | 3.4942 | 3.4952 | 3.5038 | 3.5151 | +0.0067 |
| 1500 | 3.3835 | 3.3667 | 3.3675 | 3.3730 | 3.3839 | +0.0004 |
| 1900 | 3.3436 | 3.3263 | 3.3272 | 3.3332 | 3.3433 | −0.0003 |
| 1907 (full) | 3.3443 | 3.3293 | 3.3292 | 3.3355 | 3.3447 | +0.0004 |

The untied control starts slower than `162` (its global projections start from their own
initialisation) and converges onto it by step 1500. The earlier the bridge point, the earlier the
arm leaves `162` behind.

## Loss by position, context 2048 (`eval_context`, 5606 windows; `eval_ctx2048.json` in each run)

| bucket | `162` | `184` | Δ | `185` | Δ | `186` | Δ | `187` | Δ |
|---|---|---|---|---|---|---|---|---|---|
| 0–512 (no global range) | 3.4106 | 3.4034 | −0.0071 | 3.4060 | −0.0046 | 3.4135 | +0.0030 | 3.4126 | +0.0020 |
| 512–1024 | 3.3224 | 3.3069 | −0.0156 | 3.3055 | −0.0169 | 3.3116 | −0.0109 | 3.3226 | +0.0001 |
| 1024–2048 | 3.3219 | 3.3031 | −0.0187 | 3.3024 | −0.0195 | 3.3083 | −0.0136 | 3.3216 | −0.0003 |

Totals reproduce the training evals to 0.0002. **The gain sits where $\mathcal{G}$ acts and grows
with position**: −0.016 to −0.020 beyond 512 for $j_b$ = 1 and 2, against −0.005 to −0.007 in the
first segment, where the global range is empty. That is the opposite pattern to `204`, whose cost
was already whole in 0–512. The control is flat in all three buckets. The buckets have no SE; the
0–512 differences are of the size of 2σ on the full set and are not read further.

## Copy probe (`probe_copy`, 512 windows, 1792–1855 overwritten with a copy of 256–319; no threshold, ADR-067 §5)

| | copy loss | control loss | gap (control − copy) |
|---|---|---|---|
| `162` | 3.9118 | 3.2506 | −0.661 |
| `184` | 3.8726 | 3.2374 | −0.635 |
| `185` | 3.8706 | 3.2319 | −0.639 |
| `186` | 3.8936 | 3.2435 | −0.650 |
| `187` | 3.9186 | 3.2547 | −0.664 |

No arm copies a span through $\mathcal{G}$; all gaps are negative and within 0.03 of L5d's. The
bridge changes how well far context is used on ordinary text, not whether a verbatim span can be
fetched.

## Routing and run health

Worst point 3.22 of 8 effective experts at step 13, up to four dead experts (step 12); last dead
expert at step 32; 7.99 of 8 at the end, loads 0.116–0.132 — as `162` and the other three arms.

| | |
|---|---|
| wall-clock | 8.33 h; summed `step_s` 8.240 h + evals 0.082 h + checkpoints 0.004 h → **0.0 % unaccounted** |
| throughput | 33.7 k tok/s median, flat |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 75 °C, min clock 1905 MHz |

## Interpretation

The control answers the question the first three arms left open, and not the way `185`'s write-up
guessed. Reading the global cache at the last iteration through untied projections gives L5d's
loss to 0.0004 nats, so untying is worth nothing here (unlike L5c's untied attention). Reading it
at iteration 1 or 2 instead is worth 0.015 nats (5σ against either comparator), at iteration 4
0.009, and the position buckets put that gain in the segments that actually read $\mathcal{G}$.
So an *earlier* state of the residual stream is a better key and value for later segments than
the final one, at this size and budget. A plausible reason is that $c^{(1)}$ is closer to the
token's content and $c^{(r)}$ is shaped for next-token prediction, but nothing here tests that,
and it is noted, not claimed. H20 only asked that the early source cost less than 2 % of $L_{ref}$;
at `screen` it costs nothing and the cheapest prefill point (0.34× of L5d's iterations for a 2 k
prompt, by arithmetic) is also the best loss. The cautions are the ladder's usual ones: one seed,
1 B tokens, and the L5-vs-dense tie at this budget reversed at `small`. The `small` pair for
$j_b$ = 1 (against the L5d pair `134`/`135`, by its own ADR) is what would turn this into a
verdict, and H20's 8 k end waits behind it.
