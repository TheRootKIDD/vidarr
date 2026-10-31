# 167-l5d-r4-screen — L5d at fixed $r$ = 4 (ADR-041 §3): 3.3601 nats, **+0.0158 vs `162` ($r$ = 8, +0.47 %, 1.3× 4σ)** for half the middle-block FLOPs → H7's threshold is met by a fixed schedule on L5d too; the $r$ = 4 point of the curve that scores L6/L6c

`python -m scripts.train.train --rung L5d --id 167-l5d-r4-screen --tokens 1e+09 --micro-batch 4 --seed 0 --set middle.r_min=4 middle.r_max=4 experts.d_ff=896` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (`router.segment_aux: mean`), per-step log in `log.jsonl`. Code at the `queue_screen_12` gate commit `5942ee1` (merge of `l6-adr041`).

**Bears on:** **H7**, as a control (I25, ADR-041 §3). The only change against `162` is the depth:
$r$ = 4 instead of 8, with $d_{ff}$ pinned at 896 so the width does not move (the `114` construction).
It is the low point of the L5d fixed-$r$ curve (`167` $r$ = 4, `168` $r$ = 6, `162` $r$ = 8) that
ADR-041 §4 scores L6 (`165`) and L6c (`166`) against. **It is not a verdict on a feature.** On
H7's own wording (≥ 30 % middle-block FLOPs saved at ≤ 1 % loss vs fixed $r_{max}$ = `162`), a
fixed schedule meets it: −50 % middle-block FLOPs at +0.47 %. That restates `114`'s finding for L5
on the L5d base, and it is why a learned policy has to beat the curve, not $r_{max}$. `165`/`166` are
scored after `168`.

σ = 0.0030 over twelve pairs (`161`; 2σ 0.0059, 4σ 0.0119).

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (identical to `162`; the iteration count adds no parameters) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3601** nats |
| final train loss (last step, main head) | 3.3267 |
| router aux (last step) | 0.0041 (`162` 0.0076 at $r$ = 8; `114` 0.0037 at $r$ = 4) |
| throughput (median step) | 53,054 tokens/s aggregate (`162` 33,913: 1.56×) |
| peak memory per GPU | 3.80 GiB |
| wall-clock (this session) | 5.30 h |
| seed | 0 |
| $r$ / val_r_mean | fixed 4 / 4.0000 |

The router aux is summed over iterations, so halving $r$ roughly halves it; per iteration it is the
same as `162`'s.

## Against `162`, `111` and the L5 curve

| | val loss | Δ | reading |
|---|---|---|---|
| `162-l5d-segmean-screen` (L5d, $r$ = 8, aux mean) | 3.3443 | — | the $r_{max}$ control |
| **`167-l5d-r4-screen` (L5d, $r$ = 4, aux mean)** | **3.3601** | **+0.0158 vs `162` (+0.47 %, 1.3× 4σ)** | **H7's numbers met by a fixed schedule: −50 % middle FLOPs at ≤ 1 %** |
| `111-l5d-screen` (L5d, $r$ = 8, aux summed) | 3.3477 | +0.0124 vs `167` | not the baseline (ADR-041 context: aux reduction differs) |
| `110-l5-screen` (L5, $r$ = 8) | 3.3328 | | |
| `114-l5-r4-screen` (L5, $r$ = 4) | 3.3567 | +0.0239 vs `110` (+0.72 %) | the L5 analogue |

The depth curve is **flatter on L5d than on L5**: dropping iterations 5–8 costs 0.0158 nats here and
0.0239 on L5. The gap between the two $r$ = 4 points (`167` − `114` = +0.0034, 1.1σ) is inside noise,
so most of L5d's +0.0115 tax against L5 at $r$ = 8 (`162` − `110`) is paid in the deeper iterations.
One reading: iterations 5–8 see only the final-vector cache for global range (`01 §4.3`), so they add
less on L5d than on L5, where each has its own global cache. One seed; not a claim.

**Curve.** Shallow learns faster early, deep wins from ≈ 0.3 B tokens and the gap still grows at the end:

| step | `162` ($r$ = 8) | `167` ($r$ = 4) | `167` − `162` |
|---|---|---|---|
| 100 | 5.5372 | **5.3543** | −0.1829 |
| 300 | 4.3094 | **4.2735** | −0.0359 |
| 500 | 3.8551 | 3.8532 | −0.0019 |
| 700 | **3.6599** | 3.6681 | +0.0082 |
| 1000 | **3.5084** | 3.5186 | +0.0102 |
| 1500 | **3.3835** | 3.3985 | +0.0150 |
| 1907 | **3.3443** | 3.3601 | +0.0158 |

The same shape as `114` vs `110` (L5). At `small` the $r$ = 4 cost would likely be larger, not smaller,
so H7's "a fixed schedule already meets it" is a `screen`-budget statement.

## Routing

Worst point 4.37 of 8 effective experts at step 22, up to two dead experts over steps 5–7 (`162`:
4.06, one dead at 7–11), **8.00 of 8 at the end with loads 0.119–0.130**. Same endpoint as `162`.

## Run health

| | |
|---|---|
| wall-clock | 5.30 h; summed `step_s` 5.231 h + evals 0.058 h + checkpoints 0.003 h → **0.2 % unaccounted** |
| throughput | 53.1 k tok/s median, 1.56× `162` for half the middle-block iterations (the rest of the step — embeddings, entry/exit blocks, optimizer, the host-bounced all-reduce — does not shrink) |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 75 °C |

## Interpretation

L5d at a fixed four iterations ends 0.0158 nats (+0.47 %, 1.3× 4σ) behind the same model at eight,
for half the middle-block FLOPs and 1.56× the throughput. A fixed schedule therefore already meets H7's
stated numbers (≥ 30 % saved at ≤ 1 % loss vs $r_{max}$) on the L5d base, as `114` showed for L5.
That makes the I25 control necessary: a learned policy gets credit only for beating this curve at its
own mean depth (ADR-041 §4, > 4σ below the interpolated value). L5d's depth curve is flatter than
L5's (0.0158 vs 0.0239 for iterations 5–8), so it leaves an adaptive policy less to win. **H7: no
verdict from this run.** It is the $r$ = 4 point. `168` ($r$ = 6) completes the curve, and `165` L6
($\bar r$ 7.38, 3.3452) and `166` L6c ($\bar r$ 5.52, 3.3541) are scored after it.
