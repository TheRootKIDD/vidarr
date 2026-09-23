# 136-l5-noej-screen — E-Q2, the router without the depth embedding: 3.3394 nats, +0.20 % vs L5 — **silent** (inside the 4σ band at one seed); the router does not specialise experts by iteration with or without $e_j$

`python -m scripts.train.train --rung L5-noej --id 136-l5-noej-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Sixth run of `queue_small_3.sh`; the rerun of the burnt `128`. `110-l5-screen` with the single change `use_depth_embed=False` (`model/config.py` `L5-noej`): the shared block's router sees the state $h$ only, not $[h; e_j]$. 8 192 fewer parameters (the $e_j$ rows of the router). Driver 615.71.09.

**Bears on:** **E-Q2** (`01 §4.4`, Q2 in the author's answers), which is an ablation under **H6**'s
"simulate the layered model" argument (`00 §P6`, caveat 1): that argument needs the router to know
the iteration index, and the author's view is that it can infer depth from the state alone, so the
explicit $e_j$ is optional. No ADR-013 threshold names this run; it is a one-seed `screen` run read
against `110` with the silent band, plus the diagnostic `01 §4.4` asks for — the per-iteration
expert histogram.

## Numbers

| | |
|---|---|
| params | 77.47 M total, 52.90 M non-embedding (`110`: 77.48 / 52.90) |
| tokens | 1.000 B in 1907 steps |
| **final val loss** (full held-out, 5606 windows) | **3.3394** nats |
| final train loss (last step): main / total | 3.3064 / 3.3139 (router aux 0.0075) |
| throughput (median step) | 49,330 tokens/s aggregate (`step_s` median 10.628 s, p90/median 1.004) |
| peak memory per GPU | 3.86 GiB |
| wall-clock (this session) | 5.69 h |
| seed | 0 |
| routing at the last step | `load_max` 0.130 / `load_min` 0.121, `route_ent_mean` 0.9999, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 190 telemetry rows**; `temp_c_max` ≤ 79 °C, `sm_mhz_min` ≥ 1890 |

Wall-clock accounting: summed `step_s` 5.617 h + evals 0.069 h + checkpoints 0.003 h = 5.689 h of
5.693 h, **0.07 % unaccounted**. (`n_throttled` = 4 on every row is the 615.71 `0x400` bit, `030`.)
Throughput is `110`'s and `132`'s rate to 0.2 %: the depth embedding costs nothing either way.

## Loss against `110`

| comparator | loss | Δ (nats) | Δ % | vs 4σ = 0.0123 |
|---|---|---|---|---|
| `110` L5 (with $e_j$) | 3.3328 | **+0.0066** | **+0.20 %** | 0.54× → **silent** |
| `100` L0 dense | 3.3358 | +0.0036 | +0.11 % | silent |

Along the trajectory (976-window evals, same schedule) the gap to `110` is +0.007 / +0.009 / +0.008
/ +0.008 / +0.007 at steps 400 / 800 / 1200 / 1600 / 1907 — one sign throughout, about 1σ in size.
Under ADR-013 a one-seed `screen` difference inside 4σ (0.0123 with σ = 0.0031 over six `small`
pairs; 0.0140 with the σ the queue was planned under) is **silent**: the run cannot distinguish "the
depth embedding is worth ≈ 0.2 %" from seed noise. It can say the embedding is not worth *more*
than that at this scale, which is the author's expectation.

## The diagnostic: per-iteration expert histograms (`01 §4.4`)

`scripts.analysis.probe_routing` on the final checkpoints of both runs, 8 × 4 held-out sequences on
CPU, share of tokens per expert at each iteration $j$ (uniform = 0.125); `routing.json` in each run's
folder.

| | eff. experts per iteration $j$ = 0 … 7 | worst | mean corr. between iterations' histograms | top expert by $j$ |
|---|---|---|---|---|
| `136` no $e_j$ | 8.00 7.74 7.72 7.53 7.85 7.64 7.32 7.53 | 7.32 | **0.82** | 3 0 0 0 0 6 1 1 |
| `110` with $e_j$ | 7.99 7.79 7.88 7.69 7.83 7.82 7.28 7.26 | 7.26 | **0.71** | 0 5 5 5 5 5 5 5 |

Three readings. (1) **Neither router specialises experts by iteration.** In both runs the histogram
at every $j$ is close to uniform (7.3–8.0 effective experts of 8), and the histograms at different
iterations are strongly correlated (0.71–0.82): the same experts are mildly favoured at every depth.
The "early experts at small $j$, late experts at large $j$" pattern that `docs/00 §3.2` names as the
signature of the layered-model simulation is **absent with the depth embedding as much as without
it**. (2) Iteration 0 is exactly uniform in both runs and later iterations drift to a mild
preference (max share 0.18–0.27): routing sharpens with depth, but towards the same experts. (3)
Without $e_j$ the preference is *less* concentrated at the last iterations (max share 0.20 vs 0.27
at $j$ = 7), i.e. the embedding, when present, is used to lean harder on a favourite expert late, not
to partition the experts. Caveat: this is the load-balanced router of ADR-018 at $N_e$ = 8, $k$ = 2,
after 1 B tokens; with 128 experts (`112`, `138`/`139`) there is room to partition that 8 experts
do not offer, and the same probe on those checkpoints is the natural follow-up.

## Interpretation

**Silent on E-Q2's loss question** (Δ = +0.0066 nats, +0.20 %, inside the silent band at one seed);
the depth embedding is worth at most a few tenths of a percent here, consistent with the author's
"optional" (Q2). The diagnostic is the more informative half: at $N_e$ = 8 the router does not
partition experts across iterations whether or not it is told the depth, so the caveat-1 mechanism
of `00 §P6` — the shared block imitating a layered model by routing different experts at different
depths — is **not what this model does at this scale**. What it does instead (near-uniform routing
that sharpens with depth towards the same experts) is closer to a single wide MoE layer applied
eight times. That leaves H6's matched-params test (`138`/`139`) as the thing that decides whether
the shared block is competitive, and it suggests running the histogram probe on the $N_e$ = 128
checkpoints when they land. Rig: 5.7 h, zero `n_thermal`, GPU 0 ≤ 79 °C. `137-l7b-screen` started
07:48.
