# 112-l5-ne128-screen — L5 at $N_e$ = 128, $k$ = 4 (H6's parameter-matched arm) at `screen`

`python -m scripts.train.train --rung L5-ne128 --id 112-l5-ne128-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. First run of queue 4.

**Bears on:** **H6** — *one depth-conditioned shared middle block with $r_{max} n$ experts matches
$r_{max}$ stacked MoE layers with $n$ experts each, at matched params and FLOPs.* ADR-013: "matches"
means $|\Delta| \le 2\sigma$ in both directions, and at `screen` the verdict is **silent** unless
$|\Delta| > 4\sigma$. σ is unmeasured here; against the 0.005–0.01 nats `06 §4` expects at `small`,
4σ is 0.02–0.04 nats and the measured gap is 0.018. **Silent, by the rule — and as close to the line
as a silent result gets.** The comparator is `106` (L1), as `03 §1` names it. Note the parameters are
*not* actually matched: this arm has **68 % of L1's non-embedding parameters** (ADR-025 fixes
$d_{ff}$ = 448 from the FLOPs budget, and 128 experts × 448 is what fits), so it is matched-FLOPs and
under-parameterised, which makes the reading below conservative for P6.

## Numbers

| | |
|---|---|
| params | 193.1 M total, 168.5 M non-embedding (132 M in experts) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.2232** nats |
| final train loss (last step, main head) | 3.1917 |
| throughput (median step) | 16,529 tokens/s aggregate |
| peak memory per GPU | 6.01 GiB |
| wall-clock (this session) | 16.95 h |
| seed | 0 |
| $r$ | fixed 8; $N_e$ = 128, $k$ = 4, $d_{ff}$ = 448 |

## Against the ladder

| | val loss | Δ vs this run | non-emb params |
|---|---|---|---|
| `100-l0-screen` (L0 dense, $L_{ref}$) | 3.3360 | +0.1128 nats, +3.38 % | 75.5 M |
| `110-l5-screen` (L5, $N_e$ = 8) | 3.3328 | +0.1096 nats, +3.29 % | 52.9 M |
| **`112-l5-ne128-screen` (L5, $N_e$ = 128)** | **3.2232** | — | **168.5 M** |
| `106-l1-screen` (L1 layered MoE, H6's comparator) | 3.2048 | **−0.0184 nats, −0.57 %** | 246.6 M |
| `107-l2-screen` (L2 + parallel form) | 3.1495 | −0.0737 nats, −2.34 % | 246.6 M |

**Going from 8 to 128 experts in the shared block recovers 86 % of the gap to the layered MoE**
(0.128 nats of the 0.110 + 0.018), at matched FLOPs and with 68 % of L1's parameters. The remaining
0.57 % is inside the band one seed cannot resolve. P6's claim — that the repeated layer with
"128× as many experts" loses nothing against the layered stack — is **not contradicted at this
scale**, and the direction is that a parameter-matched arm (more experts still, or wider ones at a
traded $k$) would close the rest; that arm does not exist in the ladder and would cost ≈ 25–30 h.

**The shape of the curve.** L5-ne128 is *ahead* of L1 for two thirds of training and is overtaken
only at the end:

| step | `112` L5-ne128 | `106` L1 | Δ |
|---|---|---|---|
| 100 | **5.5305** | 6.2088 | −0.68 |
| 500 | **3.7629** | 3.9453 | −0.18 |
| 1000 | **3.3959** | 3.4184 | −0.02 |
| 1300 | 3.3027 | 3.3031 | 0.00 |
| 1600 | 3.2479 | **3.2358** | +0.012 |
| 1907 | 3.2232 | **3.2048** | +0.018 |

Same pattern as `111` vs `110`: the shared block learns faster early — one weight set seen eight
times per token gets more gradient signal per parameter — and the larger, deeper-in-parameters
layered stack pulls ahead as the budget runs out. That crossing at ≈ 1300 steps is exactly why
`03 §2`'s recurrence caveat says H6 may only resolve with scale, in *either* direction: at 2.5 B
tokens the gap may widen (the layered model keeps using its extra 78 M parameters) or the shared
block's faster learning may hold. `small` decides it; a token-budget crossing is the one thing a
single `screen` run cannot extrapolate.

**Against L2, not just L1.** The ladder's best layered rung is L2 at 3.1495, 2.3 % ahead of this run.
L5 has the parallel form by construction (`110`), so that gain is already inside the shared block; the
2.3 % is the layered stack's parameter advantage plus whatever the shared attention costs, and it is
the number the recurrent branch has to earn back at `small` and `medium`.

## Routing — 128 experts through one shared router

The startup transient is the largest the ladder has seen, and it recovers completely:

| step | eff. experts (of 128) | dead | busiest expert / uniform |
|---|---|---|---|
| 23 | — | **90** | — |
| 25 | **15.1** | — | — |
| 101 | 94.8 | 14 | 3.2× |
| 500 | 121.3 | 0 | 2.6× |
| 954 | 125.8 | 0 | 1.8× |
| 1400 | 127.3 | 0 | 1.3× |
| 1907 | **127.7** | 0 | **1.18×** |

Ninety of 128 experts dead at step 23, with the router effectively using 15, is the self-reinforcing
starvation of `108` at sixteen times the pool size. The aux-loss-free bias controller pulls every one
of them back by step 400, and the endpoint (1.18× uniform for the busiest expert) is the same figure
`106` reached with 8 experts per router. The controller's 1e-3 rate is the *only* balancing knob that
was not retuned for the larger pool, and it did not need to be. Not a confound — but the transient
costs real tokens (the model runs on 15–95 effective experts for its first 100 steps), and a warm-start
or higher initial rate is a cheap ablation if a later rung shows the pool size hurting early.

## Run health

| | |
|---|---|
| wall-clock | 16.95 h; summed `step_s` 16.815 h + evals 0.112 h + checkpoints 0.019 h → **0.0 % unaccounted** |
| throughput | 16.5 k tok/s median against `018`'s 14.5 k without accumulation (+14 %); 16.95 h against the 19.2 h projection |
| thermal | **0 of 190 records flagged**, max 73 °C, min clock 1897 MHz |

Clean. The 128-expert rung is memory-bandwidth-light per card (many small GEMMs, 6.0 GiB peak), and
GPU 0 never approached its limit.

## Interpretation

At matched FLOPs and 68 % of the layered MoE's parameters, the shared middle block with 128 experts
comes within 0.57 % of L1 on held-out loss, recovering 86 % of what the 8-expert pool left on the
table, and leads L1 for the first two thirds of training before being overtaken. Under ADR-013 the
H6 verdict at `screen` is **silent** — 0.018 nats is inside 4σ of the expected seed spread — and the
crossing at step ≈ 1300 is the reason it cannot be extrapolated: `small` at 2 seeds is where H6 is
decided, and it may go either way with budget. What is settled is that P6's premise is not wrong at
this scale, that the recurrent branch's remaining deficit is against L2's parallel-form layered stack
at 2.3 %, and that the bias controller balances a 128-expert shared pool from a 90-dead start without
retuning.
