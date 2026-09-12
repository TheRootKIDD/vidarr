# 111-l5d-screen — L5d at `screen`

`python -m scripts.train.train --rung L5d --id 111-l5d-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** **H15a** — *global attention over final vectors only, with per-iteration attention kept
local ($W$ = 512), costs ≤ 2 % loss vs global per-iteration attention at 2 k–8 k context, dense, no
index.* This is the quality half of v2's P7 (ADR-021/022) and the single change against `110`: the
global range moves from every iteration's own cache to the final-vector cache $\mathcal{G}$, with
segment-recurrent training and stop-gradient memory (ADR-023). Measured at the 2 k training context
only; the 8 k end of H15a's range is an evaluation still owed. **No verdict** (ADR-013: σ unmeasured
at `screen`; L5d is on the `small` list).

## Numbers

| | |
|---|---|
| params | 77.5 M total, 52.9 M non-embedding (identical to `110` — the change is structural) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.3477** nats |
| final train loss (last step, main head) | 3.3165 |
| throughput (median step) | 33,815 tokens/s aggregate |
| peak memory per GPU | 3.89 GiB |
| wall-clock (this session) | 8.31 h |
| seed | 0 |
| $r$ | fixed 8 |

## Against `110` and the ladder

| | val loss | Δ vs `110` |
|---|---|---|
| `100-l0-screen` (L0 dense, $L_{ref}$) | 3.3360 | +0.10 % |
| `110-l5-screen` (L5, global per-iteration attention) | 3.3328 | — |
| **`111-l5d-screen` (L5d, global over $\mathcal{G}$ only)** | **3.3477** | **+0.0149 nats, +0.45 %** |

**H15a budgets ≤ 2 % and the measured cost is 0.45 %**, less than a quarter of the allowance. It is
also inside the range where one seed cannot separate it from noise with confidence: 0.015 nats is
1.5–3× the 0.005–0.01 nats seed spread `06 §4` records at `small`, so the sign is probably real and
the size is not yet. Against the dense baseline L5d is +0.35 %: the recurrent design keeps its
"matches dense at 70 % of the parameters" property with the far cheaper attention structure.

**The shape of the curve is the interesting part.** L5d is *ahead* of `110` for the first quarter of
training and falls behind only late:

| step | `110` (per-iteration global) | `111` (final-vector global) | Δ |
|---|---|---|---|
| 100 | 5.6159 | **5.5506** | −0.065 |
| 300 | 4.3664 | **4.3252** | −0.041 |
| 500 | 3.8804 | **3.8741** | −0.006 |
| 700 | 3.6601 | 3.6708 | +0.011 |
| 1000 | 3.4995 | 3.5130 | +0.014 |
| 1907 | 3.3328 | 3.3477 | +0.015 |

One entry per token in a single global cache is an easier attention problem to learn than eight
per-iteration caches, so the model gets going faster; what it gives up is the far tokens' intermediate
iterations, and that shows up as a small constant gap once both have converged. That is the trade
`03 §6` names as the risk of final-vector-only global attention, and here it is worth 0.015 nats at
1 B tokens. Whether the gap stays constant, closes, or grows with tokens is what `small` will show.

**What this buys, for the systems side.** The persistent cache is one $\mathcal{G}$ entry per token
instead of $r$ = 8 per-iteration caches — the ~8× KV reduction `02 §7` and the note's P7 and P16
rest on — for 0.45 % of loss at this scale. `costmodel/memory.py`'s `kv_reduction_factor` has the
exact figure per config.

## Routing

Same shared router as `110`: worst point 4.69 of 8 effective experts at step 19, two dead experts at
step 12, **8.00 of 8 with loads 0.119–0.129 at the end** — the same endpoint as `110`. Not a confound.

## Run health — the cleanest run of the ladder

| | |
|---|---|
| wall-clock | 8.31 h; summed `step_s` 8.217 h + evals 0.082 h + checkpoints 0.004 h → **0.05 % unaccounted** |
| throughput | 33.8 k tok/s median against `018`'s 31.9 k without accumulation (+6 %) |
| thermal | **0 of 190 records flagged**, max 81 °C, min clock 1882 MHz |

GPU 0, which ran `110` trimmed for the whole rung, held under its limit here. L5d's longer step is
the same card load with more time in attention over $\mathcal{G}$ and less in expert GEMMs, and the run
also overlapped the coolest part of the night. The throughput number is clean.

## Interpretation

Switching the global range from per-iteration caches to the final-vector cache costs 0.45 % of
held-out loss at 1 B tokens and 2 k context, against H15a's 2 % allowance, and the model trains
*faster* early with the simpler cache before settling into a small constant gap. No verdict at one
seed; the direction supports H15a and the banded test is L5d's at `small`. The 8 k-context half of
H15a is untested here and needs the context-extension evaluation `03 §7` describes. This is the rung
L6 and everything after it builds on, and it costs nothing in parameters or memory to adopt.
