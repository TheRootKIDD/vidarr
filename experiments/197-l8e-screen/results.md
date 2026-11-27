# 197-l8e-screen — L8e, a hashed 2/3-gram engram table of 2²¹ × $d$ in host RAM on L5 (ADR-071, run under ADR-083's waiver): 3.2556 nats, **−0.0772 vs `110` (26σ → H24's first clause *supported* at `screen`, one seed)**; the table closes 60 % of the L5–L1 gap at 1.61 B extra parameters and 0.63× L5's throughput

`python -m scripts.train.train --rung L8e --id 197-l8e-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml` (the L5 preset with `engram.enabled: true`, `rows: 2097152`, `ngrams: [2, 3]`, `placement: host`), per-step log in `log.jsonl`. Code at the `queue_screen_27` gate commit `0f1e85a`. Run 2026-11-26 20:45 – 2026-11-27 05:56.

**Bears on:** **H24**, first clause ("lowers held-out loss vs L5 by more than 4σ at matched active
FLOPs; table parameters reported, not matched"), by **ADR-071 §6** under ADR-066 §1. Run under
**ADR-083**: rig `052` failed ADR-071 §4's step-time gate (1.60× against 1.25×) and the gate was
waived for this run only; the overhead is reported below and does not enter the verdict. Systems
side: P16 placement data. H17: no verdict is read from this run (see the wiki-lead section).

| rule (ADR-071 §6) | measured | reading |
|---|---|---|
| Δ(`197` − `110`) < −4σ = −0.0119 supports H24's first clause at `screen` | **−0.0772** (−2.32 %, 26.1σ) | **supports** |
| within 4σ silent; worse by > 4σ weakens | — | — |
| consequence: a `small` pair (ADR-031), and ADR-065's second test (L8e + L8b-pv vs `178`/`179`), each by its own ADR | ADR-083 §3: no pair until the host update is cheaper and rig `052` is repeated | nothing queued; the user's call (I46) |

σ = 0.0030 (2σ 0.0059, 4σ 0.0119), ADR-066 §1; σ multiples use the unrounded 0.00296.

## Numbers

| | |
|---|---|
| model params | 78.07 M total, 53.49 M non-embedding (`110`: 77.48 M / 52.90 M; the 0.59 M is $W_e$, $d \times d$) |
| **engram table params (host, not in the count above)** | **1.61 B** (2²¹ × 768), ≈ 20× the model |
| active FLOPs | L5's, plus ≈ 1.2 MFLOP/token for $W_e e_t$ (0.2 % of the 600 MFLOP budget; ADR-071 §2) |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.2556** nats (`110` 3.3328, L1 `106` 3.2048) |
| final train loss (last step, main head) | 3.2236 (`110` 3.3016) |
| throughput (median step) | 30,567 tokens/s aggregate, **0.630× `110`'s** 48,553 |
| peak memory per GPU | 4.62 GiB (`110` 3.86) |
| peak host RAM (used, sampled each minute) | **59,943 MiB** (58.5 GiB); median 40,918 MiB; shared (the table and its optimiser state) peak 23,647 MiB |
| wall-clock (this session) | 9.18 h (`110` 5.79 h) |
| seed | 0 |
| $r$ | fixed 8 |
| routing at the end | entropy 0.9999 of $\ln 8$, 8.00 effective experts, 0 dead |

Of the L5–L1 gap at `screen` (3.3328 − 3.2048 = 0.1280), the table closes 0.0772, 60 %.

**Curve** (976-window evals; last row full held-out).

| step | `110` | `197` | `197` − `110` |
|---|---|---|---|
| 100 | 5.6159 | 5.4445 | −0.1714 |
| 300 | 4.3664 | 4.2236 | −0.1427 |
| 500 | 3.8804 | 3.7722 | −0.1082 |
| 700 | 3.6601 | 3.5656 | −0.0944 |
| 1000 | 3.4995 | 3.4123 | −0.0872 |
| 1300 | 3.4083 | 3.3269 | −0.0814 |
| 1500 | 3.3697 | 3.2904 | −0.0793 |
| 1800 | 3.3356 | 3.2582 | −0.0773 |
| 1900 | 3.3288 | 3.2519 | −0.0769 |
| 1907 (full) | 3.3328 | 3.2556 | −0.0772 |

**Train loss** (`197` − `110`, main head, same seed and data order, step-matched means): −0.150
over steps 100–500, −0.095 over 500–1000, −0.083 over 1000–1500, −0.079 over 1500–1907, −0.078
over the last ten steps. The lead shrinks with tokens and is still shrinking at the end, by
about 0.002 per 400 steps over the last 800.

## Host table: cost per step (ADR-083 §5)

| | per step (median) | share of the step |
|---|---|---|
| step | 17.15 s (`110`: 10.80 s → **1.59×**, as rig `052`'s 1.60×) | |
| lazy sparse Adam on the host (`engram_update_s`) | 4.08 s (3.94–4.26) | **23.8 %** |
| gradient scatter (`engram_scatter_s`) | 0.92 s (0.70–0.97) | **5.4 %** |
| remainder of the overhead (gather, injection, the rest) | ≈ 1.35 s | ≈ 8 %, not separately logged |
| rows touched per step | 547,774 (540,200–554,501), 26.1 % of the table | |

Shares are sums over all 1907 steps. Gather time is not logged by the code as built and stays
unmeasured (ADR-083 §5). Host RAM: the shared segment holds at 23.1 GiB (ADR-071 predicted
≈ 22.5 GB); "used" sits near 40 GiB and rises to 55–60 GiB in 86 of 551 one-minute samples,
with 197 GiB available at the worst sample.

## Wiki-lead eval (ADR-071 §5; `wiki_lead.json`, `wiki_lead_compare_vs_110.json`, `110`'s re-score in `experiments/110-l5-screen/wiki_lead_covered.json`)

4,000 articles, 256 tokens each; floor = 4 × max(0.0030, bootstrap SE).

| split | `110` | `197` | Δ | floor | |
|---|---|---|---|---|---|
| in-table first sentence | 3.1307 | 3.0327 | −0.0980 | 0.0166 | beats |
| in-table first sentence, covered tokens | 3.1963 | 3.0887 | −0.1076 | 0.0184 | beats |
| in-table rest of lead | 3.4080 | 3.3203 | −0.0877 | 0.0144 | beats |
| out-of-table lead (control) | 3.2613 | 3.1717 | −0.0896 | 0.0120 | beats |
| held-out (976 windows) | 3.3284 | 3.2512 | −0.0772 | | |

The gain is the same size on text inside and outside the retrieval table's articles (covered
first sentence − control = −0.018, at the floor of that difference). L8e has no retrieval
table: "in-table" here only names the ADR-053 pools. So the engram's gain is general, not
knowledge of those articles. The compare tool prints `verdict_h17: supports`; that is H17's
rule applied mechanically to an arm that beats every split including the control, and **no H17
verdict is taken from it**. H17's result stays ADR-075's.

## Run health

| | |
|---|---|
| wall-clock | 9.18 h; summed `step_s` 9.086 h + evals 0.074 h + checkpoints 0.014 h → **0.05 % unaccounted** |
| throughput | 30.6 k tok/s median, flat (first and last 300 steps agree to 0.2 %) |
| thermal | **0 of 190 records with `n_thermal` > 0**, max 69 °C, min clock 1792 MHz |

## Limits

One seed. The comparison is at matched tokens and active FLOPs, not parameters: the arm holds
1.61 B more of them, and 26 % of the rows are written every step. No parameter-matched control
exists (H24's original comparator, a product-key memory of the same size, closed with L8,
ADR-057/065). The lead is shrinking with tokens (−0.171 at step 100, −0.087 at 1000, −0.077 at
the end), so the `screen` number does not predict the `small` one; L8b's `screen` parity
turned into a deficit at 2.5 B (`176`/`177`), and here the question is how much of 0.077
survives. At `small` the L5–L1 gap is 0.1909.

## Interpretation

Adding a hashed 2/3-gram table read once per token to L5 lowers held-out loss by 0.0772 nats
at 1 B tokens, 26σ against the 4σ line ADR-071 §6 sets, so H24's first clause is supported at
`screen` on one seed. Among the arms that keep L5's eight experts it is the largest gain at
`screen` (L5c −0.056, L5e −0.038, L5f −0.018; the 128-expert L5-ne128 is −0.110), and it takes
L5 from tying dense L0 to within 0.051 of L1. The gain is general: it is the same size on Wikipedia leads inside and outside the L8
retrieval pools and on FineWeb held-out, which is what a table of local token statistics
would give and what a table of facts would not. Two things qualify it. The lead halves over
the run and is still shrinking, as expected of a memory for short-range statistics that the
shared block learns more slowly; and the table is 20× the model's parameters. On the systems
side the host placement works without staleness at 23.1 GiB of shared memory, but as built it
costs 59 % more wall-clock per step, of which the single-process sparse Adam is 4.1 s
(23.8 % of the step) and the scatter 0.9 s (5.4 %). By ADR-071 §6 and ADR-031 this result
calls for a `small` pair; ADR-083 §3 holds that pair until the host update is made cheaper and
rig `052` is repeated, by its own ADR. That, and ADR-065's second test (L8e with L8b-pv), are
the user's call (I46); nothing is queued.
