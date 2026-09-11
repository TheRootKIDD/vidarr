# 108-l3-screen — L3 at `screen`

`python -m scripts.train.train --rung L3 --id 108-l3-screen --tokens 1e+09 --micro-batch 2 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (32 × 2 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** **H13**, both clauses, **as evidence and explicitly not as a verdict** — ADR-030 (taken
2026-09-11 while this run was mid-flight, before its outcome was known) moves H13's decision to
`small`, because H13's own prior says 1 B tokens is too small a budget to interpret an MTP failure at.

## Numbers

| | |
|---|---|
| params | 272.9 M total, 248.3 M non-embedding |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.2486** nats |
| final train loss (last step, main head) | 3.2179 |
| throughput (last step) | 43,275 tokens/s aggregate |
| peak memory per GPU | 8.83 GiB |
| wall-clock (this session) | 6.51 h |
| seed | 0 |
| val_mtp_acc_h2 | 0.2149 |
| val_mtp_acc_h3 | 0.1264 |
| val_mtp_acc_h4 | 0.0890 |

## H13, both clauses

L3 is L2 plus four independent MTP heads (ADR-012: heads read the same final hidden state, one
subsampled auxiliary loss per position). Parameters differ only by the head adapters — 272.9 M vs
271.1 M.

| clause | H13 requires | measured |
|---|---|---|
| acceptance, head 2 | ≥ 70 % | **21.5 %** |
| acceptance, head 3 | ≥ 55 % | **12.6 %** |
| acceptance, head 4 | ≥ 45 % | **8.9 %** |
| main-loss regression vs L2 | $\Delta \le 2\sigma$ | **+0.0991 nats, +3.15 %** |

Both clauses miss, and not narrowly: head 2 reaches under a third of its target.

**This is what I20 predicted, in advance and in writing.** The Phase-0 literature refresh recorded
*"MTP hurts main quality below ≈ 1 B without a curriculum; expect a regression at 80 M"* as one of
four claims of the note expected to fail at our scale. It failed in the predicted direction, at the
predicted budget, by roughly the predicted mechanism. A pre-registered prediction coming true is a
better outcome than a surprise, and it is why ADR-030 exists.

**Acceptance was still climbing when the run ended, but slowly and decelerating**: 16.1 % at step 500,
18.4 % at 800, 19.9 % at 1100, 20.7 % at 1400, 21.5 % at 1907. The increment per 100 steps fell from
≈ 0.8 to ≈ 0.2. Extrapolating that curve does not reach 70 % at any budget this rig can afford; it is
not a "needs a few more tokens" shape. **If H13 is recoverable it is recoverable by a change of
method, not of budget** — which is exactly the question `small` (ADR-030) is scheduled to separate,
and which bears directly on which L3b gets built (the ADR-012 sequential-modules reading, or the I20
curriculum reading — still unresolved, see ADR-030).

**The main-loss regression is the cleaner signal.** +3.15 % against L2 at matched FLOPs, with the gap
narrowing in absolute terms through training (0.136 nats at step 100 → 0.099 at 1907) but nowhere near
closing. ADR-013 wants $\Delta \le 2\sigma$ and σ is unmeasured at `screen`; 0.099 nats is ≈ 10–20× the
`small` seed spread, so a seed artefact is not a plausible explanation even without the band.

## Against the ladder so far

| | val loss | Δ vs L0 |
|---|---|---|
| `100-l0-screen` (L0 dense, $L_{ref}$) | 3.3360 | — |
| `106-l1-screen` (L1 MoE) | 3.2048 | −3.93 % |
| `107-l2-screen` (L2 parallel) | **3.1495** | **−5.59 %** |
| `108-l3-screen` (L3 + MTP) | 3.2486 | −2.62 % |

**L3 gives back more than L1 gained.** The ladder is cumulative, so this matters: on held-out loss
alone, adding MTP moves the model back past L1 to within 2.6 % of the dense baseline. MTP is not
*supposed* to pay for itself in main-head loss — it buys speculative decode, which is a latency
property this tier cannot measure — but at 21.5 % acceptance it is not buying much of that either.

## Run health and routing

43.3 k tok/s, 6.51 h wall, **0.06 % unaccounted** — the third consecutive run with clean accounting
since I22 closed. Zero sustained thermal throttling; flags were single samples, mostly on eval steps.

**First rung with the routing trajectory** (added to the trainer the morning this run started). It
earns its place immediately: L3 dives to **35 dead experts of 96** and one router at **2.2 of 8
effective experts** around step 19, recovers to 1 dead by step 101, and finishes at **7.99 of 8, 0
dead**. That is the self-reinforcing MoE starvation being pulled back by the aux-loss-free bias
controller, watched rather than inferred. `106`'s endpoint said where it lands; this says how it gets
there.

## Interpretation

H13 misses both clauses at `screen` by a wide margin, in the direction and at the budget I20
predicted. **No verdict is recorded** (ADR-030). The acceptance curve's shape is the finding worth
carrying forward: it is decelerating, not budget-limited, which makes "more tokens will fix it" the
*less* likely of the two hypotheses `small` will separate — and that in turn favours the curriculum
reading of L3b over the sequential-modules reading. Routing is healthy throughout and is not a
confound.


## Correction (same day) — acceptance was measured against the wrong reference

**The acceptance numbers above are not the quantity H13's thresholds are written against.** ADR-013
says *"greedy acceptance of heads 2/3/4 **against the main head** on held-out"*, and ADR-012 repeats
it: *"measured greedily against head 1"*. `model/mtp.py` computes `logits.argmax(-1) == target` where
`target` is the **ground-truth token** — that is top-1 accuracy, not acceptance. CLAUDE.md: where code
and doc disagree, the doc wins. Recorded as **I27**.

The two differ because a draft is accepted when it **agrees with the verifier**, not when it is
**right**. Re-measured from this run's checkpoint with `scripts/analysis/probe_mtp_acceptance.py`
(32 held-out sequences, CPU):

| head | **acceptance vs head 1** | top-1 vs truth (what was logged) | H13 target |
|---|---|---|---|
| 2 | **38.5 %** | 21.1 % | 70 % |
| 3 | **21.6 %** | 12.0 % | 55 % |
| 4 | **15.0 %** | 8.4 % | 45 % |

Main head's own top-1 vs truth: **40.0 %**.

**H13 still misses, and the direction of the conclusion is unchanged** — 38.5 % against a 70 % target
is not a near miss. But the margin is roughly half what the trainer reported, and the *shape* of the
problem changes: head 2 is not failing to model the data, it is failing to anticipate a
better-informed head 1. Head 1 predicts token $t+j$ from position $t+j-1$; head $j$ must predict the
same token from position $t$, with $j-1$ fewer tokens of context. Acceptance measures how well head
$j$ closes that information gap, and 38.5 % is real skill rather than noise.

**What this does not license.** It is tempting to read head 1's 40 % top-1 as a ceiling on acceptance.
It is not — head $j$ only has to guess what head 1 will *say*, not be correct, so the ceiling is
100 % and the binding constraint is the context gap, not head 1's quality. That misreading was made
once while interpreting this result and is recorded so it is not made again.

**The subsampled auxiliary loss is now the leading suspect** (ADR-012): each position trains exactly
one of heads 2–4, so head 2 sees ≈ 1/3 of positions — ≈ 0.33 B tokens of the 1 B budget — and head 4
sees 1/3 of positions at 1/8 the loss weight. That was a deliberate speed choice (+25 % step instead
of +75 %). **It may also be the whole result.** `L3-full` tests it directly; see `04` I27.
