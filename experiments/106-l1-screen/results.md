# 106-l1-screen — L1 at `screen`

`python -m scripts.train.train --rung L1 --id 106-l1-screen --tokens 1e+09 --micro-batch 2 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (32 × 2 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`.

**Bears on:** H4 — as its *baseline arm*, not its test. `docs/03 §2` maps L1 to H4, but H4 compares
coarse unit-sized experts ($g$ = 1) against $g$ = $U$ sub-experts and is decided at L7; L1 supplies the
fine-grained comparator that L7 will be measured against. **On H4 itself this run is silent.** What it
does deliver is the second baseline CLAUDE.md requires of every experiment — the matched-FLOPs
fine-grained MoE — alongside `100-l0-screen`'s matched-FLOPs dense. It also **closes I22**; see below.

## Numbers

| | |
|---|---|
| params | 271.2 M total, 246.6 M non-embedding |
| tokens | 1.000 B in 1907 steps |
| final val loss (full held-out, 5606 windows) | **3.2048** nats |
| final train loss (last step, main head) | 3.1732 |
| throughput (last step) | 50,142 tokens/s aggregate |
| peak memory per GPU | 8.72 GiB |
| wall-clock (this session) | 5.62 h |
| seed | 0 |

## Against the dense baseline

| | val loss | params (total / non-emb) |
|---|---|---|
| `100-l0-screen` (L0 dense, $L_{ref}$) | 3.3360 | 100.1 M / 75.5 M |
| **`106-l1-screen` (L1 MoE)** | **3.2048** | 271.2 M / 246.6 M |
| Δ | **−0.1312 nats, −3.93 %** | |

Both at 1 B tokens, seed 0, matched FLOPs per token: L1 activates top-2 of 8 experts, so its 271 M
parameters cost the same per token as L0's 100 M. That is the intended shape of the comparison.

**Verdict withheld, deliberately.** ADR-013 states thresholds as fractions of the baseline loss inside
a measured-σ band, and takes σ from the L0 seed pair — **which exists only at `small`. At `screen` we
have one seed of each and σ is unmeasured**, so there is no band to test against and this is a
direction, not a verdict. What can be said: 3.93 % is 13–26× the 0.005–0.01 nats of seed-to-seed
spread `06 §4` records at `small`, so it is very unlikely to be noise, and the ladder's own premise —
that a fine-grained MoE beats a matched-FLOPs dense model — is not in doubt in the literature
(`docs/05`). The banded verdict comes at `small`, where L0 and L1 both run at 2 seeds.

## I22 closes: the wall-clock gap was thermal throttling, and it is gone

I22 was opened because `100-l0-screen` spent **18.0 h of wall-clock on 4.28 h of timed steps** and the
residual could not be attributed — evaluation, checkpointing to `/home`, the per-step barrier or the
host were all suspects. `106` ran the **same 1907 steps of a more expensive rung**:

| | `100-l0-screen` (L0) | **`106-l1-screen` (L1)** |
|---|---|---|
| steps | 1907 | 1907 |
| wall-clock | **18.0 h** | **5.62 h** |
| sum of `step_s` | — | 5.540 h |
| evals + checkpoints | — | 0.079 h |
| unaccounted | ≈ 76 % | **0.1 %** |

Every hour is now accounted for, on a rung whose micro-step is heavier than L0's. **None of I22's
suspects was the cause** — not `/home`, not the evals, not the barrier. It was I23's throttling, as
I23 claimed when it was opened *(closes I22)*, and `024`'s layout fix removed it. Checkpoints were on
`/mnt/nvme` for this run (queue script, checklist step 4), so `/home` is exonerated only by
implication; the 0.1 % residual leaves no room for it to have mattered.

## Interpretation

The MoE rung trains 3.93 % better than the matched-FLOPs dense baseline at 1 B tokens on one seed, and
the run cost 5.62 h against the 9.6 h `018` projects without gradient accumulation — accumulation is
worth 1.7× here, as expected for a 271 M rung whose all-reduce is ≈ 40 % of a micro-step (`026`, I21).
Throughput held at **50.2 k tokens/s** for the whole run with **zero thermal-slowdown samples**; the
only thermal flags were single samples on eval steps, where the held-out pass briefly pushes GPU 0 to
83 °C, with no clock or throughput effect. This is the first `screen` rung on this rig whose
throughput number is valid.

## Addendum — per-router expert load (`routing.json`)

Raised as a question by Annemette: how many experts does the MoE baseline have, does it collapse onto
one, and is anything balancing it? Measured after the fact with
`scripts/analysis/probe_routing.py --id 106-l1-screen`, which rebuilds the final checkpoint and runs 32
held-out sequences of 2048 tokens through it on **CPU**, so the ladder's GPUs and its throughput
numbers are untouched.

**8 experts per router, top-2, 12 routers.** Effective expert count is exp(H) of the load distribution
— 8.00 is perfectly uniform, 1.00 is total collapse.

| router | eff. experts | H/H_max | max share | min share | dead |
|---|---|---|---|---|---|
| 0–7 | **8.00** | 1.000 | 0.129–0.131 | 0.117–0.122 | 0 |
| 8–10 | **7.99** | 1.000 | 0.130–0.134 | 0.115–0.118 | 0 |
| 11 | **8.00** | 1.000 | 0.130 | 0.118 | 0 |

Uniform share is 0.1250. **No router is below 7.99 of 8 effective experts and no expert anywhere is
dead** (below 10 % of its uniform share). The worst single expert in the whole model carries 0.1341,
i.e. 7 % above uniform.

**The balancing is DeepSeek-V3-style, two mechanisms.** The primary one is an *aux-loss-free* bias
controller: each expert has a bias used **only for selection**, while the gates that weight the chosen
experts are a softmax of the **bias-free** logits. After each step the controller moves each bias by
`sign(load − target)` at rate 1e-3, so balancing never competes with the language-modelling gradient.
Behind it sits a small Switch-style sequence-level auxiliary loss at coefficient 1e-3, which logs at
0.0122–0.0159 across the run — present, gentle, not dominating.

**Convergence from the trainer's own log**, worst-to-best expert ratio over all 12 routers at once:

| step | load_min | load_max | ratio |
|---|---|---|---|
| 1 | 0.0201 | 0.2717 | 13.5 |
| 477 | 0.0533 | 0.2196 | 4.1 |
| 954 | 0.0884 | 0.1717 | 1.9 |
| 1907 | 0.1154 | 0.1326 | **1.15** |

The extremes in the raw log — one expert at 0.50, another at 0.00 — are confined to the **first 251 of
1907 steps**, the startup transient before the controller has moved its biases. Over the last 500 steps
the worst expert anywhere stays within 0.130–0.157 and the least-loaded within 0.101–0.121.

**Instrumentation gap this exposed, now fixed.** `load_max`/`load_min` are extremes over *every* router
at once, so they cannot distinguish one collapsed layer from mild spread everywhere: a model with 11
uniform routers and 1 fully collapsed router has a mean normalised entropy of 0.92, which looks fine.
The trainer now also logs `route_ent_min`, `route_eff_min`, `route_ent_mean` and `route_dead` per step,
where the *minimum* over routers is the alarm. Added 2026-09-11, so **`106` and `107` do not have the
trajectory** — only this endpoint, recovered from the checkpoint. `108` onward have both.
