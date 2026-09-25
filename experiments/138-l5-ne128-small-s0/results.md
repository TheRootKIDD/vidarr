# 138-l5-ne128-small-s0 — H6's parameter-matched arm at `small`, seed 0: 2.9588 nats, **+2.60 % vs the L1 pair** — outside the "matches" band by 12×; beats dense by 1.24 % at 68 % of L1's params (one seed; `139` completes the pair)

`python -m scripts.train.train --rung L5-ne128 --id 138-l5-ne128-small-s0 --tokens 2.5e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. First run of `queue_small_4.sh`. Shared middle block at $r$ = 8 with **$N_e$ = 128, $k$ = 4, $d_{ff}$ = 448** (ADR-025): the same eight-times-applied block as L5 but with sixteen times the experts; `112` is its `screen`. Driver 615.71.09.

**Bears on:** **H6** — *one depth-conditioned shared middle block with $r_{max} n$ experts matches
$r_{max}$ stacked MoE layers with $n$ experts each, at matched params and FLOPs.* This is I29's arm:
FLOPs match L1's, and the parameters are **168.5 M non-embedding against L1's 246.6 M (68 %)** — the
closest the ladder has to matched, not exact, and the reading below carries that caveat as I29
requires. ADR-013: "matches" = $|Δ| \le$ 2σ against the L1 pair, *weakens* above 4σ. One seed; the
pair verdict waits on `139`.

## Numbers

| | |
|---|---|
| params | 193.1 M total, **168.5 M non-embedding** (L1 pair: 246.6 M; L5: 52.9 M; L0: 75.5 M) |
| tokens | 2.500 B in 4768 steps |
| **final val loss** (full held-out, 5606 windows) | **2.9588** nats |
| final train loss (last step): main / total | 2.9622 / 2.9694 (router aux 0.0073) |
| throughput (median step) | 16,381 tokens/s aggregate (`step_s` median 32.006 s, p90/median 1.005) — a third of L5's; the reference implementation loops over 128 experts |
| peak memory per GPU | 6.01 GiB |
| wall-clock (this session) | 42.70 h — the longest run of the programme |
| seed | 0 |
| depth | fixed $r$ = 8 |
| routing at the last step (trainer) | `load_max` 0.0088 / `load_min` 0.0062 (uniform 0.0078), `route_ent_mean` 0.9996, `route_eff_min` 127.8 of 128, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 67 °C, `sm_mhz_min` ≥ 1912; `n_power_cap` 1 row — the coolest long run yet, the cards are under-occupied by the expert loop |

Wall-clock accounting: summed `step_s` 42.400 h + evals 0.240 h + checkpoints 0.047 h = 42.687 h of
42.697 h, **0.02 % unaccounted**. (`n_throttled` = 4 on every row is the 615.71 `0x400` bit, `030`.)

## Against the comparators

| comparator | non-emb. params | loss | Δ (nats) | Δ % | vs 2σ = 0.0061 / 4σ = 0.0123 |
|---|---|---|---|---|---|
| **L1 pair** (`130`/`119`), H6's comparator | 246.6 M | 2.8839 | **+0.0749** | **+2.60 %** | 12× / 6× — outside "matches", above the weakening line |
| L2 pair (`120`/`121`) | 246.5 M | 2.8851 | +0.0737 | +2.55 % | |
| L0 pair (`116`/`117`), dense | 75.5 M | 2.9958 | **−0.0370** | **−1.24 %** | 3× the 4σ band |
| L5 pair (`132`/`133`), $N_e$ = 8 | 52.9 M | 3.0748 | −0.1160 | −3.77 % | |
| at `screen`: `112` vs `106` | | 3.2232 vs 3.2048 | +0.0184 | +0.57 % | 1.5× the 4σ band |

**The `screen` reading did not survive the budget, in the same direction as L5's.** At 1 B tokens
the matched arm was 0.57 % behind L1 and its curve had crossed L1's at ≈ step 1300; at 2.5 B it is
2.60 % behind, and the trajectory against the L1 pair shows the same shape as L5's against dense:

| step (tokens) | 500 (0.26 B) | 1000 (0.52 B) | 1900 (1.0 B) | 2900 (1.5 B) | 3800 (2.0 B) | 4700 (2.46 B) |
|---|---|---|---|---|---|---|
| L5-ne128 − L1 pair (nats) | −0.095 | +0.021 | +0.055 | +0.068 | +0.073 | +0.075 |
| L5-ne128 − L0 pair (nats) | −0.228 | −0.087 | −0.050 | −0.040 | −0.038 | −0.037 |
| L5-ne128 − L5 pair (nats) | −0.120 | −0.099 | −0.104 | −0.110 | −0.114 | −0.116 |

The shared block leads early (−0.095 at 0.26 B tokens), crosses at ≈ 0.4 B, and then trails by a
gap that widens and flattens near +0.075 — the third time this shape has appeared (L5 vs L0, `132`;
L2's "improves" at `screen`, `120`). Against dense the arm's lead shrinks from −0.23 to −0.04 nats
over the run but holds; against L5 the sixteen-fold expert count is worth a stable −0.11 nats.

**Reading under ADR-013 (one seed, provisional):** Δ = +0.0749 against the L1 pair is 12× the 2σ
"matches" band and 6× the 4σ weakening line, so **the arm does not match L1** — and no plausible
seed-1 value (the largest `small` seed gap is 0.006) can bring the pair inside 0.0061. **The verdict
the pair will formalise is *weakens H6*, with I29's caveat that the arm has 68 % of L1's
parameters.** The caveat matters for the reading, not the verdict: a 32 % parameter deficit is a
plausible source of a 2.6 % loss gap (the L5 arm at 21 % of L1's params trails by 6.6 %; the dense L0
at 31 % trails by 3.9 %), so what this run refutes is the *strong* form of H6 as `03 §1` states it
against the FLOPs-matched preset, and what it leaves open is whether an exactly parameter-matched
shared block (I29's second option: $d_{ff}$ raised so 128 × $d_{ff}$ × 3$d$ + attention = 246.6 M,
at ≈ 1.45× L1's FLOPs) would close the gap. That is a different preset and a different hypothesis
— matched params at unmatched FLOPs.

## Routing at 128 experts — the histogram probe (`01 §4.4`)

`scripts.analysis.probe_routing` on the final checkpoint, 8 × 4 held-out sequences on CPU
(`routing.json`). At $N_e$ = 8 (`110`, `136`) the router did not partition experts by iteration:
histograms near-uniform at every $j$ and correlated 0.71–0.82 across iterations. **At $N_e$ = 128 it
does, partially:**

| iteration $j$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| effective experts (of 128) | 127.6 | 108.7 | 111.7 | 97.5 | 103.2 | 94.7 | 86.8 | 89.1 |
| experts below 10 % of uniform at this $j$ | 0 | 1 | 1 | 3 | 0 | 5 | 4 | 8 |
| busiest expert's share (uniform 0.0078) | 0.010 | 0.030 | 0.032 | 0.042 | 0.043 | 0.050 | 0.044 | 0.045 |

- Mean correlation between iterations' histograms **0.35** (vs 0.71–0.82 at $N_e$ = 8); $j$ = 0 vs
  $j$ = 7 is 0.23. The top-16 experts of adjacent iterations overlap by only 3–7 of 16.
- **77 of 128 experts vary their load more than 3× across iterations, 29 more than 10×**; only 5
  are flat (< 1.5×). No expert is dead at every iteration (the trainer's global `route_dead` = 0 is
  right), but 14 are dead at one or more — they are *depth-specific*, not unused.
- The first iteration routes uniformly and the histogram sharpens with depth (effective experts
  128 → 87; $L_1$ distance from the mean histogram 0.31 → 0.53), the pattern `00 §3.2` describes as
  the signature of the shared block imitating a layered model — visible here, absent at $N_e$ = 8.
  Still partial: 0.35 correlation is far from the zero a full partition would give, and the early
  and late halves' mean histograms still correlate at 0.71.

So the depth-conditioned router uses its depth signal when it has enough experts to spend on it.
That it does so and *still* trails L1 by 2.6 % says the partition, at this scale and budget, does not
recover what sharing the attention and the expert pool across depth costs.

## Interpretation

**Provisional *weakens* on H6 (one seed; `139` formalises it):** the FLOPs-matched shared block with
128 experts sits +2.60 % above the stacked MoE pair, 12× the "matches" band, and the gap grew from
0.57 % at 1 B tokens along the same crossing-then-flattening trajectory that L5 showed against dense.
With I29's caveat (68 % of L1's parameters) the refuted claim is H6 as the ladder can state it at
matched FLOPs; an exactly parameter-matched arm at ≈ 1.45× FLOPs is the one test left, and it is a
new preset and, arguably, a different hypothesis. Two results stand regardless of seed 1: the shared
block **beats dense by 1.24 % at 68 % of L1's params and 2.2× L0's** (so the recurrent design is
not a bad model, it is a worse use of parameters than stacking at this scale), and at 128 experts
**the router does partition experts by depth** where at 8 it did not (`routing.json`; the same probe
on `139` is the check that the partition is not seed-specific). Rig: 42.7 h, zero `n_thermal`,
GPU 0 ≤ 67 °C. `139-l5-ne128-small-s1` started 08:47 (≈ 43 h).
