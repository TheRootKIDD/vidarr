# 172-l7-g4-screen — L7-g4 (ADR-047: $g$ = 4, 8 of 32 sub-experts of width 224) = **3.3473**, **+0.0145 vs L5 `110` (4.8σ worse)**: finer experts lose at `screen`. That is ADR-047 §5's "unexpected" band; H4 is not weakened

`python -m scripts.train.train --rung L7-g4 --id 172-l7-g4-screen --tokens 1e+09 --micro-batch 4 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 4 × 2048 × 4) · `queue_screen_14`, gate `4fc08b5` (ruff + unit tests), 2026-11-02 15:35:36 → 2026-11-03 00:02:52 UTC+1 · driver 615.71.09 · config in `config.yaml`, per-step log in `log.jsonl`, routing probe in `routing.json` (`train.log` is gitignored and stays on disk).

**Bears on:** H4 ("coarse TP experts sized to the unit ($g$ = 1) are within 1 % loss of $g$ = $U$ sub-experts at matched FLOPs and unit count"). The reading is ADR-047 §5 under ADR-013, one seed, σ = 0.0030 over twelve pairs (2σ 0.0059, 4σ 0.0119).
**Verdict:** Δ = `172` − `110` = **+0.0145 nats (+0.44 %), 4.8σ, worse by more than 4σ.** That is the band ADR-047 §5 calls *unexpected*. The granularity prior (finer is better) does not hold here, so **H4 is not weakened**. The result also lies outside the 2σ band ADR-047 maps to "silent, supports direction", so **no *supports* verdict is claimed**. At one seed H4 stays **silent**, and the evidence leans its way: the coarse unit-sized expert loses nothing, and the fine split is what costs.

## Numbers

| | |
|---|---|
| params | 77,483,393 total, 52,907,393 non-embedding (+3,072 vs `110`: router keys for 24 extra sub-experts, ADR-047 §1) |
| active expert FLOPs | equal to `110` (8 × 224 = 2 × 896 per token per iteration; unit test `test_l7_granularity_matches_l5`) |
| tokens | 999,817,216 in 1907 steps |
| **final val loss** (full held-out, 5606 windows) | **3.3473** nats |
| final train loss (last step, main head) | 3.3152 |
| router aux (last step) | 0.0078 |
| throughput | median **33,298 tok/s** aggregate (`110` 48,553 → **0.686×**; rig `033` predicted 0.63× from 6 steps at 17.1 s; the run averaged 15.96 s/step of wall-clock including evals) |
| peak memory per GPU | 3.86 GiB (as `033`) |
| wall-clock | 30,428 s = **8.45 h** (estimate 9.2 h; `110` 5.79 h → 1.46×), of which 335 s are evals (20 × ≈ 17 s, the last full eval 78 s) |
| thermal | `n_thermal` = 0 in all 190 records; max 75 °C; SM clock ≥ 1905 MHz throughout |
| seed | 0 |

### Against the baselines

| run | rung | val loss | Δ (172 − run) | |
|---|---|---|---|---|
| `110` | L5 ($g$ = 1, matched arm) | 3.3328 | **+0.0145** (4.8σ) | the H4 comparison |
| `100` | L0 dense ($L_{ref}$) | 3.336 | +0.011 (+0.34 %, 3.8σ) | between 2σ and 4σ: L7-g4 trails dense, inconclusive at one seed |
| `106` | L1 fine-grained layered MoE (246.6 M non-emb.) | 3.2048 | +0.1425 (+4.45 %) | at 21 % of L1's non-embedding params, the same order as L5's gap (+0.128) |

### Val loss over training (vs `110`)

| step | 100 | 300 | 500 | 700 | 1000 | 1500 | 1907 |
|---|---|---|---|---|---|---|---|
| `172` L7-g4 | 5.5467 | 4.3735 | 3.9032 | 3.6803 | 3.5141 | 3.3843 | **3.3473** |
| `110` L5 | 5.6159 | 4.3664 | 3.8804 | 3.6601 | 3.4995 | 3.3697 | 3.3328 |
| Δ | −0.0692 | +0.0071 | +0.0228 | +0.0202 | +0.0147 | +0.0145 | **+0.0145** |

L7-g4 leads only at the first eval. From step 300 it trails, the gap peaks at step 500, and from step 1000 to the end it holds at +0.0145. It is neither closing nor opening, so a longer budget at this LR schedule would not obviously erase it. That is a guess from one curve, not a measurement.

### Routing health

| | `172` (32 sub-experts) | `110` (8 experts) |
|---|---|---|
| final effective experts, trainer's worst router | 31.99 of 32 | 8.00 of 8 |
| final load range (share; uniform 0.0313 / 0.125) | 0.0293–0.0324 | 0.1184–0.1285 |
| dead experts at the end | 0 | 0 |
| worst during training | **12 of 32 dead at step 15** (eff. 15.4); 0 dead from step 47 on | 1 dead, eff. ≥ 4.17 |
| probe (`probe_routing`, 8 × 4 held-out seqs on idle GPU 0 after the queue): eff. experts per iteration | 29.07–31.99 (worst iteration 6) | — |
| probe: mean correlation between iterations' load histograms | **0.65** | 0.71 (`138` at $N_e$ = 128: 0.35) |

The routers end healthy. Early on they lose up to 12 of 32 sub-experts within the first ≈ 45 steps and win them back; `110` never loses more than one. Iterations route almost the same way at 32 sub-experts as at 8, a correlation of 0.65 against 0.71. The depth partitioning that `138`/`142` showed at 128 experts does not appear at 32. The trainer's `load_*` and `route_eff_min` fields are taken over the last step's batch and the probe's over 32 held-out sequences, so the two final figures differ slightly.

## Interpretation

At matched active expert FLOPs, expert parameters and unit count, splitting each unit's expert into four 224-wide sub-experts and routing 8 of 32 costs **+0.0145 nats (4.8σ)** against L5's 2 of 8 at width 896. The direction is the opposite of the fine-grained-MoE prior that ADR-047 §5 calls the expected way for H4 to fail. So the coarse, unit-sized expert of P4 gives up nothing at `screen`, and H4 survives a test it could have failed. This is one seed, and ADR-047 §5 has no supports band for "worse by > 4σ", so H4 stays silent with the evidence leaning its way, not *supported*. The cause is untested. Candidates, none measured:

- the 224-wide sub-experts sit below the 256 tile floor that ADR-047 §3 reports but does not enforce;
- the early loss of up to 12 sub-experts in the first ≈ 45 steps may cost representation that the run never wins back, since the gap opens between steps 100 and 500;
- the router-aux weight was tuned at $N_e$ = 8, and its balance pressure per sub-expert is different at 32;
- the selection is unconstrained, so a token's 8 sub-experts can spread over up to 8 units (the unit-limited variant L7-g4u is not built).

Wall-clock was 1.46× L5's (0.686× the throughput) for the same FLOPs. That is the reference implementation's per-expert Python loop (32 gathers instead of 8), a property of `model/` and not of the design.
