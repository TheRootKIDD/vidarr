# 130-l1-small-s0 — L1 at `small`, seed 0: 2.8808 nats, −3.84 % against $L_{ref}$(small); GPU 0 held ≤ 79 °C for 14 h

`python -m scripts.train.train --rung L1 --id 130-l1-small-s0 --tokens 2.5e+09 --micro-batch 2 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (32 × 2 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Rerun of `118-l1-small-s0` (killed at step 390 for a power outage, burnt); first run of `queue_small_2.sh`, started 2026-09-16 09:33 after the mains came back.

**Bears on:** H4 — as its *baseline arm*, not its test, exactly as `106-l1-screen` at 1 B tokens. H4
compares $g$ = 1 experts against $g$ = $U$ sub-experts and is decided at L7; L1 is the fine-grained
comparator L7 is measured against, and the matched-FLOPs fine-grained MoE baseline CLAUDE.md requires
of every experiment. **On H4 itself this run is silent.** It also supplies the L1 comparator that L2
(`120`/`121`, H2) and L5-ne128 (I29, H6) are scored against at `small`, and the seed-0 half of the L1
pair whose spread re-pools σ once `119` lands.

## Numbers

| | |
|---|---|
| params | 271.2 M total, 246.6 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| final val loss (full held-out, 5606 windows) | **2.8808** nats |
| final train loss (last step, main head) | 2.8839 |
| throughput (median step) | 50,300 tokens/s aggregate (`step_s` median 10.422 s, p90 10.461, p90/median 1.004) |
| peak memory per GPU | 8.72 GiB |
| wall-clock (this session) | 14.00 h |
| seed | 0 |
| routing at the last step | `load_max` 0.131 / `load_min` 0.116 (ideal 0.125), `route_ent_mean` 0.9998, `route_eff_min` 7.99 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 476 telemetry rows**; `temp_c_max` ≤ 79 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 25 rows |

Wall-clock accounting: summed `step_s` 13.818 h + evals 0.149 h + checkpoints 0.020 h = 13.987 h of
14.00 h, **0.1 % unaccounted**.

## Against the dense baseline

| | val loss | Δ vs $L_{ref}$ | params (total / non-emb) |
|---|---|---|---|
| L0 pair (`116`/`117`), $L_{ref}$(small) | 2.9958 (σ = 0.0018) | — | 100.1 M / 75.5 M |
| **`130` L1, seed 0** | **2.8808** | **−0.1150 nats, −3.84 %** | 271.2 M / 246.6 M |
| `106` L1 vs `100` L0 at `screen`, for reference | 3.2048 vs 3.3358 | −0.1310 nats, −3.93 % | same models, 1 B tokens |

Matched FLOPs per token (top-2 of 8 experts), so the 2.7× parameter count is free per token by
construction. The gap is the same size at 2.5 B tokens as at 1 B: −3.84 % against −3.93 %, and
16× the 4σ silent band (0.0073 nats), so it is not seed noise even before the second seed lands.
At the 1 B-token eval inside this schedule (step 1900, 2 M-token subset) L1 sits at 3.1084 against
L0's 3.2206 at the same step, −0.112 nats — the advantage is flat along the trajectory, not
shrinking, over this token range.

**Verdict rule applied:** no threshold names L1 against L0, so no *supports*/*weakens* is recorded;
the resolvable statement is *"a fine-grained MoE beats matched-FLOPs dense by 3.8 % at `small`"*,
one seed. Δ, σ and the pair's own spread are finalised when `119-l1-small-s1` finishes (≈ 14 h);
σ is re-pooled over the L0 and L1 pairs in that write-up.

## Thermals

The longest single load on the rig so far (14 h, 8.72 GiB per card at micro-batch 2). Zero
thermal-slowdown rows on any card; GPU 0 (`GPU-4673cc7d`) sat at 75–79 °C and ≥ 1890 MHz throughout,
and the step time is flat to 0.4 % between median and p90. The 25 power-cap rows are GPU 0 touching
its 170 W cap, as configured (`024`). Throughput (50.3 k tok/s, 14.00 h for 2.5 B) is a valid
number: within 0.2 % of `118`'s 50.2 k over its 1.2 h, and 7 % above `018`'s no-accumulation rate for
this rung, in line with I21's ≈ 40 % all-reduce share for the 271 M layered rungs being amortised
over 32 accumulation steps.

## Interpretation

Silent on H1–H19 as a test; the run is a comparator. What it fixes is the L1 reference at `small`
for seed 0 (2.8808 nats), the size of the MoE-over-dense gap at 2.5 B tokens (−0.115 nats, −3.84 %,
unchanged from `screen`), and a clean 50.3 k tok/s rate for the 271 M layered rungs, so L1/L2/L3 at
`small` cost ≈ 14 h each and the rest of `queue_small_2.sh` (`119`–`129`) is ≈ 7 days. The
L1-vs-L0 gap becomes a banded number when `119` gives the pair's spread; L2's H2 clause (`120`/`121`)
and I29's parameter-matched H6 arm are then scored against this pair, not against `106`.
