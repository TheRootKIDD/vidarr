# 034-s0-ep-decode — S0's measured side (ADR-049): **the simulator misses at every $z$**. Throughput is 2.6–12.4× too high and the per-iteration p50 0.08–0.38× of measured. The missing term is a fixed cost per job (owner stage, expert launch), which `sim/` m1 does not model

2026-11-03 · idle rig, by hand, nothing else running · tree `847e7e7` (clean; ruff + 134 tests passed on it just before) · driver 615.71.09 · `local_3060.yaml` sha256 `2f64947e…` ·
`python -m scripts.bench.s0_ep_decode --id 034-s0-ep-decode` (defaults: `--run 132-l5-small-s0 --z 16 64 256 1024 --prefill 1536 --decode 512 --warmup 64`).
4 × RTX 3060, one process per card, NCCL with `NCCL_P2P_DISABLE=1` (host-bounced). Total wall-clock 89 s. `result.json` holds every number below and more: per-rank phase times, routing histograms and thermal samples.

**Bears on:** no hypothesis directly. This is `02 §14`'s first calibration point, which gates every S-result (H1, H2's latency clause, H5, H8's bytes half, H9–H11, H14, H16).
**Verdict: S0 not passed.** The pass rule is throughput and per-iteration p50 within ±20 % at every $z$, and the simulator misses both at all four. Under ADR-049 §6 no scenario value is fitted, and the term that misses is named below. `sim/` stays uncalibrated, and no S-result may be quoted.

## Setup (ADR-049 §2–3)

- **Model.** `132-l5-small-s0` (L5 `small`, step 4768, val 3.0778), in bf16, decoded through `model/decode.py` (`01 §4.8`).
- **Placement.** Rank $u$ owns sequences $s \equiv u$ and hosts experts $i \equiv u$ (mod 4), 2 of 8.
- **Round.** owner (DepthNorm, attention, router) → `dispatch` (a counts `all_to_all`, then a rows `all_to_all` with the expert id as an extra column) → `expert` (each hosted expert once) → `return` (`all_to_all`) → `post` (gate-weighted sum and residual). Each phase boundary has a `cuda.synchronize`.
- **Workload.** One held-out window per sequence: 1536 positions prefilled untimed, then 512 decoded teacher-forced, the first 64 as warm-up. $r$ = 8. Mean keys per timed decode token 1792.
- **Simulator.** `batch_policy = round`, `context_tokens` 1792, $z \cdot 200$ tokens with $z \cdot 40$ warm-up, seed 0. It runs with the measured routing shares (`empirical`) and with uniform routing. The two agree within 1 %, so routing skew is not the miss.

## Measured vs simulated

| $z$ | measured tok/s | sim tok/s | sim / meas | measured iter p50 | sim iter p50 | sim / meas | meas. iter p99 | token p50 |
|---|---|---|---|---|---|---|---|---|
| 16 | 802 | 9,948 | **12.4×** | 1.927 ms | 0.147 ms | **0.08×** | 2.720 ms | 19.6 ms |
| 64 | 3,153 | 25,615 | **8.1×** | 1.951 ms | 0.225 ms | **0.12×** | 2.685 ms | 20.0 ms |
| 256 | 9,832 | 41,635 | **4.2×** | 2.515 ms | 0.550 ms | **0.22×** | 2.795 ms | 25.9 ms |
| 1024 | 18,618 | 48,500 | **2.6×** | 5.045 ms | 1.895 ms | **0.38×** | 5.594 ms | 54.8 ms |

### Where a round's time goes (rank 0, p50 per round)

| $z$ | owner | dispatch | expert | return | post | round | skeleton per token (mean) | expert rows per hosted expert per round (mean / min–max) |
|---|---|---|---|---|---|---|---|---|
| 16 | 1.163 ms | 0.156 | 0.429 | 0.068 | 0.101 | 1.926 | 3.13 ms | 4.8 / 0–13 |
| 64 | 1.167 | 0.167 | 0.431 | 0.080 | 0.102 | 1.951 | 3.37 | 19.3 / 3–46 |
| 256 | 1.594 | 0.260 | 0.431 | 0.121 | 0.103 | 2.510 | 4.79 | 76.1 / 25–151 |
| 1024 | 3.546 | 0.667 | 0.444 | 0.299 | 0.103 | 5.035 | 13.28 | 305.4 / 140–569 |

| | |
|---|---|
| routing share over 8 experts ($z$ = 1024, all iterations) | 0.196 / 0.113 / 0.104 / 0.111 / 0.103 / 0.106 / 0.147 / 0.121 (expert 0 at 1.6× uniform) |
| wire bytes per token per iteration | 4,603–4,628 measured: ¾ of the $2kd$ rows leave the rank, at $(d + 1) + d$ bf16 per row; logical $2 k d\, b_{act}$ = 6,144 |
| "busy" fraction per rank (time inside owner, expert, post and skeleton phases) | 0.82–0.84 at every $z$. This is phase wall-time, which includes the CPU launching kernels, not GPU kernel occupancy |
| peak memory per GPU | 0.64 / 2.08 / 3.21 / 7.71 GiB (limit 10) |
| prefill (untimed) | 0.6 / 1.0 / 4.0 / 16.0 s |
| teacher-forced NLL (correctness check) | 3.100 / 3.089 / 2.982 / 3.024 against `132`'s 3.078 over whole windows; same range, so the decode path computes the trained model |
| thermal | `n_thermal` = 0 in every sample (every 5 s, rank 0's thread) |

## Interpretation

The simulator has the shape right: throughput grows sublinearly with $z$, and iteration latency is flat until the owner's attention and KV reads take over at $z$ = 1024. It is far too optimistic in absolute terms, most at small $z$. The time split names the cause.

**The expert phase costs 0.43–0.44 ms per round whether a hosted expert gets 5 rows or 305.** The owner phase costs 1.16 ms for 4 tokens and grows only past $z$ = 256. Both behave as fixed costs per job. The likely source is Python dispatch and kernel launches in the reference implementation, though the kernel count was not measured. The sim's rooflines (ADR-043 decisions 1 and 6) have no such term, so they charge ≈ 0.01 ms where the rig pays ≈ 0.4–1.2 ms. At $z$ = 1024 the per-row work starts to show (owner 3.5 ms, dispatch 0.67 ms), and the gap narrows to 2.6×. The collectives cost 0.22 ms per round at $z$ = 16: three host-bounced `all_to_all`s, one for counts and two for rows. The sim charges one 44 µs latency floor (`002-nccl`, 1 KiB) per leg, so ≈ 0.09 ms per round.

This is a property of the measurement stack (`model/` is a reference implementation, CLAUDE.md), not of the design. But S0's purpose is to check that the sim's terms describe a real system, and on this one they leave out the dominant term. Under ADR-049 §6 nothing was fitted. Candidate next steps, none taken:

- (a) **measure** the fixed per-job costs with their own microbenchmark (a launch cost per owner stage and per expert call, a latency per collective) and add them to `local_3060.yaml` as sourced scenario inputs, then re-run the comparison;
- (b) **reduce** them on the harness side (CUDA graphs per round, fused expert calls) until the measured system is the roofline system the sim assumes;
- (c) both.

Each needs an ADR. A fixed per-job cost is also a real term on `note64` (dispatch and kernel-issue overhead per expert step), so (a) adds a missing parameter rather than a local fudge.
