# 122-l3-small-s0 — L3 at `small`, seed 0: main head 2.9674 nats, +2.85 % vs the L2 pair; acceptance 42 / 25 / 17 % — **weakens H13 on both clauses** (one seed; the second seed did not run)

`python -m scripts.train.train --rung L3 --id 122-l3-small-s0 --tokens 2.5e+09 --micro-batch 2 --seed 0` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (32 × 2 × 2048 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. Fifth run of `queue_small_2.sh`. L2 plus four independent MTP heads (ADR-012; `mtp.subsample` at the default, as `108`), 272.9 M params vs L2's 271.1 M.

**Bears on:** **H13**, both clauses — *greedy acceptance of heads 2/3/4 against the main head ≥ 70 / 55 /
45 % on held-out, with main-head $Δ \le 2σ$ against L2*. This is the run ADR-030 deferred H13's verdict
to: `108`/`113` read +3.15 % / +2.69 % on the main head at `screen` but I20's prior says 1 B tokens is
too small a budget to interpret an MTP failure at. **`123-l3-small-s1` did not run** (driver upgrade
under the queue, see `docs/04` 2026-09-19), so this is one seed of a two-seed test and the verdict
below is provisional in form. It is not provisional in substance: the miss is 11.8× the 2σ margin
and the seed spread at `small` is ≈ 0.006 nats.

## Numbers

| | |
|---|---|
| params | 272.9 M total, 248.3 M non-embedding |
| tokens | 2.500 B in 4768 steps |
| **final val loss, main head** (full held-out, 5606 windows) | **2.9674** nats |
| final train loss (last step): main head / total incl. MTP aux | 2.9719 / 7.4263 (MTP aux 4.4422) |
| throughput (median step) | 42,320 tokens/s aggregate (`step_s` median 12.387 s, p90 12.403, p90/median 1.001) — 17 % below L2, the three extra logits passes |
| peak memory per GPU | 8.83 GiB |
| wall-clock (this session) | 16.65 h |
| seed | 0 |
| routing at the last step | `load_max` 0.134 / `load_min` 0.110, `route_ent_mean` 0.9997, `route_eff_min` 7.99 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 471 telemetry rows**; `temp_c_max` ≤ 79 °C, `sm_mhz_min` ≥ 1897; `n_power_cap` 35 rows |

Wall-clock accounting: summed `step_s` 16.412 h + evals 0.198 h + checkpoints 0.024 h = 16.634 h of
16.645 h, **0.07 % unaccounted**.

## H13, both clauses, at 2.5 B tokens

| clause | H13 requires | `screen` (`113`, 1 B) | **`small` (`122`, 2.5 B)** |
|---|---|---|---|
| acceptance vs main head, head 2 | ≥ 70 % | 39.7 % | **41.9 %** |
| acceptance vs main head, head 3 | ≥ 55 % | 23.0 % | **25.2 %** |
| acceptance vs main head, head 4 | ≥ 45 % | 16.1 % | **16.8 %** |
| head top-1 vs ground truth, heads 2 / 3 / 4 | — | — | 23.6 / 14.3 / 9.4 % |
| main-head regression vs L2 | $Δ \le 2σ$ = 0.0070 | +0.0847 (+2.69 %, vs `107`) | **+0.0823 nats, +2.85 %** vs the L2 pair (2.8851) |

Acceptance is the in-trainer `mtp_accept_h*` (I27 definition: agreement with head 1's greedy token at
the same position, on the full held-out set), the same quantity `113` logged.

**Verdict rule (ADR-013), σ = 0.0035:** main-head clause *supports* if $Δ \le 2σ$ = 0.0070, *weakens*
if $Δ > 2σ + 2σ$ = 0.0140. Δ = +0.0823 → **weakens**, by 5.9× the weakening line. Acceptance clause:
head 2 at 41.9 % against 70 % → **weakens**; heads 3 and 4 at under half their targets. **Both clauses
of H13 miss at `small` as they did at `screen`, and the extra 1.5 B tokens moved them by 2 points of
acceptance and 0.3 points of main-head loss.** I20's "1 B is too small to judge MTP" is answered:
2.5 B is not materially different. The remaining allowance is a curriculum (L3b, I20's recovery
variant), not more tokens on this recipe.

### Against the dense baseline

L3's main head is **2.9674 against the L0 pair's 2.9958: −0.0284 nats, −0.95 %**. Four MTP heads
give back almost the entire 3.7 % that the MoE gained over dense (L2 pair −3.70 %). At matched
FLOPs per token *for training* — and L3 costs 17 % more wall-clock per step for the extra heads on
top — the MTP-equipped MoE is barely distinguishable from a dense model with a third of its
parameters. The heads buy speculative-decode latency, which this tier cannot measure and which at
42 % head-2 acceptance is ≈ 1.5 accepted tokens per verify pass (`108`'s speculative analysis).

## Thermals

Longest single load yet (16.6 h at 8.83 GiB): zero thermal-slowdown rows, GPU 0 (`GPU-4673cc7d`) at
75–79 °C and ≥ 1897 MHz, step time flat to 0.1 %. Throughput 42.3 k tok/s against `108`'s 43.3 k at
`screen` (same rung, same micro-batch): −2 %, within the run-to-run band.

## What happened to the rest of the queue

`122` was the last run to complete. `123`–`129` each failed within 5 s of starting with
`nvmlInit_v2() failed: Driver/library version mismatch`: the NVIDIA user-space packages were
upgraded to 615.71.09 at 09:39–09:40 (kmod built for the running kernel 7.2.5-200) while the
kernel still ran the 610.57.04 module. `122` was already initialised and finished cleanly at
09:51; every new CUDA process fails until the machine is rebooted. Those seven ids are burnt
(append-only, as `101`/`118`) and rerun as `131`–`137` from `queue_small_3.sh` after the reboot.

## Interpretation

**Weakens H13 on both clauses at `small`, one seed** (Δ_main = +0.0823 nats against $T$ = 2σ =
0.0070, head-2 acceptance 41.9 % against 70 %). The second seed (`131-l3-small-s1`, rerun of `123`)
formally completes ADR-013's two-seed Δ and re-pools σ, but with the miss at 11.8× the margin it
cannot flip the reading; what it can add is the L3 pair's own spread. ADR-030's deferral is
discharged: the `screen` evidence was not a budget artefact. The design consequence is for `03 §2`:
L3b (forward MTP curriculum) is the only remaining route to H13 at this scale, and the four-head
configuration should not be carried into L5 onward without it.
