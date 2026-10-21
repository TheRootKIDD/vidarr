# 158-l5-8k-screen — L5 trained at context 8192: 3.3359 nats on the full held-out set at 8 k; the comparator arm of the L5d-8k pair; the verdict is in `159`

`python -m scripts.train.train --rung L5 --id 158-l5-8k-screen --tokens 1e+09 --micro-batch 1 --seed 0 --set context=8192 experts.d_ff=896` · 4 × RTX 3060, DDP, host-bounced NCCL · batch 524,288 tokens (16 × 1 × 8192 × 4) · config in `config.yaml`, per-step log in `log.jsonl`. First run of `queue_screen_10.sh` (2026-10-19 02:40 → 10:02), on the tree at `ae08987` (docs-only on top of the gate commit `36257dc`). ADR-037 with its amendment: `110-l5-screen` with context 8192 and $d_{ff}$ pinned at 896. Everything else is `110`'s: $N_e$ = 8, $k$ = 2, fixed $r$ = 8, RoPE θ 10 k, 1 B tokens, seed 0. Sized by `031`.

**Bears on:** **H15a** at 8 k, as its comparator. On its own it bears on nothing: it differs from
`110` in context and therefore in FLOPs (≈ +55 %, `06 §3`), and in what the eval measures.

## Numbers

| | |
|---|---|
| params | 77.48 M total, 52.90 M non-embedding (as `110`) |
| tokens | 1.000 B in 1907 steps |
| **final val loss** (full held-out, 1401 windows of 8192) | **3.3359** nats (`eval_context` after training: 3.3358) |
| final train loss (last step): main / total | 3.2905 / 3.2981 (router aux 0.0077) |
| throughput (median step) | 38,103 tokens/s aggregate (`step_s` median 13.76 s, p90/median 1.004); `031` measured ≈ 37.6 k; `110` at 2 k: 48.6 k |
| peak memory per GPU | 3.88 GiB (as `031`) |
| wall-clock (this session) | 7.37 h (`031` projected ≈ 7.6 h) |
| seed | 0 |
| routing at the last step | `load_max` 0.129 / `load_min` 0.122, `route_ent_mean` 0.9999, `route_eff_min` 8.00 of 8, no dead experts |
| thermal (`n_thermal`, GPU-wide) | **0 of 190 telemetry rows**; `temp_c_max` ≤ 79 °C, `sm_mhz_min` ≥ 1890; `n_power_cap` 13 rows |

Wall-clock accounting: summed `step_s` 7.278 h + evals 0.085 h + checkpoints 0.004 h = 7.367 h of
7.370 h, **0.04 % unaccounted**.

## Loss by position (`eval_context`, full held-out, 1401 windows)

| positions | 0–512 | 512–1 k | 1–2 k | 2–4 k | 4–8 k |
|---|---|---|---|---|---|
| L5-8k | 3.4827 | 3.3370 | 3.3272 | 3.3264 | 3.3240 |

Loss falls with position up to ≈ 1 k and is nearly flat after that (−0.003 from 1–2 k to 4–8 k). At
this budget, the model gets little from context beyond 1 k. A 2 k-trained L5 is > 4 nats past
position 2048 (session log 2026-10-10), so training at 8 k is what makes the long positions usable.

## Interpretation

L5 trains cleanly at 8 k, at 0.78× its 2 k throughput and the same memory. Its full-set loss, 3.3359
over 8 k windows, is H15a's comparator at 8 k; the pair verdict is in `159`. The 2 k number from `110`
(3.3328) is on 2 k windows and is not comparable. Rig: 7.4 h, zero `n_thermal`, ≤ 79 °C.
