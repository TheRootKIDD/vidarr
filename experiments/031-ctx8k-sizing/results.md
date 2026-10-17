# 031-ctx8k-sizing — L5 and L5d at context 8192 fit easily (3.88 / 3.97 GiB); L5 runs at 37.4 k tok/s, but **L5d at 11.7 k, 3.2× slower** → L5-8k ≈ 7.6 h, **L5d-8k ≈ 24 h**, not ADR-037's 8–10 h

2026-10-16 13:13–13:19 UTC+2 · `queue_screen_9.sh`, on idle GPUs, right after `153` · tree `3839aa4`
(docs-only on top of the gate `36257dc`) · per arm:
`python -m scripts.train.train --rung {L5,L5d} --id 031-ctx8k-sizing-{l5,l5d} --tokens 1e9 --micro-batch 1 --seed 0 --smoke-steps 3 --out-dir /mnt/nvme/ckpt/031-ctx8k-sizing/{l5,l5d} --set context=8192 experts.d_ff=896`.
Each arm takes 3 optimiser steps of 524,288 tokens (16 × 1 × 8192 × 4) and one full held-out eval.
`config.yaml`, `log.jsonl`, `summary.json` and `train.log` per arm are copied here from
`/mnt/nvme/ckpt/031-ctx8k-sizing/`. The checkpoints (1.8 GB) stay off-repo.
`train.attempt1.log` is the first attempt (2026-10-15 00:02), which failed at config time: the
budget solver refused 8192 because the non-expert terms alone are 0.741 GFLOP/token against the
0.6 GFLOP budget. That failure is why ADR-037 was amended to pin $d_{ff}$ at 896.

**Bears on:** no hypothesis. This is the sizing gate of ADR-037 §4 for the L5d-8k pair (H15a's 8 k
end, I30).

## Verdict: memory gate passes; the time estimate fails for L5d

| criterion (ADR-037 §4, `06 §3`) | required / estimated | **L5 @ 8192** | **L5d @ 8192** |
|---|---|---|---|
| peak memory per GPU | ≤ 10 GB; est. ≈ 4–5 GiB | **3.88 GiB** | **3.97 GiB** |
| step time (steps 2–3) | — | 13.88 / 14.02 s | 45.65 / 44.03 s |
| throughput (steps 2–3) | est. ≈ 8–10 h per arm | **37.4–37.8 k tok/s** | **11.5–11.9 k tok/s** |
| projected 1 B tokens (1907 × mean of steps 2–3) | 8–10 h | **≈ 7.4 h** | **≈ 23.8 h** |
| projected wall-clock with evals | | ≈ 7.6 h | ≈ 24.0 h |
| exit | 0 | 0 | 0 |
| full held-out eval at step 3 (1401 windows of 8192) | — | 9.0382, 70.6 s | 8.9443, 90.9 s |

## Against the 2 k anchors

| | 2 k (`110` / `111`, median) | 8 k (this probe) | 8 k / 2 k |
|---|---|---|---|
| L5 tok/s | 48,553 | ≈ 37,600 | 0.77 |
| L5d tok/s | 33,815 | ≈ 11,700 | **0.35** |
| L5d / L5 | 0.70 | **0.31** | |
| peak GiB, L5 / L5d | 3.86 / 3.89 | 3.88 / 3.97 | ≈ 1.0 |

L5's slowdown (0.77×) is milder than the ≈ +55 % FLOPs of `06 §3` would predict (0.65×), because the
causal SDPA fast path handles the longer keys efficiently. L5d's slowdown (0.35×) is far worse. `06
§3` describes L5d's attention as per 512-token segment with an explicit mask against $\mathcal{G}$, so
it does not take the fast path. At 8 k it attends up to 7680 global keys per segment where 2 k
attended up to 1536. Memory stays flat because the segment loop bounds it. The cost is time. This
reading is from the documented code path, not from a profile.

## Caveats

Three smoke steps are a short sample: step 1 includes start-up, and steps 2–3 agree within 1–4 %.
The trainer writes no thermal telemetry rows in smoke mode (`temp_c_max`/`n_thermal` are `null`),
so throttling is not excluded by this probe. The cards were idle and cold after `153`, which ran at
≤ 66 °C. The projections assume the 8 k evals cost what they cost at 2 k, scaled by the measured
full-eval time.

## Interpretation

The L5d-8k pair (`158`/`159`) passes ADR-037's memory gate with > 2× headroom, so no activation
checkpointing is needed. Its time estimate is wrong for one arm: **L5-8k ≈ 7.6 h, L5d-8k ≈ 24.0 h,
≈ 32 h for the pair**, against ADR-037's 16–20 h. L5d-8k sits on the one-node-day line. CLAUDE.md's
rule is about runs longer than a node-day without a finished `screen`, and this run *is* the
`screen`, so it may launch. It will probably run a little over 24 h, and an outage mid-run costs a day. The cost
asymmetry is itself a systems datum for H15a. In this reference implementation, an L5d training
step costs 3.2× an L5 step at 8 k (1.4× at 2 k). The two arms differ only in how global attention is
done.
`model/` is a readable reference with no fused kernel for the $\mathcal{G}$ mask (Phases 0–2), so
this says nothing about what a kernel would cost. It does mean the 8 k pair is the most
time-expensive `screen` item left in Phase 1 after L5-pm.
