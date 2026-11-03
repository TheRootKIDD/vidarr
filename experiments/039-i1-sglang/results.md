# 039-i1-sglang — calibration point (b) (ADR-052): **sim 19,349 tok/s per node against SGLang's no-TBO 17,777 (1.09×) at the stated inputs, but the verdict is *inconclusive*.** Two unpublished inputs flip it inside the §7 sweep: InfiniBand at 200 Gb/s (0.67×) and zipf-0.5 routing imbalance (0.62×)

2026-11-03 · `python -m scripts.bench.i1_sglang --id 039-i1-sglang` · CPU only (the simulator), 12 runs of ≈ 75 s each · tree `28af353` (clean) · scenario `h100_sglang.yaml` (sha256 in `result.json`). Simulator: `deepseek_v3` rung, `batch_policy = round`, `route_draw = bulk`, 72 units in nodes of 8, 288 slots (256 + 32 redundant), 256 sequences per GPU ($z$ = 18,432), 2,050 keys ($P + N/2$), $3z$ tokens with $z$ warm-up (one full decode step measured), seed 0.

**Bears on:** no hypothesis directly. This is `02 §14`'s calibration point (b), which gates every S-result together with S0 (`038`).
**Verdict: inconclusive** (ADR-052 §7). The stated run is within ±20 %, but two arms of the sweep are not. Point (b) stays open, and the two flipping inputs are named in I39. Nothing was fitted to the 17,777.

## Score

| run | input (one at a time) | sim tok/s per node | sim ÷ 17,777 | step p50 | ±20 % |
|---|---|---|---|---|---|
| **stated** | | **19,349** | **1.088** | 105.9 ms | yes |
| context_2000 | 2,000 keys (the post's KV length) | 19,410 | 1.092 | 105.5 | yes |
| **nic_200g** | InfiniBand 200 Gb/s per GPU (placeholder 400) | **11,921** | **0.671** | 171.8 | **no** |
| nvlink_intra | rows inside a node over NVLink, not RDMA | 20,541 | 1.155 | 99.7 | yes |
| **routing_zipf05** | zipf $s$ = 0.5 over the 288 slots | **10,995** | **0.619** | 186.3 | **no** |
| attn_peak | attention core at nominal peak, not FlashMLA's | 20,164 | 1.134 | 101.6 | yes |
| owner_m4096 | owner GEMMs at DeepGEMM's $M$ = 4096 rate | 20,948 | 1.178 | 97.8 | yes |
| expert_x114 | expert GEMMs × 1.14 (DeepGEMM 2025-04) | 19,598 | 1.102 | 104.5 | yes |
| pcie | H100 PCIe instead of SXM | 17,302 | 0.973 | 118.4 | yes |
| seed_1 | routing seed (noise) | 19,355 | 1.089 | 105.8 | yes |

The sim's step p50 matches the post's implied 115 ms per decode step (256 × 8 / 17,777) to the same 1.09×. As ADR-052 §2 says, it is not an independent check.

## One layer on one GPU (stated inputs, uniform load; from the step functions)

| attention + shared expert + router | dispatch | 4 expert calls (512 rows each) | combine | `post` | sum |
|---|---|---|---|---|---|
| 0.433 ms | 0.373 | 0.181 | **0.724** | 0.011 | 1.722 |

Once per step on top of that: `skel_in` 2.36 ms (3 dense layers) and `skel_out` 0.84 ms (head). The sim's round p50 is 1.768 ms, slightly above the sum, because the round waits for the busiest GPU. **The two all-to-all legs are 64 % of a layer.** Wire bytes per token per layer: 171,407 (8 × (7,392 + 14,336) B × 284/288).

Against the post's text (reported, not scored): dispatch is ≈ 0.17 ms per layer there, against 0.37 here. But the post's figure is with TBO, where half the batch dispatches at a time, and with its own overlap. The post gives no combine time in milliseconds, only "≈ 2× DeepSeek's".

**Reported only:** the sim at DeepSeek's `profile-data` configuration (EP128 on 16 nodes, 128 per GPU, 4,096 keys, 256 slots) gives 16,839 tok/s per node without overlap, against 18,598 measured with overlap. It is below, as it should be, by 9 %, inside the 25–35 % TBO gains the SGLang post reports. DeepSeek's production 14.8k per node is not comparable (unknown load).

## Interpretation

At the stated inputs the sim lands 9 % high, in the same direction and by about the same amount as S0 (`038`: 9–14 % high). So the composition that S0 checked on host-bounced PCIe also holds on NVLink + InfiniBand, as far as the stated inputs go.

The result is fragile in exactly the place where the sources are thinnest. Two thirds of a layer is all-to-all time, priced at DeepEP's measured 40 GB/s per GPU on CX7 400 Gb/s, and the SGLang cluster's NIC speed is not published. At 200 Gb/s the sim falls to 0.67×.

The routing arm is cruder than the NIC arm. Zipf $s$ = 0.5 over 288 slots puts the hottest GPU at 2.8× the mean. That is about what the post's own EPLB ablation implies for *unbalanced* routing (EPLB gives 2.54× in decode), so it brackets "no EPLB" rather than the post's EPLB-placed run. The real residual imbalance after placement is unpublished. Every other arm (keys, intra-node path, attention backend, owner GEMM rate, expert GEMM rate, SXM vs PCIe, seed) stays within ±20 %.

Under ADR-052 §7 the verdict is inconclusive, not a pass, and the rule was fixed before the run. What would settle it:
- the cluster's per-GPU NIC speed;
- the EPLB balancedness of the scored run (the post plots balancedness against throughput but gives no number for this run).

Both are I39's to source. Until then `02 §14` is not met, and no S-result may be quoted.
