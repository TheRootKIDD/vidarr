# bigmoe — Compute envelope v0.1

Status: DRAFT, 2026-09-02. Fixes what the local rig can measure, what it can only simulate, and the sizes and budgets every experiment must fit. Nominal numbers below are vendor specs; **each one is replaced by a measured value from `scripts/bench/` before it is used in a budget or a scenario** (ADR-015).

## 1. Hardware

| Component | Nominal | Notes |
|---|---|---|
| CPU | AMD Threadripper PRO 3945WX — 12 cores / 24 threads, Zen 2, 8-channel DDR4-3200 (≈ 205 GB/s), 128 PCIe 4.0 lanes | verify per-core boost and memory speed actually configured |
| Host RAM | 128 GB DDR4 | holds tokenised corpus, FAISS indexes, KV "warm" tier, optimiser offload if ever needed |
| GPUs | 4 × NVIDIA RTX 3060 12 GB (GA106, SM 8.6, 28 SMs) | consumer Ampere |
| GPU compute | BF16/FP16 tensor with FP32 accumulate ≈ 25.5 TFLOPS dense per card (≈ 51 with 2:4 sparsity); FP32 ≈ 12.7 TFLOPS | GeForce halves FP32-accumulate tensor rate; use BF16 autocast, FP32 master weights |
| GPU memory | 12 GB GDDR6, 192-bit, ≈ 360 GB/s per card | ridge point $\phi/\beta_C \approx 71$ FLOP/byte |
| GPU interconnect | PCIe 4.0 x16 per card (nominal 32 GB/s per direction); **no NVLink; no PCIe P2P on GeForce** — NCCL traffic bounces through host memory | confirm `nvidia-smi topo -m` shows x16 on all four; measure with `nccl-tests` |
| Supported numerics | BF16 ✔, FP16 ✔, TF32 ✔, **2:4 structured sparsity ✔** (SM 8.0+), FP8 ✘, FP4 ✘ | FP8/FP4 are quality-only via simulated quantisation |
| Storage | NVMe assumed (unmeasured) | measure random-read bandwidth/latency at 4 KB–1 MB blocks (KV cold tier) |
| Power | ≈ 1 kW under load (4 × 170 W + 280 W + rest) | multi-day runs: checkpoint ≤ 30 min apart |

Derived per card, BF16 weights: compute-bound batch $b_{min} = (\phi/\beta_C)(b_w/2) \approx 71$ tokens per expert per step; the local tile floor $s_{min}$ for ≥ 80 % of peak is measured by `bench_gemm` (expect 128–256 on GA106). These are the `local_3060` scenario inputs (`02 §1`).

## 2. What the rig can and cannot do

| Capability | Status | Consequence |
|---|---|---|
| Train the ablation ladder at `small` (~100 M-dense-equivalent FLOPs) | **Real** | all Tier-A hypotheses testable at `small` |
| Train `medium` (~350 M-dense-equivalent) | **Real but slow** (≈ 1 week per run with FSDP) | 3–5 confirmation runs only |
| Anything ≥ 1 B dense-equivalent FLOPs, or long-context (≥ 16 k) pretraining | ✘ | out of scope; scaling statements stay provisional |
| 2:4 sparsity speedup (P3) | **Real** (inference-time sparse GEMM on Ampere); training uses the mask on dense kernels | H3 quality from training, H3 speed from `bench_sparse24` |
| FP8 / FP4 (P3) | quality only, fake-quant | FP4-vs-2:4 *speed* comparison moves to the cost model |
| Expert parallelism across GPUs (P4/P5/P11) | **Real at toy scale**: 4 units of $U$ = 1 over host-bounced PCIe | used for S0 calibration and S10, **not** for training (experts are replicated; see §3) |
| Tensor parallel inside a unit (P4) | **Real at toy scale**: $U$ = 4 across the four cards | S10 measures where TP breaks at low link bandwidth — Q4 in miniature |
| KV / knowledge tiers (P15, P16) | **Real three-tier system**: VRAM (≈ 360 GB/s) → host DDR4 over PCIe (≈ 20–25 GB/s) → NVMe (≈ 3–7 GB/s) | L9's hit-locality model and S6's cold-tier bandwidth get measured, not assumed |
| Per-iteration external retrieval during training (P17) | ✘ (CPU index QPS is 1–2 orders of magnitude short, §5) | in-loop memory is GPU product-key memory; external retrieval only at chunk granularity, precomputed offline |
| Datacenter-scale claims (P5, P9, P10, P11, P14, P16, P19) | simulation / cost model only | unchanged from `docs/03 §3` |

## 3. Anchor sizes and parallelism

Budgets are matched-FLOPs per token (training cost = forward + backward ≈ 3 × forward, attention included at the training context). Exact widths come from `costmodel/`; the numbers here are the anchors it must reproduce within 10 %.

| Size | Dense baseline (L0) | Training cost / token | Context | Tokens | Use |
|---|---|---|---|---|---|
| `screen` | as `small` | ≈ 0.6 GFLOP | 2048 | 1.0 B | one seed, exploratory variants and sweeps |
| `small` | 12 layers, $d$ = 768, 12/4 heads, SwiGLU $d_{ff}$ = 2048, $V$ = 32 k (≈ 85 M non-embedding + 25 M embedding) | ≈ 0.6 GFLOP | 2048 | 2.5 B | decisive ladder steps, ≥ 2 seeds |
| `medium` | 24 layers, $d$ = 1024, 16/4 heads, $d_{ff}$ = 2816 (≈ 300 M + 33 M) | ≈ 2.0 GFLOP | 2048 | 7 B | confirmations, 1 seed |

`small` recurrent configuration at the same 0.6 GFLOP/token (single stream, $E$ = $F$ = 2, fixed $r$ = 8, middle-block attention window $W$ = 512): the author's local default (Q6, ADR-018) is $N_e$ = 8, $k$ = 2 → $d_{ff}$ ≈ 1.5–1.8 k, ≈ 30 M expert parameters, ≈ 80 M total; the parameter-matched variant for H6 is $N_e$ = 128, $k$ = 4 → $d_{ff}$ ≈ 0.8 k, ≈ 200 M expert parameters, ≈ 250 M total; $N_e$ = 1 is a dense recurrent block. `medium` at $r$ = 16: $N_e$ = 8, $k$ = 2 → $d_{ff}$ ≈ 3 k, ≈ 160 M total; $N_e$ = 128, $k$ = 4 → $d_{ff}$ ≈ 1.5 k, ≈ 700 M total (FSDP). Switching two streams on (L4, optional) halves every $d_{ff}$ above, down to the $U s_{min}$ floor of 512. **Thin experts are a property of matched-FLOPs recurrence at this scale, not a bug**: $r$ iterations (× 2 with two streams) multiply the number of weight passes per token. If a step needs wider experts, trade $k$ or $r$ explicitly and log it; do not silently exceed the budget. `costmodel/` is the source of truth for every width here.

Parallelism:
- **`screen` / `small`**: plain DDP, one replica per GPU, **all experts replicated** (≈ 200 M params × 16 B ≈ 3.2 GB of states per GPU + activations). Routing and dispatch are intra-GPU; no expert parallelism is needed for training. Gradient all-reduce (≈ 0.4 GB BF16 per step) is ≪ step time even host-bounced.
- **`medium`**: FSDP (sharded parameters, gradients, optimiser states across the four cards; ≈ 2 GB of states per GPU) with activation checkpointing per middle-block iteration. Per-step traffic ≈ 3 GB per GPU against a ≈ 30 s step at 0.5 M-token batches → communication stays under a few %, even through host memory. 8-bit optimiser states are the fallback, not the default (they change the experiment).
- Per-GPU footprint must stay ≤ 10 GB in every config; the remaining 2 GB is for fragmentation and eval.
- Expert / tensor parallelism across GPUs is implemented **only** in `sim/`-adjacent measurement scripts (S0, S10), never in `model/` training code.

## 4. Time budget

Assumed achieved throughput until `bench_train_step` replaces it: 8 TFLOPS per card for dense (≈ 31 % of nominal), 5 TFLOPS per card for MoE / recurrent variants (smaller GEMMs, routing, recompute). Aggregate 32 / 20 TFLOPS.

| Run | Cost | Dense | Variant |
|---|---|---|---|
| `screen` (1 B tok × 0.6 GFLOP) | 6 × 10¹⁷ | ≈ 5 h | ≈ 8 h |
| `small` (2.5 B × 0.6 GFLOP) | 1.5 × 10¹⁸ | ≈ 13 h | ≈ 21 h |
| `medium` (7 B × 2 GFLOP) | 1.4 × 10¹⁹ | ≈ 5 d | ≈ 8 d |

Programme estimate: `screen` for every ladder variant (≈ 20 × 8 h ≈ 7 days); `small` for the decisive steps L0, L1, L2, L4, L5, L6, L7, L9 at 2 seeds (≈ 16–18 days incl. baselines); `medium` for L0, L1 and the two best recurrent variants (≈ 4 weeks). **≈ 2 months of continuous GPU time; plan 3–4 calendar months with debugging and reruns.** Baselines are trained once per size and seed and reused by every ladder step.

Noise floor: at `small`, seed-to-seed spread in eval loss is typically 0.005–0.01 nats. Thresholds in `docs/03 §1` finer than ≈ 0.02 nats are not resolvable with 2 seeds; ADR-013 must respect this.

## 5. Constraints that change the plan (not just the budget)

1. **Retrieval granularity (P17, L8).** A `small` step processes ≈ 0.5 M tokens; per-iteration retrieval at $r$ = 8 is 4 M queries per ≈ 15 s step ≈ 3 × 10⁵ QPS. IVF-PQ on 12 Zen 2 cores is expected in the 10⁴–10⁵ QPS class (measure). Therefore: the in-loop knowledge branch of `01 §4.6` is a **GPU-resident product-key memory** (2²⁰–2²¹ entries × 768 × 2 B ≈ 1.6–3.2 GB per GPU; parametric values per ADR-009). External vector-DB retrieval is a separate step **L8b**: chunk-level (every 64 tokens), neighbours **precomputed offline** on a ≤ 1 B-token corpus slice, RETRO-style. This is also the honest reading of the note's "retrieve at every iteration" at scale, and it matches the author's Q10: the entailment vectors are inputs to training and the entailment-to-vector map is learned, so the table is a parametric memory layer, not a database round-trip.
2. **Recurrence depth.** `small` trains at fixed $r$ = 8 ($r_{max}$ = 8 for ACT); $r$ = 16 is a `medium` setting. `docs/01 §1` defaults updated accordingly.
3. **Numerics.** L10a (2:4): train with the mask applied on dense kernels; speed measured separately with the semi-structured sparse GEMM at inference shapes. L10b (FP4) and any FP8 step: simulated quantisation only; no speed numbers from this rig.
4. **Long context.** Pretrain at 2048 with $W$ = 512 as the segment length (v2 P7: segments of a document in order, many documents in parallel — the same per-step token count as before, arranged as 4 sequential segments per 2048-token document); indexed attention (L9) is evaluated at 8 k–32 k at inference with the real VRAM/host/NVMe tiers, after a short context-extension fine-tune. The hit-locality model for S6 comes from these runs. Because $\mathcal{G}$ is one entry per token, tier limits are set artificially small to exercise the warm/cold paths.
5. **Failure domains.** No ECC on GeForce: log loss spikes and gradient-norm outliers per step; treat an unexplained spike as a possible bit flip before as a science result.

## 6. Microbenchmark suite → `sim/scenarios/local_3060.yaml`

`scripts/bench/`, each writing a JSON result with git hash, driver/CUDA versions and date; every scenario number cites the result id.

| Script | Measures | Feeds |
|---|---|---|
| `bench_gemm.py` | BF16 GEMM TFLOPS over ($b$, $d_{ff}$, $d$) grid; grouped-GEMM efficiency vs experts-per-batch | $\phi$, $s_{min}$, $b_{min}$ curve; MoE MFU assumption |
| `bench_sparse24.py` | 2:4 semi-structured vs dense GEMM at the same shapes | H3 speed half |
| `bench_nccl.py` | all-reduce / all-gather / all-to-all bandwidth and latency, 1 KB–1 GB, 4 GPUs, host-bounced | $\lambda_{link}$, $\beta_{link}$ for `local_3060`; DDP/FSDP overhead check |
| `bench_tiers.py` | VRAM↔host (pinned/pageable) and host↔NVMe bandwidth and latency at KV-block sizes | KV tier table `02 §7`, S6 |
| `bench_faiss.py` | CPU IVF-PQ build time and QPS at 1 M–100 M vectors | §5.1 numbers; L8b offline precompute budget |
| `bench_train_step.py` | end-to-end tokens/s for `small` dense and recurrent-MoE configs | replaces the 8 / 5 TFLOPS assumptions in §4 |

**S0 (calibration, before any S-result):** run the `small` model in expert-parallel inference across the four cards (4 units, $U$ = 1) under the continuous scheduler of `02 §5`; the simulator loaded with `local_3060` must reproduce measured throughput and per-iteration latency within ± 20 %. Together with the published-deployment check (I1) this is the simulator's two-point calibration.

**S10 (measured, Q4 in miniature):** one expert tensor-parallel across 4 cards vs 4 experts of $g$ = 4 inside the "unit", same FLOPs, over host-bounced PCIe; report utilisation and the link-bandwidth ratio at which TP loses — the local-traffic check of `02 §2` with a real, deliberately poor $\lambda_C$.

## 7. Operational rules
- `screen` before `small`, `small` before `medium`. No run longer than one node-day without a completed `screen` of the same config.
- Resumable checkpoints every ≤ 30 min; every run records `nvidia-smi -q` power/thermal summaries; a throttled card invalidates throughput numbers, not loss numbers.
- NCCL: expect host-bounced transfers; if collectives hang or crawl, set `NCCL_P2P_DISABLE=1` and re-measure. Keep the four cards on x16 slots.
- The CPU is shared: FAISS builds, tokenisation and simulator sweeps (parallelised across the 12 cores with multiprocessing) should not overlap with a `medium` run's data loading.
- Tokenised corpus as `uint16` memmap on NVMe (10 B tokens ≈ 20 GB); never re-tokenise inside a run.

## 8. Data and eval candidates (ADR-014 confirms; record licences)
- **Pretraining corpus**: FineWeb-Edu, 10 B-token sample (open, deduplicated); tokenizer: a 32 k BPE trained on the sample, or an existing open 32 k vocabulary — decide once, log the choice.
- **Knowledge source (L8 / L8b)**: plain sentence embeddings of a Wikipedia slice (≈ 1–2 M sentences; embedding them on one card takes well under an hour) for the product-key table — the author's suggested simplification; the same slice for chunk retrieval; entailment-style atomic facts (Wikidata triples rendered to sentences) only as the L8d upgrade. Coverage measured separately from model quality (`docs/03 §6`).
- **Long-context evals (L9)**: synthetic needle / copy tasks at 8 k–32 k; a RULER-style subset; long-document perplexity (e.g. PG-19) — all at inference with the three real tiers.
- **General evals**: fixed held-out loss on the corpus; a small knowledge-heavy QA set; a small code set. Same suite for every ladder step.
