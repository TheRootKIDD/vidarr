# bigmoe — Research plan v0.1

## 1. Hypotheses
Numbered after the note's points. Tier A = small-scale training experiment; S = simulation; — = not testable here. Numeric thresholds are placeholders until ADR-013.

| H | Falsifiable statement | Tier | Ladder / sim |
|---|---|---|---|
| H1 | At the pool sizes S4 finds viable, systolic-array units beat low-batch designs on cost per token | S | S4, cost model |
| H2 | Parallel attn + MoE + memory in the shared block costs ≤ 1% loss at matched FLOPs and cuts the per-iteration critical path by ≥ 30% in sim | A + S | L2, S1 |
| H3 | A 2:4 mask fixed at 1% of tokens, at 2× dense-equivalent width, matches dense within 1% loss; FP4 weights add ≤ 0.5% | A | L10a, L10b |
| H4 | Coarse TP experts sized to the unit ($g$ = 1) are within 1% loss of $g$ = $U$ sub-experts at matched FLOPs and unit count | A | L7 |
| H5 | Rail clos with unit = node keeps per-port bandwidth at $1/U$ with ≤ 5% utilisation loss at scale | S | S3 |
| H6 | One depth-conditioned shared middle block with $r_{max} n$ experts matches $r_{max}$ stacked MoE layers with $n$ experts each, at matched params and FLOPs | A | L5 |
| H7 | Learned $r_t$ saves ≥ 30% middle-block FLOPs at ≤ 1% loss vs fixed $r_{max}$ | A | L6 |
| H8 | $L_e$ = 2 halves fabric bytes per FLOP at ≤ 0.5% loss | A + S | L7b, S3 |
| H9 | Co-activation placement + multicast + in-network reduction cut upper-tier bytes ≥ 3× vs random placement | S | S3 |
| H10 | Utilisation vs pool size follows $Z_{min} \approx N_e b_{min}/k$; pooling two half-size installations gains ≥ X% utilisation | S | S4 |
| H11 | Continuous scheduling reaches ≥ 90% expert utilisation with p99 iteration latency ≤ 3× p50 | S | S1, S2 |
| H12 | Two $d$-streams through shared weights match a single √2·$d$ stream at equal FLOPs (≤ 1% loss); prefill FLOPs ≈ ½ of decode; decode weight bytes ≈ ½ | A + S | L4, L4b, Q1 |
| H13 | Four MTP heads trained from scratch reach acceptance ≥ 70 / 55 / 45% for heads 2 / 3 / 4 with no main-loss regression | A | L3 |
| H14 | Managed aggregation matches disaggregation's throughput at equal p99 TPOT, without KV transfer | S | S5 |
| H15 | Indexed attention with a shared indexer: ≤ 2% loss on long-context evals; cold-tier hits per token low enough for SSD bandwidth at target throughput | A + S | L9, S6 |
| H16 | `note64` (4 GB/chip) fits resident experts + hot KV at target $Z$; training does not fit | S | S7 |
| H17 | Per-iteration knowledge memory from a clean pipeline lowers loss on knowledge-heavy evals at matched active FLOPs with ≤ 0.5% regression elsewhere | A | L8 |
| H18 | Per-vector LoRA on the shared block reaches ≥ 80% of full-fine-tune gains at rank ≤ 4; multi-tenant batching costs ≤ 5% throughput | A + S | L11, S8 |
| H19 | — (opinion) | — | — |

## 2. Ablation ladder (Tier A)
Each step = previous step + one feature, same data, same token budget, ≥ 2 seeds at `small` (1 seed at `screen`/`medium`). The two baselines (dense; fine-grained per-layer MoE) at matched FLOPs are trained once per size and seed and reused by every step.

| Step | Adds | Bears on | Notes |
|---|---|---|---|
| L0 | Dense baseline | — | pre-norm, GQA, RoPE, SwiGLU, parallel form off |
| L1 | Standard MoE baseline (fine-grained, per-layer experts, top-k, aux-free balance) | H4 | DeepSeek-style |
| L2 | Parallel attention + FF | H2 | on L1 |
| L3 | MTP heads ($m$ = 4) | H13 | |
| L4 | Two streams (default visibility); **L4b** $p$ sees only $s < t$ | H12 | prefill = $c$ only; compare to a √2·$d$ single-stream model |
| L5 | Shared middle block, depth-conditioned router/norms; **L5b** per-iteration LoRA on attention | H6 | fixed $r$ |
| L6 | Variable depth (ACT); **L6b** KV sharing across iterations | H7 | ragged semantics from `01 §4.7` |
| L7 | Hardware-quantised experts ($g$ sweep); **L7b** $L_e$ = 2 | H4, H8 | |
| L8 | Knowledge memory | H17 | |
| L9 | Indexed attention (hot/warm/cold) | H15 | produces the hit-locality model for S6 |
| L10a / L10b | 2:4 sparsity / FP4 | H3 | separately |
| L11 | Per-vector LoRA + tenant experts | H18 | |

Model sizes are fixed by `docs/06 §3` (ADR-015): `screen` and `small` are matched to a 12-layer, $d$ = 768 dense model (≈ 0.6 GFLOP/token training cost; recurrent variants ≈ 200 M total params, $r$ = 8, $d_{ff}$ = 512, $N_e$ = 128); `medium` is matched to a 24-layer, $d$ = 1024 dense model (≈ 2 GFLOP/token; recurrent variants ≈ 500 M total, $r$ = 16). Token budgets 1 B / 2.5 B / 7 B (§7). `screen` runs every variant once; `small` runs the decisive steps (L0, L1, L2, L4, L5, L6, L7, L9) at 2 seeds; `medium` confirms L0, L1 and the two best recurrent variants. L8 splits into L8 (GPU product-key memory, in-loop) and L8b (chunk-level external retrieval, neighbours precomputed offline) — `docs/06 §5`.

**Recurrence caveat.** Shared-block benefits (H6) may only appear with scale; if `small` is silent on H6, that is a recorded outcome, not a failure — fit across three sizes if the budget allows.

## 3. Simulator experiments (Tier S)
| Sim | Question | Reads (`02 §14`) |
|---|---|---|
| S0 | **Calibration on the local rig**: `small` model in expert-parallel inference over 4 × RTX 3060 under the continuous scheduler; simulator with `local_3060` must match measured throughput and per-iteration latency within ± 20 % (`docs/06 §6`) | throughput, latency |
| S1 | Per-iteration latency budget in `note64` vs `gpu_today`; effect of parallel branches | latency distributions |
| S2 | Continuous scheduler: ($b_{min}$, $w_{max}$) sweep → utilisation vs p99 | utilisation, queue wait |
| S3 | Placement + multicast + reduction → per-tier bytes; up/down asymmetry | fabric bytes |
| S4 | Pool size → utilisation; pooling installations | utilisation |
| S5 | Managed aggregation vs disaggregation | throughput, TPOT/TTFT |
| S6 | Cold-tier bandwidth using L9's hit-locality model | cold hits, bandwidth |
| S7 | Memory-capacity Pareto ($m_C$ sweep); training memory | capacity |
| S8 | Multi-tenant batching cost; pinned regulated tenants | throughput |
| S9 | Availability vs replication factor | availability |
| S10 | **Measured, not simulated**: one expert tensor-parallel across the 4 cards vs 4 sub-experts inside the "unit", same FLOPs, over host-bounced PCIe — the local-traffic check of `02 §2` and Q4 in miniature | utilisation vs link bandwidth |

Simulator validation comes first: S0 on the local rig plus the published-deployment check (`02 §14`, I1) before any S1–S9 result is reported.

## 4. Metrics
- **Quality**: loss/perplexity at matched FLOPs and tokens; a fixed eval suite (knowledge-heavy, long-context retrieval and copy, code) — ADR-014.
- **Compute**: FLOPs/token for prefill and decode; total and per-iteration parameters; parameters resident per unit.
- **Systems**: fabric bytes/token/iteration per tier; latency per iteration and per token (p50/p99); tokens in flight; utilisation; cold hits/token; availability.
- Every `results.md` carries: H numbers touched, verdict (supports / weakens / silent), config hash, seeds, hardware, wall-clock, one-paragraph interpretation.

## 5. Milestones (ordered, undated)
- **Phase 0 — Spec, cost model, rig.** Literature refresh (`docs/05`, incl. the author's linked post); ADR-006 to ADR-014 resolved; `costmodel/` with worked examples matching `01 §10`, `02 §2` and `06 §3`; eval suite and corpus chosen; `scripts/bench/` run and `sim/scenarios/local_3060.yaml` written from measured numbers; thresholds set against the noise floor in `06 §4`. Exit: no pending ADR blocks L0–L5; `bench_train_step` numbers replace the throughput assumptions.
- **Phase 1 — Reference model, L0–L5.** `model/` with all knobs (DDP, replicated experts); unit tests for masks, stream visibility, halting/ragged semantics; every variant at `screen`, then L0–L5 at `small`.
- **Phase 2 — Simulator.** `sim/` calibrated by S0 (local) and I1 (published); S1–S4; S10.
- **Phase 3 — L6–L9 and S5–S7.** Includes the L9 → S6 hand-off using the measured VRAM/host/NVMe tiers.
- **Phase 4 — L10–L11, S8–S9, `medium` confirmations (FSDP), write-up.**

## 6. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Recurrent shared block unstable at $r_{max}$ | Start $r_{max}$ = 8; norm modulation; per-iteration residual scale; MoEUT-style norm/grouping tricks |
| Router collapse or early-expert hot-spotting | Depth embedding; aux-free bias; per-$j$ histograms logged from L5 on |
| Compounded changes hide regressions | Ladder discipline; two baselines per step |
| Simulator not trusted | Calibrate against a published deployment before any S-result |
| Four 12 GB cards: `medium` ≈ 1 week per run, nothing larger | `screen` → `small` → `medium` gating (`06 §7`); `medium` for ≤ 5 runs; scaling statements marked provisional |
| Thin experts ($d_{ff}$ ≈ 512) at matched FLOPs confound H4/H6 | Trade $k$ or $r$ explicitly and log it; report $d_{ff}$ next to every loss |
| Per-iteration retrieval infeasible in training (CPU index QPS) | L8 = GPU product-key memory; L8b = chunk-level retrieval precomputed offline |
| No FP8/FP4 on Ampere | quality via fake-quant; speed claims to the cost model, labelled as such |
| Indexed attention hurts exact copy | Dense window $W$; needle/copy tasks in the eval suite |
| Knowledge-pipeline noise | Start from a clean structured source; measure coverage separately from model quality |
| Drifting into building a production trainer | `model/` stays a reference implementation on PyTorch DDP/FSDP built-ins; no custom kernels, no cross-GPU expert parallelism in training code |
| Silent hardware faults (no ECC, thermal throttling) | Log loss spikes, gradient norms and `nvidia-smi` power/thermals per run; rerun before interpreting |
| Over-fitting the spec to one hardware vendor | `U`, $s_{min}$, $\phi$, $\beta_C$ are scenario inputs; at least two scenarios per S-experiment |

## 7. Compute and data assumptions
- **Hardware**: one workstation — Threadripper PRO 3945WX, 128 GB DDR4, 4 × RTX 3060 12 GB, no NVLink/P2P (`docs/06 §1`). Planning throughput 32 TFLOPS dense / 20 TFLOPS variants aggregate until measured.
- **Token budgets**: `screen` 1.0 B, `small` 2.5 B, `medium` 7 B tokens at context 2048. Wall-clock ≈ 8 h / 21 h / 8 d per variant run (`06 §4`). Whole programme ≈ 2 months of GPU time, 3–4 calendar months.
- **Pretraining corpus**: FineWeb-Edu 10 B-token sample, tokenised once to a `uint16` memmap; 32 k tokenizer chosen in ADR-014; licence recorded there.
- **Knowledge source**: structured entailment sentences (Wikidata-derived) embedded into a 2²⁰–2²¹-entry product-key table for L8; Wikipedia slice with offline-precomputed chunk neighbours for L8b.
- **Long-context evals**: synthetic needle/copy at 8 k–32 k, a RULER-style subset, long-document perplexity (PG-19 candidate), all at inference over the real VRAM/host/NVMe tiers.
