# bigmoe — Research plan v0.1

## 1. Hypotheses
Numbered after the note's points. Tier A = small-scale training experiment; S = simulation; — = not testable here. Numeric thresholds are fixed by ADR-013 (`docs/04`): percentages are fractions of the matched baseline's held-out loss, "0.5 %" tests are non-inferiority within $2\sigma$, and a verdict is *silent* inside a $2\sigma$ band above the threshold.

| H | Falsifiable statement | Tier | Ladder / sim |
|---|---|---|---|
| H1 | At the pool sizes S4 finds viable, systolic-array units beat low-batch designs on cost per token | S | S4, cost model |
| H2 | Parallel attn + MoE + memory in the shared block costs ≤ 1% loss at matched FLOPs and cuts the per-iteration critical path by ≥ 30% in sim | A + S | L2, S1 |
| H3 | A weight-only 2:4 mask fixed at 1% of tokens (author, Q7), at 2× dense-equivalent width, matches dense within 1% loss; FP4 weights add ≤ 0.5% | A | L10a, L10b |
| H4 | Coarse TP experts sized to the unit ($g$ = 1) are within 1% loss of $g$ = $U$ sub-experts at matched FLOPs and unit count | A | L7 |
| H5 | Rail clos with unit = node keeps per-port bandwidth at $1/U$ with ≤ 5% utilisation loss at scale | S | S3 |
| H6 | One depth-conditioned shared middle block with $r_{max} n$ experts matches $r_{max}$ stacked MoE layers with $n$ experts each, at matched params and FLOPs | A | L5 — `screen` (`112`): −0.57 % vs L1 at 68 % of its params, **silent** (inside 4σ); the curve crosses at ≈ step 1300, so `small` decides it |
| H7 | Learned $r_t$ saves ≥ 30% middle-block FLOPs at ≤ 1% loss vs fixed $r_{max}$ | A | L6 — **but $r_{max}$ is the wrong control** (I25): score against fixed $r$ at the *matched mean depth* from the L5-r sweep, else adaptive depth is credited for simply being shallower. **Sweep measured** (`114`/`110`/`115`, `screen`): $r$ = 4 / 8 / 12 → 3.3567 / 3.3328 / 3.3370 — fixed $r$ = 4 already meets H7's numbers (−50 % middle FLOPs at +0.72 %), and the curve is flat past 8, so a learned policy at mean depth $\bar r$ must beat the line between the 4 and 8 points at $\bar r$ |
| H8 | $L_e$ = 2 halves fabric bytes per FLOP at ≤ 0.5% loss | A + S | L7b, S3 |
| H9 | Co-activation placement + multicast + in-network reduction cut upper-tier bytes ≥ 3× vs random placement | S | S3 |
| H10 | Utilisation vs pool size follows $Z_{min} \approx N_e b_{min}/k$; pooling two half-size installations gains ≥ 10 % utilisation (ADR-013) | S | S4 |
| H11 | Continuous scheduling reaches ≥ 90% expert utilisation with p99 iteration latency ≤ 3× p50 | S | S1, S2 |
| H12 | *(optional — author, Q1)* Two $d$-streams through shared weights match a single √2·$d$ stream at equal FLOPs (≤ 1% loss); the latency floor at full utilisation halves with half the tokens in flight, throughput unchanged (S1, incl. the non-halving fabric term) | A + S | L4 (last), S1 |
| H13 | Four MTP heads trained from scratch reach acceptance ≥ 70 / 55 / 45% for heads 2 / 3 / 4 with no main-loss regression | A | L3 — **decided at `small`, not `screen`** (ADR-030); both clauses reported at both budgets |
| H14 | Managed aggregation matches disaggregation's throughput at equal p99 TPOT, without KV transfer | S | S5 |
| H15a | *(v2)* Global attention over final vectors only, with per-iteration attention kept local ($W$ = 512), costs ≤ 2% loss vs global per-iteration attention at 2 k–8 k context, dense, no index | A | L5d |
| H15b | Indexing the final-vector cache with a shared indexer adds ≤ 1% on long-context evals; cold-tier hits per token low enough for SSD bandwidth at target throughput | A + S | L9, S6 |
| H16 | `note64` (4 GB/chip) fits resident experts + in-flight local caches + resident $\mathcal{G}$ at target $Z$; training does not fit | S | S7 |
| H17 | Per-iteration knowledge memory from a clean pipeline lowers loss on knowledge-heavy evals at matched active FLOPs with ≤ 0.5% regression elsewhere | A | L8 |
| H18 | Per-vector LoRA on the shared block reaches ≥ 80% of full-fine-tune gains at rank ≤ 4; multi-tenant batching costs ≤ 5% throughput | A + S | L11, S8 |
| H19 | — (opinion) | — | — |

## 2. Ablation ladder (Tier A)
Each step = previous step + one feature, same data, same token budget, ≥ 2 seeds at `small` (1 seed at `screen`/`medium`). The two baselines (dense; fine-grained per-layer MoE) at matched FLOPs are trained once per size and seed and reused by every step.

| Step | Adds | Bears on | Notes |
|---|---|---|---|
| L0 | Dense baseline | — | pre-norm, GQA, RoPE, SwiGLU, parallel form off; **the author's requested comparator** — a plain transformer on the same corpus at matched FLOPs/token (`04 §Author's answers`, Baseline) |
| L1 | Standard MoE baseline (fine-grained, per-layer experts, top-k, aux-free balance) | H4 | DeepSeek-style |
| L2 | Parallel attention + FF | H2 | on L1 |
| L3 | MTP heads ($m$ = 4) | H13 | heads read the single stream ($p$ only if L4 is on) |
| L5 | Shared middle block, depth-conditioned router/norms, per-iteration KV — **branches from L2** (parallel form kept, no MTP heads: ADR-030 parks H13 at `small`); measured `110`: 3.3328 nats at $N_e$ = 8, −0.10 % vs L0 at 70 % of its non-embedding params, +3.99 % vs L1 at 21 % — so `L5-ne128` is the H6 arm that matters next; run at $N_e$ ∈ {1, 8, 128} (dense recurrent block; author's local default; parameter-matched); **L5b** per-iteration LoRA on attention; **L5c** unshared per-iteration attention with a shared expert pool; **E-Q2** router without $e_j$; **L5-r** fixed-$r$ sweep $r$ ∈ {4, 8, 12} — a *control*, not a feature (I25); **$d_{ff}$ is pinned at 896** so only the depth changes (the budget solver would otherwise widen $r$ = 4 to 2496), which means the arms are *not* matched-FLOPs to each other by design — that is the point: H7 needs a loss-vs-mean-depth curve to compare a learned policy against, and this is also the first measurement of whether the architecture wants more or less depth at all; **L5e** slow/fast split — a small dense H block with its own weights every $K$ iterations beside the expert-bearing L block, testing HRM-Text's "two weight sets beat one shared loop" **on held-out loss**, which HRM-Text itself never reported (`08 §4`); at $K$ = 8, $d_{ff}$(H) = 2048 it costs +6.3 M params and **fits the 0.6 GFLOP budget with nothing traded** (`costmodel/`) — decide after L5, since L5's own result changes how interesting it is | H6 | fixed $r$ = 8 |
| L5d | *(v2 attention)* global range switched from per-iteration caches to the final-vector cache $\mathcal{G}$; local per-iteration attention within $W$; segment-recurrent training with stop-gradient memory | H15a | dense over $\mathcal{G}$, no index; the single most informative cheap run after L5 |
| L6 | Variable depth (ACT, last-state output, halted-token KV stored once — ADR-010/011) on top of L5d; **L6b** KV sharing across iterations for the transient local caches (low priority after v2); **L6c** *expert-choice* depth router with a capacity factor (MoR-style) instead of ACT halting — I24, run **beside** L6 not after it, because `docs/05` records expert-choice beating token-choice by 2.6 pts and `01 §4.7` commits to token-choice | H7 | ragged semantics from `01 §4.7`; L6c's systems cost (a per-iteration capacity rule is a synchronisation point, against P11's continuous execution) goes to `sim/` before H7 is scored |
| L7 | Hardware-quantised experts ($g$ sweep); **L7b** $L_e$ = 2; **L7c** *rank-heterogeneous* experts (FlexMoRE-style: low-rank adapters on a shared base, mixed ranks) vs uniform width at matched FLOPs **and** matched params — L7 makes experts *smaller*, L7c makes them *unequal* | H4, H8 | L7c pairs with **S11** and the two together decide whether P4's uniform-width rule survives (`00 §P4`); neither half decides it alone |
| L8 | Knowledge memory | H17 | |
| L9 | Index over $\mathcal{G}$ (hot/warm/cold tiers, shared indexer) | H15b | produces the hit-locality model for S6; one index per sequence |
| L10a / L10b | weight-only 2:4 sparsity, fixed mask (one run) / FP4 fake-quant | H3 | separately |
| L11 | Per-vector LoRA + tenant experts | H18 | |
| L4 (optional, last) | Two streams (default visibility), $F_p$ prediction-only blocks, MTP from $p$; **L4b** $p$ sees only $s < t$ | H12 | the author suggests leaving it out; only if budget remains; compare to a √2·$d$ single-stream model |

Model sizes are fixed by `docs/06 §3` (ADR-015, ADR-018): `screen` and `small` are matched to a 12-layer, $d$ = 768 dense model (≈ 0.6 GFLOP/token training cost); the recurrent default is single-stream, $r$ = 8, $N_e$ = 8, $k$ = 2 (≈ 80 M total params), with the parameter-matched $N_e$ = 128, $k$ = 4 variant (≈ 250 M total) for H6; `medium` is matched to a 24-layer, $d$ = 1024 dense model (≈ 2 GFLOP/token; $r$ = 16; ≈ 160 M / ≈ 700 M total). Token budgets 1 B / 2.5 B / 7 B (§7). `screen` runs every variant once; `small` runs the decisive steps (L0, L1, L2, **L3**, L5, L5d, L6, L7, L9) at 2 seeds — L3 added by ADR-030, because H13's own prior (I20) says 1 B tokens is too small a budget to interpret an MTP failure at; `medium` confirms L0, L1 and the two best recurrent variants. L8 splits into L8 (GPU product-key memory over sentence embeddings, parametric values, in-loop), L8b (chunk-level external retrieval, neighbours precomputed offline), L8c (encoder fine-tuned by backpropagation, budget permitting) and L8d (entailments instead of sentences, only if L8 is used) — `docs/06 §5`, ADR-009.

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
| S11 | **What non-uniform experts cost the fabric** — the systems half of the P4 question (`00 §P4`, pairs with L7c). Uniform width is what makes `02 §2`'s tile floor, `02 §4`'s placement and `02 §8`'s residency budget tractable; unequal experts mean unequal unit occupancy, so the scheduler bin-packs instead of tiling and P9's co-activation placement loses its symmetry. Sweep a rank mix against uniform width at equal total params | utilisation, fabric bytes, placement quality |
| S10 | **Measured, not simulated**: one expert tensor-parallel across the 4 cards vs 4 sub-experts inside the "unit", same FLOPs, over host-bounced PCIe — the local-traffic check of `02 §2` and Q4 in miniature | utilisation vs link bandwidth |

Simulator validation comes first: S0 on the local rig plus the published-deployment check (`02 §14`, I1) before any S1–S9 result is reported.

## 4. Metrics
- **Quality**: loss/perplexity at matched FLOPs and tokens; a fixed eval suite (knowledge-heavy, long-context retrieval and copy, code) — ADR-014.
- **Compute**: FLOPs/token for prefill and decode; total and per-iteration parameters; parameters resident per unit.
- **Systems**: fabric bytes/token/iteration per tier; latency per iteration and per token (p50/p99); tokens in flight; utilisation; cold hits/token; availability.
- Every `results.md` carries: H numbers touched, verdict (supports / weakens / silent), config hash, seeds, hardware, wall-clock, one-paragraph interpretation.

## 5. Milestones (ordered, undated)
- **Phase 0 — Spec, cost model, rig.** Literature refresh (`docs/05`, incl. the author's linked post); ADR-006 to ADR-014 resolved; `costmodel/` with worked examples matching `01 §10`, `02 §2` and `06 §3`; eval suite and corpus chosen; `scripts/bench/` run and `sim/scenarios/local_3060.yaml` written from measured numbers; thresholds set against the noise floor in `06 §4`. Exit: no pending ADR blocks L0–L3 or L5; `bench_train_step` numbers replace the throughput assumptions; the author's updated note is in `docs/source/` and `docs/00`–`01` re-checked against it (I7). **Reached 2026-09-02**: rig measured (`000`–`013`), corpus tokenised (ADR-014), thresholds set (ADR-013), widths re-derived (ADR-025/027), literature verified (`docs/05`), the author's hardware document read (`docs/07`); ADR-007/008/024/028 remain proposed and block only Phase 2–3.
- **Phase 1 — Reference model, L0–L3 and L5.** `model/` with all knobs (DDP, replicated experts, single-stream default); unit tests for masks, halting/ragged semantics, per-iteration KV; every variant at `screen`, then L0–L3 and L5 at `small`.
- **Phase 2 — Simulator.** `sim/` calibrated by S0 (local) and I1 (published); S1–S4; S10.
- **Phase 3 — L6–L9 and S5–S7.** Includes the L9 → S6 hand-off using the measured VRAM/host/NVMe tiers.
- **Phase 3.5 — FSDP characterisation (cheap, and it gates Phase 4).** **FSDP has never been run on this rig.** `06 §3` specifies it for `medium` and `06 §4`'s `medium` row is a projection from DDP behaviour, so every multi-day `medium` estimate rests on an unmeasured assumption — and the assumption is load-bearing, because collectives here bounce through host memory (`002-nccl`, 3.59 GB/s) and FSDP gathers parameters *per layer per step* rather than reducing gradients once. Run `bench_train_step` at `medium` shape under FSDP for ~150 steps under its own `0NN` id: measure step time, per-GPU peak memory and the gather overhead, and replace the projected `medium` column with measured numbers. ≈ 1 h. **Do this before committing to any `medium` run**; if FSDP costs materially more than projected, `medium` gets rescoped rather than discovered mid-run. Also settles where the per-GPU ceiling actually is: at `small` dense, optimiser states are 1.68 GiB of a measured 7.6–8.0 GiB, so **activations bind, not states**, and the ≈ 1.2 B-parameter figure that states-only arithmetic gives is not the real limit.
- **Phase 4 — L10–L11, S8–S9, `medium` confirmations (FSDP), write-up.**
- **Phase 5 — objective sensitivity (after the ladder, not during).** The note says nothing about training objectives, so this is outside its 19 points — but HRM-Text attributes roughly *half* its gain to its objective (response-only loss + PrefixLM) rather than its architecture (`05` line 113, `08 §2`), which is uncomfortable for an architecture programme to leave unexamined. It cannot run earlier: ADR-014 fixes the corpus, tokenizer and held-out set, and changing the objective makes every ladder number **incomparable rather than merely worse** — response-only masking scores a different subset of positions, and FineWeb-Edu has no instruction/response split at all, so it means a different corpus and a full re-tokenisation. Two forms, cheap first:
  - **5a — ordering stability (cheap, ≈ 1 node-day at `screen`).** Do not compare losses across objectives at all; compare **rank order**. Retrain L0, L1, L2 and the best recurrent rung under the new objective on a small instruction corpus, and ask whether the ordering survives. **If it holds, every Tier-A conclusion the ladder reached generalises beyond one training setup — a materially stronger claim than the ladder can currently make. If it flips, that is the finding.** Needs a new ADR for the corpus and its own $L_{ref}$ per objective; no cross-objective loss comparison is reported.
  - **5b — the parallel ladder (expensive, weeks).** Every decisive rung retrained under the new objective, with its own baselines, as a second ladder beside the first. Only worth it if 5a shows the ordering is *not* stable, because a stable ordering means 5b would re-derive the same conclusions at many times the cost. **Gate 5b on 5a's outcome.**
  - Whatever 5a spawns is scheduled the same way the rest of the programme is: one feature at a time, matched FLOPs, two baselines, ADR-013 bands. Candidate spawns already visible: **L5e** (H/L split, `08 §4`) and **L6c** (expert-choice depth, I24) are both objective-sensitive in principle — HRM-Text's H/L result was measured under *its* objective, so if 5a shows ordering instability, L5e's own verdict may be objective-dependent too, and that is worth stating rather than discovering.

## 6. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Recurrent shared block unstable at $r_{max}$ | Start $r_{max}$ = 8; norm modulation; per-iteration residual scale; MoEUT-style norm/grouping tricks |
| Router collapse or early-expert hot-spotting | Depth embedding; aux-free bias; per-$j$ histograms logged from L5 on |
| Compounded changes hide regressions | Ladder discipline; two baselines per step |
| Simulator not trusted | Calibrate against a published deployment before any S-result |
| Four 12 GB cards: `medium` ≈ 1 week per run, nothing larger | `screen` → `small` → `medium` gating (`06 §7`); `medium` for ≤ 5 runs; scaling statements marked provisional |
| Thin experts at matched FLOPs confound H4/H6 (worst in the $N_e$ = 128 and two-stream variants) | Author's local default is $N_e$ = 8 (Q6); trade $k$ or $r$ explicitly and log it; report $d_{ff}$ next to every loss |
| Per-iteration retrieval infeasible in training (CPU index QPS) | L8 = GPU product-key memory; L8b = chunk-level retrieval precomputed offline |
| No FP8/FP4 on Ampere | quality via fake-quant; speed claims to the cost model, labelled as such |
| Indexed attention hurts exact copy | Dense window $W$; needle/copy tasks in the eval suite |
| Final-vector-only global attention (v2) loses what intermediate iterations could have offered far tokens | L5d measures it before anything is built on it; fallback variant: global over the final vector plus one mid-iteration vector |
| Knowledge-pipeline noise | Start from a clean structured source; measure coverage separately from model quality |
| Drifting into building a production trainer | `model/` stays a reference implementation on PyTorch DDP/FSDP built-ins; no custom kernels, no cross-GPU expert parallelism in training code |
| Silent hardware faults (no ECC, thermal throttling) | Log loss spikes, gradient norms and `nvidia-smi` power/thermals per run; rerun before interpreting |
| Over-fitting the spec to one hardware vendor | `U`, $s_{min}$, $\phi$, $\beta_C$ are scenario inputs; at least two scenarios per S-experiment |

## 7. Compute and data assumptions
- **Hardware**: one workstation — Threadripper PRO 3945WX, 128 GB DDR4, 4 × RTX 3060 12 GB, no NVLink/P2P (`docs/06 §1`). Planning throughput 32 TFLOPS dense / 20 TFLOPS variants aggregate until measured.
- **Token budgets**: `screen` 1.0 B, `small` 2.5 B, `medium` 7 B tokens at context 2048. Wall-clock ≈ 8 h / 21 h / 8 d per variant run (`06 §4`). Whole programme ≈ 2 months of GPU time, 3–4 calendar months.
- **Pretraining corpus**: FineWeb-Edu `sample/10BT` (ODC-By), tokenised once to a `uint16` memmap on `/mnt/nvme/corpus`; tokenizer = Mistral-7B-v0.1 32 k (Apache 2.0); held-out = first 10 000 documents of shard 000 — ADR-014.
- **Knowledge source**: a Wikipedia sentence slice (e.g. article lead sentences) embedded with a pretrained sentence encoder into a 2²⁰–2²¹-entry product-key table for L8 (author's addendum: no entailment extraction); the same slice with offline-precomputed chunk neighbours for L8b; entailment-style atomic facts only for L8d.
- **Long-context evals**: synthetic needle/copy at 8 k–32 k, a RULER-style subset, long-document perplexity (PG-19 candidate), all at inference over the real VRAM/host/NVMe tiers. Note: with one $\mathcal{G}$ entry per token, 32 k of context is ≈ 33 MB per sequence at `small` — S6's tier limits must be set artificially small to exercise the warm/cold paths at our context lengths.
