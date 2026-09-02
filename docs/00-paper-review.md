# Review: "The big-DC MoE LLM design"

Reviewed 2026-09-02 against the 19-point note. Purpose: decide what to build, what to simulate, and what to push back on.

Status labels:
- **Established** — validated in published work at meaningful scale.
- **Plausible** — sound reasoning, no public validation at scale.
- **Unclear** — under-specified or not derivable from the note as written.
- **Overstated** — direction is fine; strength of the claim isn't supported.
- **Opinion** — economics/geography; not testable by this project.

## 1. Summary verdict

The note is a hardware–model co-design, and its strongest idea is the organizing principle: **size the expert to the smallest hardware unit that runs at full utilization, make every expert that size, and build the fabric so that the unit — not the chip — is the addressable node.** Almost everything else (repeated middle layer, continuous scheduling, huge expert pools, one installation per DC, tiny HBM) follows from wanting that unit busy at all times. Read this way it is coherent, and several pieces already have literature behind them.

Main weaknesses:
1. It is an *inference* design that says nothing about how a model with variable depth, non-lockstep execution and index-in-the-loop attention gets **trained** from scratch. That is where most cost and risk lives.
2. The **router** — top-k over ~10⁵ experts at every one of ~128 iterations, depth-aware, balanced in space *and time*, tenant-masked — is the crux and is not specified.
3. Its **expert granularity** (one coarse expert per 64-chip unit) runs against current evidence that finer experts are better per FLOP.
4. Several latency claims are asserted rather than derived (P2, P12).
5. The **minimum scale** at which the design beats conventional serving is very large, so nearly every systems claim can only be simulated here.
6. Some "nobody does this" / "silly" statements are out of date: MTP trained from scratch, chunked prefill, multi-LoRA serving and rail-optimized fabrics all exist.

None of this makes the project uninteresting. The *combination* of a repeated all-expert middle block, two latent streams, depth-adaptive compute and hardware-quantized experts has not been studied together, and each piece is testable at small scale.

## 2. Pros

1. **Expert = hardware unit** turns the granularity debate into a measurable quantity (utilization at batch b) and makes fabric cost per node independent of expert count.
2. **The fabric (P5) is sound and buildable today.** A fat local domain of 64–72 chips plus one clos "rail" per chip position is how NVL72-class domains with rail-optimized InfiniBand/Ethernet are built. The novelty is *usage*: node = expert group with tensor parallelism inside the node, instead of spreading small experts one-per-chip.
3. **Repeated middle layer with the whole expert pool (P6)** composes naturally with variable depth (P7) and continuous execution (P11); the "you can simulate the layered model" argument is right for the FF path (with caveats, §4).
4. **Two-stream token state (P12)** cleanly separates "what I mean to others" from "what comes next", makes prefill cheaper by construction and gives MTP heads (P13) a natural home.
5. **Managed aggregation (P14)** is the chunked-prefill side of a real debate and fits a design where all units are identical.
6. **Multicast, in-network reduction and co-activation placement (P9)** are real levers that current MoE deployments underuse.
7. **Per-vector adapters + bespoke experts on a shared spine (P18)** is a good product idea and mostly engineering, given existing multi-LoRA kernels.
8. The note is honest about being untested and asks readers to have a reason for rejecting each point — that is the project's method.

## 3. Cons and gaps (cross-cutting)

1. **No training story.** Variable depth (P7), continuous scheduling (P11), index-in-the-loop attention (P15) and per-iteration retrieval (P17) all need a synchronous, gradient-consistent training procedure. Options exist (sampled fixed depth during pretraining, ACT-style ponder cost, straight-through routing, indexer distilled from a dense-attention teacher) but the note has none. Also: expert gradients are sparse — with 10⁵ experts each sees few tokens per step.
2. **Router unspecified.** Product-key / hierarchical routing exists for large pools; balancing a *continuous* system over time (all tokens want "early" experts at early iterations) does not.
3. **Granularity conflict.** Fine-grained MoE scaling-law work says smaller experts win per FLOP. The note's experts are deliberately coarse. Either tensor-parallel coarse experts recover this through the larger pool available at every iteration, or the unit should host many small experts (expert-parallel inside the unit). Same fabric traffic either way — so this is the first thing to measure (H4).
4. **Latency accounting.** 128 sequential fabric round trips per token remain. Parallel branches (P2) and MTP (P13) reduce it; "halve" claims are upper bounds (P2) or rest on an unstated baseline (P12).
5. **Minimum viable scale.** Keeping N_e experts busy at b_min tokens per step needs ≈ N_e·b_min/k tokens in flight per iteration. For 10⁵ experts, b_min = 256, k = 8: ~3 M tokens in flight. Below that the design under-utilizes. There is no small deployment of this system, only simulations.
6. **Failure domains.** A tensor-parallel 64-wide expert means one chip failure removes every expert on that unit. Replication is implied (P9) but not designed; in-flight tokens need recompute semantics.
7. **19 coupled changes.** Attribution is impossible without a strict ablation order (fixed in `docs/03`).
8. **Tone vs evidence.** "Silly" appears where the field has open trade-offs (serial vs parallel blocks, disaggregation vs chunked prefill, HBM). Treat as priors, not conclusions.

## 4. Point by point

### P1 Systolic arrays, not LPUs — *Established (economics), with a caveat*
GPUs/TPUs give the best dense-matmul throughput per dollar and watt; SRAM-resident low-batch designs buy latency with capital cost. Caveat: this design's premise is that batch is always available; any regime where it isn't (small tenants, off-peak, early deployment, agentic workloads that pay for tail latency) is exactly where the other designs win. Nothing to train; encode as a cost-model assumption.

### P2 Parallel attention + FF — *Established*
The parallel block (GPT-J, PaLM, Falcon) trains ~15% faster at scale with negligible loss at ≥60B in PaLM's ablations and a small loss at ~8B. "Halve decode latency" holds only if attention and FF wall-clock are equal and fully overlapped; per-iteration latency becomes max(attn, FF) + dispatch, so it is an upper bound. In this design the two branches run on different hardware (attention owner vs expert units), which is what makes the overlap real. Pitfall: the FF no longer sees this iteration's attention output; with a repeated block it sees last iteration's, and with 2-layer experts (P8) the interleaving gets coarser. Ladder step L2.

### P3 2:4 sparsity from the start; FP4 — *Plausible, high risk*
Hardware 2:4 sparsity accelerates one operand (weights). "Train dense for 1% of tokens, then fix a 50% mask" is an early-bird lottery ticket; evidence for masks found that early at LLM pretraining scale is thin, and a fixed mask removes plasticity. Compare fixed-early-mask vs dynamic sparse training (RigL-style periodic mask updates) vs dense. Compounding FP4 + 2:4 + a weight-tied recurrence is a numerics risk: do sparsity and precision as separate ladder steps (L10a/b). FP4 *pretraining* recipes exist (NVFP4, 2025) but need care (block scaling, stochastic rounding, high-precision layers). Check whether the target hardware's FP4 datapath supports sparsity at all.

### P4 Expert size = smallest FF that fills U chips — *Plausible; central; conflicts with fine-grained evidence*
Sizing rule (derived in `02 §2`): tensor-parallel over U chips needs the per-chip slice d_ff/U ≥ s_min (tile width) and a token batch b ≥ b_min = (chip FLOP/byte ratio × bytes/param)/2 to be compute-bound. With d = 8192, U = 64, s_min ∈ [128, 512]: d_ff ∈ [8k, 32k], i.e. 0.2–0.8 B params per expert (SwiGLU). That is a coarse expert. The alternative that keeps the fabric argument intact is expert-parallel *inside* the unit (U small experts, one per chip): identical clos traffic, lower per-chip batch, finer granularity. Which is better per FLOP at fixed unit count is the project's first real question (H4, L7). The note's "will never process fewer weights than this size at a time" is a scheduling invariant kept either way.

### P5 N × 64 fabric — *Established (topology) / Plausible (usage)*
This is a rail-optimized clos over fat nodes. Verified properties: per-rail bandwidth is 1/U of the node's; a node's aggregate clos bandwidth equals one full rail set; per-port cost is that of a small port. Things to model: multi-tier clos latency (two traversals per iteration), oversubscription, the "asymmetric-bw 4×4×4 torus" inside the unit (meaning unclear — Q8), and the all-gather/reduce per expert step inside the unit. "No MoE complications" is true for *dispatch*; it is not true for placement, replication and failure (P9, §3.6).

### P6 Few early/final layers + one repeated middle layer with all experts — *Plausible; well precedented; two caveats*
Precedents: Universal Transformer, ALBERT (cost of sharing), MoEUT (UT + MoE with fixes for norms and grouping), recurrent-depth models (2025), Relaxed Recursive Transformers (sharing + per-layer LoRA), HRM-style recurrent modules. Caveat 1: the "simulate the layered model" argument requires the router to *know the iteration index* — add a depth embedding to router, attention and norms, otherwise a token picks the same experts at every iteration. Caveat 2: the argument covers FF experts only; attention weights are shared across iterations, so per-layer attention cannot be simulated unless attention also gets per-iteration parameters (per-iteration LoRA is the cheap fix). Also: sharing weights does not shrink the KV cache — 128 iterations still store 128 KV sets per token unless KV is shared across iterations. Ladder L5.

### P7 Variable iteration count; RL for compute allocation — *Plausible; training risk*
Adaptive depth has precedent (ACT, Mixture-of-Depths, early exit/CALM, recurrent-depth models evaluated at different recurrence counts). Two hard parts the note skips: (i) a token that halted at iteration r_s < j has no state at iteration j for later tokens to attend to — define state propagation (copy final state) or masking; (ii) learning the halting policy: ACT-style ponder cost first, RL only as a fine-tuning stage — pretraining-scale RL over discrete compute decisions is high-variance and tends to collapse to "always max" or "always min". "Reduce KV cache significantly" is true only with early halting *and* a cache policy for propagated states. Ladder L6.

### P8 Experts with ≥ 2 internal layers — *Plausible*
Halves fabric bytes per FLOP. Cost: coarser interleaving of attention and FF. Cheap ablation (L7b). Interacts with P2.

### P9 Multicast + co-activation placement — *Plausible; needs a simulator*
Up-traffic dedup (one token copy up, fan-out at the lowest common switch) and in-network reduction of the k expert outputs on the way back (SHARP-style weighted sum) are real. Placement is dynamic graph partitioning + replication (DeepSeek's EPLB is a static precedent); co-activation statistics drift during training and differ per tenant. The up/down asymmetry claim depends on where attention owners sit — make it an *output* of the simulator (S3), not an assumption.

### P10 One combined installation per DC — *Plausible (statistics) / Opinion (operations)*
Statistical multiplexing is correct: the tokens-in-flight needed to keep experts busy (§3.5) only exist in one big pool. Against: blast radius, upgrades, data residency (EU customers cannot be served from a single non-EU pool), single-model lock-in (partly addressed by P18). Model as a utilization-vs-pool-size curve (S4).

### P11 Continuous, non-lockstep execution — *Plausible; the largest engineering unknown*
Per-expert queues with (min batch, max wait); expert-choice routing is the natural algorithmic partner. Risks: tail latency, per-sequence ordering constraints at attention, deadlock without timeouts, and the fact that today's EP runtimes are barrier-based all-to-all — this is a µs-scale dataflow/actor runtime nobody has shipped at this granularity. Little's law makes latency and tokens-in-flight one budget; model it (S1, S2). Training still needs lockstep or bounded-staleness semantics (§3.1).

### P12 Two latent streams per token — *Plausible (structure) / Unclear (baseline for the latency claim)*
Closest precedent: XLNet's two-stream attention (content stream vs query stream). Our best reading of the claims: compare at **equal decode FLOPs per token**. A single-stream model with 2d² FLOPs-scale has width √2·d and 2d² parameters; the two-stream model at width d pushes two vectors through d² parameters. Hence: half the parameters → half the weight bytes per decode step → half the decode latency *in the bandwidth-bound regime*; prefill runs one stream → half the FLOPs. Two caveats: (i) the quality premise — two d-vectors through shared weights matching one √2·d-vector at equal FLOPs — is an untested hypothesis (that is L4); (ii) this design's expert path is compute-bound *by construction* (P4), so the latency mechanism applies only to whatever remains bandwidth-bound; what survives everywhere is "half the resident weights for the same FLOPs", which is still valuable. Confirm the intended baseline with the author (Q1). Further risk: the context stream is never directly supervised; it receives gradient only through the prediction stream.

### P13 Multi-token prediction trained from the start — *Established*
MTP from scratch improves sample efficiency at scale and enables self-speculative decoding (Meta 2024; DeepSeek-V3 ships it with one extra depth and reports ~85–90% acceptance). Four heads is aggressive — acceptance decays per head. Caveat specific to this design: speculation saves *latency*, not FLOPs; a throughput-bound batched system gains less, and verifying 4 draft tokens is a variable-depth pass here. Reading heads off the prediction stream only is consistent with P12. Ladder L3.

### P14 No prefill/decode disaggregation; managed aggregation — *Plausible; overstated*
This is chunked prefill (Sarathi-Serve) generalized to a per-unit prefill:decode ratio. It removes KV transfer and SLO interference. Disaggregation's *other* reasons — different parallelism configs and different hardware for the two phases — vanish only because this design makes all units identical and HBM-light. So the claim holds inside the design's assumptions, not in general. Tension: P12 creates exactly the "prefill-only encoder" the author says would justify disaggregation. Simulate both (S5).

### P15 Indexed attention, cold KV on SSD — *Plausible; constants dominate*
Sparse/indexed attention has a large literature (Memorizing Transformers, Unlimiformer, Landmark, Quest, MagicPIG, RetrievalAttention, NSA, DeepSeek's "lightning indexer"). What the note asks for beyond it: sub-linear index structures (HNSW/IVF) instead of O(n) scoring, and training with the index in the loop. Pitfalls: (i) query count — per generated token, iterations × heads ANN queries unless one low-dimensional indexer per iteration is shared across heads and selects *blocks*; (ii) incremental index maintenance per decoded token; (iii) SSD bandwidth works only if cold hits are rare — that needs a measured locality model, not a hope; (iv) exact copy/induction over far context degrades when the index misses — keep a dense local window. The search-engine analogy has a flaw the author would appreciate: search runs ~1 query per request; attention runs thousands per token. Ladder L9 (quality) + S6 (bandwidth).

### P16 ~4 GB HBM per chip, or no HBM — *Plausible for inference in this design / Overstated in general*
Inside this design, expert units are compute-bound by construction (batched tokens, thin per-chip weight slices), so per-chip memory need is capacity for resident experts (~256 GB per 64-chip unit holds hundreds of FP4 experts) plus a hot KV window. The argument works *here*. It does not transfer to (a) training, where weights + grads + optimizer state ≈ 16 B/param and capacity is the binding constraint; (b) dense or long-context serving with dense attention; (c) the knowledge table (P17), which is capacity-hungry. "Global run on DRAM for no reason" is opinion. No commodity accelerator ships with 4 GB, so this is advice to chip designers, not a saving available to us. Cost model only (S7).

### P17 Per-iteration knowledge retrieval from a vector DB — *Plausible; precedents exist*
This is a memory layer (product-key memory; Memory Layers at Scale; PEER) whose values are *populated from a knowledge pipeline* rather than learned from scratch, plus RETRO's "don't spend weights on facts". Architecturally it is one more expert reached over the same fabric. Pitfalls: entailment extraction at scale is expensive and noisy; values must live in the model's latent space (learned projection, or learned values with pipeline-initialized keys); the model may learn to ignore a noisy table; poisoning and versioning become model-quality issues. Ladder L8 with a clean, small source.

### P18 Per-vector tiny QLoRAs; bespoke experts; shared spine — *Plausible; mostly engineering*
Multi-LoRA serving with per-token adapter gather exists (Punica/S-LoRA kernels). Specific to this design: an adapter on the *shared* middle block applies ~128 times per token — cheap and powerful, possibly unstable. Bespoke experts need a tenant mask in the router. Isolation: shared batches leak timing; regulated tenants may refuse. Ladder L11 + S8.

### P19 One giant DC, arctic or orbital — *Opinion*
The latency point is fair for most workloads; data residency, blast radius, power procurement and politics are why nobody does it. Record, don't test.

## 5. Claim-status table

| Claim (note) | Status | Handled in |
|---|---|---|
| Parallel attn/FF halves decode latency | Overstated (upper bound) | 01 §4.1, cost model |
| Serial att-then-FF models are "silly" | Opinion | — |
| Early-mask 2:4 sparsity matches dense | Unclear | L10a |
| Coarse 64-wide experts lose nothing | Unclear; conflicts with fine-grained scaling laws | L7 |
| Fabric costs 1/64 per node "without MoE complications" | Established (cost) / Overstated (complications) | 02 §3–4 |
| Repeated layer can simulate the layered model | Established for FF with depth conditioning; not for shared attention | 01 §4.2 |
| Variable depth cuts KV cache | Plausible with defined semantics | 01 §4.7, §6.4 |
| Two streams halve decode latency | Plausible under an equal-FLOPs, bandwidth-bound baseline; unclear otherwise | Q1, L4 |
| Two streams halve prefill FLOPs | Plausible (relative to two-stream decode cost) | 01 §10 |
| MTP from the start is standard | Established | L3 |
| Disaggregation is "silly" | Overstated | 02 §6, S5 |
| O(log n) attention with KV on SSD has enough bandwidth | Unclear; needs a locality model | S6 |
| 4 GB HBM per chip suffices | Plausible (inference, this design only) | S7 |
| Nobody sells bespoke experts / multi-LoRA | Partly out of date | 02 §10 |

## 6. Questions for the author
See `docs/04 §Open questions` (Q1–Q10). Three block design decisions: the intended baseline for the P12 latency claim; whether attention parameters are shared across middle-layer iterations; how training is meant to interact with continuous execution and variable depth.
