# bigmoe — Embla 1 follow-ups

Status: PROPOSED, 2026-11-18. Handoff from the spec session to the Claude Code instance on the rig. Everything below is additive: new hypotheses, ladder steps, one cost-model item, proposed ADRs and reading-list entries. Nothing here changes an accepted ADR or an existing result.

## 0. How to use this document (for Claude Code)

1. Read `README.md`, `CLAUDE.md`, `docs/03-research-plan.md` and `docs/04-decision-log.md` in the repo first. The ids below (H20–H26, L5e, L5g, L5d-nope, L9-relu2, L8e, L5f, E-router-EQB, L10b′, S11, ADR-025–029) were chosen against the spec as of 2026-09-02. If the repo has moved on, renumber to the next free ids and keep the names. Never reuse an id.
2. Merge, don't paste: hypotheses into `docs/03 §1`, ladder steps into `docs/03 §2` (order in §7 below), S11 into `docs/03 §3`, ADRs into `docs/04` under "Proposed / pending", reading-list entries into `docs/05`, the precedent notes into `docs/00` (P12, P15, P16, P17), config keys into the schema in `docs/01 §9`.
3. Every item keeps the project's rules: `screen` before `small`, thresholds fixed against the noise floor before running (ADR-013), one change per rung, two matched-FLOPs baselines, append-only experiments, measured hardware numbers only.
4. Verify every citation in §8. The 2026 entries are post-cutoff for the model that wrote this; the arXiv ids come from the Embla posts.
5. Commit as `docs: embla follow-ups (H20–H26, L5e/L5g/L5f/L8e/L9-relu2, S11, ADR-025–029)`, or per the repo's current prefixes.

## 1. Source and regime

Sevren's Embla 1 (Copenhagen, November 2026): 10 B total / 1 B active MoE built for phones. 32 layers of width 2048; 8 of 96 routed experts plus one shared expert per token; 15 sliding-window layers (128 tokens, RoPE, softmax), 5 full-attention layers (NoPE, ReLU² scoring) and 12 sparse layers that keep no K/V of their own and reuse the preceding full layer's cache restricted to its 1,024 highest-scored keys plus the 128 most recent; KV bridges that let prefill stop after 16 of 32 layers; block attention residuals; a 3 B-parameter hashed n-gram "engram" table read from memory rather than computed; ReLU² experts; Exact Quantile Balancing for the router; 1.4 T pretraining tokens with distillation from the first step; QAT in the last 2,000 SFT steps; Muon for weight matrices and AdamW for the rest.

Posts: `https://sevren.ai/blog/embla-1` and `https://sevren.ai/blog/embla-1-architecture`.

**Regime difference.** Embla minimises bytes read per token at batch 1 on a phone. The note's design is compute-bound at a U-chip unit with Z ≈ N_e · b_min / k tokens in flight. Two tests decide whether an Embla idea transfers:

- **Batching test.** A saving that comes from skipping *weights* (dead expert neurons) disappears once a batch unions the rows it needs: expected fraction of rows read ≈ 1 − (1 − p)^b for per-token firing rate p and batch b. A saving that comes from skipping *KV* (zero attention weights, selection reuse, bridges) survives batching, because each decode token reads its own sequence's cache and batching across sequences unions nothing.
- **Recurrence test.** Embla is layered; our middle block is one weight-shared block run r times. "Layer" maps to "iteration", "block of four layers" to "group of iterations", "the preceding full layer" to "the last full iteration".

## 2. Transfer table

| Embla idea | Batching test | Recurrence test | Disposition |
|---|---|---|---|
| KV bridges, prefill early exit | passes (prefill is per sequence) | G built from c^(j_b), j_b < r | L5e, ADR-026 |
| Full / sparse / window layer schedule with selection reuse | passes (KV-side) | full at every m-th iteration, reuse in between | L5g |
| RoPE only in local attention, NoPE in full | n/a | local range RoPE, G NoPE | L5d-nope, ADR-027 |
| ReLU² attention scoring, skip V reads | passes (per sequence) | global range only; local stays softmax | L9-relu2 |
| Engram (hashed n-gram table) | passes (one read per token, reused by every iteration) | second input to every iteration | L8e, ADR-029 |
| Block attention residuals | n/a | token-dependent mix over finished iterations | L5f |
| Exact Quantile Balancing | n/a | quantile per iteration vs per step is a real design choice | E-router-EQB, ADR-028 |
| ReLU² MLP, skip dead down-projection rows | fails at the note's batch (p ≈ 0.22 → 98 % of rows read at b = 16) | — | S11, cost model only; optional one-run quality ablation |
| QAT in the last 2,000 steps | n/a | — | L10b′ |
| Muon + AdamW | n/a | — | ADR-025, decided before L0 |
| Distillation from step one; embedding init from the teacher's principal components | — | — | not for the ladder: needs a teacher the rig cannot host at ladder throughput, and confounds architecture comparisons. Phase-5 recipe for useful models |
| OPD / GRPO / MOPD, Metal kernels, 256 K extension | — | — | out of scope |

## 3. New hypotheses

Thresholds are placeholders until ADR-013 sets them against the measured noise floor (`docs/06 §4`).

| H | Statement | Tier | Steps |
|---|---|---|---|
| H20 | G built from an early iteration (j_b ≤ r/4) costs ≤ 2 % loss vs G from the final vector at 2 k–8 k context, and prefill compute per prompt token drops by ≈ r/j_b | A + S | L5e, S5 |
| H21 | Full global attention at every m-th iteration with selection reuse in between (m = 4) matches full global attention at every iteration within 1 % loss; global reads per token drop ≈ m× | A + S | L5g, S6 |
| H22 | NoPE on the global range (RoPE local only) is within noise of RoPE everywhere, and ANN recall@K over NoPE keys is higher than over RoPE'd keys | A | L5d-nope, L9 |
| H23 | ReLU² scoring on global attention reaches ≥ 90 % exact zeros without a sparsity loss at ≤ 1 % loss cost; capping to the model's own nonzero set is lossless by construction; capping to K = 1024 costs ≤ 1 % at 8 k context | A | L9-relu2 |
| H24 | A hashed n-gram table with parameter count P matches a product-key memory of the same P within 1 % loss at matched FLOPs; the two together beat either alone | A | L8e, L8 |
| H25 | Attention residuals over iterations close ≥ 25 % of the L5-vs-L1 gap (H6) | A | L5f |
| H26 | EQB per iteration flattens the per-iteration expert histogram (entropy within 5 % of uniform at every j) and costs ≥ 1 % loss vs per-step EQB; per-step EQB matches or beats bias-update balancing | A | E-router-EQB |

## 4. New ladder steps

Common: same corpus, token budget and eval suite as the rung each builds on; one seed at `screen`, two at `small`; report d_ff, N_e, k and r next to every loss. Costs are in `screen` runs (≈ 8 h each on the rig, `docs/06 §4`).

### L5e — Bridged global cache (prefill early exit)

Builds on L5d. The global entry of token t is K_t = W_K^b · LN(c_t^(j_b)) and V_t = W_V^b · LN(c_t^(j_b)), with a separate bridge projection pair (not the attention's own W_K, W_V, which stay for the local range), j_b ∈ {1, 2, 4, r}; j_b = r is L5d.

- Prefill: prompt tokens run j_b iterations; the last W tokens of the prompt run all r iterations, because the first decode token needs their local per-iteration caches. Cost per prompt ≈ L · j_b + W · r iterations instead of L · r.
- Training: segment-recurrent as in ADR-021; G of earlier segments is bridged from their j_b states; stop-gradient per the ADR-023 default.
- Decode: unchanged (every token runs r iterations; its G entry is written after iteration j_b).
- Measure: loss; needle/copy and long-document perplexity at 2 k–8 k; S5 time-to-first-token for sequential vs bridged prefill.
- Prediction: j_b = r/2 is near-free; j_b = 1 costs more on retrieval tasks than on perplexity.
- Config: `attention.global_source: final_vector | bridged`, `attention.bridge_iteration: int`.
- Cost: 3 (j_b = 1, 2, 4).
- Precedents: YOCO; HySparse2; DeepSeek-V4.1-Flash's causal encoder-decoder; Embla's layers 20, 24 and 28.
- ADR-024: with j_b = 1 the segment-sequential critical path is one iteration per segment, which is the author's "less effective parallel prefill mode" made concrete; E-Q12 compares it with local-only prefill.

### L5g — Global-attention schedule with selection reuse

Builds on L5d or L5e. Iterations are classed full, sparse or local. Full iterations attend over all of G (dense; ReLU² if L9-relu2 is on). Sparse iterations attend over the K_sel keys scored highest by the last full iteration (selection by score, not by survival) plus the local window, with their own softmax. Local iterations attend to the window only. Schedules to run: `m = 1` (all full; equals L5d), `[F,S,S,S] × 2` (m = 4), `[F,S,S,S,L,L,L,L]`, `[F,S,S,S,S,S,S,S]` (m = r).

- With variable depth (L6) a token that halts before the next full iteration keeps its last selection.
- Measure: loss; long-context evals; global reads per token (counted), which feed S6.
- Config: `attention.schedule: list[str]`, `attention.k_sel: 1024`.
- Cost: 3.
- Precedent: Embla's 5 full / 12 sparse / 15 window schedule.

### L5d-nope — Positional split

Builds on L5d. RoPE is applied to q and k in the local range only; the global range uses no positional encoding. Variant: NoPE everywhere (expected worse).

- Also report ANN recall@K of an HNSW index over G keys, NoPE vs RoPE: the Q9 motivation.
- Config: `attention.rope_scope: all | local`.
- Cost: 1 (+1 for NoPE everywhere).
- Precedents: Embla (RoPE only in window layers); Kazemnejad et al. 2023 (NoPE).

### L9-relu2 — ReLU² global attention

Builds on L5d-nope. Global weight w_s = relu(⟨q̂, k̂_s⟩)², normalised by the sum of w_s over the global range, with q̂ and k̂ RMS-normalised. The local range keeps softmax. Guard: when the sum is zero the global branch contributes nothing; this needs the two-branch combination of ADR-008, so the one-softmax combination does not apply to this variant.

- Measure: loss; sparsity fraction over training, per iteration index j and per head; loss vs a cap K ∈ {256, 1024, 4096} applied at eval to a model trained uncapped; the fraction of keys for which all query heads sharing a KV head give zero (the V-skip rate).
- Indexer consequence: train the `01 §6.2` indexer as a classifier of "block contains a surviving key" against the model's own exact-zero support, instead of KL against a softmax teacher. Compare recall and loss with the dense-teacher plan.
- Config: `attention.global_scoring: softmax | relu2`.
- Cost: 2, plus eval sweeps.
- Precedents: Embla (50.8 % zeros at initialisation, about 95 % after training, no sparsity loss); So et al. 2021, Primer (squared ReLU).

### L8e — Engram table

Builds on L5, not on L8: it is L8's baseline. Hash the last-2 and last-3 token ids, with two different hash seeds, into a table of 2^21 rows × d; the two rows are summed (or concatenated and projected) into an engram vector e_t computed once per token before the middle block and exposed to every iteration: either as an additive branch with a zero-initialised output projection, or as a source in L5f's attention-residual mix.

- Size at `small` (d = 768): 2^21 rows ≈ 1.6 B parameters, ≈ 3.2 GB in bf16, ≈ 22 GB with fp32 master weights and Adam states; the 2^22 variant doubles these (≈ 45 GB). Both fit in host RAM, neither in VRAM.
- Rig placement: table and optimiser states in host RAM; rows gathered per step ≈ 2 × d × 2 B per token ≈ 1.5 GB at 0.5 M tokens per step; sparse gradient scatter and a CPU Adam update on touched rows. This is the P16/P17 "table off-chip" claim measured for real: log gather and scatter time per step.
- Compare at matched parameter count: L8e alone, L8 alone, L8e + L8.
- Config: `engram: { enabled, rows: 2097152, ngrams: [2, 3], placement: host | gpu }`.
- Cost: 2.
- Precedents: Embla's engram (3 B parameters, SSD-resident); DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next n-gram memories, prefetched from host memory.

### L5f — Attention residuals over iterations

Builds on L5. Before attention and again before MoE at iteration j, the input is a softmax-weighted mix of the sources {embedding, engram or memory read if present, outputs of iterations 1..j−1, partial sum of the current iteration}, with one learned query vector per mix (two per block), the depth embedding e_j added to the query, and per-token weights summing to one. Ragged depth: halted tokens stop contributing sources, so the softmax simply has fewer entries.

- Measure: loss; the mix weights per j (which iterations deep iterations read from).
- Config: `block.input_mix: residual | attention_residual`.
- Cost: 1–2.
- Precedents: Kimi Team 2026, Attention Residuals (arXiv 2603.15031); Embla's 8 blocks of 4.

### E-router-EQB — Exact Quantile Balancing

Ablation inside L5 at N_e = 8 and N_e = 128. Variants: (a) bias-update, aux-loss-free plus a small auxiliary loss (current ADR-004); (b) EQB per iteration: after every step, each expert's selection bias is set to the exact cutoff that would have given it k/N_e of that iteration's tokens; (c) EQB per step: the cutoff is taken over all (token, iteration) pairs of the step. Selection uses the biased scores; gate weights use the unbiased scores. Cutoffs by exact selection over bf16 scores (one block, so one all-reduce pair per iteration per step).

- Measure: loss; per-iteration expert-histogram entropy; co-activation statistics for the placement controller.
- Prediction: (b) kills depth specialisation and costs loss; (c) matches or beats (a).
- Config: `router.balance: bias_update | eqb_iteration | eqb_step`.
- Cost: 2 (variants b and c; a is L5 itself).
- Precedents: Neitemeier et al. 2026 (arXiv 2609.28053); Aleph Alpha, Kolibri-1 (384 experts per layer).

### L10b′ — Late QAT

Builds on the L0 and L1 checkpoints. Fake-quant to 4 bits (round-to-nearest, groups of 64, bf16 scale and offset per group, weights only) in the last 2 % of steps, vs from step 0, vs post-training round-to-nearest. Quality only; Ampere has no FP4 kernels.

- Cost: ≈ 0.2 per variant.
- Precedents: Embla (QAT in the last 2,000 SFT steps); Jacob et al. 2018.

### S11 — Batch dependence of activation sparsity (cost model, no GPU)

For per-token firing probability p and batch b, the expected fraction of expert down-projection rows read is 1 − (1 − p)^b. Tabulate for p ∈ {0.1, 0.22, 0.5} and b ∈ {1, 4, 16, 64, b_min(scenario)}, and add the per-sequence KV case, where the fraction read equals the per-query sparsity and does not depend on b. Put the table in `docs/00` under P3 and P16 as the reason activation sparsity is a phone-regime trick and KV sparsity is not.

### ADR-025 — Optimiser (decide before L0)

Muon for 2-D weight matrices, experts included; AdamW for embeddings, norms, router, halting head and the bridge, engram and memory tables; constant learning rate with a short cooldown. Decide by one `screen` pair on L0 (AdamW vs Muon + AdamW); the winner is the ladder optimiser for every rung. Cost: 1 extra `screen`. If the ladder has already started under AdamW, do not switch mid-ladder; record Muon as a Phase-5 item instead.

## 5. Proposed ADRs (pending; results decide)

- **ADR-025** — Optimiser, as above.
- **ADR-026** — Global cache source: bridged from iteration j_b (default j_b = r/2 if H20 holds), superseding "final vector" as the default in ADR-021 while keeping one entry per token.
- **ADR-027** — Positional encoding: RoPE in the local range only, NoPE on the global range, if H22 holds; makes ADR-008's two-branch combination the default.
- **ADR-028** — Router balancing: EQB per step replaces bias-update if H26 holds; EQB per iteration is recorded as rejected, with its histogram evidence.
- **ADR-029** — Knowledge-memory baseline: L8 (product-key memory) must beat L8e (engram) at matched parameters to stay in the design; otherwise the engram becomes the P17 mechanism and the product-key memory an optional addition.

## 6. Spec touch-ups that need no experiment

- `docs/00` P16 and P17: add the precedent that a 3 B-parameter n-gram table read from SSD or host memory shipped in Embla 1, and that DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next use the same mechanism; raise "parameters read from memory rather than computed" to Established as a mechanism, with quality still Unclear.
- `docs/00` P12: the prefill-halving half of P12's claim is delivered by L5e without two streams.
- `docs/00` P15: ReLU² sparsity is O(n) in K reads and about O(0.05 n) in V reads; the index is still needed on the K side.
- `docs/01 §4.3` and `§6`: note that NoPE global keys are directly indexable by content.
- `docs/02 §5` and `§6`: prefill modes gain "bridged" (ADR-024 option b with j_b).
- `docs/04` open question Q9: add that Embla- and HySparse2-style selection reuse answers the query-count part; the sub-linear K-side structure remains ours.

## 7. Suggested order and budget

ADR-025 pair → L5d-nope → L5e (j_b sweep) → L9-relu2 → E-router-EQB → L5g → L8e → L5f → L10b′; S11 whenever the GPUs are busy. `small` confirmations: L5e and L9-relu2 first.

Budget: about 16 `screen` runs (≈ 5–6 days of GPU time) plus 2 `small` pairs (≈ 4 days).

## 8. Reading-list additions (verify each)

- Sevren 2026, Embla 1 (blog, November 2026); Sevren 2026, The Architecture of Embla 1 (blog, November 2026).
- Sun et al. 2024, YOCO (arXiv 2405.05254) — already listed under P15.
- Wei et al. 2026, HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing (arXiv 2609.26368).
- DeepSeek-AI 2026, DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression (arXiv 2609.19969).
- Kimi Team 2026, Attention Residuals (arXiv 2603.15031).
- Qwen Team 2026, Qwen3.8-Flash-Next technical report (GitHub).
- Neitemeier et al. 2026, Exact Quantile Balancing and Load-Error Injection for Mixture-of-Experts (arXiv 2609.28053); Aleph Alpha 2026, Kolibri technical report, §2.1.5.
- So et al. 2021, Primer: Searching for Efficient Transformers for Language Modeling (squared ReLU).
- Kazemnejad et al. 2023, The Impact of Positional Encoding on Length Generalization in Transformers (NoPE).
- Jordan et al. 2024, Muon (blog); Jacob et al. 2018, Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (CVPR).
- Phase 5 only: Xu et al. 2024, Initializing Models with Larger Ones (ICLR); Agarwal et al. 2024, On-Policy Distillation of Language Models (ICLR).

## 9. Implementation traps

1. L5e: the final W prompt tokens must run full depth, or the first decode token has no local per-iteration caches to attend to.
2. L5e: the bridge projections are separate parameters; do not reuse the local attention's W_K and W_V, which see inputs from a different iteration.
3. L9-relu2: the normaliser can be zero for a query; use the two-branch combination (ADR-008) so the global branch can contribute nothing without producing NaNs.
4. L9-relu2 with L5g: selection is by score, not by survival; keep the pre-ReLU scores for ranking so that keys just below zero can still be selected by sparse iterations.
5. E-router-EQB: "per iteration" means the quantile over the tokens at that iteration index within the step; with ragged depth (L6) later iterations hold fewer tokens, so compute each cutoff over the tokens present.
6. L8e: hash collisions are intended; use different seeds for the 2-gram and 3-gram; hash across segment boundaries the same way in training and inference.
7. L8e on host RAM: pin the gathered rows; overlap the gather with the previous step's backward pass; the CPU Adam update on touched rows must finish before the next gather of the same rows, or allow one step of staleness and log it.
8. L5f with ACT: sources from iterations beyond a halted token's r_t do not exist; mask them in the softmax rather than zero-filling.
9. All attention steps: add attention-side bytes and reads counters to the simulator outputs (`docs/02 §14`) before running L5g or L9-relu2, so S6 gets its inputs.
