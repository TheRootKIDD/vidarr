# bigmoe — Decision log and open questions

## ADR template
```
### ADR-NNN — Title
Date · Status (proposed | accepted | superseded by ADR-MMM)
Context · Decision · Consequences · Bears on (H / L / S ids)
```

## Accepted

### ADR-001 — The hardware unit width $U$ is a parameter
2026-09-02 · accepted
Context: the note fixes 64. Decision: `U` and `s_min` are config knobs; Tier-A code simulates them by setting $d_{ff}$ and reporting utilisation. Consequences: results transfer across hardware generations; no kernel work before Phase 3. Bears on H4, H5.

### ADR-002 — Stream visibility
2026-09-02 · accepted (default), variant kept
Context: note P12 says the prediction stream cannot be seen by attention. Decision: $c$ is the only K/V source; $p$ queries $\{c_s : s \le t\}$. Variant L4b: $p$ queries $\{c_s : s < t\}$ plus its own embedding. Consequences: prefill is $c$-only. Amended 2026-09-02 after Q1: the latency claim is now derived (author's mechanism = ours at the utilisation floor, `docs/00 §P12`); `costmodel/` reports the latency floor with and without two streams; the feature itself is optional (ADR-020). Bears on H12.

### ADR-003 — Depth conditioning
2026-09-02 · accepted
Decision: learned depth embedding $e_j$ feeds norm modulation, router and halting head; per-iteration attention LoRA is a separate ladder step (L5b). Amended after Q2: the author considers the router's depth input optional (it can track depth from the state), so E-Q2 ablates the router without $e_j$; $e_j$ stays for norms and halting. Consequences: the "simulate the layered model" argument becomes testable. Bears on H6.

### ADR-004 — Router shape
2026-09-02 · accepted
Decision: product-key scoring by default; hierarchical unit-then-expert routing as an option aligned with placement; aux-loss-free bias balancing plus a small auxiliary loss; additive tenant mask ($0 / -\infty$). Bears on H4, H9, H11.

### ADR-005 — Methodology
2026-09-02 · accepted
Decision: ablation ladder with two matched-FLOPs baselines; append-only experiments; the simulator arbitrates systems claims; numeric thresholds fixed before runs (ADR-013).

### ADR-015 — Compute envelope
2026-09-02 · accepted
Context: the project runs on one workstation — 4 × RTX 3060 12 GB, Threadripper PRO 3945WX, 128 GB DDR4, no NVLink or PCIe P2P (`docs/06`). Decision: three sizes `screen` / `small` / `medium` matched to 12-layer-768 and 24-layer-1024 dense models at 1 B / 2.5 B / 7 B tokens; DDP with replicated experts up to `small`, FSDP at `medium`; ≤ 10 GB per GPU; cross-GPU expert/tensor parallelism only in measurement scripts (S0, S10); FP8/FP4 quality-only; 2:4 measured at inference; in-training retrieval restricted to GPU product-key memory (L8) with chunk-level offline retrieval as L8b; every hardware number used in a budget or scenario is measured by `scripts/bench/`, nominal specs are placeholders. Consequences: anything ≥ 1 B dense-equivalent or long-context pretraining is out of scope; scaling statements are provisional; `r` = 8 at `small`. Bears on every H; supersedes the size placeholders in `docs/03 §2`.

### ADR-009 — Knowledge-memory values are parametric
2026-09-02 · accepted (resolves Q10)
Context: the author learns the entailment-to-vector map during training, optionally starting from a pretrained embedding updated by backpropagation or through a layer in front; the table can be refreshed. Decision: $v_n = g_\theta(\mathrm{enc}(\text{entailment}_n))$, $\kappa_n = h_\theta(\mathrm{enc}(\text{entailment}_n))$ with a frozen pretrained sentence encoder and trainable $g_\theta, h_\theta$ (L8); fine-tuning $\mathrm{enc}$ by backpropagation is L8c, budget permitting; entries are precomputed and GPU-resident (ADR-015). Consequences: the table is extensible by running new entailments through the learned map; Amended after the author's addendum: the table is populated from plain **sentence embeddings** of a corpus slice (no entailment extraction) — the author's own simplification; entailment-style atomic facts (e.g. Wikidata triples rendered to sentences) are a later upgrade, L8d, to run only if L8 shows the table is used. Bears on H17.

### ADR-010 — ACT output is the last computed state
2026-09-02 · accepted (author's Q5). Bears on H7.

### ADR-011 — Halted tokens: K/V from the last state, stored once
2026-09-02 · accepted (author's Q5)
Decision: for $j > r_s$ later tokens attend to K/V computed from $c_s^{(r_s)}$; with attention weights shared across iterations this is one entry shared by all later iterations (exact); it is recomputed per iteration only when per-iteration attention adapters (L5b) are on. Consequence: a token halted at $r_s$ stores $r_s + 1$ KV entries instead of $r_{max}$. Bears on H7, H15.

### ADR-016 — Training is lockstep; continuous execution is inference-only
2026-09-02 · accepted (author's Q3)
Decision: synchronous SGD with a barrier per iteration and ragged-depth masks (ACT) for variable $r_t$; the continuous scheduler of `02 §5` is an inference mechanism. The staleness-emulation experiment (E-Q3) and any asynchronous training variant are dropped. Bears on H7, H11.

### ADR-017 — One KV cache per middle-block iteration
2026-09-02 · accepted (author's Q2)
Decision: `kv_share_every: 1` is the design; K/V sharing across iterations is an optimisation tested as L6b. Amended after v2 of the note (ADR-021): per-iteration caches exist only for tokens in the local range $W$; the persistent cache $\mathcal{G}$ holds one final-vector entry per token. Consequence: weight sharing does not shrink the local caches, but the persistent cache is ~$r_{max}$× smaller than a per-iteration global cache; memory budgets in `02 §8` count $W \cdot r_{max}$ transient entries per in-flight sequence plus one $\mathcal{G}$ entry per token. Bears on H6, H15, H16.

### ADR-018 — Local expert count
2026-09-02 · accepted (author's Q6)
Context: many experts in a billion-dollar installation, one or eight in a small local one. Decision: `small`/`medium` default $N_e$ = 8, $k$ = 2; $N_e$ = 128 ($k$ = 4) is the parameter-matched variant needed to test H6's "one block with $r n$ experts ≈ $r$ layers with $n$"; $N_e$ = 1 (dense recurrent block) is the cheapest rung. L5 runs all three at `screen`. Consequences: experts are wide at the default (`06 §3`); the thin-expert confound is confined to the 128 variant. Bears on H4, H6.

### ADR-019 — Sparsity is weight-only with a fixed early mask
2026-09-02 · accepted (author's Q7)
Decision: 2:4 on expert weights only, mask by magnitude after 1 % of tokens, then fixed; compared against dense at matched FLOPs. RigL-style updates and activation sparsity are out of scope. L10a is one run. Bears on H3.

### ADR-021 — Attention: local per-iteration, global final-vector (v2 P7)
2026-09-02 · accepted (from v2 of the note)
Context: v2 adds that per-iteration KV caches are seen only with local attention, while global attention sees only the final middle-layer vector of each token; a token is globally visible once complete; far tokens are complete because they sit in earlier batches. Decision: `01 §4.3` — local range $W$ over same-iteration caches; global range over $\mathcal{G}$ (one entry per completed token); segments of a document in order, documents in parallel; training as segment recurrence with stop-gradient memory by default (ADR-023). The index of `01 §6` lives over $\mathcal{G}$: one structure per sequence, shared by iterations and heads. Ladder: L5 keeps the standard semantics as baseline, L5d switches to $\mathcal{G}$ (H15a). Consequences: KV tiers, memory budgets, prefill modes and training regime revised (`02 §5.4, §6, §7, §13`); the SSD/HBM claims (P15, P16) now rest on a cache $r_{max}$× smaller. Bears on H7, H15, H16.

### ADR-022 — Local range is a sliding window over per-iteration caches
2026-09-02 · accepted (derived from v2 P7, not asked)
Decision: local attention is a sliding window of $W$ tokens over same-iteration caches; it crosses the segment boundary into the previous segment, whose per-iteration caches are retained until they leave the window (they are complete, so there is no serialisation cost). In the sequential regime the in-flight region per document is one segment plus any speculative drafts; only the pipelined prefill mode of ADR-024 makes it deeper (≤ $r$ segments). Bears on H15a, H16.

### ADR-020 — Two streams are optional
2026-09-02 · accepted (author's Q1: "nice idea, not standard, easier to avoid")
Decision: single-stream default; MTP heads read the single stream; $F_p$ prediction-only blocks exist only with two streams; L4/L4b move to the end of the ladder and run only if budget remains. Consequences: every $d_{ff}$ in `06 §3` is the single-stream value; H12 becomes optional. Bears on H12, H13.

## Proposed / pending
- **ADR-006** — Early/final blocks: dense, or a small local MoE.
- **ADR-007** — Fixed $r$ schedule for prefill vs learned halting in prefill.
- **ADR-008** — Local and indexed attention: one softmax, or two gated branches.
- **ADR-012** — MTP heads: sequential modules vs independent heads.
- **ADR-013** — Numeric thresholds for H1–H18 and the `small` / `medium` sizes.
- **ADR-014** — Eval suite and pretraining corpus.
- **ADR-024** — Parallel prefill mode for long prompts (the author's "less effective mode"): (a) all segments at once with local attention only, global cache filled afterwards from context-poor final vectors; (b) pipelined segments, segment $k+1$ at iteration $j$ attending globally to segment $k$'s iteration-$j$ caches (transient per-iteration global cache during prefill, in-flight depth up to $r$ segments). Both are inference-time modes of a sequentially trained model; E-Q12 evaluates both against sequential prefill on continuation loss and TTFT (S5). If both degrade, a mixed-mode training variant becomes a later ladder step.
- **ADR-025** — Skeleton block count and width in the FLOPs budget; derived $d_{ff}$ rounded to a multiple of 64. Context: `costmodel/` reproduces `06 §3`'s recurrent widths (1665, 833) exactly only with **two** skeleton blocks in total, while `01 §1` states $E$ = $F$ = 2 (four → 1153, 577); the skeleton's own $d_{ff}$ is stated nowhere (assumed 4$d$); and odd derived widths cost 12–27 % of GEMM throughput (`004-gemm-alignment`). Full text: `costmodel/README.md §A`.
- **ADR-026** — L4 (two streams) requires an explicit $k$ or $r$ trade at `small`: the "halves every $d_{ff}$" sentence of `06 §3` is false (the width left is 362, below the tile floor; at the floor the config overspends by 11 %). `costmodel/README.md §B`.
- **ADR-027** — Key-count convention for FLOPs budgets ($W$ alone vs $W + |\mathcal{G}_{visible}|$ at the training context): a 13 % swing in $d_{ff}$; the doc's ranges match only the local-window reading. `costmodel/README.md §D`.
- **ADR-023** — Segment memory gradient: stop-gradient into earlier segments' final vectors (Transformer-XL / Memorizing-Transformer practice, cheap) vs backpropagation through the previous segment (more faithful, ~2× activation memory). Default: stop.

## Open questions

### For the note's author
- **Q1** *(answered — see below; ADR-020)* P12: we read the decode-latency claim as: at equal FLOPs per token the two-stream model has half the parameters, hence half the weight bytes per decode step, hence half the latency *when bandwidth-bound*. Is that the intended baseline? If so — since P4 makes the expert path compute-bound by construction — where do you expect the halving to apply: attention/skeleton only, or do you expect units to run bandwidth-bound in practice?
- **Q2** *(answered in part — ADR-017; e_j optional; the author added detail to the note, see I7)* P6: are attention weights and norms shared across the 128 middle-layer iterations, or only the expert pool? Is the router given the iteration index?
- **Q3** *(answered — ADR-016)* P7 / P11: during *training*, is execution lockstep? How do variable $r_t$ and the continuous scheduler reconcile with synchronous SGD?
- **Q4** *(answered: latency and clos traffic, not granularity; H4 stays ours)* P4: one tensor-parallel expert across 64 chips, versus 64 small experts inside the unit — was TP chosen for utilisation, for granularity, or both?
- **Q5** *(answered — ADR-010/011)* P7: when token $s$ halts at $r_s$, what do later iterations of later tokens attend to for $s$?
- **Q6** *(answered — ADR-018)* What total expert count and parameter budget is envisaged? (Our estimate for a 64-chip unit at $d$ = 8192: 0.2–0.8 B params per expert depending on tile width, hundreds of experts resident per unit.)
- **Q7** *(answered — ADR-019)* P3: weight sparsity only, or activation sparsity too? Fixed mask after 1% of tokens, or periodically updated?
- **Q8** *(answered — `note64` scenario input)* P5: what is the "asymmetric-bw 4×4×4 torus" — different bandwidth per dimension, or per direction?
- **Q9** *(half-answered by v2: the index lives over the final-vector cache, per sequence, shared across iterations; the indexer and training remain ours)* P15: is the index per head, per iteration, or shared? Any view on training with the index in the loop?
- **Q11** *(assumed, not asked — ADR-021 amendment)* "See all the prior layer KV caches with local attention": read as "the per-iteration caches are the ones local attention touches", with same-iteration visibility (standard layered semantics). Derivation: P6's "simulate the standard layer structure" presumes it; the only constraint the author states concerns *final* vectors; seeing iterations $j' \le j$ as well is a superset that needs no serialisation either, so it is an optional ablation, not a question.
- **Q12** *(decided by us — ADR-022 accepted, ADR-024 proposed)* The in-flight depth follows from the segment-sequential regime (one segment per document, plus speculative drafts); the previous segment is complete when the current one starts, so a sliding local window may cross the boundary at no cost. Which "less effective" parallel prefill mode the author means is not derivable from the text, but it is an inference-time choice between two obvious candidates, evaluated on the L5d model in hours (E-Q12) — no training runs, no answer needed.
- **Q10** *(answered — ADR-009)* P17: is the entailment table frozen after construction, and are the value vectors learned or fixed?

### Author's answers (received 2026-09-02, in Danish; paraphrased)
- **Q1** — Nice idea, not standard; easier to leave out. Mechanism: each vector is half the work, the two run in one batch so a step takes half as long; each token fills two slots so a unit serves half as many tokens at once and throughput is unchanged; tokens in flight get answers twice as fast, with 2× hardware everyone does. → Same as our reading at the utilisation floor (`docs/00 §P12`); feature made optional (ADR-020).
- **Q2** — Not fully designed in the note. Easiest is to keep one KV cache per iteration (128 at full scale, fewer locally). The router need not be given the iteration index — it can keep track itself — but it may be. Detail was added to the note. → ADR-017; E-Q2 ablation; I7.
- **Q3** — Training must be lockstep or it is not deterministic; inference need not be, since weights do not change (but could be). → ADR-016; E-Q3 dropped.
- **Q4** — Tensor parallelism gives lower per-token latency at the same throughput, and it is what saves 64× on the network across the torus. Irrelevant for a small installation; this is for billion-dollar installations. → recorded in `02 §2`; H4 and S10 unchanged.
- **Q5** — Later tokens attend to the last computed vector. → ADR-010/011.
- **Q6** — Many experts in a large installation; maybe 1 or 8 in a small local one. → ADR-018.
- **Q7** — Weights only; fixed mask. → ADR-019.
- **Q8** — Bandwidths 1, 4, 16, 64; can run all-reduce at full speed with an adapted algorithm; not relevant on GPUs. → `note64` unit fabric in `02 §1`; confirm the per-level reading against the updated note.
- **Q9** — Does not know what "index" refers to; $n$ is the width of attention; a research project in itself. → open; terminology fixed in `01 §6.2`; our design.
- **Q10** — Can be refreshed continuously but need not be. Pipeline: all of Wikipedia → entailments → one or more vectors each, as input to training; the entailment-to-vector map is learned during training, or start from a pretrained embedding and update it by backpropagating into it (best if the database will be extended later) or by a layer in front; entailment design is a research project in itself. → ADR-009.
- **Q10, addendum** — One can also skip entailments altogether: make sentence embeddings and use those from there. → ADR-009 amended: L8 populates the table from plain sentence embeddings; entailment extraction becomes a later upgrade (L8d).

### v2 diff (word-level, v1 → v2; everything not listed is identical)
- **P4** — added: the unit width may be 32, 64, 128 or any number; wider halves the clos bandwidth requirement but enlarges the failure domain, the minimum expert size and the torus dimensions. → ADR-001 confirmed; $U$ becomes a sweep axis in S3/S9.
- **P5** — the local fabric list grew: fast router, asymmetric-bandwidth 4×4×4 torus, even-bandwidth 3-D 4×4×4 torus, 64-wide 1-D torus, 8×8 torus, doubled-bandwidth 8×8 mesh. → scenario variable in `02 §1`.
- **P7** — added (≈150 words): per-iteration KV caches are attended locally; global attention sees only each token's final middle-layer vector; using final vectors locally would serialise tokens and defeat prefill/training/speculative-decode parallelism; far tokens are in earlier batches and already complete, so global attention over final vectors needs no serialisation; this yields almost the full 128× KV reduction; batches of the same document become sequential; a super-parallel prefill would run in a weaker mode. → ADR-021, L5d, H15a, Q11–Q12.
- P13, P15, P18 show only page-break differences in extraction; the author's Q10 addendum (sentence embeddings without entailments) is already in ADR-009.

### Internal
- **I1** Which published expert-parallel deployment to calibrate the simulator against (candidate: DeepSeek-V3's published inference figures). The second calibration point is the local S0 run (`docs/06 §6`).
- **I2** Whether `small` can show anything about H6 (recurrence may need scale).
- **I3** Licence and provenance of the knowledge source for L8.
- **I4** Which `note64` / `gpu_today` scenario numbers can be sourced from public vendor data vs must remain placeholders (`local_3060` is fully measured).
- **I5** *(answered 2026-09-02, `002-nccl`)* $\lambda_{link}$ = 3.59 GB/s per rank, 44 µs floor. **S10 is viable and well-posed**: the `02 §2` TP condition is missed by 7× at the `small` default and 14× at the $N_e$ = 128 variant, so TP loses decisively and the crossover is swept by rung rather than observed. **S0 is measurable but fabric-bound by 2.7–4×**; record it as a fabric-dominated calibration point and take compute-side calibration from `003-gemm`; two cards is not a fix.
- **I6** Tokenizer: train a 32 k BPE on the corpus vs reuse an open 32 k vocabulary (feeds ADR-014).
- **I7** *(closed 2026-09-02)* v2 stored as `docs/source/The_big-DC_MoE_LLM_design_v2.pdf` with extracted text; diff below; `docs/00`, `01`, `02`, `03` revised (ADR-021).
- **I8** Which pretrained sentence encoder for the table (ADR-009), and whether its licence permits fine-tuning (L8c).
- **I9** Which corpus slice to embed for L8 at 2²⁰–2²¹ entries (e.g. lead sentences of Wikipedia articles), and how to sample the 2²¹ from it.
- **I10** *(downgraded to sequencing)* No NVMe was present (`000-env`: one SATA 850 PRO); a 4 TB NVMe is pending install. `bench_tiers --tiers storage` and $\beta_{cold}$ wait for it; never measure the cold tier on the SATA drive.
- **I11** *(closed, `003-gemm`/`010`)* 3 × GA104 + 1 × GA106: per-card $\phi$ spread 2.5 %, identical DDP step times — immaterial.
- **I12** *(resolved by the NVMe + 1 TB HDD)* 31 GiB free vs a ≈ 20 GB corpus; NVMe takes corpus, cold tier and indexes, HDD is bulk storage.
- **I13** The empirical batch floor (512 for 80 % of $\phi$) is 7× the roofline $b_{min}$ (79). `02 §2` tells the simulator's queue to use $b_{min}$; it must use the larger. Is the gap a GA106-class property or intrinsic to the roofline argument — i.e. what is it on `note64` hardware? Scenario files carry both numbers.
- **I14** `007-faiss-1m` recall is degenerate on random 768-d vectors; QPS stands (7.2 k at nprobe 8, 40× short of per-iteration retrieval, confirming `06 §5.1`). Re-measure recall on real sentence embeddings once I8/I9 settle. The 10 M run exceeded the 15-minute rule and has no result.
- **I15** 2:4 sparsity measured at 1.3–1.6× best and a loss below $b$ ≈ 2–4 k (`005/006`) against a nominal 2×. Framework path (PyTorch semi-structured) or silicon ceiling? A Phase-3 kernel question; until then ADR-013's H3 threshold prices the trade at ≈ 1.4×, not 2×.

### Resolution plan — what we can settle ourselves, and at what cost
*Revised after the author's answers. Author-only parts are now closed; the table keeps the experiments that still decide the spec.* Dropped: E-Q3 staleness emulation (Q3), the halted-token variants (Q5), the RigL and activation-sparsity variants (Q7), the frozen-values variant (Q10). Downgraded: L4 (Q1) to optional. Kept: E-Q2, L5 at three $N_e$, L5b/L5c, L7 sweep + S10, L9 variants, L8 vs L8c. Extra `screen` runs beyond the ladder ≈ 7 (≈ 2.5 days); `small` confirmations for Q2 and Q4 ≈ 8 days.

Cost units from `docs/06 §4`: one `screen` run ≈ 8 h (1 B tokens, 1 seed); one `small` pair ≈ 2 days (2 seeds); one `medium` run ≈ 8 days; *bench* = hours on the rig; *sim* = CPU hours once `sim/` exists (Phase-2 engineering is not charged per question). "Author-only" is the part no experiment can answer: what they meant. "Scale-only" is what our envelope cannot reach at any budget.

| Q | Author-only / scale-only | What we can determine at our scale | How | GPU cost | Prior from the literature (Phase 0, free) |
|---|---|---|---|---|---|
| Q1 | *closed* (mechanism given; feature optional) | quality premise H12 only, if budget remains | L4 + √2·$d$ comparator, after L11 | 2 `screen` ≈ 16 h (optional) | — |
| Q2 | *closed on KV (per-iteration) and router input (optional)*; scale behaviour at $r$ = 128, $d$ = 8192 stays scale-only | which sharing wins at our scale: L5 vs L5b (per-iteration attention LoRA) vs L5c (unshared attention, shared pool); router with vs without $e_j$ (E-Q2); L5 at $N_e$ ∈ {1, 8, 128} | ladder L5 family | 3 extra `screen` ≈ 1 d; top two at `small` ≈ 4 d | MoEUT; Relaxed Recursive Transformers |
| Q3 | *closed* (lockstep training, ADR-016) | that the lockstep ACT recipe trains (L6, in ladder) | L6 | 0 extra | — |
| Q4 | *closed on motive* (latency, clos traffic); a real 64-chip unit stays scale-only | quality of $g$ = 1 vs $g$ = $U$ at matched FLOPs and unit count (H4); where TP inside a unit loses on a poor link (S10, measured) | L7 $g$-sweep; S10 | 2 extra `screen` ≈ 16 h; 2 `small` pairs ≈ 4 d; S10 ≈ 1 d engineering + hours | Krajewski 2024, DeepSeekMoE |
| Q5 | *closed* (last state; KV stored once, ADR-010/011) | — | — | 0 | — |
| Q6 | *closed* (1–8 experts locally, ADR-018) | what fits per unit at scale | `costmodel/` | hours, no GPU | — |
| Q7 | *closed* (weights only, fixed mask, ADR-019) | H3 at our scale; 2:4 speed at inference shapes | L10a (one run); `bench_sparse24` | in ladder; bench hours | lottery-ticket literature |
| Q8 | *closed* (1 : 4 : 16 : 64) | how the hierarchical unit fabric behaves in S1/S3; confirm the per-level reading against the updated note (I7) | `note64` scenario | sim, CPU hours | — |
| Q9 | *half-closed by v2* (index over $\mathcal{G}$, per sequence); beyond 32 k stays scale-only | first the cost of final-vector-only global attention (L5d, H15a, no index); then shared vs per-head indexer, index-in-loop vs dense-then-distil (L9) | L5d; 3 L9 variants + context-extension fine-tune + 8–32 k evals over the real tiers | L5d 1 `screen` ≈ 8 h (+ `small` pair ≈ 2 d); L9: 2 extra `screen` + evals ≈ 1.5 d; winner at `small` ≈ 2 d | Transformer-XL, YOCO; DSA (V3.2-Exp), NSA |
| Q11–12 | *not asked*: Q11 assumed (same-iteration local), Q12 decided by us (ADR-022/024) | E-Q12: prefill modes (a) local-only vs (b) pipelined, evaluated on the trained L5d model — continuation loss and TTFT; optional Q11 ablation (local range over all iterations $\le j$) | evaluation only; one `screen` run if the ablation is wanted | Transformer-XL, Block-Recurrent Transformers |
| Q10 | *closed* (parametric values from sentence embeddings, ADR-009) | projection over a frozen encoder (L8) vs encoder fine-tuned by backprop (L8c); entailments vs plain sentences (L8d, only if L8 is used) | L8, L8c; embedding a corpus slice | 1 extra `screen` ≈ 8 h + embedding ≈ 1 h on one card | Memory Layers at Scale; RETRO |

**Totals (revised).** ≈ 16 `screen` runs, ≈ 9 of them ladder steps already budgeted → ≈ 7 extra runs, ≈ 2.5 days at `screen`; `small` confirmations for Q2 and Q4 ≈ 8 days. Still not determinable locally: the scale-dependent parts (Q2 at 128 iterations, Q4 with a real 64-chip unit, Q9 beyond 32 k) — labelled simulator estimates.

**Order.** Free first: I7 (updated note), literature priors for Q2, Q4, Q9 during the Phase-0 refresh; `costmodel/` for the two-stream latency floor and for what fits per unit; `bench_nccl` for I5 (≈ 1 h, gates S0/S10). Then the `screen` variants in ladder order.

| I | How we settle it | Cost |
|---|---|---|
| I1 | pick the deployment, encode its published figures as `gpu_today`, reproduce within ± 20 % | Phase-2 engineering, days |
| I2 | L5 at `screen` then `small`; if silent, one `medium` L5 | 8 h / 2 d / 8 d |
| I3 | licence check on the knowledge source | hours |
| I4 | vendor data for `note64` / `gpu_today` | hours–days |
| I5 | `bench_nccl` on all four cards | ≈ 1 h, run first |
| I6 | train a 32 k BPE on a 1 B-token subset; decide by convention, no A/B | ≈ 1 h CPU |

## Session log

### 2026-09-02 — Phase 0 bootstrap and measurements
Repo bootstrapped (`.venv` py3.12 / torch 2.13+cu130; faiss has no cp314 wheel), `costmodel/` with 29 worked-example tests, all of `scripts/bench/` except the storage tier, `sim/scenarios/local_3060.yaml`. Measured: $\phi$ 27.1 TFLOPS (above nominal), $\beta_C$ 342 GB/s, $s_{min}$ 256, batch floor 512 vs roofline 79, $\lambda_{link}$ 3.59 GB/s / 44 µs, host link 24–27 GB/s pinned with a 64 KiB batching knee, ANN 7.2 k QPS, `small` dense 22.1 k tok/s per card and 67.5 k aggregate. **Surprises:** no NVMe in the box; three of four cards are GA104 (harmless); 2:4 is worth 1.3–1.6× not 2× and loses below $b$ ≈ 2–4 k; odd expert widths lose 12–27 %; the empirical batch floor is 7× the roofline; DDP all-reduce is 31 % per micro-step; `06 §3`'s parameter counts are MHA figures under a GQA spec, and its recurrent widths only reproduce with two skeleton blocks (ADR-025/026/027 proposed, `costmodel/README.md`). Storage tier deferred to the NVMe.
