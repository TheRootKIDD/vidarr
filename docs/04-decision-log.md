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
Context: note P12 says the prediction stream cannot be seen by attention. Decision: $c$ is the only K/V source; $p$ queries $\{c_s : s \le t\}$. Variant L4b: $p$ queries $\{c_s : s < t\}$ plus its own embedding. Consequences: prefill is $c$-only; the note's decode-latency claim is evaluated against an equal-FLOPs, bandwidth-bound baseline (see Q1). Bears on H12.

### ADR-003 — Depth conditioning
2026-09-02 · accepted
Decision: learned depth embedding $e_j$ feeds norm modulation, router and halting head; per-iteration attention LoRA is a separate ladder step (L5b). Consequences: the "simulate the layered model" argument becomes testable. Bears on H6.

### ADR-004 — Router shape
2026-09-02 · accepted
Decision: product-key scoring by default; hierarchical unit-then-expert routing as an option aligned with placement; aux-loss-free bias balancing plus a small auxiliary loss; additive tenant mask ($0 / -\infty$). Bears on H4, H9, H11.

### ADR-005 — Methodology
2026-09-02 · accepted
Decision: ablation ladder with two matched-FLOPs baselines; append-only experiments; the simulator arbitrates systems claims; numeric thresholds fixed before runs (ADR-013).

### ADR-015 — Compute envelope
2026-09-02 · accepted
Context: the project runs on one workstation — 4 × RTX 3060 12 GB, Threadripper PRO 3945WX, 128 GB DDR4, no NVLink or PCIe P2P (`docs/06`). Decision: three sizes `screen` / `small` / `medium` matched to 12-layer-768 and 24-layer-1024 dense models at 1 B / 2.5 B / 7 B tokens; DDP with replicated experts up to `small`, FSDP at `medium`; ≤ 10 GB per GPU; cross-GPU expert/tensor parallelism only in measurement scripts (S0, S10); FP8/FP4 quality-only; 2:4 measured at inference; in-training retrieval restricted to GPU product-key memory (L8) with chunk-level offline retrieval as L8b; every hardware number used in a budget or scenario is measured by `scripts/bench/`, nominal specs are placeholders. Consequences: anything ≥ 1 B dense-equivalent or long-context pretraining is out of scope; scaling statements are provisional; `r` = 8 at `small`. Bears on every H; supersedes the size placeholders in `docs/03 §2`.

## Proposed / pending
- **ADR-006** — Early/final blocks: dense, or a small local MoE.
- **ADR-007** — Fixed $r$ schedule for prefill vs learned halting in prefill.
- **ADR-008** — Local and indexed attention: one softmax, or two gated branches.
- **ADR-009** — Knowledge-memory values: frozen (pipeline embeddings + learned $W_o$) vs trainable. Constraint from ADR-015: the in-loop table is GPU-resident (2²⁰–2²¹ entries at `small`); external retrieval is chunk-level and precomputed (L8b).
- **ADR-010** — ACT output: halting-weighted average vs last state.
- **ADR-011** — KV policy for copied states of halted tokens (compute-once-and-cache vs lazy).
- **ADR-012** — MTP heads: sequential modules vs independent heads.
- **ADR-013** — Numeric thresholds for H1–H18 and the `small` / `medium` sizes.
- **ADR-014** — Eval suite and pretraining corpus.
- **ADR-016** — Training regime for variable depth: lockstep SGD with ragged-depth masks (ACT), continuous-scheduler effects studied by staleness emulation (E-Q3 below) rather than by asynchronous training. Proposed as the working interpretation of Q3.

## Open questions

### For the note's author
- **Q1** *(interpretation fixed by ADR-002; a contrary answer supersedes it)* P12: we read the decode-latency claim as: at equal FLOPs per token the two-stream model has half the parameters, hence half the weight bytes per decode step, hence half the latency *when bandwidth-bound*. Is that the intended baseline? If so — since P4 makes the expert path compute-bound by construction — where do you expect the halving to apply: attention/skeleton only, or do you expect units to run bandwidth-bound in practice?
- **Q2** *(interpretation fixed by ADR-003; a contrary answer supersedes it)* P6: are attention weights and norms shared across the 128 middle-layer iterations, or only the expert pool? Is the router given the iteration index?
- **Q3** *(interpretation fixed by ADR-016 once accepted)* P7 / P11: during *training*, is execution lockstep? How do variable $r_t$ and the continuous scheduler reconcile with synchronous SGD?
- **Q4** P4: one tensor-parallel expert across 64 chips, versus 64 small experts inside the unit — was TP chosen for utilisation, for granularity, or both?
- **Q5** P7: when token $s$ halts at $r_s$, what do later iterations of later tokens attend to for $s$?
- **Q6** What total expert count and parameter budget is envisaged? (Our estimate for a 64-chip unit at $d$ = 8192: 0.2–0.8 B params per expert depending on tile width, hundreds of experts resident per unit.)
- **Q7** P3: weight sparsity only, or activation sparsity too? Fixed mask after 1% of tokens, or periodically updated?
- **Q8** P5: what is the "asymmetric-bw 4×4×4 torus" — different bandwidth per dimension, or per direction?
- **Q9** P15: is the index per head, per iteration, or shared? Any view on training with the index in the loop?
- **Q10** P17: is the entailment table frozen after construction, and are the value vectors learned or fixed?

### Internal
- **I1** Which published expert-parallel deployment to calibrate the simulator against (candidate: DeepSeek-V3's published inference figures). The second calibration point is the local S0 run (`docs/06 §6`).
- **I2** Whether `small` can show anything about H6 (recurrence may need scale).
- **I3** Licence and provenance of the knowledge source for L8.
- **I4** Which `note64` / `gpu_today` scenario numbers can be sourced from public vendor data vs must remain placeholders (`local_3060` is fully measured).
- **I5** Does host-bounced NCCL on the four GeForce cards deliver enough all-to-all bandwidth for S0/S10 to be meaningful, or must those be run with 2 cards / smaller shapes? Decide from `bench_nccl` results.
- **I6** Tokenizer: train a 32 k BPE on the corpus vs reuse an open 32 k vocabulary (feeds ADR-014).

### Resolution plan — what we can settle ourselves, and at what cost
Cost units from `docs/06 §4`: one `screen` run ≈ 8 h (1 B tokens, 1 seed); one `small` pair ≈ 2 days (2 seeds); one `medium` run ≈ 8 days; *bench* = hours on the rig; *sim* = CPU hours once `sim/` exists (Phase-2 engineering is not charged per question). "Author-only" is the part no experiment can answer: what they meant. "Scale-only" is what our envelope cannot reach at any budget.

| Q | Author-only / scale-only | What we can determine at our scale | How | GPU cost | Prior from the literature (Phase 0, free) |
|---|---|---|---|---|---|
| Q1 | which baseline they meant; whether their units run bandwidth-bound | (a) *mechanism*: on which parts of the design the 2× applies and above which batch it vanishes; (b) *quality premise*: two $d$-streams vs one √2·$d$ stream at equal FLOPs (H12) | (a) `bench_gemm` roofline + `costmodel/`; (b) L4 + a √2·$d$ single-stream comparator at the same depth (attention-score FLOPs differ by √2 — match totals in `costmodel/`) | (a) hours; (b) 2 `screen` ≈ 16 h; decisive: 2 `small` pairs ≈ 4 d | — |
| Q2 | what they intended to share; behaviour at $r$ = 128, $d$ = 8192 | which sharing wins at our scale: experts only (**L5c**, per-iteration attention) vs everything (L5) vs everything + per-iteration LoRA (L5b); router with vs without $e_j$ | L5 / L5b / L5c + one router ablation | 4 `screen` ≈ 1.3 d; top two at `small` ≈ 4 d | MoEUT (shares attention with grouping tricks); Relaxed Recursive Transformers (per-iteration LoRA recovers most of the gap) |
| Q3 | whether they imagine lockstep training; DC-scale behaviour | (a) that a lockstep recipe for variable $r_t$ trains (L6); (b) **E-Q3 staleness emulation**: each expert reads and applies weights delayed by $\delta_e$ steps drawn from S2's queue-wait distribution — does non-lockstep execution cost convergence? | L6 (in ladder) + 3 staleness settings; needs S2 first for the $\delta$ distribution | 3 extra `screen` ≈ 1 d | asynchronous / stale-gradient SGD results |
| Q4 | why TP was chosen; a real 64-chip unit | quality of $g$ = 1 vs $g$ = $U$ at matched FLOPs and unit count (H4); where TP inside a unit loses on a poor link (S10, measured) | L7 $g$-sweep; S10 | 3 `screen` ≈ 1 d; 2 `small` pairs ≈ 4 d; S10 ≈ 1 d engineering + hours | Krajewski 2024, DeepSeekMoE (granularity helps) |
| Q5 | intent | best semantics for halted tokens: copy final state and cache KV once vs recompute per iteration vs mask out | 3 L6 variants (settles ADR-010/011) | 3 `screen` ≈ 1 d; winner at `small` ≈ 2 d | CALM |
| Q6 | their target budget | what *fits*: params per expert and experts per unit as functions of $U$, $s_{min}$, $d$, $m_C$ | `costmodel/` | hours, no GPU | — |
| Q7 | intent | fixed early mask vs periodically updated mask vs dense at 2× width; 2:4 speed at inference shapes | L10a variants; `bench_sparse24` | 3 `screen` ≈ 1 d; bench hours | lottery-ticket / RigL: fixed early masks lose at high sparsity; 50 % is mild |
| Q8 | what "asymmetric-bw torus" means | which reading (per-dimension vs per-direction) works better as a unit fabric | S1/S3 with two torus scenarios | sim, CPU hours | — |
| Q9 | intent; behaviour beyond 32 k context | shared vs per-head indexer; index-in-loop vs dense-then-distil vs trainable compression | 3 L9 variants + context-extension fine-tune + 8–32 k evals over the real tiers | 3 `screen` + evals ≈ 2 d; winner at `small` ≈ 2 d | DSA (V3.2-Exp: one shared indexer, dense-teacher warm-up); NSA |
| Q10 | intent | frozen vs trainable values; frozen vs refreshed table | 2 L8 variants; pipeline build | 2 `screen` ≈ 16 h + pipeline ≈ 1 d (CPU + hours of embedding on one card) | Memory Layers at Scale (trainable); RETRO (frozen retriever) |

**Totals.** ≈ 23 `screen` runs ≈ 8 days of GPU time, of which ≈ 10 are ladder steps already budgeted → ≈ 5 extra days at `screen`. `small` confirmations for the three questions that shape the spec (Q1, Q2, Q4) ≈ 2 weeks. Not determinable at any local budget: the author's intent (every Q), Q6's target, and the scale-dependent parts (Q2 at 128 iterations, Q3 at DC scale, Q4 with a real 64-chip unit, Q9 beyond 32 k) — those remain author questions or labelled simulator estimates.

**Order.** Free first: literature priors for Q2, Q4, Q7, Q9, Q10 during the Phase-0 refresh; `costmodel/` for Q1(a) and Q6; `bench_nccl` for I5 (≈ 1 h, gates S0/S10). Then the `screen` variants in ladder order.

| I | How we settle it | Cost |
|---|---|---|
| I1 | pick the deployment, encode its published figures as `gpu_today`, reproduce within ± 20 % | Phase-2 engineering, days |
| I2 | L5 at `screen` then `small`; if silent, one `medium` L5 | 8 h / 2 d / 8 d |
| I3 | licence check on the knowledge source | hours |
| I4 | vendor data for `note64` / `gpu_today` | hours–days |
| I5 | `bench_nccl` on all four cards | ≈ 1 h, run first |
| I6 | train a 32 k BPE on a 1 B-token subset; decide by convention, no A/B | ≈ 1 h CPU |
