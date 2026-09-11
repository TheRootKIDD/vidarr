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

### ADR-014 — Pretraining corpus, tokenizer and eval suite
2026-09-02 · accepted
Context: `06 §8` named FineWeb-Edu and a 32 k vocabulary; I6 said decide the tokenizer by convention, not by A/B; I12 (disk) is closed, so the `06 §7` memmap-on-NVMe rule is now satisfiable. Decision: **corpus** = FineWeb-Edu `sample/10BT` (HuggingFaceFW/fineweb-edu, 14 parquet shards, 26.6 GiB, licence ODC-By 1.0 — attribution to the FineWeb authors and Common Crawl's terms of use; no restriction on training), stored raw and tokenised under `/mnt/nvme/corpus/fineweb-edu/`. **Tokenizer** = the Mistral-7B-v0.1 SentencePiece BPE (32 000 entries, Apache 2.0, ungated; BOS 1, EOS 2, no added BOS in the memmap): the most-used open 32 k Llama-style vocabulary, so $V$ = 32 000 in every FLOPs budget and `06 §3` count, and its `tokenizer.json` is pinned by SHA-256 in the corpus manifest. Documents are encoded without special tokens and joined with a single EOS; `uint16` memmap in 100 M-token shards (`scripts/data/tokenize_fineweb.py`), tokenised once. **Held-out split** = the first 10 000 documents of shard `000`, excluded from training, 11.48 M tokens — the fixed "held-out loss on the corpus" for every ladder step. **Tokenised 2026-09-02**: 9 672 101 documents, 11.01 B tokens (11.00 B train in 111 shards + val), 22.0 GB, 28 min on 12 cores; the Mistral vocabulary yields ≈ 10 % more tokens than the sample's GPT-2 count; tokenizer SHA-256 `11c08db2…bd1720`, full value in `manifest.json`. **Eval suite** (same for every step; loss-based, no generation, so an 80 M model scores above chance): (1) held-out FineWeb-Edu loss; (2) zero-shot log-likelihood accuracy on HellaSwag, ARC-Easy, ARC-Challenge, PIQA and LAMBADA under lm-evaluation-harness conventions — ARC and knowledge-heavy loss serve H17; (3) code: held-out loss on a fixed Python slice of a permissively licensed code corpus (The Stack, permissive subset; slice and licence recorded when L-runs start); (4) long context, from L5d/L9 on: synthetic needle and copy at 8–32 k, a RULER-style subset, PG-19 long-document perplexity, all over the real three tiers (`06 §8`). Thresholds per hypothesis are ADR-013's. Consequences: $V$ = 32 000 fixed; the tokenizer is never retrained mid-programme (a change is a new ADR and a full re-tokenisation); the raw parquet stays on the NVMe for provenance; the HDD holds nothing the runs read. Bears on every L step; H17 (knowledge evals), H15b (long-context evals).

### ADR-013 — Numeric thresholds and the resolvability rule
2026-09-02 · accepted
Context: `docs/03 §1` states thresholds as percentages of "loss"; `06 §4` puts the two-seed noise floor at 0.005–0.01 nats seed-to-seed, so nothing finer than ≈ 0.02 nats is resolvable at `small`; sizes are fixed by ADR-015/018; S0's calibration tolerance is ± 20 %. Decision — **quality (Tier A).** Every quality threshold is relative to the matched-FLOPs baseline of the same size and seed: $\Delta = \overline{L_{variant} - L_{ref}}$ over seeds on the ADR-014 held-out set, $L_{ref}$ = the L0 (dense) loss unless the step names L1; $\sigma$ = the pooled seed spread **measured from the L0 pair at `small`** and recorded in its `results.md` (it replaces the 0.005–0.01 estimate; if it exceeds 0.01 nats, every "1 %" test below needs a third seed). Percent thresholds are fractions of $L_{ref}$: "1 %" = 0.01 $L_{ref}$ (≈ 0.03 nats at $L_{ref}$ ≈ 3), "2 %" = 0.02 $L_{ref}$. **"0.5 %" thresholds (H8, H17-elsewhere, H3-FP4) are below the floor and become "no detectable regression": $\Delta \le 2\sigma$.** Verdict rule for a threshold $T$: *supports* if $\Delta \le T$; *weakens* if $\Delta > T + 2\sigma$; *silent* in between (and always silent at `screen`, one seed, unless $|\Delta| > 4\sigma$). "Matches" (H6, H12) means $|\Delta| \le 2\sigma$ in both directions. Per hypothesis: H2 $\Delta \le 0.01 L_{ref}$ and S1 per-iteration p50 latency ≥ 30 % lower with parallel branches than sequential in the same scenario; H3 quality at 2× width as stated ($\Delta \le 0.01 L_{ref}$; FP4 fake-quant on top: $\le 2\sigma$), but its *cost* is scored with the measured 2:4 factor **1.4×, not 2×** (I15): the cost-matched dense comparator is the width that 1.4× buys, and both comparisons are reported; H4, H12 $\le 0.01 L_{ref}$; H6 within $2\sigma$ at matched params and FLOPs; H7 mean $r_t \le 0.7\, r_{max}$ on held-out with $\Delta \le 0.01 L_{ref}$; H8 $\le 2\sigma$ with fabric bytes/FLOP halved in `costmodel/`; H13 greedy acceptance of heads 2/3/4 against the main head on held-out ≥ 70 / 55 / 45 % with main-head $\Delta \le 2\sigma$; H15a $\le 0.02 L_{ref}$ at 2 k, and at 8 k after the context-extension fine-tune; H15b PG-19 $\Delta \le 0.01 L_{ref}$ and needle/copy accuracy within 5 points of the un-indexed L5d at 8–32 k; H17 ARC-Easy + ARC-Challenge log-likelihood accuracy up ≥ 1 point (mean of the two) **and** knowledge-slice loss down by ≥ $2\sigma$, with held-out $\Delta \le 2\sigma$; H18 LoRA gain ≥ 0.8 × full-fine-tune gain on the same fine-tune eval at rank ≤ 4. **Systems (Tier S).** Thresholds are comparative — two scenarios in the same calibrated simulator — so systematic calibration error largely cancels; any *absolute* figure carries the ± 20 % S0 band and is reported with it. H1 cost/token ≥ 20 % lower (above the band); H5 ≤ 5 % utilisation loss vs an ideal fabric; H9 ≥ 3× fewer upper-tier bytes than random placement; **H10 $X$ = 10 %** (pooling two half-size installations gains ≥ 10 % utilisation — below that the gain is inside the calibration band); H11 ≥ 90 % expert utilisation with p99 ≤ 3× p50; H12-S the latency floor halves within 10 %; H14 throughput within 5 % of disaggregation at equal p99 TPOT; H15b-S cold hits × block bytes × throughput ≤ 0.5 $\beta_{cold}$ at the QD-16 block-size value of `local_3060` (2× headroom for the tail); H16 boolean fit at target $Z$; H18-S ≤ 5 % throughput. **Sizes** stay ADR-015/018. Consequences: `03 §1` reads "thresholds per ADR-013"; every `results.md` states $\Delta$, $\sigma$, $T$ and the verdict rule applied; a step whose $\Delta$ lands in the silent band is rerun with a third seed before anything is built on it; "0.5 %" claims of the note are downgraded to non-inferiority tests at our scale and said so in the write-up. Bears on H1–H18.

### ADR-025 — Skeleton block count and width; widths re-derived
2026-09-02 · accepted
Context: `06 §3`'s recurrent widths (1.5–1.8 k / 0.8 k) reproduce only with two skeleton blocks in total, while `01 §1` states $E$ = $F$ = 2; the skeleton's $d_{ff}$ is stated nowhere; derived odd widths cost 12–27 % of GEMM throughput (`004`); the dense parameter counts in `06 §3` are MHA figures under a GQA spec (`costmodel/README.md §A, §C`). Decision: **$E$ = $F$ = 2 means two early and two final blocks, as `01 §1` says** — the model spec wins and `06 §3`'s numbers were wrong; the skeleton's $d_{ff}$ is the conventional **4$d$** (SwiGLU, 3072 at `small`, 4096 at `medium`); every derived $d_{ff}$ is **rounded down to a multiple of 64**; all widths and counts in `06 §3` and `01 §1` are re-derived from `costmodel/` under ADR-027's key convention and become: `small` (0.6 GFLOP/token, $r$ = 8) $N_e$ = 8, $k$ = 2 → $d_{ff}$ = **896**, 16.5 M expert / 77 M total params; $N_e$ = 128, $k$ = 4 → **448**, 132 M / 193 M; $N_e$ = 1 → **1792**, 65 M total; `medium` (2.0 GFLOP, $r$ = 16) → **1920** (47 M / 144 M), **960** (378 M / 474 M), **3776** (108 M). Dense counts corrected to GQA: `small` 75.5 M non-embedding + 24.6 M embedding, `medium` 270 M + 33 M. Consequences: experts are thinner than the doc promised (the skeleton is 65 % of the non-expert budget); the tile floor is $U s_{min}$ with the **measured** $s_{min}$ = 256 — 256 at $U$ = 1 (training replicas), 1024 at $U$ = 4 — so **L7's $g$-sweep at $U$ = 4 cannot hold 896 and must trade $r$ or $k$ explicitly** (log it; $U$ ∈ {1, 2} is free); the per-GPU footprint at `small` drops (193 M params at $N_e$ = 128, not 250 M). `costmodel/` tests pin the new widths. Bears on H4, H6, ADR-015, ADR-018.

### ADR-026 — L4 (two streams) runs at 2× budget against a √2·d comparator
2026-09-02 · accepted
Context: `06 §3` claimed two streams "halve every $d_{ff}$ down to the $U s_{min}$ floor". Under $E$ = $F$ = 2 the non-expert terms of a two-stream `small` model already cost 0.66–0.78 GFLOP/token at $r$ = 8 — over the 0.6 GFLOP budget with **no experts at all** — and $r$ ≤ 3 is needed before any width is left (`costmodel/README.md §B`, tests). Decision: L4 is **not** budget-matched by shrinking $d_{ff}$; it runs at **2× the rung's budget** (1.2 GFLOP/token at `small`) with the single-stream widths, and its comparator is what H12 states: a **single-stream model at $d$ = √2 · 768 → 1088** (multiple of 64), same $r$, $N_e$, $k$, widths solved by `costmodel/` at the same 1.2 GFLOP. The "halves every $d_{ff}$" sentence is struck. Consequences: an L4 pair costs ≈ 2 × 2 `screen` runs (≈ 32 h, not 16); H12's quality half is a matched-FLOPs test at a *different* budget from the rest of the ladder, said so in the report; the systems half (latency floor halves, S1) is unaffected. Bears on H12, ADR-020.

### ADR-027 — Key-count and fabric-bytes conventions for budgets
2026-09-02 · accepted
Context: `01 §10` scores attention as $2 \cdot 2 d (W + |\mathcal{G}_{visible}|)$ per stream but no document says what $|\mathcal{G}_{visible}|$ is when budgeting, a 13 % swing in $d_{ff}$ (`costmodel/README.md §D`); `01 §10`'s fabric-bytes line is ambiguous between per stream and both (§E). Decision: **budgets count the FLOPs actually spent at the training context**: un-indexed global attention over a causal 2048-token context averages $(L+1)/2$ ≈ **1024 keys** per query ($W$ = 512 local + ≈ 512 global on average), and that is the number every anchor in `06 §3` assumes — the same rule charges the dense baseline its full causal attention, so matched FLOPs stay matched. Local-window-only counting ($W$ = 512) is *not* used for budgets; it remains the decode-time figure once the index bounds $|\mathcal{G}_{visible}|$ to $K_{idx} B_{idx}$. **Fabric bytes per token per iteration $2 k d\, b_{act}$ are per stream** (dispatch + combine of one $d$-vector to $k$ experts); two streams double it. Consequences: `01 §10` states both conventions; `costmodel.solve.budget_n_keys(context)` is the one place the 1024 comes from; every fabric number in `02 §3`, S0 and S10 is per stream unless the scenario says otherwise. Bears on every `06 §3` anchor; H8, H9 (fabric).

### ADR-006 — Early and final blocks are dense
2026-09-02 · accepted
Context: `01 §5` leaves the skeleton blocks "dense by default (ADR-006 pending)" with a small local MoE as the alternative; ADR-025 already budgets them as dense SwiGLU at $d_{ff}$ = 4$d$. Decision: **dense.** The $E$ + $F$ skeleton blocks are standard pre-norm parallel-form blocks with their own weights and a dense SwiGLU FF of width 4$d$; no routing outside the shared middle block. Rationale: the ladder isolates the shared block's MoE (H6); a second routed component in the skeleton would confound every L5 comparison and add a router per block to `model/` for no hypothesis of the note; the cost model's skeleton term is fixed by it. A skeleton MoE, if ever wanted, is a later variant (L5e) with its own ADR. Consequences: `01 §5` reads "dense"; the skeleton is ≈ 65 % of the non-expert budget (ADR-025), which is the price of four unshared blocks around a shared one. Bears on H6, L5.

### ADR-012 — MTP heads are independent, with subsampled auxiliary losses
2026-09-02 · accepted
Context: `01 §7` allows "small sequential MTP modules or independent heads"; the note's decode drafts all $m$ tokens from the heads in one pass, which is what independent heads give (sequential modules need $m$ − 1 dependent passes); each extra head costs a full logits pass ($2 d V$ ≈ 49 MFLOP/token forward at `small`, and the output head is already ≈ 25 % of the measured step, `009`), so $m$ = 4 on every position would add ≈ +75 % to the step. Decision: heads $2..m$ are **independent**: head $j$ = RMSNorm → a linear $d \to d$ adapter on $p_t^{out}$ (on $c_t^{out}$ single-stream) → the tied output embedding; no cross-head dependency. $m$ = 4 as the note states, so H13 tests heads 2/3/4. **The auxiliary losses are subsampled**: each position contributes to exactly one of heads $2..m$, chosen uniformly per position per step (`mtp.subsample` = 1/($m$ − 1)), so the extra logits cost is one head, ≈ +25 % step time, not three; loss weights $\lambda_2 \ge \lambda_3 \ge \lambda_4$ apply on top; each head's loss is the mean over its own subsampled positions, an unbiased estimate of its full-position loss, so no rescaling is needed (corrected during implementation, 2026-09-02). Acceptance per head is measured greedily against head 1 on held-out (ADR-013). Consequences: L3's training step is ≈ 1.25× L2's, stated in its `results.md`; sequential (DeepSeek-V3-style) modules become an optional L3b if H13's acceptance targets are missed by the independent heads; `01 §7` and the config schema carry `mtp: { m: 4, style: independent, subsample: auto }`. Bears on H13, L3.

### ADR-023 — Segment memory gradient: stop by default
2026-09-02 · accepted
Context: proposed with the default already in `01 §4.3`, `02 §2b` and the config (`segment_memory_grad: stop`). Decision: **stop-gradient** into earlier segments' final vectors (Transformer-XL / Memorizing-Transformer practice); `through` stays a config knob, ≈ 2× activation memory, for one ablation only if L5d is silent or weak on H15a. Consequences: L5d trains with detached memory; the per-GPU footprint of `06 §3` holds. Bears on H15a, L5d.

### ADR-029 — Reference-implementation decisions (Phase 1)
2026-09-02 · accepted
Context: `model/` had to settle details `docs/01` leaves open. Decision (full list in `model/README.md`): (1) the $\mathcal{G}$ entry is the shared block's $W_k/W_v$ applied to a dedicated RMSNorm of the final vector, without depth modulation; under ADR-023's `stop` the *input* is detached so the cache's own parameters train; (2) skeleton blocks attend causally within the current segment — only $M$ reads $\mathcal{G}$; (3) `per_iteration` = plain causal attention over the whole context per iteration (L5 baseline), `final_vector` = window $W$ within the segment plus $\mathcal{G}$ in one softmax; (4) a halted token's K/V is frozen at the first iteration after halting and reused unchanged; (5) MTP auxiliary losses are per-head means over subsampled positions, no ($m$ − 1) rescale; (6) per-iteration activation checkpointing and chunked tied-head cross-entropy are training-time knobs with no semantic effect; (7) the layered MoE baselines L1–L3 size each expert at $d_{ff}^{dense}/k$ (matched FLOPs), giving 271 M parameters at `small`. Consequences: `01 §4.3`, `§4.7`, `§7` read with these clarifications; ADR-008's gated two-branch attention remains unimplemented. Bears on L0–L7.

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

### ADR-030 — L3 joins the `small` list; H13 is not decided at `screen`
2026-09-11 · accepted
Context: `03 §2` lists the `small` steps as L0, L1, L2, L5, L5d, L6, L7, L9 — **L3 is absent**, so as written MTP gets exactly one `screen` run at 1 B tokens and is never revisited. That collides with **I20**, whose literature prior is that *"MTP hurts main quality below ≈ 1 B without a curriculum"*: the one budget at which L3 is scheduled is precisely the budget the prior says is too small. A failure there would be uninterpretable — indistinguishable between "the independent-head design is wrong" (ADR-012's premise) and "the token budget is too small" (I20's prior) — and the ladder would carry the wrong conclusion forward with no experiment able to correct it. `108-l3-screen` is mid-run and is tracking the prior: a main-loss gap of ≈ 0.13 nats against L2 that is flat in absolute terms, and head-2 acceptance climbing ≈ 0.8 points per 100 steps from 16.1 % at step 500, which extrapolates to ≈ 27 % against H13's 70 % target. **Decision is taken now, before `108` finishes, so it cannot be read as fitting the schedule to a result.**
Decision: **L3 is added to the `small` list** and runs at 2.5 B tokens, 2 seeds, alongside L2 as its comparator. **H13 is not decided at `screen`**: `108`'s outcome is recorded as evidence and explicitly not as a verdict, whichever way it falls. Acceptance per head *and* the main-loss delta against L2 are both reported at both budgets, so the budget effect is measurable as a difference rather than asserted.
Consequences: ≈ 17 h of extra `small` time at the measured L3 rate (`018`: 628 ms DDP micro-step, ≈ 26.1 k tok/s without accumulation; ≈ 43.3 k with, measured in `108`), against a `small` programme already budgeted at 16–18 days — under 5 %. `03 §2`'s `small` list is edited. **The identity of L3b remains unresolved and this ADR does not settle it**: ADR-012 and `01 §7` name it as *sequential DeepSeek-style modules*, while I20 names it as *a forward MTP curriculum*. Those are different experiments with different costs — sequential modules need $m-1$ dependent passes at decode, which cuts against the note's reason for wanting MTP at all, whereas a curriculum keeps the single-pass draft and changes only the training schedule. Whichever is built needs its own ADR, and the `small` result should inform that choice rather than the other way round. If L3 at 2.5 B closes most of the gap, the budget was the cause and L3b may be unnecessary.
Bears on H13, L3, L3b; touches I20.

## Proposed / pending
- **ADR-007** — Fixed $r$ schedule for prefill vs learned halting in prefill.
- **ADR-008** — Local and indexed attention: one softmax, or two gated branches.
- **ADR-028** — Unit model revisions from the author's hardware document (`docs/07 §4`): (a) two compute classes per chip — an FF core with a large array ($s_{min,ff}$ = 256–1024) and attention cores with small arrays ($s_{min,attn}$ ≈ 64), hence $\phi_{ff}$, $\phi_{attn}$ as separate scenario inputs; (b) the tile floor is a *row* floor, $N$ = $M$ × (LHS bw / RHS bw) rows per weight tile, with each token contributing 1 + $m$ rows under $m$-token speculation — the scheduler's queues count rows; (c) the reduce leg of the local fabric carries ≥ 16-bit partial sums: `02 §2` traffic becomes $b d\, b_{act} + b d\, b_{sum}$, $b_{sum}$ ≥ 2; (d) full pipelining inside the unit doubles tokens in flight and per-token latency — the Little's-law budget $Z$ of `02 §5.5` carries the factor; (e) router-based unit fabrics are ≤ 50 % link-efficient (avg. 2 hops), torus collectives 100 %: `02 §1`'s "switch" option is not equivalent to a torus; (f) the asymmetric 4×4×4 torus all-gather costs ≈ 3X/(8B) vs ≈ X/(2B) on a flat ring (our derivation from the author's algorithm) — the number S1/S3 must reproduce; (g) the failure domain is the 4-chip ring board, and a failed rail port is bypassed through torus neighbours at ¼ load (`02 §11`); (h) the replica-rounding utilisation floor $\ell / \lceil \ell \rceil$ reported beside the Little's-law model in S4. Also fills `note64.yaml` with the sourced ratios of `docs/07 §6`; absolute $\phi$, $\beta_C$, $\lambda_C$, $\nu_C$ stay placeholders (I4). Bears on H1, H5, H8, H9, H10, H11, H16; changes `02 §1, §2, §5, §11`.
- **ADR-024** — Parallel prefill mode for long prompts (the author's "less effective mode"): (a) all segments at once with local attention only, global cache filled afterwards from context-poor final vectors; (b) pipelined segments, segment $k+1$ at iteration $j$ attending globally to segment $k$'s iteration-$j$ caches (transient per-iteration global cache during prefill, in-flight depth up to $r$ segments). Both are inference-time modes of a sequentially trained model; E-Q12 evaluates both against sequential prefill on continuation loss and TTFT (S5). If both degrade, a mixed-mode training variant becomes a later ladder step.

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
- **Q14** *(from `docs/07`)* The document's recommended unit is an even-bandwidth 4×4 donut of 16 chips; the 64-chip 1:4:16:64 torus is a "nerd box" variant the author endorses only if every expert is 64-way tensor-parallel. Is the note's $U$ = 64 that variant — i.e. does P4 exist to make the asymmetric torus work — and is 16 the fallback if experts stay smaller?
- **Q15** *(from `docs/07`)* The document predicts that within-layer retrieval (P17) will shrink MoE to 4–8 experts; the note designs for thousands. Which is the intended end state, and does P17 reduce $N_e$ in the note's design?
- **Q16** *(rig, `022` — **answered 2026-09-10 by inspection**)* The bracket blocking the bottom four PCIe slots is a Lian Li 4-slot vertical GPU kit (VG4 family), shipped with a **200 mm** riser — which is why it could only sit at the bottom-*back*, why two cards had to go side by side, and why one of them holds 93 °C at 100 % fan (`022`). **The user reports the VG4's riser cable is replaceable: it detaches from the frame and a longer cable of the same kind fits — specifically the 900 mm `PW-PCIV-4-90X` type already in the case, which is what feeds the upright bracket on the side panel.** So the cable length is no longer what pins the bracket to the bottom-back, and the bracket can be relocated to free the slots that force the side-by-side pair. **The 900 mm cable on hand is not spare** — it is carrying the side-mounted upright bracket — so relocating the VG4 needs a second long riser unless the two brackets swap cables. Q16 is closed as a compatibility question; what remains is procurement and the physical move, not a fact to establish. See the session log entry below.
- **Q13** *(from the author's baseline remark)* Is the plain-transformer comparison our own L0 (same corpus, matched FLOPs per token, `06 §3`), or a specific published model? If the latter, which, and matched on what — parameters, training FLOPs, or tokens?

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
- **Baseline (unprompted, 2026-09-02)** — A plain transformer trained on the same corpus at a size relevant to ours should be the comparison point for everything we do on the new architecture. → Already the ladder's L0 and the "baselines travel with every experiment" rule (ADR-005): a dense pre-norm transformer (GQA, RoPE, SwiGLU) at matched training FLOPs per token, the same token budget, the same corpus (ADR-014) and the same eval suite, trained once per size and seed and reported next to every ladder step; L1 adds the fine-grained per-layer MoE as the second comparator. The author's intent confirms L0 as the *primary* comparator. "Relevant size" is read as **matched training FLOPs per token** (`06 §3`: 12-layer $d$ = 768 at `small`, 24-layer $d$ = 1024 at `medium`), the note's own cost unit, not matched parameters — a parameter-matched dense model would be a different, larger baseline; the parameter-matched comparison lives inside L5 ($N_e$ = 128) against L1. If the author means a specific published model rather than our own L0, that is a new question (Q13, below).

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
- **I6** *(closed 2026-09-02, ADR-014)* Reuse the Mistral-7B-v0.1 32 k vocabulary; decided by convention, no A/B.
- **I7** *(closed 2026-09-02)* v2 stored as `docs/source/The_big-DC_MoE_LLM_design_v2.pdf` with extracted text; diff below; `docs/00`, `01`, `02`, `03` revised (ADR-021).
- **I8** Which pretrained sentence encoder for the table (ADR-009), and whether its licence permits fine-tuning (L8c).
- **I9** Which corpus slice to embed for L8 at 2²⁰–2²¹ entries (e.g. lead sentences of Wikipedia articles), and how to sample the 2²¹ from it.
- **I10** *(closed 2026-09-02, `012-tiers-nvme`)* The 4 TB NVMe is installed at `/mnt/nvme`; $\beta_{cold}$ = 7.3 GB/s sequential, 6 GB/s random from 4 MiB, 46 µs / 70 µs random latency at 4 / 16 KiB (queue depth 1). The SATA drive was never measured. Cold is 10× warm in latency at KV-block sizes, so S6 batches cold fetches to ≥ 1 MiB or models the stall; see I16.
- **I11** *(closed, `003-gemm`/`010`)* 3 × GA104 + 1 × GA106: per-card $\phi$ spread 2.5 %, identical DDP step times — immaterial.
- **I12** *(closed 2026-09-02)* 3.6 TB free on `/mnt/nvme` (corpus, cold tier, indexes) and 916 GB on `/mnt/hdd` (bulk storage, never a KV tier) against a ≈ 20 GB corpus; the `06 §7` memmap-on-NVMe rule is satisfiable as written.
- **I13** The empirical batch floor (512 for 80 % of $\phi$) is 7× the roofline $b_{min}$ (79). `02 §2` tells the simulator's queue to use $b_{min}$; it must use the larger. Is the gap a GA106-class property or intrinsic to the roofline argument — i.e. what is it on `note64` hardware? Scenario files carry both numbers.
- **I14** `007-faiss-1m` recall is degenerate on random 768-d vectors; QPS stands (7.2 k at nprobe 8, 40× short of per-iteration retrieval, confirming `06 §5.1`). Re-measure recall on real sentence embeddings once I8/I9 settle, and run 10 M and `faiss-gpu` in that same job. The 10 M run exceeded the 15-minute rule; **1 M is accepted as the Phase-0 measurement** (`007-faiss-1m` §Decision): the 40× shortfall is structural at note scale (≈ 10⁸ QPS needed), not a property of this CPU, so ADR-009's memory-layer reading of P17 transfers to any hardware; only the cold-tier lookup ceiling and the offline L8b budget are rig-specific numbers.
- **I15** 2:4 sparsity measured at 1.3–1.6× best and a loss below $b$ ≈ 2–4 k (`005/006`) against a nominal 2×. Framework path (PyTorch semi-structured) or silicon ceiling? A Phase-3 kernel question; until then ADR-013's H3 threshold prices the trade at ≈ 1.4×, not 2×.
- **I20** Priors from the literature refresh (`docs/05 §Headline`) expect four of the note's claims to *fail* ADR-013's thresholds at our scale: H4 (granularity law: $g$ = 1 ≈ 3 % worse than $g$ = 4 at `small`), H3 (2:4 at equal width costs 1.5–3.8 %; no precedent for the 2×-width claim), H13 (MTP hurts main quality below ≈ 1 B without a curriculum), H15b (1 % is below Landmark's +1.5 % at ≈ 100 M). Thresholds stay — they are the note's claims — but each of these steps gets its recovery variant scheduled beside it rather than after: L7b ($L_e$ = 2) with L7, a forward MTP curriculum as L3b, and for H15b the RULER/PG-19 subset at 8–32 k that ≈ 100 M models can resolve (HELMET and LongBench v2 cannot). Also from `docs/05`: `007-faiss-1m` recall must be re-measured on real *block keys* (attention queries are out-of-distribution for IVF; 30–50 % of lists scanned in RetrievalAttention) before H15b-S is scored — extends I14.
- **I23** *(closes I22)* **Thermal throttling.** During `101-l1-screen` (14:46, after ≈ 19 h of continuous load) three cards read 91–93 °C with `sw_thermal_slowdown` active and SM clocks of 270 / 1320 / 1620 MHz against 2115 max; the fourth (GPU 2) sat at 58 °C and 1950 MHz. DDP runs at the slowest card, so the wall-clock gap of I22 is the 270 MHz card. The timed step missed it because the timer had no `cuda.synchronize` (fixed). Consequences per `06 §7`: **throughput numbers from `100-l0-screen` and the queued runs are invalid; loss numbers stand.** The trainer now logs min SM clock, max temperature and throttle count every 10 steps. `015-thermal-soak-170w` shows the three hot cards throttle within a minute while drawing only 85–105 W of their 170 W limit, so a power cap is not a remedy; the fix is airflow and spacing (`015` §Interpretation), verified by a 15-minute un-throttled soak before any run. The `014` bench ran cold and is unaffected. **2026-09-04, first cooling step** (`016-thermal-soak-bottom-fans`: three 140 mm bottom intakes plus top exhaust fans, cards not yet re-slotted): time to first throttle 50–70 s → 110–130 s, worst-card steady clock 328 → 1518 MHz, DDP step 1277 → 826 ms, but the three sandwiched cards still reach 93 °C with `sw_thermal_slowdown`; GPU 2 (free slot beside it) unchanged at 62 °C. Spacing is the remaining constraint; the acceptance soak waits for the bracket/riser and needs ≈ 1100 steps to last 15 min. **2026-09-10, after the rebuild and two re-slottings** (`017`, `019`, `021`, `022`): spacing is confirmed as the constraint — three cards now hold 1921–1958 MHz at 63–66 °C with zero throttled samples, while the fourth sits beside another card and throttles at 93 °C with its fan at 100 % and 113 W of a 170 W limit, costing the lockstep DDP step 13.4 %. No fan fault and no power fault: `019`'s 0 % fan reading was the bracket fouling the header and is retired (`021`). The remaining lever was the one slot position. **Closed 2026-09-10 by `024-thermal-soak-bracket-move`**: after the user's bracket and GPU work, 1100 steps from cold gave **zero thermal-slowdown samples on all four cards over a 13.7-minute load window**, peaks of 82 / 70 / 71 / 67 °C, every card's clock minimum within 20 MHz of its mean, and a `step_s_median` of **737.5 ms** — below the ≈ 760 ms cold floor the criterion asked it to approach. The starved card of `022` (`GPU-88074816`) runs at 70 °C and 1931 MHz. The DDP step falls 968 → 737.5 ms against `022`, **23.8 %**, which is ≈ 1.9 h per `screen` rung. One caveat carried forward: GPU 0 holds 82 °C with its fan at 97 % and has almost no margin, where the other three keep 15–20 points in reserve — a `screen` rung runs for hours, not fourteen minutes, so checklist step 6's watch on `106` stands. **The acceptance soak was `024`, not `023`**: ids are allocated in order and the env re-capture came first in the order of work.
- **I26** **Speculative decode has no global attention over its own draft window.** Under ADR-021/022 the global range is the final-vector cache $\mathcal{G}$, one entry per token, and a token's entry exists only after its **final** iteration. P13 drafts $m$ tokens in one pass and verifies them in the next. So within a single speculative window, draft token $t+2$ has no $\mathcal{G}$ entry for draft token $t+1$ — the drafts are mutually invisible to global attention, and only become visible once verified and committed. **Neither `01 §4.3` (the v2 attention structure) nor `01 §7` (MTP) says what happens**, and they were specified independently; this is an interaction, not an error in either. Found while answering the HRM brief's Q4 (`08 §6`), which asked exactly where the serialisation the note's author warns about bites.

  **Why it may not matter much, and why that needs checking rather than assuming.** ADR-012 makes the MTP heads *independent* — head $j$ reads the same final hidden state and predicts $x_{t+j}$ directly, with no cross-head dependency — so the drafts are generated in one pass from one state and never attend to each other at all. On that reading the gap is vacuous for the *current* design. It bites for (a) sequential MTP modules, which is exactly the L3b fallback ADR-012 names, (b) any draft window longer than the heads produce, and (c) tree/verification schemes that re-run drafts. **So the question is not "is the spec wrong" but "does the spec's cheap MTP survive being the only MTP we can have".** If L3b goes sequential, this must be answered first.

  Resolution: state the interaction explicitly in `01 §7` once H13 is decided at `small` (ADR-030), and price it in S5 if a sequential L3b is ever built. **Not blocking** — nothing before L9 touches it. Bears on H13, H15b; touches `01 §4.3`, `01 §7`, L3b.

- **I25** **H7's comparator is the wrong control.** H7 reads *"learned $r_t$ saves ≥ 30 % middle-block FLOPs at ≤ 1 % loss **vs fixed $r_{max}$**"*. Beating fixed $r_{max}$ is not the interesting claim: a learned policy that settles at a mean depth of, say, 5.6 is competing against $r$ = 8, so it is being credited for the FLOPs saved by *being shallower*, which any fixed $r$ = 6 run would also save. **The honest control is fixed $r$ at the same mean depth**, and the ladder does not currently have one — `03 §2` runs L5 at fixed $r$ = 8 only. Without it, "adaptive depth works" and "this model wanted to be shallower" are indistinguishable, and the ARC Prize HRM ablation (`05` line 92) is a live example of exactly that confusion resolving the boring way: a same-size plain transformer within ~5 pp, *"the outer refinement loop, not the hierarchy, carries it"*. Found while answering the HRM brief (`08 §5`), not from the brief itself.

  **Resolution — a fixed-$r$ sweep as a control, not a feature**: L5 at $r$ ∈ {4, 8, 12}, one seed each at `screen`, ≈ 16 h total at the measured rate. It gives H7 a loss-vs-mean-depth curve to compare a learned policy against, and it is independently useful — it is the first measurement of whether this architecture wants more or less depth at all, which H6 also benefits from. **H7 should not be scored before it exists.** Cheap, and the cheapest thing on `08 §9`'s list. Bears on H7, H6; touches `03 §1`, `03 §2`.

- **I24** **H7's depth mechanism is token-choice; its strongest prior says expert-choice is better.** `01 §4.7` specifies per-token depth as an **ACT halting head**: each token accumulates $h_t^{(j)}$ and stops when the sum crosses $1-\epsilon$. That is a *token-choice* decision — each token decides for itself, locally. Mixture-of-Recursions (Bae et al. 2025, arXiv:2507.10524), which `docs/05` records as **the strongest prior for both H6 and H7**, instead uses an *expert-choice* router: at each recursion the router picks which tokens continue, under a capacity constraint. The same entry records that **expert-choice beats token-choice by 2.6 points** in their ablation. So the mechanism the spec commits to is the one its own best prior measured as worse, and no ladder step currently compares them. Raised by Annemette 2026-09-11 (as "let the router decide how many times each token goes around").

  **Why the spec may be right anyway, and why this is a trade rather than a fix.** Expert-choice depth routing needs batch-level information — you cannot select the top tokens to continue without seeing the others — so it is a *synchronisation point*. The note's P11 execution model is continuous and non-lockstep at inference, with tokens flowing through units independently (`02 §5`); ACT halting is a purely local decision and composes with that, while expert-choice does not, or not without a capacity rule evaluated over whatever happens to be in flight. Training is lockstep either way (ADR-016), so the conflict is inference-only — but inference is where the note's whole scheduling argument lives. **If expert-choice wins on quality but costs the continuous execution model, that is a trade to price, not a free upgrade.**

  **Proposed resolution — L6c, beside L6 rather than after it** (the I20 pattern): L6 as specified (ACT halting, token-choice) vs **L6c** (expert-choice depth router with a capacity factor, MoR-style) at `screen`, same budget, same seed, on top of L5d. Quality half is one cheap run. The systems half belongs to `sim/`: an S-experiment measuring what a per-iteration capacity rule costs the continuous scheduler in utilisation and tokens in flight, against ACT's ragged depth. Report both before H7 is scored. Note the interaction with ADR-010/011: expert-choice changes *which* tokens stop, not the ragged-depth semantics or the halted-KV rule, so those stay as accepted. Bears on H7, H11; touches `01 §4.7`, `02 §5`.
- **I22** *(closed by I23; confirmed 2026-09-11 by `106-l1-screen`)* `100-l0-screen` took 18 h of wall-clock for 4.28 h of timed steps; the first checkpoint came on time (30 min), so the loss is later and outside the timed region — evaluation, checkpointing to `/home` (SATA, 31 GiB free, btrfs zstd), the per-step barrier, or the host. The trainer now records absolute time, `eval_s` and `ckpt_s` per step. **Confirmed closed by `106-l1-screen`**, which ran the *same 1907 steps* of a *more expensive* rung in **5.62 h of wall-clock against 5.540 h of summed `step_s` plus 0.079 h of evals and checkpoints — 0.1 % unaccounted**, where `100` left ≈ 76 % unexplained. **None of I22's suspects was the cause**: not `/home`, not the evals, not the per-step barrier, not the host. It was I23's thermal throttling throughout, and `024`'s layout fix removed it. Every `results.md` still reports timed-step hours *and* wall-clock, now as a cheap regression check rather than an open question.
- **I21** *(closed 2026-09-11 by `026-train-step-rungs-1gpu`)* `014-train-step-rungs`: DDP's per-micro-step all-reduce costs the recurrent rungs ≈ 1.9–2.9 s against ≈ 0.25 s for the dense baseline, for a smaller gradient. Likely no bucket/compute overlap because the shared block's gradient completes only at the end of backward, plus many small expert matrices and checkpoint recompute; unverified. Gradient accumulation (16 micro-steps per step) hides it in training; measure the reducer timeline before any throughput statement about the recurrent design on this rig. **Premise withdrawn 2026-09-10 (`024`, and the erratum appended to `014`).** `024` measures the identical DDP L5 configuration at **737.5 ms** where `014` reported 2526 ms, so against `014`'s own 651 ms single-GPU figure the DDP overhead is **≈ 86 ms — below the dense rung's ≈ 250 ms**, not eight times above it. `014`'s minimum sits 1 % under its median, i.e. it never ran a fast step, which dates the loss to cards already saturated before the rung began rather than to the reducer. **I21 is not closed**: `024` measures one rung, and `014` kept no thermal telemetry, so the recurrent rungs' true all-reduce cost is simply unmeasured. **Closed by `026`**, which re-measured the single-GPU column on the candidate-C layout and paired it with `018`'s DDP column. The DDP overhead is a ring all-reduce of the gradient and nothing else: at $N$ = 4 that is $2(N-1)/N$ = 1.5× the gradient, BF16 at 2 bytes per parameter, over the measured $\lambda_{link}$ = 3.59 GB/s (`002-nccl`). That parameter-free prediction matches measurement within 10 % on six of nine rungs, **including the recurrent L5 at ratio 1.02**. **L5 pays 66 ms on 77.5 M parameters against L0's 65 ms on 100.1 M** — the same cost per parameter, not eight times it, and cheaper per parameter than the 271 M layered rungs at 0.85–0.89 ms/M. There is nothing left to explain: the ≈ 1.9–2.9 s was thermal throttling on the old layout, in both columns. Two rungs sit above ratio 1.2 — L5d at 1.57 and L5-ne128 at 1.31, the latter plausibly many small expert buckets rather than a few large ones — and are second-order against a prediction that assumes a perfect ring. **Consequence for the design:** the recurrent rungs carry no distributed-training penalty beyond their parameter count, so accumulation is worth having for the 271 M layered rungs (all-reduce ≈ 40 % of a micro-step) and close to irrelevant for the 77.5 M recurrent ones (9–11 %).
- **I17** `docs/07 §4.1`: the author's baseline unit is $U$ = 16 (even 4×4 torus + cheap routers), the note's is 64 (asymmetric 1:4:16:64). `note64.yaml` carries both; every S-experiment that depends on $U$ reports the pair until Q14 is answered.
- **I18** `docs/07 §4.3`: the tile floor is a row floor and speculation multiplies rows per token (1 + $m$). Our `003-gemm` floor (512 tokens for 80 % of $\phi$) was measured without speculation rows; the roofline $b_{min}$, the tile-row floor and the empirical occupancy floor are three separate constraints and the scenario file must name which one the scheduler uses (extends I13).
- **I19** `docs/07 §6`: the document gives ratios and structural constants only; absolute $\phi$, $\beta_C$, $\lambda_C$, $\nu_C$, \$/chip and W/chip for `note64` still need vendor sources (I4). Candidates: TPU v5p/v6 torus link bandwidth and NVLink 5 for `gpu_today`; GDDR7 / CXL vendor figures for the copper-trace and "other RAM" guesses.
- **I16** *(closed 2026-09-02, `013-tiers-nvme-qd`)* `bench_tiers --queue-depth 1 4 16 64` (threads, synchronous `preadv`): a 16 KiB block goes from 0.23 GB/s / 14 k IOPS at QD 1 to 2.1 GB/s / 127 k at QD 16 (9×), saturating there; latency 70 → 114 µs median (p99 227 µs). The drive gives its full 7.3 GB/s from 256 KiB at QD 16 and 6.9 GB/s at 64 KiB at QD 64, so the cold tier follows the warm tier's rule: aggregate to ≥ 64 KiB, ≥ 8 in flight. Below 64 KiB the ceiling (≈ 1.3 × 10⁵ IOPS) is one Python process, not the drive (rated 1.2 M); if an S6 scenario needs more, remeasure with `io_uring` under a new id. `local_3060` carries both the QD 1 and the QD 16 values.

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
Repo bootstrapped (`.venv` py3.12 / torch 2.13+cu130; faiss has no cp314 wheel), `costmodel/` with 29 worked-example tests, all of `scripts/bench/` except the storage tier, `sim/scenarios/local_3060.yaml`. Measured: $\phi$ 27.1 TFLOPS (above nominal), $\beta_C$ 342 GB/s, $s_{min}$ 256, batch floor 512 vs roofline 79, $\lambda_{link}$ 3.59 GB/s / 44 µs, host link 24–27 GB/s pinned with a 64 KiB batching knee, ANN 7.2 k QPS, `small` dense 22.1 k tok/s per card and 67.5 k aggregate. **Surprises:** no NVMe in the box; three of four cards are GA104 (harmless); 2:4 is worth 1.3–1.6× not 2× and loses below $b$ ≈ 2–4 k; odd expert widths lose 12–27 %; the empirical batch floor is 7× the roofline; DDP all-reduce is 31 % per micro-step; `06 §3`'s parameter counts are MHA figures under a GQA spec, and its recurrent widths only reproduce with two skeleton blocks (ADR-025/026/027 proposed, `costmodel/README.md`). Storage tier measured after the NVMe and HDD were installed the same day (`012-tiers-nvme`): $\beta_{cold}$ 7.3 GB/s bulk, 70 µs per 16 KiB block at queue depth 1 — cold is one order of magnitude below warm in both bandwidth and latency; I10/I12 closed. I16 answered the same day (`013-tiers-nvme-qd`): concurrency lifts a 16 KiB cold fetch 9× to 2.1 GB/s by QD 16 and the drive reaches 7.3 GB/s from 256 KiB; the single-process reader caps ≤ 16 KiB at ≈ 1.3 × 10⁵ IOPS. ADR-014 accepted: FineWeb-Edu `sample/10BT` (ODC-By) under the Mistral-7B 32 k tokenizer (Apache 2.0), tokenised once to `/mnt/nvme/corpus`; held-out = first 10 000 documents of shard 000; loss-based eval suite fixed (I6 closed). Corpus downloaded (27 GiB) and tokenised the same day: 11.01 B tokens, 22.0 GB memmap on `/mnt/nvme`. The author's baseline remark recorded (§Author's answers, Baseline; Q13): L0 is the requested plain-transformer comparator. ADR-013 accepted: thresholds as fractions of the baseline loss with a measured-σ verdict band; 0.5 % claims become non-inferiority tests; H10's $X$ = 10 %; H3 cost priced at 1.4×. ADR-025/026/027 accepted: $E$ = $F$ = 2 stands and every width is re-derived (`small` $d_{ff}$ 896 / 448 / 1792, `medium` 1920 / 960 / 3776, multiples of 64, 1024 budget keys); two streams overspend the budget with zero experts, so L4 runs at 2× budget against a √2·$d$ comparator; dense counts corrected to GQA (75.5 M / 270 M). ADR-006 (dense skeleton), ADR-012 (independent MTP heads, subsampled aux losses, ≈ +25 % step) and ADR-023 (stop-gradient memory) accepted: no pending ADR blocks L0–L3 or L5/L5d; ADR-007/008 stay proposed for Phase 3. Literature refresh started: the author's linked document (a public Google Doc, 154 pp.) fetched to `docs/source/` and summarised in `docs/07`; ADR-028 (unit-model revisions) proposed; I17–I19 and Q14–Q15 opened. **Phase 1 started**: `model/` reference implementation (config, layers, router, dropless MoE, layered block, shared middle block with depth conditioning / global cache / ragged depth, MTP heads, 2:4 masks), 30 semantic unit tests, DDP trainer over the ADR-014 corpus, `bench_train_step --rung` (`014-train-step-rungs`); ADR-029 records the implementation decisions. **Runs:** `100-l0-screen` finished — $L_{ref}$ = 3.336 nats at 1 B tokens (one seed). `101-l1-screen` aborted at step 180 on the user's decision (I23, thermal throttling); the queued L2, L3, L5, L5d screen runs are cancelled; **the system is going down for a cooling fix**. On restart: (1) confirm the cards hold clocks under load — 30 min of `bench_train_step --ddp --rung L5`, temps < 85 °C, no `sw_thermal_slowdown` in `nvidia-smi`; (2) rerun `bench_train_step --ddp --rung ...` cold vs warm to put a *valid* throughput in `06 §4`; (3) rerun the ladder from L1 with new ids 106+ (101 is burnt) using the queue script pattern on `/mnt/nvme`, now with thermal columns in `log.jsonl`. Reading list verified end to end (docs/05: every entry with id and a quantitative takeaway, ≈ 100 additions through mid-2026, priors per H, flagged entries); I20 opened for the priors that run against H3, H4, H13 and H15b. **Phase 0 exit criteria met** (`03 §5`): the remaining pending ADRs (007, 008, 024, 028) block only Phase 2–3 work.

### 2026-09-04 — Cooling rebuild, part 1, and the restart checklist

**State.** Nothing is running; nothing may be launched until the acceptance soak below passes. Installed today: three 140 mm bottom intakes and the top exhaust fans (`016-thermal-soak-bottom-fans`, which measured both). Still to arrive and install: the Lian Li O11DEXL-1X upright bracket, the VG4-5 (V4) vertical kit with its riser, and the 900 mm PW-PCIV-4-90X riser. The four cards are still back to back in slots 1/3/5/7 (bus 01/02/41/42 = GPU 0/1/2/3; GPU 2 is the GA106 and the only one with a free slot beside it). `016` after the fans: GPUs 0/1/3 still reach 93 °C and `sw_thermal_slowdown` within ≈ 2 min, worst-card steady clock 1518 MHz, DDP L5 step 826 ms against a cold floor of ≈ 760 ms; GPU 2 at 62 °C is the un-throttled reference. `thermal_soak.sh` now defaults to 1100 steps (≈ 15 min un-throttled). Ids 100–105 are used or burnt (101 aborted, 102–105 were queued and cancelled); the next rig-characterisation id is 017, the next ladder id is 106. Disk: `/mnt/nvme` 3.6 TB free (corpus, queue script, log); `/home` has 27 GB free (89 % full) and `experiments/*/ckpt` lives there — see step 4.

**Checklist when the parts are installed** (in order; each step's outcome goes in the session log):

1. **Record the new physical layout** in `06 §1`: which slot or bracket holds which card, and the bus-ID → GPU-index map from `nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv`. Indices may move when a card sits on a riser. Check every riser card negotiates the expected link (`nvidia-smi -q | grep -A2 "Link Width"` and `PCIe Generation`, or `lspci -vv -s <bus>`): x16 gen4 wanted, x16 gen3 acceptable (collectives are host-bounced at 3.6 GB/s, `002-nccl`, so the link is not the bottleneck); x8 or link errors in `dmesg` → force that slot to gen3 in BIOS and re-check. Intended layout per the order: two cards on the board three slots apart (1 and 5), one vertical at the rear, one upright at the front facing the side intakes; slots 6–7 free for the vertical bracket.
2. **Acceptance soak**, cards cold: `scripts/bench/thermal_soak.sh 017-thermal-soak-<config>` (1100 steps, ≈ 15 min). Pass = every card < 85 °C and 0 throttled samples in the last-5-min summary the script prints, and the four `step_s_median` values in `result.json` within a few percent of the ≈ 760 ms cold floor. Fail = write `results.md` with the same table as 015/016, note which card and where it sits, and stop; the next lever is per-card (fan curve, undervolt via `nvidia-smi -pl` only if the failing card is power-bound, which 015 showed it is not). Write `results.md` either way (bears on no hypothesis; closes or extends I23).
3. **Re-bench throughput warm** so `06 §4` gets a valid number: run `bench_train_step --ddp --steps 350 --rung L0:4 L1:2 L2:2 L3:2 L5:4 L5d:4` as `018-train-step-rungs-warm` immediately after the soak (cards warm), and compare with `014` (cold). If warm and cold agree within ≈ 5 %, `014`'s per-rung numbers stand and the `screen` time budget in `06 §4` can be recomputed from them; if not, the warm number is the budget. Also decide whether L5's 2526 ms in `014` versus 1277 / 826 ms in `015` / `016` is a `014` artefact (it ran nine rungs back to back) — the soak's L5:4 number is the one measured in isolation.
4. **Checkpoint location.** `100-l0-screen` wrote checkpoints to `/home` (SATA, btrfs zstd, 27 GB free), one of I22's suspects. Before the ladder restarts, either point the trainer's `ckpt` directory at `/mnt/nvme/ckpt/<id>` and symlink it back, or prune `experiments/100-l0-screen/ckpt` to the final checkpoint only. Check `scripts/train/train.py` for the ckpt path argument first.
5. **Relaunch the ladder from L1 with new ids** by copying `/mnt/nvme/queue_screen.sh` to `queue_screen_2.sh`, dropping the L0 wait loop, and setting the specs to `106-l1-screen L1 2`, `107-l2-screen L2 2`, `108-l3-screen L3 2`, `109-l5-screen L5 4`, `110-l5d-screen L5d 4` (same `--tokens 1e9 --seed 0`). Run it under `nohup … &` and log to `/mnt/nvme/queue_screen.log`. Do not resume `101` from its checkpoint: its throughput is invalid and the ladder is append-only. The 101 partial (180 steps, loss 5.47 at 0.094 B tokens) stays as a record only.
6. **Watch the first eval of 106** (about 30 min in): in `experiments/106-l1-screen/log.jsonl` the trainer writes `sm_mhz_min`, `temp_c_max`, `n_throttled`, `n_thermal` and `n_power_cap` every 10 steps; abort if **`n_thermal`** > 0 for more than a few consecutive records or `temp_c_max` ≥ 90 °C. **Read `n_thermal`, not `n_throttled`** (`024`): NVML packs every reason into one bitmask, and a well-cooled card at full clock hits the 170 W software power cap often, which `n_throttled` counts as throttling. `n_throttled` is retained only so records before `024` stay comparable. `train.log` shows tok/s per step; L1 ran at ≈ 45 k tok/s aggregate at step 180 while still cold, which is the number to beat. Report timed-step hours *and* wall-clock in every `results.md` until I22's residual (eval, ckpt, host) is separated by the `eval_s` / `ckpt_s` columns.
7. **Results.** Each rung's `results.md` states the H it bears on and the verdict against $L_{ref}$ = 3.336 nats (ADR-013 bands), with `100-l0-screen` as the dense comparator; the matched-FLOPs fine-grained MoE baseline (CLAUDE.md rule) is still owed for the ladder — L5-ne128 in `014` is the candidate config; open a queue entry for it once L1–L5d are in.

**What not to do:** no power cap (015: the cards are thermally, not power, limited); no run > 1 node-day without a finished screen; no `git push` (the user pushes); no edits to numbers in existing `experiments/*` (new id + note).

### 2026-09-10 — Re-slotting measured; the RAM upgrade did not take; two GPUs at x8

**State.** Machine rebuilt and rebooted 19:16 (down 17:00–19:16). The riser, upright bracket and
vertical kit are installed and the four cards are no longer back to back. Nothing is running; the
machine is going down again for a BIOS memory investigation.

**Checklist step 1 (layout), partly done.** Bus-ID → index map changed: bus 01/21/22/41 = GPU 0/1/2/3,
the GA106 is now GPU 1 (was GPU 2 at bus 41). **GPUs 1 and 2 negotiate PCIe x8, not x16**, on two root
ports under one host bridge — one x16 slot bifurcated x8/x8. Not expected to bind (collectives are
host-bounced at 3.59 GB/s, `002-nccl`) but it needs the BIOS check the checklist asks for. No AER or
PCIe errors in the boot journal. Table in `017/results.md`. **Still owed:** which slot or bracket
physically holds which card — that needs eyes on the case, not a command.

**Checklist step 2 (acceptance soak): passes on thermals, incomplete on duration.**
`017-thermal-soak-riser-respaced`, cards cold at 40–41 °C, stopped by the user at 11.8 min of the
requested 1100 steps. Every card held 1905–1950 MHz for the whole 11.6-minute load window with
**0 throttled samples** and peaks of 78 / 70 / 70 / 66 °C, against 93 °C on three of four cards in
both 015 and 016; the worst-card clock floor went 225 → 990 → **1905 MHz** across the three soaks and
per-card power 84–104 → 120–132 → **135–142 W**. On every card the last-5-min clock minimum equals
the mean to the MHz. **Spacing was the constraint and the re-slotting removes it.** But the run was
killed before `bench_train_step` wrote `result.json`, so there is **no step time** for 017, and
11.6 min is short of the 15 min the criterion asks for. **I23 stays open.** Rerun the full soak as
`019-thermal-soak-…` after the BIOS work and record the verdict there; `018` stays reserved for the
warm re-bench of checklist step 3. No training run may start before `019`.

**The host-RAM upgrade did not take effect.** Kernel at boot: `DMI: Memory slots populated: 4/8`;
`MemTotal` 125.6 GiB, unchanged from `06 §1`. `dmidecode -t 17`: channels A–D hold 32 GiB
`HMA84GR7MFR4N-TF` (SK Hynix 2Rx4 registered ECC), channels E–H report `No Module Installed` — the
BIOS could not read their SPD at all. The user reports all eight sticks are physically inserted.
Configured speed is **2133 MT/s**, the JEDEC fallback, on a 2933 MT/s part; `06 §1` claims DDR4-3200
and is wrong twice. With 4 DIMMs on an 8-channel controller the machine runs **4-channel**, about
half the ≈ 205 GB/s `06 §1` assumes — and no host-memory bandwidth number in `06` has ever been
measured. Ignore `amd64_edac`: it reports 192 GiB over 6 channels, contradicting both SMBIOS and
`MemTotal`, and is a driver misreport. Leading hypothesis is stale cached memory training (Memory
Context Restore / Fast Boot) surviving the DIMM change, which fits the unenumerated channels and the
fallback speed together. Order of work: BIOS first (Memory Context Restore off, Fast Boot off, DRAM
frequency Auto, interleaving and NPS Auto, then a CMOS clear), reseat second, and if both fail move a
known-good stick from channel A to channel E — E reads 32 GiB ⇒ the new sticks are at fault, E still
empty ⇒ the slot or the CPU socket is. **Before any CMOS clear, note the current PCIe settings**: a
clear will also discard whatever produced the x8 links above.

**Owed when the machine comes back:** (1) `019` full soak, cards cold, then I23's verdict; (2)
checklist step 3, the warm re-bench as `018`; (3) `06 §1` corrections — RAM speed, channel count and
the measured-vs-nominal memory bandwidth, plus the new bus map; (4) checklist steps 4–7 unchanged.

**Update, 20:04 boot (after the user re-slotted the PCIe risers; no DIMM was moved).** Same board and
BIOS (`WRX80 Creator R2.0`, BIOS 10.03 12/19/2025). **5/8 slots now populated**, `MemTotal` 157.0 GiB
(160 GB), up from 4/8 and 125.6 GiB, with nothing touched on the memory side — so **detection is
intermittent, which is a contact symptom, not a settings one**. `dmidecode -t 17`: channels A, B, C,
D, F hold 32 GiB; **E, G, H report `No Module Installed`**. All eight sticks are the *same* part,
`HMA84GR7MFR4N-TF` (SK Hynix 32 GiB 2Rx4 RDIMM, DDR4-2933), so the mixed-module hypothesis is dead.
Serials split into two batches: `277BCC7C` / `277BCD8D` / `277BCD67` in A / B / F and `914DD9A6` /
`914DD9D6` in C / D. On the reading that `277B…` is the original four and `914D…` the new four,
**the three dark channels are one old stick and two new ones** — failure crosses both batches, which
again points at contact or socket rather than at modules. Configured speed is still **2133 MT/s** on
a 2933 part, the JEDEC fallback, consistent with training that cannot close.

The user reports the original four sticks sit in the slots silkscreened A, B, G and H and were not
moved; SMBIOS shows an original-batch serial in **channel F**. So either the setup-screen/SMBIOS
channel letters do not follow the silkscreen on this board, or the recollection is by position rather
than by label. **Unresolved, and it blocks any slot-level conclusion.** Settle it physically: with the
machine off, record slot silkscreen → serial from each DIMM's own label and pair that with the table
above. That is the definitive map and costs no boot.

**Memory Context Restore and the stale-training-cache hypothesis are withdrawn**: that option is Zen 4
/ AM5 AGESA and does not exist on this Zen 2 Threadripper PRO. ASRock's Fast Boot is under *Boot*,
not *Advanced*; DRAM frequency was already Auto. **No BIOS setting is missing or wrong**, so the CMOS
clear drops to last resort — it would also discard whatever currently gives the PCIe layout below.
Revised order of work: (a) physical slot → serial map; (b) reseat all eight hard, both latches
closing unaided, inspect the dark slots for debris; (c) re-read `dmidecode`; (d) if the same channels
stay dark under different sticks it is the board or socket pins, if the darkness follows the sticks it
is the modules. Open question for the user: whether the CPU cooler was removed or retorqued during the
cooling rebuild — uneven sWRX8 mount pressure drops whole channels.

**PCIe, after the riser move.** Bus map changed again: **01 / 21 / 41 / 42 = GPU 0 / 1 / 2 / 3**, and
the GA106 is now **GPU 2** (it was GPU 1 at bus 21 earlier the same day, and GPU 2 at bus 41 before
the rebuild — the index is not stable across re-slotting and must be re-read, never assumed). **Only
bus 21 still negotiates x8**; 01, 41 and 42 are x16. So the move cleared one of the two x8 links of
`017`. Idle link gen reads 1 on all four, which is power-save down-training, not a fault; re-check gen
under load. No AER or PCIe errors in the boot journal (the platform does not export AER at all:
`_OSC: platform does not support [AER LTR DPC]`).

**`017`'s thermal result is now stale**: it measured a physical arrangement that the riser move has
changed. The `019` soak must run on the *final* layout, and any further re-slotting invalidates it
again. Settle the DIMMs and the risers first, then soak.

**Cooler ruled out; and the pre-upgrade channel map was never recorded.** The user confirms the CPU
and its cooler were not touched during the cooling rebuild, so uneven sWRX8 mount pressure is out as
a *newly introduced* cause. `000-env` logged host RAM only as "125 GiB total" — **no per-channel or
per-slot record exists from before the upgrade**, so we cannot distinguish a channel that stopped
working from one that never worked. What the serials do say: one original-batch stick is dark in a
slot the user did not move it out of, i.e. **one channel that carried a working DIMM before the
upgrade is dark now**, alongside two channels whose history is unknown. Mixed, and still consistent
with contact. If the reseat does not clear it, run the **revert test**: pull the four new sticks, put
the original four back in their original slots, boot. 125.6 GiB and 4/8 restores the known-good
baseline and puts the fault on loading or on the new modules; anything less is a regression in a
configuration that demonstrably worked, which is a cleaner fault to chase. Record slot → serial either
way — that record is what `000-env` is missing.

### 2026-09-10 (late) — RAM upgrade takes; soak aborts on a dead GPU fan

**Host RAM is fixed.** After the user reseated the DIMMs, the 20:39 boot reports `DMI: Memory slots
populated: 8/8` and `MemTotal` 251.5 GiB (263 750 232 kB). `dmidecode -t 17`: all eight channels A–H
hold 32 GiB `HMA84GR7MFR4N-TF`. The machine runs **8-channel**. It was a contact fault, as the
intermittent 4/8 → 5/8 → 8/8 progression suggested; **no BIOS setting was ever at fault and no CMOS
clear was needed**. The batch reading in the 20:04 update above is withdrawn: the serials split
**six `277B…` / two `914D…`**, not four and four, so neither batch is "the original four". The
per-slot map now on record (A–H): `277BCC7C`, `277BCD8D`, `914DD9A6`, `914DD9D6`, `277BCBA1`,
`277BCD67`, `277BCD6E`, `277BCD81`. The revert test and the slot-level diagnosis in the 20:04 update
are moot.

**The 2133 MT/s fallback is not.** All eight modules still report a configured speed of 2133 MT/s
against the part's rated 2933. It survived a correct 8/8 population, so it is a BIOS/AGESA setting
(DRAM frequency is on Auto), not a symptom of the missing sticks. Recovering 2933 is ≈ 37 % of host
memory bandwidth for free. **`06 §1` is wrong twice**: DDR4-3200 is not what is installed (2933 is
the ceiling) and 2133 is what is running.

**First host memory-bandwidth measurement on this box** (ad-hoc threaded STREAM-style probe,
`experiments/019-…/membw_probe.py`, 12 threads, 4 GiB arrays, best of 5): copy 70.8, scale 47.1,
add 51.3, triad 29.3 GB/s. Crude, so lower bounds — but copy exceeds the 68.3 GB/s 4-channel
DDR4-2133 peak, which independently confirms the 8-channel population, and every figure is far under
`06 §1`'s ≈ 205 GB/s, which is unreachable at 2133 MT/s. **Not yet a scenario input**: it needs a
real `scripts/bench/bench_membw.py` under its own id before it may enter
`sim/scenarios/local_3060.yaml` (CLAUDE.md: every number carries a source line).

**PCIe (`020-env-post-rebuild`).** Bus map held across the reboot: 01 / 21 / 41 / 42 = GPU 0 / 1 /
2 / 3, GA106 at bus 41 = GPU 2. **One card still negotiates x8** (GPU 1, bus 21), down from two in
`017` — the riser move cleared one. All four reach **gen 4 under load**; the gen-1 idle reading is
ASPM, not a fault. BF16 autocast passes on all four. Still owed: the BIOS check of checklist step 1,
and which slot or bracket physically holds which card.

**Acceptance soak `019` failed on hardware, not on cooling.** `019-thermal-soak-riser-respaced-full`
started cold at 20:43 and was stopped at 5.2 min. **GPU 0's fan reported 0 % for every sample**, idle
through 93 °C, while GPUs 1–3 ramped to 77–81 %. GPU 0 raised `sw_thermal_slowdown` at 91 s and sat
pinned at 92–93 °C with its clock sawing 525–1492 MHz; once idle it took over 4 min to fall to 88 °C,
fan still at 0 %, while the others dropped to 45–50 °C. GPUs 1–3 reproduced `017` exactly (1920–1961
MHz flat, 63–67 °C, 0 throttled samples), so **the spacing fix still holds for three of four cards**.
`nvidia-smi` reports commanded fan duty, so 0 % at 93 °C is a fan that is not driven, not a cooling
margin that is too small. Decisive: in `017`, 70 min and two case-openings earlier, **GPU 0 was the
card whose fan ran hardest (93 %)** and it peaked at 78 °C — same bus, same chip, same root port,
same physical card. The fault therefore dates to the 20:04 riser re-slotting or the RAM reseat.
Leading candidate is the 900 mm riser cable fouling the fan or its header (the user's own guess);
next is the fan lead unseated at the PCB. **I23 stays open.** `018` stays reserved for the warm
re-bench; the next acceptance soak is **`021`** (`020` is the env re-check).

**Operational note.** Killing `thermal_soak.sh` leaves the `torchrun` children and the
`multiprocessing` workers training at full power — it took three rounds of `SIGTERM` to release the
GPUs, in both `017` and `019`. The script's `trap` should `pkill -f bench_train_step`.

**Next shutdown closes three things at once:** (1) GPU 0's fan / the riser cable; (2) DRAM frequency
to 2933 MT/s in BIOS; (3) the x8 link on bus 21. Then run the full soak as `021` and let it write
`result.json`. No training run before that.

### 2026-09-10 (night) — the x8 link and the "dead fan" both close; one card is starved; the DRAM instruction was wrong

**State.** Machine booted 21:26 after the user re-slotted the GPUs "to match the PCIe
configurations". Two runs tonight: `021-env-gpu-remap` (environment capture) and
`022-ddp-throttle-cost` (a 420-step DDP soak, deliberately *not* the I23 acceptance soak). Nothing is
training. Host RAM unchanged and healthy: 8/8 slots, `MemTotal` 251.5 GiB, and `amd64_edac` now agrees
with SMBIOS at 262 144 MB with `ce_count` = `ue_count` = 0, so `019`'s 192 GiB / 6-channel misreport
is gone.

**PCIe: fixed.** All four cards negotiate **x16 and gen 4 under load** (`021`) — the first time this
rig has done so. Bus map 01 / 02 / 41 / 42 = GPU 0 / 1 / 2 / 3, root ports 00:01.1, 00:03.1, 40:01.1,
40:03.1; the GA106 is GPU 2. Item (3) of the previous entry's three-item list is closed. It changes no
throughput number — collectives are host-bounced at 3.59 GB/s (`002-nccl`) — but `06 §1` asserted
x16-on-all-four as measured fact and that was false until tonight.

**Card identity is now on record.** The GPU index and the bus id are properties of *position*, so
neither survives a re-slotting; the UUID does. `021` records UUID → chip → VBIOS for all four.
`bench_env` should capture `index,pci.bus_id,uuid` from now on.

**There is no fan fault, and there never was.** The user confirms only the x8 card moved. Pairing that
with the root ports, `019`'s GPU 0 (root port 00:03.1, fan at 0 % through 93 °C) is tonight's GPU 1 —
the same physical card, unmoved — and tonight it runs its fan to **100 %**. Nothing was done to the
card except the bracket work in the same downtime. The "fan lead unseated at the PCB" candidate is
retired; the mechanical one stands: the riser or the bracket fouled the fan or its header, and
disassembling the bracket freed it. Item (1) of the three-item list is closed.

**But the same card is now thermally starved.** In `022` GPU 1 enters the load window at 73 °C, raises
`sw_thermal_slowdown` at 40 s, holds it for 37 of 41 samples and settles at 1499 MHz mean / 1320 min
against a 2115 max, at 93 °C, **with its fan at 100 % and drawing 113 W of a 170 W limit**. Full fan
duty and well under the power cap is `015`'s conclusion again: this is airflow, not the controller and
not power. The user reports the card sits in an ordinary motherboard slot **side by side with another
card, because the bottom bracket takes up too much space** — so the bracket is not holding the hot
card, it is consuming the slots that would let the cards be spaced.

**The other three cards are the best this rig has recorded**: 63–66 °C peak, clocks flat within 8 MHz
of their means at 1921–1958 MHz, zero throttled samples. The cooling rebuild works. It is being wasted
by one slot position.

**What the starvation costs.** DDP is lockstep, so one card sets the step for all four.

| | DDP L5:4 step | vs tonight's median |
|---|---|---|
| `022` median | 968 ms | — |
| `022` min, GPU 1 not yet throttled | 838 ms | −13.4 % |
| `016`, three cards at 93 °C, bottom fans only | 826 ms | −14.7 % |
| cold floor, checklist step 2 | ≈ 760 ms | −21.5 % |

The `016` row is a cross-session comparison and its cards were cold where GPU 1 was not, so do not
quote it as a controlled result — but the direction is not in doubt and it is the useful finding:
**tonight's arrangement is no faster in DDP than the pre-bracket arrangement `016` measured.**
Concentrating the throttling in one card instead of spreading it over three buys nothing, because DDP
runs at the slowest rank either way. Recovering the 13.4 % takes a `screen` rung from 8.2 h to ≈ 7.1 h,
≈ 5.5 h over the five queued rungs 106–110.

**The DRAM instruction in the previous entry was wrong and is withdrawn.** `019` read the modules as
"`HMA84GR7MFR4N-TF`, DDR4-2933" and concluded 2133 MT/s was a JEDEC fallback worth ≈ 37 % of host
bandwidth. In SK Hynix's DDR4 module part numbering the trailing code carries the grade *and* the
timings — `TF` = 2133 15-15-15, `UH` = 2400 17-17-17, `VK` = 2666 19-19-19, `WM` = 2933 21-21-21,
`XN` = 3200 22-22-22 — so `…-TF` is a **DDR4-2133 CL15** part and a 2933 part would be `…-WM`. Every
vendor listing of the part number agrees. **2133 MT/s is the rated speed, correctly applied; there is
nothing to recover, and "(2) DRAM frequency to 2933 MT/s in BIOS" must not be carried out** — it would
be a 37 % overclock of registered ECC modules. `019`'s 70.8 GB/s copy is 52 % of the 136.5 GB/s
8-channel DDR4-2133 peak, an ordinary figure for a threaded numpy probe, and no longer evidence of
headroom; it still exceeds the 68.3 GB/s 4-channel ceiling, so it still confirms the population.
Confirmation owed before `06 §1` is edited: `sudo dmidecode -t 17`, comparing `Speed:` (the SPD
maximum) with `Configured Memory Speed:`.

**`06 §1` corrections still owed**, now with a different content than the previous entry said: host
RAM is **256 GB of DDR4-2133 registered ECC across 8 channels**, not 128 GB and not DDR4-3200; the
≈ 205 GB/s figure is unreachable and the measured lower bound is 70.8 GB/s copy; the GPU-interconnect
row's "measured x16 gen4 on all four" is true again as of `021` and should cite `021`, not `000-env`.
Host memory bandwidth needs a real `scripts/bench/bench_membw.py` under its own id before it enters
`sim/scenarios/local_3060.yaml`.

**Next.** The remaining lever is one slot position, not a setting. Two candidate moves, both physical:
free the bottom slots so the side-by-side pair can be separated, or give the starved card its own air.
The bracket's own riser is the limit on where the bracket can sit — see the open question below. Then
run the I23 acceptance soak **cold, 1100 steps, on the final layout, as `023`**; `018` stays reserved
for the warm re-bench of checklist step 3. No ladder run before I23 closes.

**Open question — Q16, the bottom bracket.** The bracket blocking the bottom four PCIe slots is a
Lian Li 4-slot vertical GPU kit (VG4 family), which ships with a **200 mm** riser; that short cable is
why it can only sit at the bottom-*back*. The user already owns a **900 mm PW-PCIV-4-90X**, bought for
the O11DEXL-1X upright bracket, which Lian Li ships *without* a cable — so a Lian Li bracket taking an
arbitrary separately-bought riser is a configuration Lian Li itself sells. What is **not** confirmed is
whether the VG4's riser detaches from *its* bracket: the slot-end of these kits is a small PCB screwed
to the frame, and the question is whether the 900 mm cable's PCB matches the VG4's mounting holes and
its three 10.16 mm height positions. Settle it by looking at the VG4 frame with the machine off — is
the slot-end PCB removable with the same screws, or moulded into the frame? Vendor pages do not say.

**DRAM confirmed the same night, and the thread is closed.** `sudo dmidecode -t 17` on all eight
modules, channels A–H: `Part Number: HMA84GR7MFR4N-TF`, **`Speed: 2133 MT/s`**, `Configured Memory
Speed: 2133 MT/s`, `Configured Voltage: 1.2 V`. `Speed:` is the SPD *maximum*, so the modules declare
2133 as their own ceiling and the configured value equals it. The part-number reading above is
confirmed by SMBIOS: **DDR4-2133 is the rating, not a fallback**, there is no 37 % to recover, and the
2933 instruction stays withdrawn. `06 §1` corrected: the CPU row no longer carries ≈ 205 GB/s as a
usable figure, the host-RAM row now reads 256 GB of DDR4-2133 registered ECC over 8 channels with the
136.5 GB/s theoretical peak and the 70.8 GB/s measured lower bound, and the GPU-interconnect row cites
`021` rather than `000-env` for x16 gen4 on all four. The only remaining host-memory item is a real
`scripts/bench/bench_membw.py` under its own id before any bandwidth number enters
`sim/scenarios/local_3060.yaml`.

### 2026-09-10 (late night) — Q16 closes: the bottom bracket's riser is replaceable, so the bracket can move

**State.** Nothing is running. No hardware moved since the 21:26 boot that `021`/`022` measured, so
the rig is still in `022`'s arrangement: three cards at 63–66 °C with flat clocks, and GPU 1 (UUID
`GPU-88074816…`) side by side with another card, holding 93 °C at 100 % fan and setting the DDP step
for all four ranks.

**Q16 is answered, and the answer is the good one.** The user reports by inspection that the VG4
bottom bracket's riser cable **detaches and takes a longer replacement** of the same kind — the
900 mm `PW-PCIV-4-90X` type that already feeds the upright bracket on the side panel. The 200 mm
cable was the only reason the bracket had to sit at the bottom-*back*; that constraint is gone.
**The last lever on I23 is therefore available**, and it is the one `022` argued for: the bracket is
not holding the hot card, it is consuming the slots that would let the two board cards be spaced.

**One thing this does not supply: a spare cable.** The 900 mm riser already owned is in use on the
side-mounted upright bracket. Relocating the VG4 needs either a second long riser or a swap that
leaves the upright bracket reachable on a shorter one — and the upright is the position that most
needs the length. Treat "order a second `PW-PCIV-4-90X`" as the assumption until the user says
otherwise; nothing else on the critical path is blocked by it.

**What the move is worth.** `022` prices the starvation at **13.4 %** of the DDP step (968 ms median
against an 838 ms un-throttled min), which is ≈ 1.1 h per `screen` rung and ≈ 5.5 h over the five
queued rungs 106–110. That is the whole return; the three healthy cards are already at their best
and cannot contribute more while the fourth throttles, because DDP runs at the slowest rank.

**When the cable arrives — order of work.**

1. **Move the bracket, machine off**, and note which slot or bracket physically holds which card.
   This is checklist step 1's outstanding half and it has been owed since `017`.
2. **Re-read the bus map and re-key it by UUID.** Index and bus id are properties of position and
   will move again; `021` records UUID → chip → VBIOS for all four. Confirm all four still negotiate
   **x16 gen 4 under load** — that result is one re-slotting old and `017`/`020` both lost links to a
   case opening. `bench_env` should be capturing `index,pci.bus_id,uuid` by then (`021`, still owed).
3. **Check the fan on the card that gets touched.** `019`'s "dead fan" was the bracket fouling a fan
   header, and this move disturbs the same region. `nvidia-smi` fan duty at idle and under 30 s of
   load, before committing to a 15-minute soak.
4. **I23 acceptance soak as `023`**: cold cards, 1100 steps, final layout. Pass = every card < 85 °C
   with 0 throttled samples in the last-5-min summary **and** a `step_s_median` in `result.json`
   within a few percent of the ≈ 760 ms cold floor. `018` stays reserved for the warm re-bench of
   checklist step 3.
5. Only then the ladder restarts at **L1 with id 106**.

**Everything measured before the move is thermally stale**, per the standing rule that re-slotting
invalidates a thermal result — `017`, `019` and `022` included. Their non-thermal content (the bus
map, the UUIDs, the DRAM finding, the 13.4 % cost of one starved rank) stands.

**Candidate B, raised the same night: replace the bracket instead of its cable.** The user proposes the
be quiet! PCIe 4.0 riser (accessory 4430, 20 cm) in place of the VG4. Product photographs identify it
as a riser cable whose **slot end is screwed to a rigid pedestal** — two cross-head screws hold the
female x16 PCB to the block, and the block's top face carries several spare holes, so the slot
position is adjustable along it. There is **no multi-slot frame, no slot covers and no rotation
adjustment**: the pedestal bolts to a case's vertical-GPU mounting points and the card stands inboard
of the expansion-slot covers, which stay in place.

**This is a better read of `022` than the long-cable plan.** The complaint in `022` is *footprint*, not
reach: the VG4's four-slot frame consumes the bottom four PCIe slots and that is what forces two cards
side by side. A pedestal that occupies no slot positions frees them directly. Cable length is
unchanged at 20 cm, i.e. the same 200 mm the VG4 ships, so the card lands in much the same region —
which is fine, because length was never the constraint that mattered.

**Three things to check before buying**, none of them answerable from vendor pages:

1. **Hole pattern.** The pedestal is designed for be quiet! case mounting points. This case is a Lian
   Li O11D EVO XL. Compare the pedestal's footprint with the floor and bracket holes actually present.
2. **How much floor it leaves.** *(Corrected 2026-09-10 late night: the user reports the VG4 bracket
   and the three 140 mm bottom intakes cannot both be fitted — the fans came out when the bracket went
   in, so `017`, `019` and `022` all ran with no bottom intakes. See `017` §Erratum.)* This inverts the
   check: the question is not whether the pedestal fouls the fans, it is **how many of the three fans
   the pedestal lets back in**. A footprint that leaves two or three of them clear buys spacing and
   bottom intake at once, which no soak on this rig has yet had.
3. **Second anchor.** The kit brings no slot-cover bracket, so the card is held by the pedestal alone
   unless the case's vertical slots take its I/O plate. This rig runs multi-day jobs; a cantilevered
   card is not acceptable for that.

If Candidate B is taken, the "order a second `PW-PCIV-4-90X`" assumption above is dropped and Q16's
answer stops mattering, because this unit's cable is not the part being replaced. Either way the
sequence is unchanged: move, re-map by UUID, re-check links, check the fan of any card touched, then
`023` cold for 1100 steps.

**Correction that reframes the whole cooling story: the bracket and the bottom fans are mutually
exclusive.** The user reports there was never room for the three 140 mm bottom intakes once the VG4
bottom bracket was fitted. The fans measured in `016` came out when the bracket went in, so `017`,
`019` and `022` all ran with **top exhaust, 3 side intakes, 1 rear exhaust and no bottom intakes**.
Errata appended to all three results files. No measured number moves; the reading of them does.

| soak | bottom intakes | cards spaced | worst card |
|---|---|---|---|
| `015` | no | no | 93 °C, 328 MHz steady |
| `016` | **yes** | no | 93 °C, 1518 MHz steady |
| `017` | no | yes | 78 °C, 1912 MHz steady |
| `022` | no | yes for three of four | 93 °C on the unspaced card, 63–66 °C on the rest |

**Spacing with no bottom fans beats bottom fans with no spacing, by a wide margin.** The 63–66 °C the
three healthy cards hold in `022` is reached with *less* case airflow than `016` had. It also means
the starved card has never had bottom intake air at any point on this rig, so its 93 °C is not the
ceiling of what its position can do.

**This settles the choice between the two candidates.** Candidate A moves the VG4 to the bottom-front
on a longer cable, but a four-slot frame on the floor keeps the fans out wherever it sits, so A buys
the freed slots and nothing else. Candidate B replaces the frame with a pedestal and can buy the freed
slots **and** some or all of the bottom intakes. **B is the preferred route**, subject to its three
checks above, and the fan count it leaves clear is the number to establish before ordering.

`016`'s own contribution is now smaller than `04` has been claiming: with the cards back to back the
bottom fans moved the worst-card steady clock from 328 to 1518 MHz and the DDP step from 1277 to
826 ms, but they never got a card under 93 °C. Their value in a *spaced* layout is unmeasured, and
`023` on a Candidate-B layout would be the first run to measure it.

**Candidate C, the user's own plan and the one to try first: both brackets at the bottom, cables
swapped.** Mount the vertical kit under the PCIe slots and move the VG4 to the bottom-front, using the
cable swap Q16 established — the 900 mm on the long run to the bottom-front, the short 200 mm on the
one under the slots. **This costs nothing and needs no purchase**, since both brackets and both cables
are already in the case. It frees the PCIe slots that force the side-by-side pair, which is the entire
13.4 % on the table.

What it does not do is bring the bottom intakes back: two frames on the floor keep the floor occupied,
so C is a no-bottom-fans layout like `017`/`019`/`022`. That is very likely fine — spacing with no
bottom fans already holds three cards at 63–66 °C — and it is the cheapest way to find out. **Try C
first; keep B (the pedestal) as the fallback if `023` shows the fourth card still short of the bar.**
Ordering hardware before `023` has run on C would be buying against an untested assumption.

### 2026-09-10 (very late) — I23 closes; and `014` turns out to have been measuring heat, not the reducer

**State.** The user reports the RAM and GPU hardware work finished; the machine booted 22:27. Three
runs tonight: `023-env-post-bracket-move` (environment capture), a 60 s fan check (scratch, no id) and
`024-thermal-soak-bracket-move` (the I23 acceptance soak). Nothing is training. **The ladder is
cleared to restart.**

**Ids: the soak is `024`, not `023`.** The previous entry reserved `023` for the soak, but the order of
work puts the env re-capture first and ids are allocated in order and never reused, so the capture took
`023`. `018` still stands reserved for the per-rung re-bench.

**`023` — the layout moved again and one card is back on the x8 link.** Bus map 01 / 21 / 41 / 42 =
GPU 0 / 1 / 2 / 3, root ports 00:03.1, **20:03.3**, 40:01.1, 40:03.1. Keyed by UUID, **three of the
four cards changed position**; `GPU-88074816` — `022`'s starved card — now sits on `20:03.3`, the x8
port that `017` and `020` flagged and `021` had emptied. All four reach gen 4 under load, BF16 autocast
passes on all four (3.10e-3 relative error). The x8 link costs nothing measurable, since collectives are
host-bounced at 3.59 GB/s, but `06 §1` asserted x16-on-all-four as measured fact two hours ago and that
is false again; the row is corrected to say the width follows the *slot* and must be re-read after every
re-slotting, never assumed. `bench_env` now captures `uuid` and the sysfs root port on every PCIe row —
the fix `021` asked for, so that tracking a card across a move is a capture rather than a
reconstruction.

**`024` — I23 closes, on every clause.** Cards cold at 42/45/44/36 °C, 1100 steps, 13.7 min of load.
**Zero thermal-slowdown samples on all four cards**, peaks 82 / 70 / 71 / 67 °C, every card's clock
minimum within 20 MHz of its mean across 82 samples, and `step_s_median` **737.5 ms**, *below* the
≈ 760 ms cold floor the criterion asked it to approach. `GPU-88074816` runs at 70 °C and 1931 MHz
against 93 °C and 1499 MHz in `022`. Against `022` the DDP step falls **968 → 737.5 ms, 23.8 %** —
a `screen` rung 8.2 h → 6.3 h, ≈ 9.5 h over the five queued rungs.

**The pass criterion had to be repaired before it could be read.** The script printed
`GPU0: throttled 3/30`, which reads as a fail; all three samples are `0x4`, the software power cap, on
a card at full clock drawing its 170 W. NVML packs every reason into one bitmask and the summary counted
anything not idle — backwards here, because these cards are thermally, not power, limited (`015`), so
**better cooling yields more power-cap samples, not fewer**. `thermal_soak.sh` and `train.py` now split
`SwThermalSlowdown | HwThermalSlowdown` from `SwPowerCap`; the trainer logs `n_thermal` and
`n_power_cap` alongside the old `n_throttled`, and checklist step 6's abort rule reads `n_thermal`.

**The larger finding: `014-train-step-rungs` was measuring heat.** Five runs now measure the same rung
at the same micro-batch with the same 3.59 GiB peak.

| run | median | min | p90 | p90/median |
|---|---|---|---|---|
| `024` | **737.5 ms** | 723.0 | 742.4 | **1.007** |
| `022` | 968.1 | 838.0 | 999.2 | 1.032 |
| `016` | 825.8 | 758.6 | 1086.0 | 1.315 |
| `015` | 1277.2 | 760.6 | 2546.0 | 1.993 |
| `014` | 2526.4 | 2495.7 | 3808.8 | 1.507 |

`014`'s minimum is 1 % under its median: over 40 timed micro-steps it never ran one fast step, where
`015` and `016` both reach ≈ 760 ms cold and only then degrade. That is a run whose cards were
**already saturated when the rung began** — `014` ran nine rungs back to back, L5 fifth, on the layout
`015` measured at 328 MHz steady the same day. `014` kept no thermal telemetry, so this is inference
from the step-time distribution, not a reading.

**What it costs us.** `014`'s L0 row (first, cold) stands, as do its parameter counts, peak-memory
column and the "memory is not the constraint" conclusion. Its ms/micro-step and tokens/s columns for L1
through L7b do not, nor do the `screen` hour estimates built on them, nor `06 §4`'s variant column.
**I21 loses its premise**: `014` read a 1.9 s gap between single-GPU 651 ms and DDP 2526 ms as the
per-micro-step all-reduce and concluded the recurrent rungs pay ≈ 8× the dense DDP overhead; `024` puts
that overhead at **≈ 86 ms, below the dense rung's ≈ 250 ms**. I21 is *not* closed — the recurrent
rungs' true reducer cost is simply unmeasured, and `024` measures one rung. Erratum appended to
`014/results.md`; no number in it is edited.

**Also done.** `scripts/bench/bench_membw.py` written — the real host-memory benchmark owed since `019`,
a STREAM sweep over thread count whose plateau, not a single-thread figure, is the scenario input. Not
yet run. `ruff` now excludes `experiments/`, which is an append-only record rather than source.
`/mnt/nvme/queue_screen_2.sh` drafted for ids 106–110 with checkpoints on the NVMe (checklist step 4;
`/home` is no longer tight at 179 GB free, but `/home` is SATA with btrfs zstd and was one of I22's
suspects). **Not launched.**

**Next, in order.** (1) Run `bench_membw` under its own id and put the plateau into
`sim/scenarios/local_3060.yaml`. (2) `018`, the per-rung re-bench, **one rung at a time from cold with a
cooldown between rungs** — it now carries `014`'s replacement and I21's test, not just warm-vs-cold.
(3) Recompute `06 §4`'s variant column and the per-rung micro-batches from `018`. (4) Restart the ladder
at L1 with id 106 and watch the first eval per checklist step 6. GPU 0 is the card to watch: it passes
at 82 °C but with its fan at 97 % and almost no margin, over a run measured in hours rather than
minutes.

**Still owed and not answerable from a command:** which slot or bracket physically holds which card,
and whether the layout built is candidate C. Open since `017`; `024` is the accepted layout on record.

### 2026-09-11 (small hours) — `018` and `026`: the whole per-rung table re-measured; I21 closes

**State.** The user confirms the layout is **candidate C — all four cards spaced on the board**, the
vertical kit under the PCIe slots and the VG4 moved to the bottom-front with the cables swapped. That
answers the half of checklist step 1 open since `017`, and it is recorded in `06 §1`. Two benches ran
after `024`: `018-train-step-rungs-recheck` (all nine rungs, 4-way DDP) and `026-train-step-rungs-1gpu`
(the same nine on one card). **Nothing is training yet; the ladder is cleared to start.**

**`018` replaces `014` outright.** Same nine rungs, same order, same command — and 21.7 minutes of
continuous load with **zero thermal-slowdown samples**, max 79 / 69 / 71 / 67 °C, every rung's p90
within 1.5 % of its median except L5d at 12 %. So a back-to-back sequence is fine on this layout; the
old one was the fault. L5 lands at 737 ms against `024`'s 737.5 ms from an isolated 1100-step run, so
the sequence costs nothing.

| rung | `014` | `018` | ratio | `026` 1 GPU | DDP overhead |
|---|---|---|---|---|---|
| L0 | 694 ms | **519** | 1.34 | 454 | 65 ms |
| L1 | 1013 | **565** | 1.79 | 334 | 230 |
| L2 | 1363 | **570** | 2.39 | 329 | 241 |
| L3 | 1938 | **628** | 3.09 | 384 | 244 |
| L5 | 2526 | **737** | 3.43 | 671 | **66** |
| L5d | 3779 | **1026** | 3.68 | 924 | 102 |
| L5-ne128 | 5862 | **2263** | 2.59 | 2052 | 211 |
| L5-ne1 | 2652 | **636** | 4.17 | 579 | 57 |
| L7b | 2678 | **815** | 3.28 | 744 | 71 |

The ratio column rises **monotonically across the first six rungs in run order** — 1.34, 1.79, 2.39,
3.09, 3.43, 3.68. That is a property of *when* a rung ran, not of the rung. **The previous entry's
erratum was itself too kind to `014`**: it said the L0 row stood, and L0 was 34 % slow. Corrected in
place.

**I21 is closed, and the answer is that there was never anything to explain.** The DDP overhead is a
ring all-reduce of the gradient and nothing else. At $N$ = 4 that is $2(N-1)/N$ = 1.5× the gradient,
BF16 at 2 bytes per parameter, over the measured $\lambda_{link}$ = 3.59 GB/s (`002-nccl`) — a
prediction with nothing fitted. It lands within 10 % on six of nine rungs, **L5 at ratio 1.02**.
**L5 pays 66 ms on 77.5 M parameters against L0's 65 ms on 100.1 M**: the same cost per parameter, not
the 8× I21 was opened on, and cheaper per parameter than the 271 M layered rungs at 0.85–0.89 ms/M.
The ≈ 1.9–2.9 s figure was thermal throttling, present in `014`'s single-GPU column as well as its DDP
one — which is why `018`'s DDP L3 (628 ms) beat `014`'s *single-GPU* L3 (673 ms), an impossibility that
was the first clue.

**Consequence for the design, not just the rig:** the recurrent rungs carry **no distributed-training
penalty beyond their parameter count**. Accumulation is worth having for the 271 M layered rungs, where
the all-reduce is ≈ 40 % of a micro-step, and close to irrelevant for the 77.5 M recurrent ones at
9–11 %. `06 §4` said the opposite until tonight.

**One number to treat with care.** `026` ran entirely on GPU 0, the warmest card, and its L5-ne128
(2052 ms) is *slower* than `014`'s (1544 ms) where every other rung is equal or faster. GPU 0 peaked at
75 °C with zero thermal samples over `026`'s 17.3 min, so it was not throttled — but a single-card
bench measures whichever card it lands on, and these four are not interchangeable. A per-card sweep
would settle it. I21's closure does not rest on it.

**`06 §4` and `sim/scenarios/local_3060.yaml` rewritten** from `018`/`026`: the full nine-rung table
with DDP and single-GPU times, parameter counts and the all-reduce model. `screen` costs **4.4 h** dense
(L0), **6.2 h** L5, **8.7 h** L5d, **19.2 h** L5-ne128; the five queued rungs 106–110 total **44.8 h**
without accumulation.

**Next: the ladder starts at L1, id 106.**

### 2026-09-11 (morning) — `106-l1-screen` finishes: the first valid `screen` rung, and I22 confirmed closed

**`106-l1-screen` completed, exit 0**, 1.000 B tokens in 1907 steps, 20 evals, **final held-out loss
3.2048 nats** against $L_{ref}$ = 3.3360 from `100-l0-screen` — **−0.1312 nats, −3.93 %**, at matched
FLOPs (top-2 of 8 experts, so 271.2 M parameters cost what L0's 100.1 M cost per token). Seed 0, one
seed.

**No verdict is recorded, on purpose.** ADR-013 wants a fraction of the baseline loss inside a
measured-σ band and takes σ from the L0 seed pair, **which exists only at `small`**. At `screen` there
is one seed of each and σ is unmeasured, so there is no band to test against. 3.93 % is 13–26× the
0.005–0.01 nats of seed spread `06 §4` records at `small`, so it is very unlikely to be noise — but
that is a direction, not a banded verdict, and the verdict belongs at `small` where both rungs run at
2 seeds. L1's role here is the **matched-FLOPs fine-grained MoE baseline** CLAUDE.md requires of every
experiment; it is H4's baseline arm and silent on H4 itself, which L7 decides.

**I22 is confirmed closed, and none of its suspects was the cause.** It was opened because `100` spent
18.0 h of wall-clock on 4.28 h of timed steps. `106` ran the **same 1907 steps of a heavier rung** in
**5.62 h**, of which 5.540 h is summed `step_s` and 0.079 h is evals and checkpoints — **0.1 %
unaccounted**, against ≈ 76 % for `100`. Not `/home`, not the evals, not the barrier, not the host: it
was I23's throttling all along, exactly as I23 claimed when it was opened *(closes I22)*, and `024`'s
layout fix removed it.

**Run health.** 50.2 k tokens/s held for the entire 5.62 h with **zero thermal-slowdown samples**. The
only thermal flags were single samples landing on eval steps, where the held-out pass briefly pushes
GPU 0 to 83 °C against 80 °C in steady training, with no effect on clock (1890 MHz minimum) or
throughput. Accumulation is worth **1.7×** here — 5.62 h against the 9.6 h `018` projects without it —
which is what I21's closure predicts for a 271 M rung whose all-reduce is ≈ 40 % of a micro-step.

**`107-l2-screen` started automatically** at 05:59 and is running. The queue continues to 108, 109, 110.

### 2026-09-11 (midday) — `107-l2-screen`: the parallel form gains where H2 budgeted a loss

**`107-l2-screen` completed, exit 0**, 1.000 B tokens in 1907 steps, **final held-out 3.1495 nats**.
Against `106`'s 3.2048 that is **−0.0553 nats, −1.73 %**, at unchanged parameters (271.1 M vs 271.2 M;
the parallel form drops one norm) and matched FLOPs. Against $L_{ref}$ = 3.3360 the ladder now stands at
**−5.59 %**.

**H2's quality clause budgeted a cost of ≤ 1 % and we measured a 1.73 % gain.** H2 is phrased as a
tolerance — the parallel form is wanted for the critical path, and ≤ 1 % is what we were willing to pay
— so the direction is favourable and the trade may not be a trade at all here. Two limits on what that
means. **H2 is stated for the shared block**, and L2 applies the parallel form to the *layered* stack,
so this tests the form and not yet the setting the hypothesis is finally about; that arrives with L5.
And **the latency clause is untouched** — it is Tier S and belongs to S1 in `sim/`.

**No verdict, same reason as `106`.** ADR-013's bands need σ from the L0 seed pair, which exists only at
`small`. 1.73 % is 5–11× the 0.005–0.01 nats of seed spread `06 §4` records there, so a seed artefact is
unlikely, but that is not the banded test. L2 is on the `small` list (`03 §2`) and gets the real verdict
at 2 seeds.

**Worth flagging in the interpretation, not just the number:** a parallel block is not a reordering. With
attention and FF reading the same input, gradients reach both directly from the residual, and at
$d$ = 768 over 12 layers that optimisation effect may be worth more than the representational loss costs.
Plausible story, not a measured mechanism, and the kind of thing that can shrink or invert with depth
and width — `medium` is where it would show.

**Run health.** 50.9 k tok/s, 5.56 h wall, **0.07 % unaccounted**, matching the clean accounting `106`
established when I22 closed. Thermal flags were single samples, mostly on eval steps, minimum clock never
below 1845 MHz, no throughput effect. Ambient rose through the morning and GPU 0 tracked it; the user
opened windows around 10:00 and it settled.

**`108-l3-screen` started 11:32** and is the first run with the **routing trajectory** the trainer gained
this morning. It is already earning its place: L3 dives to **35 dead experts of 96** and one router at
**2.2 of 8 effective experts** around step 19, then recovers to **1 dead by step 101** and keeps climbing.
That is the classic self-reinforcing MoE starvation being pulled back by the aux-loss-free bias
controller at its 1e-3 step, and it is the first time on this project we have *watched* it rather than
inferred it from a converged checkpoint. `106`'s endpoint (7.99–8.00 of 8 effective experts, 0 dead)
says where it lands.

**Owed:** routing endpoints for `106` *(done)* and `107` *(deferred)* via
`scripts/analysis/probe_routing.py` — it saturates ~10 of 12 cores, so it waits for an idle ladder rather
than denting a running rung's throughput. Each run keeps its own checkpoint, so nothing expires.
