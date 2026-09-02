# bigmoe — Architecture specification v0.1

Status: DRAFT. Each section names the decisions it depends on (ADR-nnn, `docs/04`) or the questions it leaves open (Qn). Hardware appears only as abstract parameters; see `docs/02`.

## 0. Scope
Defines the model as a function from token sequences to next-token distributions, its parameters, its per-token compute, and the invariants the systems layer relies on. Excludes fabric, scheduling and placement (`docs/02`) and experiment design (`docs/03`).

## 1. Notation

| Symbol | Meaning | Tier-A `small` default |
|---|---|---|
| $V$ | vocabulary size | 32k |
| $d$ | model width (per stream) | 768 |
| $H, H_{kv}, d_h$ | query heads, KV heads (GQA), head dim | 12, 4, 64 |
| $E, F$ | number of early / final blocks (unshared weights) | 2, 2 |
| $F_p$ | of the $F$ final blocks, how many are prediction-only | 1 |
| $M$ | the single shared middle block | — |
| $j$ | middle-block iteration index, $1..r_t$ | — |
| $r_t$ | number of middle-block iterations executed for token $t$ | — |
| $r_{min}, r_{max}$ | bounds on $r_t$ | 2, 8 (`medium`: 16) |
| $e_j$ | depth embedding for iteration $j$ ($\in \mathbb{R}^{d_e}$) | $d_e$ = 64 |
| $c_t^{(j)}, p_t^{(j)}$ | context / prediction stream of token $t$ after iteration $j$ | $\in \mathbb{R}^d$ |
| $N_e$ | experts in $M$ | 8 (author, Q6); 128 for the parameter-matched H6 variant; 1 = dense recurrent block |
| $k$ | active experts per token per iteration | 2 (4 in the $N_e$ = 128 variant) |
| $U$ | hardware unit width, chips per unit (abstract) | 4 (simulated) |
| $s_{min}$ | minimum per-chip tile width | 128 |
| $d_{ff}$ | expert intermediate width | $\ge U s_{min}$ |
| $g$ | sub-experts per unit (granularity knob) | 1 |
| $L_e$ | FF layers inside one expert | 1 |
| $W$ | local attention range: tokens still in flight / the current segment (per-iteration KV) | 512 |
| $\mathcal{G}$ | global cache: one K/V entry per completed token, from its final middle-layer vector $c_s^{(r_s)}$ | — |
| $B_{idx}, K_{idx}$ | KV block size; blocks retrieved per query | 16, 32 |
| $d_{idx}$ | indexer dimension | 32 |
| $N_m, d_m, d_v, k_m$ | knowledge-table size, key dim, value dim, values retrieved | $2^{20}$, 128, 768, 1 |
| $m$ | MTP depth (tokens predicted ahead, incl. next) | 4 |
| $\tau$ | tenant id | — |
| $b_w, b_{act}, b_{kv}$ | bytes per weight / activation / KV element | — |

Positions $t$ are causal. `screen`/`small`/`medium` sizes are anchored in `docs/06 §3` (ADR-015, ADR-018); `costmodel/` derives $d_{ff}$ from the FLOPs budget (single-stream `small`, $r$ = 8, $E$ = $F$ = 2, 1024 budget keys: 896 at $N_e$ = 8, $k$ = 2; 448 at $N_e$ = 128, $k$ = 4; 1792 dense — ADR-025/027; two streams do not fit the budget and run at 2× against a √2·$d$ comparator, ADR-026). Quality thresholds remain ADR-013.

## 2. Macro-structure

Embedding → $E$ early blocks → middle block $M$ applied $r_t$ times → $F$ final blocks (last $F_p$ prediction-only) → output heads.

- Early and final blocks are standard pre-norm blocks (parallel form, §4.1) with their own parameters; dense, SwiGLU width 4$d$ (ADR-006). They give the shared block a stable input distribution and specialise the output.
- $M$ holds "almost all" experts. One weight set, $r_t$ applications, conditioned on $j$ through $e_j$ (§4.2).
- $r_t$ is per token (§4.7). Prefill may use a fixed schedule for throughput (ADR-007 pending).
- The final middle-layer vector $c_t^{(r_t)}$ is the token's *global representation*: it is the only thing far tokens ever attend to (§4.3, v2 P7).

Rationale (note P6): $r_{max}$ stacked layers with $n$ experts each is a special case of one block applied $r_{max}$ times with $r_{max} n$ experts *if* the router can condition on $j$; the general case lets any expert be used at any depth. Caveat: attention parameters are shared across $j$ unless the per-iteration adapters of §4.2 are on. Only *weights* are shared: there is one KV cache per iteration (author, Q2; ADR-017).

## 3. Token state: two streams (note P12) — *optional* (ADR-020)

Status: the author calls this a nice but non-standard idea and suggests leaving it out; the default configuration is single-stream (`two_stream: false`), MTP heads read the single stream, and the two-stream model is the last ladder step (L4). Everything below applies when it is on.

Each token carries two vectors through every block:

- **Context stream** $c_t$ — what other tokens see. The *only* source of keys and values. Supervised only indirectly (gradient arrives through $p$).
- **Prediction stream** $p_t$ — predicts $x_{t+1..t+m}$. Never a key or value ("cannot be seen by attention"). Queries the context stream.

Initialisation, with a learned stream tag $e_p \in \mathbb{R}^d$:
$$c_t^{(0)} = \mathrm{Emb}(x_t), \qquad p_t^{(0)} = c_t^{(0)} + e_p .$$
Both streams pass through the *same* weights (batched as two rows). Only $c$ produces K/V; both produce Q.

Visibility (ADR-002):
- $c_t^{(j)}$ attends to $\{c_s^{(j)} : s \le t\}$.
- $p_t^{(j)}$ attends to $\{c_s^{(j)} : s \le t\}$ — including its own token's context.
- Variant L4b: $p_t^{(j)}$ attends to $\{c_s^{(j)} : s < t\}$ only; $c_t$ can then be computed one step behind $p_t$ (pipelined).

Consequences:
- **Prefill** computes only $c$ for prompt tokens (plus $p$ for the last prompt token) and skips the $F_p$ prediction-only blocks for all but the last token.
- **Decode** computes $c_t$ and $p_t$ for each accepted token; the $m$ MTP heads read only $p$ (§7).
- **KV cache** is a function of $c$ only.

Accounting (details §10). Relative to a single-stream model of equal *decode FLOPs per token* (width $\approx\sqrt2 d$): parameters halve, weight bytes per decode step halve, prefill FLOPs halve (one stream). Latency (author, Q1, reconciled in `docs/00 §P12`): the utilisation floor $b_{min}$ counts rows and a token is two rows, so a unit at the floor holds $b_{min}/2$ tokens and its step time halves; throughput per unit is unchanged; the halving applies to the weight-streaming part of the iteration, not to fabric round-trips. `costmodel/` reports the latency floor at full utilisation with and without two streams. The quality premise (two $d$-vectors through shared weights $\approx$ one $\sqrt2 d$-vector) is hypothesis H12.

## 4. The middle block $M$

For a stream vector $x \in \{c_t^{(j-1)}, p_t^{(j-1)}\}$ at iteration $j$:
$$u = \mathrm{LN}(x)\odot\big(1+\gamma(e_j)\big) + \beta(e_j)$$
$$x^{(j)} = x + \underbrace{\mathrm{Attn}\big(u;\ \mathcal{K}^{(j)}_{\le t}\big)}_{\text{parallel}} + \underbrace{\mathrm{MoE}(u, e_j, \tau)}_{\text{parallel}} + \underbrace{\mathrm{Mem}(u, e_j, \tau)}_{\text{parallel}}$$

### 4.1 Parallel branches (note P2)
All branches read the same normalised input and are summed; none sees another's output within the iteration. Systems consequence: the attention owner dispatches the MoE and memory requests *before* computing attention and sums on return (`02 §5.2`).

### 4.2 Depth conditioning (ADR-003)
$e_j$ is a learned embedding of $j$ (optionally also of $r_{max}$ and remaining budget). It enters the norm modulation above (AdaLN-style scale/shift), the router (§4.4), the halting head (§4.7) and, optionally, attention/expert projections through a per-iteration low-rank adapter
$$W \mapsto W + A_j B_j, \qquad \mathrm{rank}\ r_a \ll d .$$
Per-iteration adapters on attention are what lets the shared block simulate per-layer attention weights (`docs/00 §P6`, caveat 2). Default off; ladder step L5b.

### 4.3 Attention (v2 P7, ADR-021)
Grouped-query attention with rotary positions. Two key ranges, combined either in one softmax or as two gated branches (ADR-008 pending):
- **Local, per-iteration**: at iteration $j$, token $t$ attends to $\{c_s^{(j)} : t - W < s \le t\}$ — the per-iteration KV of tokens still in flight (the current segment). This is the only place per-iteration KV is used; these caches are transient and are discarded once a token leaves the local range (Q11 asks whether iterations $j' < j$ should also be visible locally; default: same iteration only).
- **Global, final-vector**: at every iteration, token $t$ attends to $\mathcal{G}_{<t-W} = \{K/V(c_s^{(r_s)})\}$ — one entry per *completed* token, computed once from its final middle-layer vector. Dense over $\mathcal{G}$ at Tier-A context lengths; indexed (§6) beyond that.
- **Visibility rule**: a token becomes globally visible only when its iterations are complete. Hence batches (segments) of one document are processed in order, each attending globally to the previous ones' final vectors, while parallelism comes from many documents in flight — Transformer-XL-style segment recurrence with a single global cache. Whether local attention should also reach into the previous segment's per-iteration caches is ADR-022; gradients do not flow into earlier segments' final vectors (ADR-023: stop-gradient; `through` is an ablation knob).
- **Consequence**: the persistent KV cache is one entry per token; only the last $W$ tokens of each in-flight sequence carry $r_{max}$ caches (ADR-017 as amended). Sharing K/V across iterations (L6b) now concerns the transient local caches only.
- **Baseline semantics** (ladder L5, before L5d): global attention over per-iteration caches of the same iteration, i.e. the standard layered semantics; L5d switches the global range to $\mathcal{G}$ and measures the cost (H15a).

### 4.4 Router
Input: $u$ (projected to $q = W_q u \in \mathbb{R}^{d_r}$), $e_j$, tenant $\tau$. With expert keys $\kappa_i$, per-expert bias $b_i$ and an additive mask $\mu_\tau \in \{0, -\infty\}^{N_e}$:
$$\ell_i = q(u, e_j)^\top \kappa_i + b_i + \mu_{\tau,i}, \qquad \mathcal{S} = \mathrm{Top}\text{-}k(\ell), \qquad g_i = \mathrm{softmax}_{i \in \mathcal{S}}(\ell)_i .$$
- **Large $N_e$**: product-key factorisation $\kappa_{(a,b)} = [\kappa^{(1)}_a; \kappa^{(2)}_b]$ so top-k costs $O(\sqrt{N_e})$. Hierarchical variant: pick unit first, then expert within unit — aligned with placement (ADR-004).
- **Balancing**: $b_i$ updated by a load controller (aux-loss-free balancing) plus a small sequence-level auxiliary loss. Balancing must be *temporal* as well as spatial (`02 §5.3`); the router exposes $\ell$ so the scheduler can run expert-choice mode.
- **Tenant mask**: shared experts always allowed; tenant-owned experts only for $\tau$.
- **Depth awareness**: $e_j$ is an input by default; the author expects the router can infer depth from the state alone (Q2), so E-Q2 ablates the router without $e_j$. Log the per-$j$ expert histogram — concentration on "early" experts at small $j$ is the diagnostic for `docs/00 §3.2`.

### 4.5 Experts (note P4, P8)
Gated FF stack with $L_e \in \{1, 2\}$ layers (residual + norm between layers when $L_e = 2$):
$$\mathrm{Exp}_i(u) = W^{down}_i\big(\mathrm{SiLU}(W^{gate}_i u)\odot W^{up}_i u\big) \quad (L_e = 1).$$

**Sizing invariant (hardware quantisation).** All experts in $M$ share one $d_{ff}$:
$$d_{ff} = \max\big(U \cdot s_{min},\ d_{ff}^{quality}\big),$$
the smallest width whose per-chip tensor-parallel slice $d_{ff}/U$ fills a tile, or a multiple. The systems layer never executes fewer weights than one expert at a time. In Tier A, $U$ and $s_{min}$ are simulated: they set $d_{ff}$ and the reported utilisation; no kernel changes.

**Granularity knob $g$.** One unit may host $g$ sub-experts of width $d_{ff}/g$, executed expert-parallel inside the unit (each chip = one sub-expert) instead of one tensor-parallel expert. Fabric traffic is identical; per-chip batch is $1/g$. This is the H4 experiment (L7).

Parameters per expert ($L_e = 1$, SwiGLU): $3\, d\, d_{ff}$. With 2:4 sparsity the dense-equivalent width is $2 d_{ff}$ at equal FLOPs (§8).

### 4.6 Knowledge memory (note P17)
A retrieval branch with an expert's interface:
$$q_m = W_m [u; e_j], \qquad \mathcal{N} = \text{top-}k_m\ \text{of}\ q_m^\top \kappa_n, \qquad \mathrm{Mem}(u,e_j,\tau) = W_o \sum_{n \in \mathcal{N}} \alpha_n v_n .$$
- Table: $N_m$ entries with keys $\kappa_n \in \mathbb{R}^{d_m}$ (product keys for sub-linear lookup) and values $v_n \in \mathbb{R}^{d_v}$; entries carry a tenant tag (shared or $\tau$).
- **Population** (ADR-009, accepted; author's Q10 and addendum): corpus sentences → sentence vectors; no entailment extraction at first. Values are *parametric*: $v_n = g_\theta(\mathrm{enc}(s_n))$ with $s_n$ a sentence, $\mathrm{enc}$ a pretrained sentence encoder and $g_\theta$ trained with the model (the author's "layer in front"); keys $\kappa_n = h_\theta(\mathrm{enc}(s_n))$ likewise, product-key factorised for lookup. Fine-tuning $\mathrm{enc}$ itself by backpropagation (the author's preferred form if the table will grow) is variant L8c; replacing sentences by extracted entailments (atomic facts) is L8d, only if L8 shows the table is used. Because the map is learned rather than the entries, new sentences can be added later by running them through the same map.
- The table is "correct but not complete": training must not depend on coverage. Coverage is measured separately from model quality.

### 4.7 Iteration count $r_t$ (note P7)
Halting head on the context stream at each iteration:
$$h_t^{(j)} = \sigma\big(w_h^\top [u; e_j]\big), \qquad r_t = \min\Big\{ j : \sum_{j' \le j} h_t^{(j')} \ge 1-\epsilon \Big\}\ \text{clipped to}\ [r_{min}, r_{max}] .$$
Training regimes, in order:
1. **Fixed** $r$, sampled per batch from a schedule — establishes the recurrent model.
2. **ACT-style**: ponder cost $\lambda_p \sum_t r_t$; output = the last computed state (ADR-010, accepted; author's Q5).
3. **RL fine-tuning** of the halting policy (optionally also $k$ and $K_{idx}$) against a compute-aware reward — only after 1–2 are stable.

**Ragged-depth semantics (binding; author's Q5).** If token $s$ halted at $r_s < j$, then for all $j > r_s$: $c_s^{(j)} := c_s^{(r_s)}$ (state copy — later tokens attend to the last computed vector). Its K/V for iterations $> r_s$ are computed once from that state and stored as a single entry shared by all later iterations (ADR-011, accepted): exact when attention weights are shared across iterations, recomputed per iteration only if per-iteration attention adapters (L5b) are on. Consequence: "fewer iterations → less KV cache" holds — a token halted at $r_s$ stores $r_s + 1$ KV entries instead of $r_{max}$.

## 5. Early and final blocks
Standard blocks (attention + FF, parallel form, own weights). Final blocks $1..F-F_p$ process both streams; the last $F_p$ process only $p$ and are skipped in prefill except for the last token. Output head tied to the embedding by default.

## 6. Indexed attention (note P15)

### 6.1 Tiers
| Tier | Contents | Access | Selection |
|---|---|---|---|
| Hot | last $W$ tokens: per-iteration caches (transient) + recent $\mathcal{G}$ entries | dense | none |
| Warm | older $\mathcal{G}$ blocks | block scores | indexer top-$K_{idx}$ |
| Cold | ancient $\mathcal{G}$ blocks | ANN index (HNSW/IVF) | indexer query → ANN |

After v2 every tier beyond the local range holds **final-vector entries only** — one per token — and the index is one structure per sequence, shared by all iterations and heads; only the query changes per (token, iteration).

### 6.2 Indexer
Terminology: the note's "index" is the sub-linear structure over the keys (the author: $n$ is the width of attention, i.e. the number of keys — Q9); the *indexer* below is our scoring module that decides which blocks to fetch from it. The author has not designed this part; §6 is ours.

One low-dimensional indexer per iteration, shared across heads: $q_{idx} = W_{idx}[u; e_j] \in \mathbb{R}^{d_{idx}}$. Each block has a summary key $\kappa_{blk}$ (mean or learned pooling of its final vectors $c_s^{(r_s)}$). Score → top-$K_{idx}$ blocks → all heads attend to those blocks' tokens with normal K/V. This bounds queries to one per (token, iteration) rather than one per head (lightning-indexer / NSA-style block selection).

### 6.3 Training with the index in the loop
- Warm up with dense attention; enable the indexer; train its scores against the dense teacher's block-attention mass (KL); then end-to-end through the selected values only (top-k is non-differentiable; gradient flows through the selected set).
- Curriculum on $W$ and $K_{idx}$.

### 6.4 Cold tier and ragged depth
Blocks of $\mathcal{G}$ enter the cold index when older than the warm horizon; the ANN key is $\kappa_{blk}$; insertion is incremental per block, once, when the block's tokens complete. A halted token's global entry is simply its last computed vector (ADR-010/011). Cold-tier hits per token are a *measured* quantity (`02 §7`); the "KV on SSD" claim rests entirely on them being rare.

## 7. Multi-token prediction (note P13)
$m$ heads on $p_t^{out}$ ($c_t^{out}$ single-stream): head 1 predicts $x_{t+1}$ (main loss); heads $2..m$ predict $x_{t+2..t+m}$ as **independent heads** — RMSNorm → linear $d \to d$ adapter → tied output embedding — with each position contributing to exactly one auxiliary head per step (`mtp.subsample`, ADR-012; sequential modules are the optional L3b). Loss weights $\lambda_2 \ge \lambda_3 \ge \lambda_4$. Decode: draft $m$ tokens from the heads, verify with one pass computing $c$ for all drafts and $p$ for the last accepted token. Acceptance rate per head is a first-class metric.

## 8. Numerics and sparsity (note P3)
- Precision ladder: BF16 → FP8 weights+activations (block scaling) → FP4 weights (NVFP4-style) — separate ladder steps.
- Sparsity (ADR-019, accepted; author's Q7): weights only, no activation sparsity; after 1 % of the token budget, derive a 2:4 mask per expert weight matrix by magnitude and keep it fixed. Compared against dense at matched FLOPs (dense-equivalent width doubles). Periodic mask updates (RigL) are out of scope.
- Weight-tied recurrence: monitor residual-norm growth across $j$; use the norm modulation of §4.2 and, if needed, a learned per-iteration residual scale.

## 9. Per-tenant adapters (note P18)
- Tenant id $\tau$ travels with the token. Adapter: rank-$r_a$ LoRA on the shared block's $W_Q$, $W_O$ and router projection $W_q$ (not on experts), applied per vector via batched-gather matmul.
- Tenant experts: extra experts unlocked by $\mu_\tau$; trained on tenant data with the spine frozen.
- Tenant memory entries: appended to the knowledge table with the tenant tag.

## 10. Accounting (per token, per iteration, both streams unless noted; FLOPs count multiply-add as 2)
- Attention projections: Q and O on both streams: $2 \cdot 2 \cdot 2 d^2$; K and V on $c$ only: $2 \cdot 2\, d\, H_{kv} d_h$.
- Attention scores: $2 \cdot 2\, d\,(W + |\mathcal{G}_{visible}|)$ per stream, with $|\mathcal{G}_{visible}| = K_{idx} B_{idx}$ once indexed. **Budget convention (ADR-027):** at the training context un-indexed, $W + |\mathcal{G}_{visible}|$ averages $(L+1)/2$ = 1024 keys at $L$ = 2048; that is the figure every `06 §3` anchor uses.
- KV bytes: persistent $2 H_{kv} d_h b_{kv}$ per token (one $\mathcal{G}$ entry); transient $\le W \cdot r_{max} \cdot 2 H_{kv} d_h b_{kv}$ per in-flight sequence.
- MoE: $2 \cdot k \cdot 3\, d\, d_{ff}\, L_e$ per stream; router $O(d_r \sqrt{N_e})$.
- Memory: $O(d_m \sqrt{N_m}) + 2\, k_m\, d_v\, d$ per stream.
- Prefill: $c$ only → drop the per-stream doubling → ≈ ½ of decode per token.
- Fabric bytes per token per iteration (activations): $2 \cdot k \cdot d \cdot b_{act}$ **per stream** (dispatch + combine; two streams double it — ADR-027), before multicast/reduction dedup.

Worked examples for the Tier-A `small` config and for a note-scale config ($d$ = 8192, $U$ = 64, $s_{min}$ = 256 → $d_{ff}$ = 16384, $N_e$ = 65 536, $k$ = 8, $r_{max}$ = 128, $L_e$ = 1: ≈ 0.4 B params per expert, ≈ 13 GFLOP per token per iteration for two streams) live in `costmodel/` and must match `docs/02 §2`.

## 11. Config schema (YAML sketch)
```yaml
model:
  d: 768
  heads: 12
  kv_heads: 4
  head_dim: 64
  vocab: 32000
  early_blocks: 2
  final_blocks: 2
  pred_only_final: 1
  middle:
    r_min: 2
    r_max: 8                       # 16 at `medium`
    depth_embed: 64
    per_iter_lora_rank: 0          # L5b
  streams:
    two_stream: false              # true = L4 (optional, last ladder step)
    p_sees_own_context: true       # false = L4b
  router:
    k: 2                           # 4 in the n_experts=128 variant
    n_experts: 8                   # author's local default; 128 = parameter-matched H6 variant; 1 = dense recurrent block
    product_keys: true
    hierarchical: false
    balance: aux_free
  experts:
    unit_width: 4                  # U (simulated)
    tile_min: 128                  # s_min
    d_ff: null                     # derived unless overridden
    layers: 1                      # L_e
    granularity: 1                 # g
  attention:
    window: 512                    # local range W (in-flight / segment)
    global_source: per_iteration   # L5 baseline; final_vector = v2 design (L5d, ADR-021)
    segment_memory_grad: stop      # ADR-023: stop | through
    prev_segment_local: false      # ADR-022
    kv_share_every: 1              # L6b (transient local caches only)
    index: { enabled: false, block: 16, topk: 32, indexer_dim: 32 }
  memory: { enabled: false, n_entries: 1048576, key_dim: 128, topk: 1, values: parametric, encoder_finetune: false }   # GPU product-key memory (L8); L8b = chunk retrieval; L8c = encoder fine-tune
  halting: { mode: fixed, ponder_cost: 0.0, output: last, halted_kv: once }
  mtp: { m: 4, style: independent, subsample: auto }   # ADR-012; m: 1 disables
  numerics: { dtype: bf16, sparsity_2_4: false }   # mask, when on, is fixed after 1% of tokens (ADR-019)
  tenants: { lora_rank: 0 }
```

## 12. Dependencies
Accepted: ADR-001 ($U$ abstract), ADR-002 (stream visibility), ADR-003 (depth conditioning), ADR-004 (router shape), ADR-009 (parametric memory values), ADR-010 (last state), ADR-011 (halted-token KV once), ADR-016 (lockstep training), ADR-017 (per-iteration KV), ADR-018 (local expert count), ADR-019 (weight-only fixed 2:4 mask), ADR-020 (two streams optional), ADR-021 (v2 attention: local per-iteration, global final-vector). ADR-022 (sliding local window), ADR-006 (dense skeleton), ADR-012 (independent MTP heads), ADR-023 (stop-gradient memory), ADR-025/026/027 (widths, L4 budget, key convention), ADR-013/014/015 (thresholds, corpus, envelope). Pending: ADR-007, 008, 024. Open: Q9 (indexer is ours; the index lives over $\mathcal{G}$), Q11, Q12.
