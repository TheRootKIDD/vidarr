# bigmoe — Systems specification v0.1

Status: DRAFT. Implemented by `sim/` (discrete-event simulator) and `costmodel/`. Hardware numbers are *scenario inputs* with a source line each, never constants.

## 1. Hardware abstraction
- **Chip** $C$: peak $\phi$ FLOP/s (per precision), local memory capacity $m_C$ and bandwidth $\beta_C$, local-fabric bandwidth $\lambda_C$, rail NIC bandwidth $\nu_C$.
- **Unit**: $U$ chips on a local fabric (switch, or a $4{\times}4{\times}4$ torus for $U$ = 64 — Q8). Unit memory $m_U = U m_C$.
- **Fabric**: $U$ rail planes; rail $i$ connects chip $i$ of every unit through a clos of $N$ ports. A unit's aggregate clos bandwidth is $U \nu_C$ while each port is a cheap $\nu_C$ port — this is the note's P5 economy.
- **Scenarios** (`sim/scenarios/*.yaml`): `note64` ($U$ = 64, $m_C$ = 4 GB, unit fabric = one of the v2 P5 options as a scenario variable: fast router, 4×4×4 asymmetric-bandwidth torus (author's Q8: relative bandwidths 1 : 4 : 16 : 64 per level), even-bandwidth 3-D torus, 64-wide 1-D torus, 8×8 torus, doubled-bandwidth 8×8 mesh; $U$ ∈ {32, 64, 128} likewise (v2 P4: wider halves clos bandwidth, enlarges failure domain and minimum expert size) — and placeholder $\phi$, $\beta_C$, $\lambda_C$, $\nu_C$ to be sourced in Phase 0), `gpu_today` (a current 72-chip domain with commodity HBM and rail-optimized networking), and `local_3060` (the project's own rig: 4 units of $U$ = 1, $\phi$ ≈ 25.5 TFLOPS BF16, $\beta_C$ ≈ 360 GB/s, $m_C$ = 12 GB, host-bounced PCIe links — every value measured by `scripts/bench/`, see `docs/06`).

## 2. Unit-internal execution of an expert step
Why tensor-parallel (author, Q4): at the utilisation floor a unit's step time is $1/U$ of a single chip's for the same batch, so per-token latency drops $U$-fold at the same throughput; and a token enters the unit once, with the internal broadcast/reduce on the local fabric, so clos traffic per chip drops $U$-fold. Granularity was not the motive; the quality question (H4) stays ours, and none of this applies to a small installation except by simulation.

Tensor-parallel over $U$ chips, two-step pipeline: (i) broadcast/all-gather the token batch $X \in \mathbb{R}^{b \times d}$ to all chips; each chip computes its $d_{ff}/U$ slice of gate/up, the nonlinearity, and its slice of down; (ii) reduce-scatter/all-reduce the partial outputs. With $L_e$ = 2 the intermediate residual is reduced once between the two FF layers.

**Compute-bound condition.** Per chip per step: FLOPs $= 2\, b \cdot 3 d\, (d_{ff}/U)\, L_e$; weight bytes $= 3 d\, (d_{ff}/U)\, L_e\, b_w$. Arithmetic intensity $= 2b/b_w$ FLOP/byte, so the step is compute-bound iff
$$b \ \ge\ b_{min} = \frac{\phi}{\beta_C} \cdot \frac{b_w}{2} .$$
Example: $\phi/\beta_C$ = 1000 FLOP/byte and FP8 ($b_w$ = 1) → $b_{min}$ = 500; FP4 → 250.
**Tile condition**: $d_{ff}/U \ge s_{min}$ and $b \ge s_{min}$.
Both live in `costmodel.expert_step()`; the simulator's per-expert queue uses $b_{min}$ as its "enough tokens" threshold.

**Local traffic per step**: $2\, b\, d\, b_{act}$ bytes over the local fabric (broadcast + reduce); ring time $\approx 2 b d\, b_{act} / \lambda_C$ must be $\ll$ compute time $2 b \cdot 3 d (d_{ff}/U) L_e / \phi$ — i.e. $\lambda_C \gg \phi\, b_{act} / (3 (d_{ff}/U) L_e)$. Check per scenario.

**Alternative (granularity $g$)**: expert-parallel inside the unit — chip $i$ holds sub-expert $i$ whole; batch per chip $b/g$; no all-reduce, one extra dispatch hop instead. Same clos traffic. The simulator models both.

## 3. Fabric: rail-optimized clos, multicast, reduction
- Traffic unit: a token's stream vector ($d\, b_{act}$ bytes) plus metadata (expert ids and gates, tenant, sequence id, iteration, deadline).
- Per token per iteration: $k$ dispatches out and $k$ results back, before dedup.
- **Multicast** (P9): one copy leaves the attention owner; fan-out to the $k$ destination units at the lowest common switch.
- **In-network reduction**: the $g_i$-weighted sum of the $k$ expert outputs is formed at the lowest common switch on the way back (SHARP-style), so one copy returns. With both, top-tier bytes per token per iteration fall from $2 k\, d\, b_{act}$ toward $2\, d\, b_{act}$; leaf-tier bytes stay $2 k\, d\, b_{act}$.
- **Asymmetric provisioning**: for a given placement the simulator reports per-tier up vs down utilisation. The note's "less up bandwidth" claim is an *output* of S3, not an input.
- Clos tiers $T \in \{2, 3\}$, per-hop latency $\ell_{hop}$; per-iteration network latency $\approx 2 \cdot 2T\, \ell_{hop}$ plus queueing.

## 4. Placement controller
Inputs: windowed co-activation matrix $A_{ii'}$ (from router logs), per-expert load $\lambda_i$, unit capacities, tenant constraints. Objective: minimise expected upper-tier traffic (graph partitioning of $A$ over leaves) subject to load balance. Replicate experts whose $\lambda_i$ exceeds one unit's throughput; migrate at a bounded rate (moving an expert costs $3 d\, d_{ff} L_e\, b_w$ bytes). Precedent: DeepSeek's EPLB (static per-phase replication/placement). Re-run every $T_{place}$; the simulator measures utilisation loss between re-placements.

## 5. Continuous scheduler (P11)

### 5.1 Entities
- **Attention owner**: the unit holding a sequence's hot KV and token state. Runs attention for that sequence's tokens at each iteration, issues expert/memory requests, sums returns, runs halting. Fixed per sequence for its lifetime.
- **Expert unit**: per-expert queue $Q_i$; fires when $|Q_i| \ge b_{min}$ or the oldest entry has waited $\ge w_{max}$.
- **Memory shards**: knowledge-table partitions with the same queue policy.
Every unit may play all three roles (managed aggregation, §6).

### 5.2 Token lifecycle per iteration
dispatch ($k$ experts + memory) → attention (local, overlapped) → collect → sum → halting check → next iteration, or exit to final blocks and output.

### 5.3 Balancing in time
All tokens starting an iteration together is the lockstep pathology (hot "early" experts). Mitigations the simulator exposes: admission jitter; expert-choice mode where an expert with spare capacity pulls tokens whose second-best score is within $\delta$; per-$j$ replication of experts that are hot at small $j$. Metric: per-expert utilisation variance over time windows.

### 5.4 Per-sequence constraints
Within a sequence, token $t$ at iteration $j$ needs the per-iteration states $c^{(j)}_{s}$ of the tokens in its local range (or copied final states for halted $s$, `01 §4.7`) and the global entries $\mathcal{G}$ of every earlier *completed* token (v2 P7, ADR-021). So: a segment's tokens proceed in parallel across iterations (standard layered dependency); segments of one document run in order, each waiting for the previous segment to complete before it can attend globally; documents run in parallel. Decode is sequential per sequence; a decode token attends locally to the last $W$ tokens (per-iteration caches, still resident) and globally to $\mathcal{G}$. Ordering is enforced at the owner; expert units are stateless.

### 5.5 Liveness and Little's law
Timeouts $w_{max}$ prevent starvation; admission control bounds tokens in flight $Z$. Throughput $X$ and per-token latency $L \approx \sum_j (\ell_{net} + \ell_{queue,j} + \ell_{expert} + \ell_{attn})$ are tied by $Z = X L$. $Z$ is also the in-flight memory budget (§8). Deadlock analysis: owners wait on experts, experts wait on batch; $w_{max}$ breaks the cycle; the simulator asserts progress.

### 5.6 Pool size vs utilisation (P10)
Tokens in flight needed to keep $N_e$ experts busy: $Z_{min} \approx N_e\, b_{min} / k$ per iteration in flight. S4 sweeps pool size to produce the utilisation curve, including "two half-size installations vs one pooled".

## 6. Prefill/decode managed aggregation (P14)
Each unit receives a controlled mix of prefill chunks (compute-heavy, many tokens per sequence, $c$-stream only) and decode tokens (both streams, latency-sensitive). After v2, prefill of one prompt is segment-sequential ($\lceil L/W \rceil$ segments × $r$ iterations on the critical path) unless it runs in the author's "less effective" parallel mode (local attention only, or global attention over a coarser summary — Q12); S5 models both and reports TTFT for each. Controller target: FLOP utilisation and memory-bandwidth utilisation both near ceiling; knobs: ratio $\rho$ (prefill : decode tokens per batch) and chunk size. No KV transfer is needed because the owner is fixed per sequence. S5 compares against a disaggregated baseline (separate prefill and decode pools, KV transfer cost, phase-specific parallelism) on identical hardware, at equal p99 time-per-output-token.

## 7. KV tiers and the cold index (P15)
| Tier | Location | Capacity | Bandwidth | Access |
|---|---|---|---|---|
| Hot | unit memory | per in-flight sequence: $W$ tokens × $r_{max}$ per-iteration entries (transient) + recent $\mathcal{G}$ entries | $\beta_C$ | dense, every iteration |
| Warm | unit memory / host DRAM | horizon $H_w$ blocks of $\mathcal{G}$ (one entry per token) | $\beta_C$ or host link | top-$K_{idx}$ blocks |
| Cold | SSD / storage cluster | unbounded, $\mathcal{G}$ only | $\beta_{cold}$ | ANN hits only |

KV bytes: one $\mathcal{G}$ entry per token $= 2 H_{kv} d_h b_{kv}$, persistent; per-iteration entries exist only inside the local range (v2 P7: "almost the full 128×" reduction relative to per-iteration global caches). Cold demand per generated token $= r_t \cdot (\text{cold hits}) \cdot B_{idx} \cdot 2 H_{kv} d_h b_{kv}$ — the same formula, now against a cache $r_{max}$× smaller. S6 requires a **hit-locality model** — fraction of selected blocks per tier as a function of context length and task — produced by Tier-A runs with the index enabled (L9). P15's bandwidth claim holds iff cold hits per token × block bytes × throughput $< \beta_{cold}$, with headroom for tail behaviour.

## 8. Memory budget per unit ($m_U = U m_C$)
Resident experts $n_e^{unit} \cdot 3 d\, d_{ff} L_e\, b_w$ + hot KV for owned sequences + in-flight token state ($Z_{unit} \cdot 2 d\, b_{act}$, plus partial sums awaiting returns) + adapter store + index summaries. The note's "4 GB per chip" becomes a concrete test: does `note64` fit its resident experts and hot KV at target $Z$? S7 reports the capacity/bandwidth Pareto for $m_C \in \{4, 8, 16, 32, 96\}$ GB. Training memory (weights + grads + optimiser state ≈ 16 B/param, activations for $r_{max}$ iterations × 2 streams) is reported separately — it is why the 4 GB claim does not transfer to training.

## 9. Knowledge-table service (P17)
Sharded by product-key half; a lookup is one rail hop to the shard owning the key range, returning $k_m$ values. Shards are placed by the same controller as experts (co-locate with experts that co-fire with them). Tables are versioned; a request pins one version; tenant-tagged entries (§10).

## 10. Multi-tenancy (P18)
- Adapter store per unit: all tenants' LoRA weights for the shared block ($|\tau| \times$ rank $\times d \times$ matrices) — small; gathered per vector.
- Tenant experts are placed like any expert; the router mask is enforced at the owner.
- Isolation: mixed-tenant batches leak timing; per-tenant quotas; option to pin regulated tenants to dedicated units at a utilisation cost that S8 quantifies.
- Multi-model: one spine; products = adapters + tenant experts + memory entries. Note: multi-LoRA serving and fine-tuning-on-shared-infrastructure already exist commercially; the new part is tenant experts inside one routed pool.

## 11. Failure model
Unit failure = its experts and its owned sequences become unavailable. Experts: replicas on other units (placement keeps ≥ 2 copies of hot experts; cold experts may be single-copy and reloaded from storage). Sequences: re-prefill on another owner from the token log — no in-flight checkpointing. A chip failure inside a TP unit is a unit failure unless the local fabric supports degraded $U-1$ operation (re-sharding $d_{ff}$ slices — expensive; default: fail the unit). S9 measures availability vs replication factor and unit failure rate.

## 12. Geography (P19)
Cost-model option only (latency, power price, cooling PUE). No simulator work.

## 13. Training-system implications (the note's gap)
The inference design is asynchronous and per-token adaptive; training is neither by default.
1. **Depth**: pretrain with $r$ sampled per batch (fixed within a batch); enable ACT after the recurrent model is stable; variable-depth micro-batching later.
2. **Routing during training**: dropless (no capacity factor), expert-parallel groups = units, aux-loss-free bias balancing; log co-activation for the placement controller.
2b. **Segments (v2 P7)**: a training step holds many documents × one segment each; each segment attends locally through per-iteration caches and globally to the final vectors of its document's earlier segments, held as a memory (stop-gradient by default, ADR-023 — Transformer-XL practice); segments of one document are processed in order across steps. This is the training-side counterpart of the visibility rule in `01 §4.3`.
3. **Scheduler**: training uses lockstep all-to-all with a barrier per iteration — confirmed by the author (Q3: lockstep for determinism; continuous execution only at inference, where weights are fixed). The asynchronous training variant is dropped (ADR-016).
4. **Memory**: activations of $r_{max}$ iterations × 2 streams — checkpoint per iteration; gradient through the shared block accumulates over $r$ applications — track gradient norm per $j$.
5. **Index in the loop**: dense-attention warm-up, then indexer distillation (`01 §6.3`).
6. **Sparse expert gradients**: with $N_e$ large each expert sees few tokens per step; memory-layer / PEER practice (higher LR for sparse parameters, no weight decay on keys) applies.

## 14. Simulator outputs (what `sim/` must report)
Per scenario: utilisation per unit over time; persistent $\mathcal{G}$ bytes per token and transient local-cache bytes per in-flight sequence; fabric bytes per token per iteration per tier, up and down; per-iteration and per-token latency distributions (p50/p99); tokens in flight; expert queue-wait distribution; cold-tier hit rate and bandwidth; placement drift; availability. Every S-experiment in `docs/03` names which of these it reads.

**Calibration rule**: before any S-result is trusted, the simulator must reproduce (a) the local S0 run on `local_3060` and (b) a published expert-parallel deployment's throughput and latency, each within ±20% (S0 in `docs/03`, I1 in `docs/04`). Two calibration points at very different $\lambda_C/\phi$ ratios are what make extrapolation to `note64` defensible.
