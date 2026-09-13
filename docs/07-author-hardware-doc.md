# bigmoe — The author's hardware document, read against the note

**What this is.** A summary, in our words, of Bjarke Hammersholt Roune, *Designing AI Chip Hardware and Software* (2026, 154 pp.) — the "in-depth post" linked at the end of the design note (`docs/source/Designing_AI_chip_software_and_hardware.{pdf,txt}`; a public Google Doc). Compiled 2026-09-02 by reading the whole text. It settles *intent* for the hardware side of the note the way `docs/04 §Author's answers` does for the model side; every number in it remains a scenario input, not a constant (CLAUDE.md).

**How to use it.** §2.7 and §6 are the source lines for `sim/scenarios/note64.yaml`; §3 is the per-point reading; §4 lists what disagrees with `docs/02` — each item there is either an open question (I17–I19) or part of the proposed ADR-028 in `docs/04`.


Reading rule used throughout: **A** = author asserts / opines; **C** = author cites someone else; **M** = author reports something measured or observed first-hand (TPU history, MLPerf, etc.). The author himself says the document contains few numbers *on purpose* (L1975 "As determined by an undisclosed process"): he refuses to give bandwidths, latencies or prices he cannot source. So nearly every number below is a structural or ratio number, not an absolute spec.

---

## 1. Document map

| # | Section (txt line, TOC page) | Gist |
|---|---|---|
| 0 | Front matter, L1–45 | Who (TPUv3 software lead, ex-Nvidia/Amazon/Facebook), why (decided not to found a chip company, publishes the plan instead), "AI CPU" thesis in one paragraph (L33). |
| 1 | Proposal for a hardware design (summary), L115, p.4 | The 11-bullet chip: CPU cores + one systolic array per core, MB-scale software-managed SRAM, DMAs everywhere incl. SRAM→net→SRAM with 8-bit Huffman, no host, little HBM, an SSD per chip for KV, 7-bit multiplies only, 1:2 sparsity, 1-D/2-D copper-trace torus on snap-together boards, XLA backend + C++ intrinsics, inference first. Nerd box contrasts with Groq. |
| 2 | Systolic arrays — the foundation, L142, p.5 | Every AI chip is systolic arrays plus glue; brand names are marketing. |
| 3 | Larger systolic arrays are more efficient, L180, p.7 | Doubling vector width quarters scalar work and halves vector work outside the array; 4→128 = 1024× scalar, 32× vector. Why a 30-person TPU team kept up with Nvidia. |
| 4 | Forces limiting the size, L248, p.11 | The model's reduction dimension K bounds the array; attention K is 16–128, FF K is d_model. Half-width models get 25 %/50 % utilisation. |
| 5 | Numerics, L306, p.14 | FP32→FP16→BF16→FP8→FP4 history; int summation for FP4; proposes an Int8-only or Int7 array; 5 bits "usually enough". |
| 6 | Structured sparsity, L362, p.18 | 2:4 on Nvidia; proposes **Int7+1 = 1:2 sparse** format, dense at half speed; a research programme to make 1:2 work; training-time sparsity needs transposable masks. |
| 7 | Mono-sized arrays are unbalanced, L406, p.21 | Attention wants small K, FF wants large K → **heterogeneous cores on one chip** (one huge FF core, several small attention cores), same RTL parametrised on VECTOR_WIDTH; parallel attn+FF as the model-side fix. |
| 8 | Chips with larger arrays are easier to make, L456, p.24 | The bigger the array, the less the surrounding CPU matters → the future is "AI CPUs". |
| 9 | Non-square systolic arrays, L488, p.26 | The N/K/M dimensions; **batch floor N = M × (LHS bw / RHS bw)**; low-batch chips need M× the bandwidth; LPU discussion; large K for FF limits TP degree. |
| 10 | Tyranny of the cycles / systolic-array DMAs, L588, p.31 | The driving core must feed the array every cycle; 2-D registers, queues, slow clocks, and a proposed DMA that streams thousands of rows straight into the array (also from the network). |
| 11 | Anecdote: arrays are hot, L626, p.33 | TPUv3 heat, compiler throttling = "Turbo Mode" (+25 % clock). |
| 12 | Compression, L664, p.36 | Huffman tables in DMAs at memory/network speed; same tables do lossy formats (Int8→FP4); ~20 % lossless gain cited; variability handled by dropping low-priority tokens. |
| 13 | Kinds of AI workload, L702, p.38 | Training/prefill compute-bound if done right; decode is the hard one; RAG will make prefill dominate. |
| 14 | Training, L728, p.39 | Don't build a training chip first; what training needs (transpose, more vector, 16-bit, ops, flexibility). |
| 15 | Decode, L775, p.42 | Naive decode <1 % array utilisation; GQA (G=8) × speculation (4) = 64×; agent multiplication; batch=1 still has a niche. |
| 16 | Prefill, L854, p.47 | Plenty of queries; GQA helps K. |
| 17 | Managed aggregation vs disaggregation, L869, p.48 | Coins "managed aggregation" (≈ chunked prefill): mix prefill:decode per chip in a controlled ratio; pricing / filler work / training to reach balance; a chip designed for it can ship less memory bandwidth. |
| 18 | Expensive HBM, L895, p.50 | KV-cache formula 2·B·L·N·G·W·S·M·P → 4 194 GB at N=1 M; weights are cheap to distribute, KV is not (pipelining doesn't help KV). |
| 19 | Reducing the need for large memories, L969, p.54 | Dense attention on the last ~1 024 tokens + retrieval for the rest → 1 000× less; a "32-bit chip with 4 GB"; 1 TB model needs ≥256 such chips; 16/32 GB as milder options; HBM is bloat protected by bloat. |
| 20 | Parallelization of an AI assistant, L1028, p.59 | Three surprising facts: network bw limits within-layer parallelism; 2× bw → up to ½ latency; pipelining shrinks weights/chip but not KV/chip. |
| 21 | Parallelization concepts, L1056, p.60 | Throughput vs token latency vs first-token latency; minimum installation; failure domain. |
| 22 | Duplication, L1104, p.62 | D copies: throughput ×D, nothing else changes. |
| 23 | Between-layer pipelining (decode), L1124, p.63 | Partial vs **full pipelining** (2× stages: latency ×2, KV ×2, network off the critical path). |
| 24 | Between-layer pipelining (prefill), L1205, p.67 | Same, but KV/chip does divide by S. |
| 25 | Within-layer parallelism, L1234, p.68 | Attention 8 (GQA) × 4–8 (KV split); FF up to 16k·4/256 = 256 chips; all-reduce = 2X(1−1/P); **per-chip bw ∝ P**; between-layer bw neutral; 16-bit sums on the wire. |
| 26 | Mixture of Experts, L1293, p.72 | Load imbalance; replica rounding (120 % → 60 % util, 1 020 % → 93 %); priority queues; prediction that within-layer retrieval will shrink MoE to 4–8 experts. |
| 27 | Network topologies for a MoE world, L1337, p.75 | Router vs torus; his proposal: cheap slow routers over a fast slim torus. |
| 28 | Networking concepts, L1354, p.76 | Node/link/latency/bandwidth pitfalls (bits/bytes, bidirectional, GiB, goodput, realised). |
| 29 | Link types, L1505, p.83 | Cost ladder from on-die to optical; copper traces on snap-together boards. |
| 30 | Multi-hop waste, L1587, p.86 | useful bw = link_util × node_bw / avg_hops; a router is ≤50 % efficient. |
| 31 | The big 3 collectives, L1654, p.89 | All-gather / reduce-scatter / all-reduce at 100 % on tori, incl. **the asymmetric-bandwidth torus algorithm** (pipelined, slow dimension first). |
| 32 | MoE vs topology, L1733, p.92 | Random-destination hop counts: 4-ring 1, router 1.5, 4×4 2, 4×4×4 3, v5p 16×16×24 = 14. |
| 33 | A proposal for a networking approach, L1777, p.93 | **Ring board (4 chips) → donut (4×4=16) → box (N×4×4, N=16 → 256 chips)** with 16 cheap routers per box; failure domain = donut; nerd box: **asymmetric N×4×4×4 with ¼ bw per dimension → routers at 1/64, only if every expert is 64-way TP**; no host, tool-use CPUs elsewhere; custom 16-port router alternative. |
| 34 | Co-design, L1871, p.100 | Rules and stories; "derived consequences"; numbers are meaningless without meaning; combinatorial search instead of hill-climbing a plan of record. |
| 35 | A critical look at GPU features, L2048, p.112 | Kernels/warp specialisation (→ megakernels = CPUs), path divergence, software-managed cache (keep), warp scheduling (= hardware software-pipelining, expensive), host (drop), thread startup overhead. |
| 36 | Software pipelining, L2140, p.119 | What it is, why by hand is hard, **write a tiled software-pipelining library with op fusion in C++** (what XLA:TPU does internally), power-of-2 shapes, XLA + C++ + LLVM as the minimum interface, AI-written kernels. |
| 37 | Hiring, L2400, p.132 | Sourcing, interviewing, talking to engineers. |
| 38 | Objections and answers, L2492, p.142 | OS interrupts; who owns a CPU; "the science is missing"; generality; tokens/$ isn't everything; analog; **"Why more FLOPs?" (Zhao, cites the two arXiv papers)**; "structured sparsity sucks" (Kübler, 9 counter-ideas); depthwise conv; routers vs tori (→ chapter 27); existing AI CPUs (SME/AMX/QMX are 32×32 or smaller); fixed-function (Etched/Taalas). |

---

## 2. The proposed hardware ("AI CPU")

### 2.1 Compute organisation (A unless marked)
- A conventional CPU (x86 if you are Intel/AMD, else licensed/own, VLIW considered — L2514) with wide vector registers; **one systolic array attached to each compute core**; extra scalar cores without arrays run Linux, the input pipeline, network algorithms, retrieval and the OS (L122, L125, L625, L2499). The chip is the computer: no host (L2103).
- **Heterogeneous cores**: one FF core with a very large array (K e.g. 256, possibly K=1024 non-square) and several attention cores with small arrays (K e.g. 64); same RTL and binary, parametrised on VECTOR_WIDTH; suggested default split 50/50 by OPs (L416–L425, L558).
- Systolic array does A·Bᵀ natively so inference needs no transpose unit (L750–L753, TPU precedent — M).
- **Systolic-array DMAs**: a DMA that streams thousands of rows from SRAM/HBM/another array/the network straight into the array, so the core need not service it every cycle (L619).
- **Batch floor rule**: an M-wide weight tile needs N = M × (LHS bw / RHS bw) activation rows to reach 100 % — 128 rows for a 128×128 array at 1:1 bandwidth; N=1 gives ~1 % (L515–L524). Low-batch (LPU-style) chips need M× the bandwidth (L536).
- Utilisation when the model's reduction dim is half the array: 25 % naive, 50 % with the diagonal-block trick (L265–L275).

### 2.2 Memory hierarchy
- Large **software-managed SRAM cache (MBs)** that can double as an automatic cache when unused (L123, L2085).
- **Little HBM: "4 GB should do it"** — pushed "to its limit" for effect; 16 GB or 32 GB per chip are the milder options (L979, L982). HBM is not sold in 4 GB stacks; consider GDDR/other RAM because HBM's interposer cost makes it not the best GB/s per dollar (L979). No absolute β_C is given.
- **An SSD per chip** (on the ring board) holding cold KV, reachable only via sparse/indexed attention (L127, L973).
- KV plan: dense attention over the last ~1 024 tokens, retrieval beyond → N_eff 1 M → 1 k, 4 194 GB → 4 GB before TP splitting (L973).
- Idle conversations: prefer faster SSDs to holding idle KV in HBM (L947).

### 2.3 Fabric
- Base proposal (chapter 33): **ring board** = 4 chips + memory/SSD on one motherboard (fast passive copper traces); **donut** = 4 ring boards folded into a 4×4 torus with edge-to-edge connectors; **box** = N donuts joined by 16 cheap off-the-shelf routers, router i connecting position i of every donut (this *is* the note's "64 parallel clos planes", at 16). N=16 → 256 chips. Routers are "lower bandwidth", the torus is high bandwidth; the N dimension is for between-layer parallelism, weight loading and the DC network (L1782–L1800).
- **Asymmetric variant** (nerd box L1840–L1843): on-board dimension at bandwidth B, between-board ¼B, short copper cables 1/16 B, routers 1/64 B → an N×4×4×4 topology whose routers "only need to run at 1/64th the speed". He *cannot recommend it for MoE* — unless "each expert in MoE will be large enough to parallelize 64 ways", in which case the torus carries only within-expert TP traffic and MoE sees a pure routed topology. This is exactly the note's P4+P5, and the wiring order he gives (routers/long cables → short copper → board edge → on-board) is the 1 : 4 : 16 : 64 of the author's Q8 answer.
- Router efficiency: a single router is at most 50 % efficient (avg hops 2); multi-level routers worse (L1615–L1641). Torus neighbour-only collectives are 100 % (L1655). Mesh = half speed of torus (L1729).
- Random-destination hop counts (L1740–L1773): 4-ring 1.0; router 1.5; 4×4 2; 4×4×4 3; v5p 16×16×24 pod 14 (8 if the 24-dim is TP).
- Failure model: donut is the failure domain; a failed router is bypassed through the 4 torus neighbours at ¼ load each; the same path gives up to 16× (64× for N×4×4×4) burst bandwidth for weight loading (L1794, L1803).
- Link latency matters little if software-pipelined (TPUv3 link latency "100s of cycles" was fine — M, L1384). Bandwidth matters.
- **Partial sums on the wire need ~16 bits**, tori carry wider partial sums than routers; 32-bit sums double network cost (L1292).

### 2.4 Numerics and sparsity
- Arrays multiply in **7-bit integer only**, sums wider and saturating; **1:2 structured sparsity for the RHS (weights)** in an 8-bit "Int7+1" container (1 position bit + 7 magnitude bits); dense mode at half speed (L128–L129, L381–L390). Int6+2 / 2:4 is the fallback; 1:2 converts losslessly to 2:4 (L396).
- FP8 always suffices for inference if quantisation is done right; Int8-only chip is a "real contender"; FP4 sometimes; 5 bits "usually enough" (L355–L358, L384).
- Structured sparsity does not help activation×activation matmuls (attention) nor KV size (L393).
- Recipe claim (A): train a 50 % bigger model and sparsify it → faster *and* more accurate (L375). Contested by Kübler's objection; author lists 9 further techniques incl. lottery-ticket retraining (L2648–L2656) and cites MLPerf and Tencent as 2:4 users (C, L2603–L2612).
- Huffman (de)compression in DMAs at memory/link speed, applied to weights *and* activations/KV; cited paper claims ~20 % lossless (C, L674); encode table 2^8, decode table 2^10 entries (L683).

### 2.5 Software (see §5)
XLA backend; C++ with intrinsics; LLVM; tiled software-pipelining + op-fusion library; megakernel-style single binary; DMAs SRAM→network→SRAM bypassing HBM.

### 2.6 Power / cost claims (all A, none quantified in $)
- HBM ≈ 3× regular RAM per GB (estimate, L898); HBM 20 % (his guess) to 50 % (others' estimates) of accelerator cost (L991).
- Optical links are the most expensive; copper traces the cheapest; routers "expensive bandwidth" (L1509–L1586).
- Correct metrics: tokens per dollar of comparable quality, then token latency; networking $ and HBM GB must be normalised per unit of realised performance (L1000, L1870).
- ">2× left to gain" in hardware+software without new lithography (L2567).

### 2.7 Every concrete number stated (scenario-input candidates)

| Quantity | Value | Where (txt line) | Kind |
|---|---|---|---|
| Systolic array sizes | Volta 4×4, Turing 16×16, Blackwell 64×64; TPU 128×128 / 256×256; TPUv3 128×128 | L211, L410, L2695 | M/C |
| TPU vector register | 8×128 | L229, L518, L610 | M |
| Scalar-work reduction 4→128 wide | 4^5 = 1024× (8192× with 8-row regs) | L226–L229 | A (derivation) |
| Vector-work reduction 4→128 wide | 2^5 = 32× (upper bound) | L235 | A |
| Utilisation, model width = ½ array | 25 % naive / 50 % with trick | L265–L275 | A |
| Attention K | 16–128 elements | L407 | A |
| FF K (d_model) | e.g. 8 192; FF feature dims >16 000 | L407, L561 | A |
| Attention-core K / FF-core K | e.g. 64 / 256; FF K up to 1 024 | L558, L561 | A |
| Max TP degree with array K | model K / array K, e.g. 16 384/1 024 = 16 | L561 | A |
| FF TP degree at 100 % | 16k·4/256 = 256 chips | L1247 | A |
| Attention TP degree | 8 (GQA groups) × 4–8 (KV split) ≈ 64 | L1241–L1244 | A |
| Batch floor | N = M × LHS_bw/RHS_bw; N=1 → 1/128 | L515–L524 | A |
| LPU3 SRAM | 500 MB; 100 B FP8 model ≥ 200 chips | L539 | C |
| FP16 vs FP32 array | < ½ area & power | L316 | A |
| FP8 / FP4 throughput | 2× BF16 / 2× FP8 (Nvidia) | L328, L334 | C |
| MXFP4 | 4.25 bits/number | L343 | C |
| 2:4 metadata | 3 bits per 4 entries | L369 | C |
| Proposed weight format | Int7+1, 1:2, 8 bits/element, dense half speed | L381–L387 | A |
| "enough" bits | 5 usually; 7 definitely; 8 more than enough | L346, L384 | A |
| Huffman lossless gain | ~20 % (one paper) | L674 | C |
| Nvidia decompression | ~1/10 memory speed | L671 | C |
| TurboQuant | "~8×" = 32→4 bit | L701 | C (debunked) |
| Turbo Mode | +25 % TPUv3 clock | L639 | M |
| Naive decode utilisation | < 1 % | L789 | A |
| GQA + speculation gain | 128/G with G=8 → 16; ×4 spec → 64×; 256 needed for a 256 array | L802–L811 | A |
| Agent multiplication | ×4 → batch 16 per user | L835–L841 | A |
| Speculation width | 4 (not 16) | L805, L817 | A |
| Prefill vs decode tokens per step | e.g. 256 vs 1–4 | L870 | A |
| KV formula | 2·B·L·N·G·W·S·M·P | L913 | A |
| KV example | B=64, L=32, N=1e6, G=8, W=128, S=1 → 4 194 GB | L959 | A |
| Full pipelining factor P | up to 4× KV | L953 | A |
| Current HBM per chip | 100–200 GB; Blackwell Ultra 288 GB HBM3e | L898, L965 | C |
| KV per chip at 32-way TP | 4 194/32 = 131 GB | L965 | A |
| Weights per chip | 1 000 GB / (32×32 = 1 024 chips) = 1 GB | L968 | A |
| Dense window | ~1 024 tokens (N 1e6 → 1e3, 1 000×) | L973 | A |
| Proposed memory per chip | 4 GB (32-bit addressing); 16 GB / 32 GB alternatives | L979, L982 | A |
| Min installation, 1 TB model, 4 GB chips | ≥ 256 chips | L982 | A |
| HBM cost share | 20 % (guess) … 50 % (estimates) | L991 | A/C |
| HBM vs DRAM $/GB | ~3× | L898 | A |
| Token latency "fast enough" | 100 vs 50 tok/s both above reading speed | L1177 | A |
| All-reduce bytes per chip | 2X(1 − 1/P) | L1253 | A |
| Within-layer bw scaling | per-chip bw ∝ P | L1259 | A |
| Pipeline every k-th layer | bw ÷ k (example k=4) | L1183 | A |
| Training activation storage | 2(L−1) instances per first stage | L1233 | A |
| Full-pipelining example | 25 % net overhead → 4× more TP, latency ½ | L1277–L1280 | A |
| Network sum precision | 8 not enough, 16 probably OK, 32 too many (=2× bw) | L1292 | A |
| Mixtral / Llama 4 Maverick / Switch-C | 8 experts (25 % active) / 128 experts, 17 B of 400 B (4 %) / 2 048 experts, 1.6 T | L1303 | C |
| Replica rounding | 120 % load → 2 replicas at 60 %; 1 020 % → 11 at 93 % | L1315–L1318 | A |
| Future MoE | 4 or 8 experts + within-layer retrieval | L1330 | A (prediction) |
| Dojo link latency / TPUv3 | 1 cycle / 100s of cycles | L1384 | C/M |
| TPU pod / cable failure | 1 024 chips; fault tolerance cost ½ torus bw | L1100 | M |
| Router efficiency | ≤ 50 % (avg hops 2) | L1615 | A |
| Random-pair hop counts | 4-ring 1; router 1.5; 4×4 2; 4×4×4 3; 16×16×24 14 (or 8) | L1740–L1773 | A |
| Proposal sizes | ring 4, donut 16, box N×16 (N=16 → 256 chips), 16 routers/box, 4 ports/board | L1782–L1834 | A |
| Router-failure bypass | 4 neighbours × ¼; burst 16× (64× for N×4×4×4); 4× for a 4-ring tenant | L1803–L1806 | A |
| Asymmetric torus | ¼ bw per dimension; routers 1/64 | L1840 | A |
| Custom router alt. | 16-port router chips, 2 cm board pitch → 32 cm backplane, 4 routers → 16×4, amortised over 64 chips | L1867 | A |
| Google TPU spend | ~$100 B (Gemini estimate); 1 % = $1 B | L1350 | C |
| TPUv2 team | ~30 people, XLA 5 | L217, L2335 | M |
| Int7 SME/AMX/QMX arrays | 32×32 or smaller | L2695 | C |
| Fixed-function FF gains | "1000×" only at batch 1 and 16/32-bit | L2725 | A |

---

## 3. Mapping to the 19 points

**P1 Systolic arrays, not LPUs.** Strongly reinforced and *explained*: chapters 2–4 and 9 give the mechanism (larger arrays shrink everything else; LPUs are "not really a systolic array" because N≈1 needs M× bandwidth; Jensen at GTC 2026 positions LPUs for premium tokens — C). What the doc *adds* that the note lacks: (i) the array is bounded by the model's reduction dimension, so attention and FF want different arrays → heterogeneous cores; (ii) the **batch floor N = M·(bw ratio)** is a tile-geometry floor separate from the roofline floor; (iii) a clocked-down larger array is still more efficient, so "more FLOPs" and "bigger arrays" are independent (L2579). Silent on how big *U* should be from the array's point of view except via the TP-degree limit model-K/array-K (L561).

**P2 Parallel attention + FF.** Added twice (L428, L1268): halves token latency *and* halves the number of pipeline stages (hence KV in flight); lets attention (memory-heavy) overlap FF (compute-heavy) so the chip needs less memory bandwidth; cost "a slight degradation in converged model accuracy" (A, no number). Also the reason his heterogeneous FF/attention cores can both stay busy.

**P3 2:4 from the start; FP4.** The doc prefers **1:2 Int7+1** over 2:4 and FP4, wants dense at half speed, and lists the recipe space (column matching, outlier columns/rows dense, gradual L1, lottery-ticket retraining "most effective", train-time dropout of the smaller element, FF-only). The note's "train dense for 1 % then lottery" is item 5 here. Training-time acceleration needs transposable masks and hardware Nvidia does not have (L405) — so on our rig L10a is quality-only, as CLAUDE.md already says. Contradiction inside the doc: Kübler's objection that 2:4 trade-offs are "pretty bad" — the author answers with ideas, not results.

**P4 Expert = smallest FF that fills U chips.** The doc supplies the *why* for a big unit in the networking chapter: 64-way within-expert TP is the one case that makes an asymmetric N×4×4×4 torus work with MoE (L1843). It also supplies the *sizing constraint* that our `docs/02 §2` lacks: with an FF-core array of K=256 (or 1 024), the row-parallel down-projection slice d_ff/U must be ≥ K, and the batch must supply ≥ M rows per weight tile → d_ff ≥ 64·256 = 16 384 (65 536 at K=1 024). At d=8 192 that is 0.4–1.6 B params per expert, above the 0.2–0.8 B in `docs/00 §P4`. Contradiction with the note's spirit: the doc predicts within-layer retrieval will leave "only a few experts, like 4 or 8" (L1330), whereas the note wants "huge amounts of experts".

**P5 N × 64 fabric.** Confirmed in detail: rail planes = one cheap router per torus position; the torus is where the bandwidth is; the N dimension is "much lower bandwidth" and doubles as weight-loading and DC network; 1 : 4 : 16 : 64 per level (Q8) matches the wiring order routers → short cables → board edge → on-board. Two additions: the *base* recommendation is an **even-bandwidth 4×4 donut (U=16)** — the 64-wide asymmetric unit is the nerd-box variant conditioned on P4; and a failed router degrades (bypass via torus), while a failed chip takes out the donut. The doc also says N=16 (256 chips) "should be enough for most inference use cases", and that going beyond one router level raises hops above 2 and needs custom algorithms — the note's DC-scale N implies the multi-tier clos in `docs/02 §3`.

**P6 Few early/final layers, one repeated middle layer.** Silent. Nothing on weight sharing or recurrence.

**P7 Variable iteration count; RL; local vs global attention.** Silent on adaptive depth. Adjacent material: "agent multiplication" (parallel chains of thought as a batch multiplier, L835) and "vary the number of speculated tokens depending on confidence" (L817).

**P8 Experts with ≥ 2 internal layers.** Same idea in different clothes: "pipeline only every 4th layer … drops bandwidth requirements by a factor of 4" (L1183) and "do several layers on the same set of chips" (L1262). Supports H8's fabric claim; silent on quality.

**P9 Multicast + co-activation placement.** No multicast or in-network reduction anywhere. Partial support: when a token goes to several experts, MoE traffic "may start to look like all-gather traffic" and dedup favours the torus (L1764); and document-level relocation — move a whole conversation (with its KV) to the region whose experts it prefers (L1755). The up/down asymmetry claim is absent. The doc's replica-rounding model (L1315) is the *placement* side of P9/P10.

**P10 One combined installation.** Supported with a formula: load ℓ (in units of one expert's capacity) → ⌈ℓ⌉ replicas at ℓ/⌈ℓ⌉ utilisation, so utilisation "approaches 100 % … at high multiples" but never reaches it without queues (L1318–L1321); priority queues (live vs offline tokens) close the gap. He also states, as industry chatter (not M), that deployed MoE utilisation is "MUCH lower" than 100 %. This gives S4 a closed-form baseline to compare the Little's-law model against.

**P11 Continuous, non-lockstep.** Supported by the queue discussion (L1315–L1324): experts fire on high-priority tokens immediately and fill with queued low-priority tokens; waiting costs token latency and KV; low-priority tokens should have short attention. Also "drop some low priority tokens and rerun them later" to absorb variability (L677). The doc says nothing about lockstep vs continuous at the µs level; the software chapter's megakernel/OS-on-chip argument is what makes an actor-style runtime natural.

**P12 Two latent streams.** Silent.

**P13 Multi-token prediction from the start.** Speculation with 4 tokens is a central multiplier (×4 on both bandwidth and utilisation, L805–L811) and allows B=64 instead of 256 for a 256-wide array (L935). The doc assumes a separate small draft model; the note's MTP heads are a different mechanism with the same systems effect. New for us: **the per-expert batch threshold is in rows, and each token contributes (1 + speculated) rows**.

**P14 Managed aggregation.** Chapter 17 is the note's P14 written out: term coined by the author, "very similar or even the same" as chunked prefill; balance via pricing, self-generated filler work, or mixing in training; hybrid overflow pools; a chip built for it can ship less memory bandwidth. Objection section (L2527) concedes the market may go the other way. Consistent with `docs/02 §6`; adds the "minimum required ratio of prefill to decode" as a design output.

**P15 Indexed attention, cold KV on SSD.** Reasserted (L293, L823, L973): "all attention should be sparse (indexed)", may allow "even SSD bandwidth to be enough for decode (!!!)", with a dense window of ~1 024 tokens. Still no design, no numbers, no report — "you'll have to do some AI research" (L138). The Groq box (L141) makes explicit that indexed attention is what *lets* the design keep large batch without SRAM. Adds the per-chip SSD as the tier location. Our `docs/00 §P15` objection (thousands of queries per token) is not addressed.

**P16 ~4 GB HBM per chip.** Full argument is chapter 18–19: KV, not weights, is why HBM is large; pipelining does not shrink KV/chip but within-layer TP does; 4 GB is "the idea to its limit", 16–32 GB are reasonable; minimum installation ≥ 256 chips for a 1 TB model; HBM may not even be the best GB/s per dollar once interposers are counted; "bloat protects bloat". The "global run on DRAM" line appears here too (L976) as opinion. Adds one thing our spec lacks: **full pipelining inside the unit doubles KV in flight** (L1171), which the note's "2-step pipeline" (P5) implies.

**P17 Per-iteration retrieval.** Reinforced as a prediction (L1330): move knowledge out of weights into a within-layer vector database (explicitly "not the same as RAG"); expects it to shrink MoE. The doc's on-chip scalar cores + local SSD are where retrieval would run (L625, L1855).

**P18 Per-vector QLoRAs; bespoke experts.** One sentence: "much more retrieval of small amounts of weights, like QLoRAs" (L1330). Multi-tenancy of a donut (a customer on a 4-ring, burst bandwidth borrowed from neighbours, L1806) is the only tenancy discussion.

**P19 One giant DC.** Silent on geography and on the 50–100 ms argument. Relevant: minimum installation size is acceptable because "prominent AI assistants require a lot of throughput anyway" (L982); tool-use CPUs belong in the same DC, not in a host (L1855).

---

## 4. Disagreements and surprises (vs `docs/02 §1–§2` and `docs/00`)

1. **The author's own default unit is 16 chips, not 64.** The recommended topology is an even-bandwidth 4×4 donut with cheap routers; the 64-chip asymmetric unit exists only in a nerd box and only "if you could guarantee that each expert … will parallelize 64 ways". `docs/02 §1` treats "switch" and the six torus variants as interchangeable scenario options; the doc ranks them: torus ≫ router for cost, and a router-based unit fabric is ≤ 50 % link-efficient. `note64.yaml` should carry U ∈ {16, 64} with the 16-chip even torus as the author's baseline and the 64-chip 1:4:16:64 torus as the note's.
2. **Heterogeneous cores → two φ per chip.** `docs/02 §1` has one φ. The doc's chip has an FF core (huge array) and attention cores (small arrays); the attention owner and the expert unit of `docs/02 §5` run on different silicon with different peak rates and different tile floors (K_attn ≈ 64, K_ff ≈ 256–1 024). Suggest `phi_ff`, `phi_attn`, `s_min_ff`, `s_min_attn`.
3. **Tile floor is larger than we assumed and is a *row* floor.** `docs/02 §2` uses s_min ∈ [128, 512] from GPU experience (measured 256 on the 3060). The doc's FF core implies s_min = K_ff = 256–1 024 on the reduction side *and* N = M rows per weight tile on the batch side. With 4-way speculation each token is 4 rows, so b_min in tokens = 256/4 = 64 — the note's queues should count rows, not tokens. The roofline b_min = (φ/β_C)(b_w/2) in `docs/02 §2` and the tile floor are two separate constraints; the empirical 512 floor on the 3060 (I13) is a third.
4. **Local-fabric traffic must carry 16-bit partial sums.** `docs/02 §2` charges 2·b·d·b_act for broadcast + reduce. The doc says the reduce leg needs ~16 bits regardless of the multiply precision, and tori carry wider partial sums than routers. Corrected: b·d·b_act (gather) + b·d·b_sum (reduce), b_sum ≥ 2 bytes.
5. **Asymmetric-torus collective cost is derivable.** The doc gives the algorithm (pipelined, slowest dimension first, each stage moves 4× the data at 4× the bandwidth, L1694–L1707). Our derivation (not the author's): an all-gather of X over a 4×4×4 torus with per-dimension link bandwidths B/16, B/4, B (two links per dimension per chip) takes ≈ 3X/(8B) per stage, all three stages balanced and overlapped, versus ≈ X/(2B) on a 64-ring at B everywhere. So the 1:4:16:64 unit is ~25 % faster than a flat ring while provisioning far fewer fast links. That is the number S1/S3 should reproduce.
6. **Full pipelining doubles tokens in flight in the unit.** The note's "2-step pipeline" in P5 = the doc's full pipelining: network off the critical path at the cost of 2× latency and 2× in-flight state (L1171). `docs/02 §5.5` Little's-law budget Z should include this factor explicitly.
7. **Replica-rounding utilisation model.** `docs/02 §5.6` derives pool utilisation from Z_min ≈ N_e b_min/k. The doc adds a static-placement model (⌈ℓ⌉ replicas at ℓ/⌈ℓ⌉) that gives a utilisation floor without queues; S4 should report both.
8. **Router-failure degradation.** `docs/02 §11` fails the unit on any internal failure. The doc: a failed *router* (rail port) is bypassed through torus neighbours at ¼ load; a failed *chip* kills its ring board (4 chips, soldered) and therefore the donut. For U=64 the natural failure domain is the ring board, not the chip.
9. **Numbers are deliberately absent.** No β_C, λ_C, ν_C, φ, $/chip, W/chip, or link latency is given; the author says so (L1975). I4 cannot be closed from this document for absolute values; only ratios and structural constants (table in §2.7) are sourced.
10. **No multicast / in-network reduction** (P9) and no attention-owner concept: in the doc, attention is parallelised by GQA group and KV split across the torus, not owned by one unit. `docs/02 §5.1`'s "attention owner" is our construction.
11. **MoE shrinks in the author's future.** The doc predicts retrieval will reduce MoE to 4–8 experts (L1330); the note designs for thousands. Both are the same author; the note is later. Record in `docs/04` as an open question rather than resolve.
12. **Managed aggregation extended to training** (L885): mixing training into an inference pool to reach balance — outside our scope (`docs/02 §13` keeps training lockstep) but relevant to P14's balance argument.
13. **Doc's KV-cache formula uses per-layer caches** (L=32); the note's v2 P7 collapses global caches to final vectors. The doc's 4 194 GB example is the baseline the note claims to reduce ~128×; use it as the S6/S7 reference point.
14. **Sparsity in training is not accelerated** (L405) — confirms our quality-only treatment; and the author's own preferred format (1:2, Int7) does not exist on any hardware we can measure.

---

## 5. Software claims (vs `docs/02 §5` scheduler)

- **No host; the accelerator runs Linux** on scalar cores; OS interrupts stay off the array-bearing cores (L125, L2499). Implication for us: the continuous scheduler of `docs/02 §5` is an on-device program, not a host-side orchestrator; per-expert queues live in on-chip SRAM.
- **Megakernel / single-binary execution** (L2069–L2081): network progress (all-reduce steps) runs as threads beside compute; this is the runtime shape P11 needs and the doc's argument for why GPUs approximate it badly.
- **DMAs SRAM→network→SRAM bypassing HBM** (L2102) so collectives cost no memory bandwidth — assume λ_C traffic is free of β_C in the unit model.
- **Systolic-array DMAs** stream weights/KV straight into the array; op fusion into GEMM may defeat this (L619, L2517).
- **Software pipelining as a tiled C++ library with declared stages and tile coordinates**, op fusion by composition, communication queues as pipeline operands (L2305–L2332) — what XLA:TPU does internally (M). Not directly a scheduler concern, but the 3-stage load/compute/store pattern is the per-expert-step microstructure our cost model assumes when it treats weight loading as hidden.
- **Priority classes**: live tokens go first, offline tokens fill queues; low-priority tokens should have small attention windows so waiting is cheap in KV; drop-and-rerun for variability (L677, L1324). Extends `docs/02 §5.1`'s (b_min, w_max) policy with a class-dependent w_max and a KV-aware admission rule.
- **Forced even routing at inference degrades answers randomly** (L1336) — an argument against capacity-factor dropping; matches our dropless choice (`docs/02 §13`).
- **Document-level placement**: move a conversation (and its KV) to where its preferred experts live (L1755). A coarser, cheaper version of the P9 co-activation controller; candidate policy for S3.
- **Managed aggregation = chunked prefill with a target ratio**, plus filler work and pricing as balance levers (L879–L891). Same as `docs/02 §6`; adds the "minimum prefill:decode ratio to stay compute-bound" as a reported quantity.
- **Weight loading and burst bandwidth** through idle router links of neighbours (L1803–L1806) — relevant to `docs/02 §4` migration cost (3·d·d_ff·L_e·b_w bytes per move) which can use the burst path.
- **Power-of-2 shapes only** (L2314) — consistent with our measured odd-width penalty (`local_3060.yaml`).
- **Combinatorial search over designs** with sensitivity analysis rather than hill-climbing a plan of record (L2038–L2044) — an argument for making `sim/` sweep U, m_C, fabric variant and b_min jointly (S3/S7/S9) rather than one at a time.
- **Nothing on MoE dispatch mechanics** (all-to-all, token permutation, capacity) beyond "send to the expert and back, optionally pipelined" (L1300).

---

## 6. Numbers we can use

`sim/scenarios/note64.yaml` does not exist yet (only `local_3060.yaml` and `README.md`); the placeholders are the ones named in `docs/02 §1` (φ, β_C, λ_C, ν_C, m_C = 4 GB, U = 64, unit-fabric variant).

**Replace / fill from the document (each with a source line):**
- `unit.U`: 16 (author's baseline donut, L1782) and 64 (note; asymmetric nerd box, L1840–L1843). Sweep both.
- `unit.fabric`: `asym_torus_4x4x4` with per-dimension link bandwidths λ_C × {1, 1/4, 1/16} and rail ν_C = λ_C/64 (L1840; Q8). Two links per chip per dimension. Also `even_torus_4x4` for U=16.
- `fabric.router_efficiency`: ≤ 0.5 per router level (L1615); `avg_hops`: 2 per single router, > 2 for multi-level (L1641, L1758).
- `chip.s_min_ff`: 256 (FF-core K; L558, L1247), variant 1 024 (L561); `chip.s_min_attn`: 64 (L558).
- `chip.rows_per_tile`: N = M × (LHS bw/RHS bw), default 256 at 1:1 (L515–L524); `tokens_per_row`: 1 + m (speculation width 4, L805) → b_min_tokens = 64 for M = 256 (L935).
- `chip.b_sum_bytes`: 2 (16-bit partial sums on the fabric, L1292); keep b_act = 1.
- `chip.weight_format`: Int7+1 1:2 (8 bits/element, 2× array throughput vs dense, L381–L387) — quality-only for us; fallback 2:4 with 3/4 bit metadata (L369).
- `chip.m_c_bytes`: 4e9 primary; sweep 16e9, 32e9 (L979, L982).
- `tiers.cold`: per-chip SSD (L127); dense window W = 1 024 tokens (L973).
- `unit.full_pipelining`: true → tokens-in-flight ×2, latency ×2 (L1171).
- `moe.replica_rounding`: utilisation = ℓ/⌈ℓ⌉ (L1315–L1318).
- `kv.reference`: 2·B·L·N·G·W·S·M·P with B=64, L=32, N=1e6, G=8, W=128, S=1 → 4 194 GB (L959) as the S6/S7 baseline.
- `failure.domain`: ring board of 4 chips; router failure = bypass at ¼ load through 4 neighbours (L1794–L1803).

**Still placeholders after this document (I4 stays open):** absolute φ (FLOP/s per core type), β_C (GB/s per chip, memory type unspecified — "ganging up other kinds of RAM"), λ_C (on-board copper-trace bandwidth), ν_C (router port bandwidth), ℓ_hop, chip and network $ and W. The author explicitly declines to give these (L1975). Candidates for sourcing: TPU v5p/v6 torus link bandwidth and Blackwell NVLink for `gpu_today`; PCIe/CXL or GDDR7 vendor figures for a copper-trace guess.

**What the author's design predicts differently from our `local_3060` measurements:**
- `s_min` 256 (measured) → 256–1 024 on the FF core: the same or larger, so experts get *bigger*, not smaller, than `docs/00 §P4` assumed.
- `b_floor_empirical` 512 vs roofline 79 (I13): the doc's tile-row floor N = M gives 256 rows for a 256-wide array; with 4-row tokens that is 64 tokens — the doc predicts the *row* floor, not the token floor, is intrinsic, and that the 7× gap we see is occupancy on a small GPU, not the roofline.
- `sparsity_2_4.speedup_best` 1.58 (measured) vs 2× asserted for a systolic array with a "very modest increase" in area (L369). The doc's 2× is per-array throughput; our measurement includes everything around the array — consistent with the doc's own thesis that GPUs are inefficient around the array.
- `lambda_c` 3.59 GB/s host-bounced vs the doc's on-board copper traces: the doc argues per-chip fabric bandwidth must scale ∝ U (L1259); on our rig U = 1 so nothing local can be checked; S10 remains the only local fabric measurement.
- `grouped_gemm` cliff below 128 tokens/expert vs the doc's replica-rounding and priority-queue model: the doc predicts that without queues utilisation is bounded by ℓ/⌈ℓ⌉, i.e. the cliff is a placement effect, not a kernel effect.
- Cold tier: our NVMe 46–70 µs random latency and the 64 KiB / QD 16 knee are exactly the constants the doc waves past with "even SSD bandwidth" — the doc gives no hit-rate model, so S6's locality model remains ours.

---

## 7. The two arXiv papers cited under the LinkedIn post

**arXiv:2509.09505 — "Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference" (PLENA), H. Wu et al., v1 Sep 2025, v3 Apr 2026.** A hardware–software co-design for long-context agentic inference: a systolic-array-based accelerator with asymmetric quantisation support and native FlashAttention, plus a custom ISA, compiler, transaction-level simulator and an automated design-space-exploration flow; reports higher throughput and power efficiency than existing accelerators on LLaMA inference. The author cites it (as "Wu et al.", L570, L2594) for the idea of enlarging only the K dimension of the array. Relevance: **H1** (systolic economics at long context), **H3** (asymmetric/low-bit quantisation results), and methodologically **S-tier work** — their transaction-level simulator + DSE is the closest published analogue of `sim/` and of the "combinatorial search" the author recommends; worth reading for how they validate a simulator (I1, I4).

**arXiv:2601.22001 — "Heterogeneous Computing: The Key to Powering the Future of AI Agent Inference", Y. Zhao and J. Liu, Jan 2026.** A short position paper: agent inference is memory-bound, not compute-bound; proposes two metrics beyond roofline — Operational Intensity and Capacity Footprint — to show that KV capacity, not FLOPs, binds long-context decode; recommends phase-specific accelerators, better networking, and optically disaggregated memory. This is the **counter-position to P14/P15/P16**: it argues *for* disaggregation and *for* large disaggregated memory precisely because dense long-context KV is assumed. The author's reply (L2585–L2591): correct that adding FLOPs to a memory-bound system is useless, but the paper does not include indexed attention, managed aggregation, GQA×speculation or the other optimisations, so its conclusions hold only for that software. Relevance: **H14** (their disaggregation case is the baseline S5 must beat), **H15b/H16** (Capacity Footprint is the quantity P15/P16 claim to shrink ~128× — adopt their metric in S6/S7 reporting), **H1** (their "not compute bound" observation is the regime where P1's premise fails, cf. `docs/00 §P1` caveat).
