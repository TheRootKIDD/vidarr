# 09 — Findings so far (2026-09-13)

Written for Annemette. Everything here is measured at `screen`: 1 B tokens, one seed, on the local rig. Under ADR-013 a one-seed result is a *direction*, not a verdict; verdicts come at `small` (2.5 B tokens, two seeds), which has not started. Full write-ups are in `experiments/<id>/results.md`; decisions in `docs/04`.

## 1. What we are testing, and how

The project takes the 19 points of "The big-DC MoE LLM design" as hypotheses H1–H19 and switches the model's features on one at a time (the *ablation ladder*, `docs/03 §2`). Every run is trained the same way and scored the same way, so the only thing that differs between two adjacent rungs is the feature.

**Training recipe, identical for every run**

| | |
|---|---|
| corpus | FineWeb-Edu `sample/10BT` (ODC-By), tokenised once with the Mistral-7B 32 k tokenizer to a memmap on NVMe (ADR-014) |
| held-out set | the first 10 000 documents of shard 000, never trained on; 5606 windows of 2048 tokens |
| budget | 1.0 B training tokens in 1907 steps of 524 288 tokens (context 2048) |
| optimiser | AdamW, β = (0.9, 0.95), weight decay 0.1, grad-clip 1.0; lr 6e-4, 2 % warm-up, cosine to 6e-5 |
| numerics | bf16 autocast; per-iteration activation checkpointing on the recurrent rungs |
| hardware | 4 × RTX 3060 12 GB, DDP with gradient accumulation, NCCL bounced through host memory (no NVLink, no P2P) |
| matched FLOPs | every rung costs ≈ 0.6 GFLOP per training token, the same as the 12-layer dense baseline; widths are solved from that budget by `costmodel/` (ADR-025) |

**Evaluation.** One number decides everything: held-out cross-entropy in nats per token on the fixed set, at the end of training, compared with the matched-FLOPs baseline of the same size and seed. Thresholds (ADR-013) are fractions of the baseline loss; "matches" means within two seed-spreads, and one seed cannot supply a seed spread, so `screen` is a direction and `small` is the verdict. Beside the loss we log the routing state every ten steps (effective expert count, dead experts), the thermal state of each card (a throttled card invalidates throughput numbers, never loss numbers), and, for the multi-token rung, speculative-decode acceptance measured against the main head.

## 2. The architectures, rung by rung

All rungs share $d$ = 768, 12 query heads and 4 KV heads of 64, RoPE, SwiGLU, a tied 32 k output embedding, and 2048-token context.

| Rung | Architecture | Tests | Params (non-emb) |
|---|---|---|---|
| **L0** dense | 12 pre-norm blocks, sequential attention → FF, $d_{ff}$ = 2048 | the baseline the note asks for | 75.5 M |
| **L1** layered MoE | L0 with each FF replaced by 8 experts of $d_{ff}$ = 1024, top-2 routing, DeepSeek-V3-style aux-loss-free bias balancing | the fine-grained MoE comparator (H4's baseline arm) | 246.6 M |
| **L2** parallel form | L1 with attention and the expert FF reading the same normalised input and summed (note P2) | H2: is the latency-friendly form free in quality? | 246.6 M |
| **L3** MTP | L2 plus three extra independent heads predicting $x_{t+2..t+4}$ from the final hidden state (note P13, ADR-012); each position trains one head per step | H13: do trained-from-scratch draft heads reach useful acceptance? | 248.3 M |
| **L3-full** | L3 with every head trained on every position (I27 #3) | whether L3's subsampling caused its miss | 248.3 M |
| **L5** shared block | 2 dense early blocks → **one** MoE block applied $r$ = 8 times with a learned depth embedding conditioning its norms and router → 2 dense final blocks (note P6). Parallel form. 8 experts of 896, top-2. Local attention window 512 per iteration, one KV cache per iteration | H6: does one repeated layer match a stack? (author's default pool) | 52.9 M |
| **L5-ne128** | L5 with 128 experts of 448, top-4 (the note's "128× as many experts") | H6's parameter-matched arm | 168.5 M |
| **L5d** final-vector global | L5 with global attention over the *final-vector cache* $\mathcal{G}$ only — one entry per token, written after its last iteration — and local attention per iteration (note P7 v2, ADR-021/022/023) | H15a: the KV-cache reduction the note rests on | 52.9 M |

## 3. Results

| Id | Rung | Held-out (nats) | vs L0 | Δ vs its comparator | One-line reading |
|---|---|---|---|---|---|
| 100 | L0 | **3.3360** | — | — | $L_{ref}$ |
| 106 | L1 | 3.2048 | −3.93 % | — | MoE beats dense at matched FLOPs, as expected; routing perfectly balanced at the end |
| 107 | L2 | **3.1495** | −5.59 % | **−1.73 %** vs L1 | H2 budgeted a ≤ 1 % *cost*; the parallel form is a gain here |
| 108 | L3 | 3.2486 | −2.62 % | +3.15 % vs L2 | H13 misses both clauses, as the literature prior (I20) predicted |
| 113 | L3-full | 3.2342 | −3.05 % | +2.69 % vs L2 | subsampling was worth a few tenths of a point of acceptance, not the miss |
| 110 | L5 | 3.3328 | **−0.10 %** | +3.99 % vs L1 | the shared block equals dense at 70 % of its parameters |
| 112 | L5-ne128 | 3.2232 | −3.38 % | **+0.57 %** vs L1 | 128 experts recover 86 % of the gap to the layered MoE at 68 % of its params |
| 111 | L5d | 3.3477 | +0.35 % | **+0.45 %** vs L5 | final-vector-only global attention costs a quarter of H15a's 2 % allowance |

**Speculative decode (H13).** Acceptance of heads 2/3/4 against the main head is 38.5 / 21.6 / 15.0 % against targets of 70 / 55 / 45 %. The number that matters for decode speed, the expected accepted run length, is 0.51 heads, i.e. 1.5 tokens committed per verify pass. Acceptance rises to 56 % on the third of positions where the main head is near-certain and is flat elsewhere; training every head on every position (113) changes none of this by more than a few tenths of a point.

**Routing.** The aux-loss-free bias controller balances every configuration we have tried without retuning: 12 separate routers of 8 (L1–L3), one shared router of 8 used at every iteration (L5), and one shared router of 128 (L5-ne128, which starts with 90 of 128 experts dead at step 23 and ends with the busiest expert at 1.18× its uniform share).

## 4. What the results say about the note

- **P2, parallel attention and FF.** Not a cost at this scale but a 1.7 % gain. The latency half is untested and belongs to the simulator.
- **P6, one repeated middle layer.** With the author's default pool of 8 experts it matches dense exactly at fewer parameters. With 128 experts it comes within 0.57 % of the layered MoE at 68 % of the parameters, and *leads* it for the first two thirds of training before being overtaken. Whether that crossing moves with budget is the H6 question `small` will answer, in either direction.
- **P7 v2, global attention over final vectors only.** Costs 0.45 % of loss for an ≈ 8× smaller persistent KV cache. The most favourable result on the ladder relative to its threshold.
- **P13, multi-token prediction from the start.** Fails its acceptance targets at this budget by a wide margin, in the direction and for the reason the literature predicted. Two cheap probes and one 10-hour run have ruled out our own optimisation as the cause. Decided at `small` (ADR-030); the fallback design (curriculum vs sequential modules) is open.
- **P4, one uniform expert width.** Challenged by FlexMoRE and FlexMoE (`docs/00 §P4`); L7c and S11 are planned to decide it. Adjacent evidence: 8 experts of 896 vs 128 of 448 at equal FLOPs is 3.3 % in favour of the fine pool.

## 5. What the rig taught us

Thermal throttling invalidated the first throughput table entirely (I21–I23): three cards sat at 91–93 °C in the original layout. After a cooling rebuild, spacing the cards, and moving one to a bracket, all four hold full clock for hours; one card (GPU 0) still runs at its limit under the heaviest rung and is trimmed ≈ 4 % of clock, which touches throughput but not loss. Every run now accounts for its wall-clock to within 0.1 %. Two runs at the same seed differ by 0.04 nats early in training: DDP over host-bounced NCCL is not bit-reproducible, one more reason verdicts wait for two seeds.

## 6. Running now and next

Queue 4 is finishing the L5 depth sweep (r = 4 and r = 12 with widths pinned, the control H7 needs so that learned depth is compared against fixed depth at the same mean, I25). After that, in order of information per GPU-hour: the H/L slow-fast split L5e (`docs/08 §4`, needs a model change), the expert-choice depth router L6c (I24), then the `small` programme at two seeds for L0, L1, L2, L3, L5, L5d. Post-ladder direction on record: composing separately trained small models on a frozen spine (I28).
