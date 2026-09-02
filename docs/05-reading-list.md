# bigmoe — Reading list (Phase-0 literature refresh)

Compiled from memory on 2026-09-02. **Verify every entry**, add work published after mid-2026, and record a one-line takeaway per entry in this file. Also fetch and summarise, in our own words, the author's in-depth hardware/software post linked at the end of the note (lnkd.in short link in `docs/source/`).

## P2 — Parallel attention + FF
- Wang & Komatsuzaki 2021 — GPT-J-6B (parallel block).
- Chowdhery et al. 2022 — PaLM (parallel layers; 8B vs 62B ablation).
- Almazrouei et al. 2023 — The Falcon series (parallel attention + multi-query).

## P3 — Sparsity and low precision
- Mishra et al. 2021 — Accelerating Sparse Deep Neural Networks (2:4).
- Frankle & Carbin 2019 — Lottery Ticket Hypothesis; You et al. 2020 — Early-Bird Tickets.
- Evci et al. 2020 — RigL (dynamic sparse training).
- NVIDIA 2025 — NVFP4 pretraining recipe; DeepSeek-V3 2024 — FP8 training details.

## P4 / P8 — Expert granularity
- Krajewski et al. 2024 — Scaling Laws for Fine-Grained Mixture of Experts.
- Dai et al. 2024 — DeepSeekMoE; DeepSeek-AI 2024 — DeepSeek-V3 (256 routed experts per layer, top-8, aux-loss-free balance, MTP).
- Clark et al. 2022 — Unified Scaling Laws for Routed Language Models.
- Gale et al. 2022 — MegaBlocks (dropless MoE).
- Fedus et al. 2021 — Switch Transformer; Lepikhin et al. 2020 — GShard.

## P5 / P9 — Fabric, placement, collectives
- NVIDIA DGX SuperPOD reference architecture — rail-optimized topology; GB200 NVL72 domain.
- Mellanox/NVIDIA SHARP — in-network reduction.
- DeepSeek 2025 — DeepEP (expert-parallel all-to-all) and EPLB (expert placement / replication).
- Ultra Ethernet Consortium — status of multicast and in-network collectives (verify).

## P6 / P7 — Repeated block, adaptive depth
- Dehghani et al. 2019 — Universal Transformers (with ACT).
- Lan et al. 2020 — ALBERT (cost of parameter sharing).
- Csordás et al. 2024 — MoEUT: Mixture-of-Experts Universal Transformers.
- Bae et al. 2024 — Relaxed Recursive Transformers (sharing + layer-wise LoRA).
- Geiping et al. 2025 — Scaling up Test-Time Compute with Latent Reasoning (recurrent depth).
- Wang et al. 2025 — Hierarchical Reasoning Model (recurrent modules with ACT halting; the DFM Mimir line builds on it).
- Graves 2016 — Adaptive Computation Time.
- Raposo et al. 2024 — Mixture-of-Depths.
- Schuster et al. 2022 — CALM (early exit with state propagation).
- Dai et al. 2019 — Transformer-XL (segment recurrence, stop-gradient memory); Rae et al. 2020 — Compressive Transformer; Hutchins et al. 2022 — Block-Recurrent Transformers — precedents for v2 P7's segment-sequential processing with a memory of earlier segments.

## P10 / P11 / P14 — Serving systems
- Yu et al. 2022 — Orca (continuous batching).
- Agrawal et al. 2024 — Sarathi-Serve (chunked prefill).
- Zhong et al. 2024 — DistServe; Patel et al. 2024 — Splitwise; Qin et al. 2024 — Mooncake (prefill/decode disaggregation).
- Zhou et al. 2022 — Mixture-of-Experts with Expert Choice routing.
- Wang et al. 2024 — Auxiliary-Loss-Free Load Balancing for MoE.

## P12 / P13 — Streams and multi-token prediction
- Yang et al. 2019 — XLNet (two-stream self-attention).
- Gloeckle et al. 2024 — Better & Faster Large Language Models via Multi-token Prediction.
- Leviathan et al. 2023; Chen et al. 2023 — speculative decoding / sampling.
- Cai et al. 2024 — Medusa; Li et al. 2024 — EAGLE.

## P15 — Indexed / sparse attention, KV offload
- Wu et al. 2022 — Memorizing Transformers; Bertsch et al. 2023 — Unlimiformer.
- Mohtashami & Jaggi 2023 — Landmark Attention.
- Tang et al. 2024 — Quest; Chen et al. 2024 — MagicPIG; Liu et al. 2024 — RetrievalAttention.
- Yuan et al. 2025 — Native Sparse Attention (NSA); DeepSeek-AI 2025 — DeepSeek-V3.2-Exp / DeepSeek Sparse Attention (lightning indexer).
- Brandon et al. 2024 — Cross-Layer Attention (KV sharing); Sun et al. 2024 — You Only Cache Once — the closest precedent for v2 P7's single global cache built from one representation per token.
- Sheng et al. 2023 — FlexGen; Lee et al. 2024 — InfiniGen (KV offload).
- Malkov & Yashunin 2018 — HNSW; Johnson et al. 2019 — FAISS / IVF-PQ.

## P17 — Memory and retrieval
- Lample et al. 2019 — Large Memory Layers with Product Keys.
- He 2024 — Mixture of A Million Experts (PEER).
- Berges et al. 2024 — Memory Layers at Scale.
- Borgeaud et al. 2022 — RETRO; Khandelwal et al. 2020 — kNN-LM.

## P18 — Multi-tenant adapters
- Sheng et al. 2023 — S-LoRA; Chen et al. 2023 — Punica (batched-gather LoRA kernels).
- Dettmers et al. 2023 — QLoRA.

## P16 / P19 — Economics (cost model only)
- Public HBM / GDDR / LPDDR bandwidth-per-capacity and price data — source every number placed in `sim/scenarios/`.
