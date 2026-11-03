# 040-wiki-memory — L8's table built in 288 s: 2²⁰ Wikipedia lead sentences (of 5.65 M eligible), BGE-base, on a 1024 × 1024 product grid; 74.6 % of entries in their top-32 × top-32 centroid pairs, 2.8 % placed by the last-resort tier

2026-11-03 22:26–22:31 UTC+1 · idle rig, by hand · `python -m scripts.data.build_wiki_memory --id 040-wiki-memory` · tree `ba89ed2` · download (11.6 GB parquet, 2 min 45 s over IPv4; IPv6 stalls on this network) done beforehand.

**Bears on:** nothing directly. This is the data build of ADR-053 §9a, which L8 (`173`, H17) runs on. Rig series id `040`.

Text from English Wikipedia (CC BY-SA 3.0 / GFDL). The table, sentences and eval pools are derived data. They stay at `/mnt/nvme/memory/wiki-20231101-lead/` and are not committed. This directory holds the manifest (`result.json`, with every file's sha256) and the sampled article ids only (`sampled_article_ids.npz`: `first` plus `uint32` gaps; ids, no text).

| | |
|---|---|
| articles in `20231101.en` | 6,407,814 |
| dropped: title ("List of", "(disambiguation)") | 168,896 |
| dropped: "may refer to" | 257,401 |
| dropped: first sentence outside 40–300 chars | 335,025 |
| **eligible** | **5,646,492** |
| table $N_m$ (seed 0) | 1,048,576; mean first sentence 122 chars |
| eval pools (seed 1) | 2,000 in-table (from the table's articles), 2,000 out-of-table (from the eligible rest) |
| encoding | 145 s on four cards, fp16 autocast; against fp32 on 256 sentences, cosine ≥ 0.9999994 |
| k-means, 1024 per 384-d half | 11 s; mean squared distance 0.430 → 0.2417 and 0.434 → 0.2437 over 25 iterations (flat after ≈ 15) |
| placement tiers | top-32²: 782,207 (74.6 %) · top-128²: 237,204 (22.6 %) · any free cell: 29,165 (2.8 %) |
| placement cost | unconstrained nearest pair 0.485 (mean squared distance, both halves); placed excess mean 0.116 (p50 0.107, p99 0.301), **+24 %** |
| files | `table.npy` 1.61 GB (float16, cell order), `mu1/mu2.npy` 1.57 MB each, `sentences.jsonl` 217 MB, `eval_pools.jsonl` 1.6 MB |
| wall-clock | 288 s (extract 35, encode 145, k-means 11, place 79, write 8) against ADR-053's ≈ 1 h |

**Two things the ADR did not fix, recorded here:**
1. **A third placement tier.** ADR-053 §4 names top-32² and then top-128². Late in the greedy order, 29,165 entries found every one of their 16,384 top-128 pairs taken, so they take the nearest free cell anywhere on the grid. These are the entries whose key describes them worst. The lookup-accuracy probe of `173` is read with that in mind.
2. **The eval pools.** ADR-053 §2 says both pools are drawn "from the rest", but §7 defines the in-table pool as articles whose first sentence *is* in the table. The build follows §7: in-table from the 2²⁰ table articles, out-of-table from the 4.6 M eligible articles not sampled (session log 2026-11-03, 22:40).

**Interpretation.** The table is built as specified, in a twentieth of the time budgeted. Balanced placement costs 24 % in quantisation distance against an unconstrained product quantiser. That is the price of one entry per cell and no collisions. Whether it matters shows in `173`'s lookup accuracy (own cell among the 32 retrieved), not here. A first look at the cell order is a sanity check of the grid's semantics: cells 0–2 are a funk box set, a hip-hop compilation album and a live compilation by The Who. Neighbouring cells share a row centroid, and these share a topic.
