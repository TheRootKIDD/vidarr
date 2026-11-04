# 044-l8k-gate — ADR-056's gate **fails**: a perfect query retrieves its own cell 16.8 % of the time with PCA-64 keys (line 50 %), and only 22.8 % with the full 384-d codebooks. The limit is the table's placement on the product grid, not the keys → **no `175` run**

2026-11-04 22:35–22:43 UTC+1 · GPU 0, rig otherwise idle · driver 615.71.09 · tree `672e3b4` (ruff and the full test suite passed on it) · commands and inputs in `config.yaml`. PCA build 5.9 s, ceiling probe 2.9 s. Outputs: `pca.json` (the projection's hashes and the variance kept), `ceiling.json` (the gate), and `ceiling_by_k.json` (the same queries at larger $k_m$ / half_topk, a follow-up in the same session, not a gate input).

**Bears on:** I40, and ADR-056 §5's gate. It bears on no hypothesis directly: no model was trained, and H17's verdict stays `174`'s. Rig series id `044`.

**Method.** The 2,000 in-table wiki-lead articles are used. Each query is the article's own first-sentence embedding, its row in `040`'s table, which is what a perfect query in the encoder's space would be. It is mapped into each key space and run through the lookup's product search (top 32 per half, 1,024 candidates, top $k_m$ = 32). The ceiling is the share whose own cell comes back.

## Numbers

| keys | own cell retrieved ($k_m$ 32) | own-cell rank among 2²⁰, median (p10 – p90) | own row / column rank per half, median |
|---|---|---|---|
| **PCA-64, distance** (ADR-056) | **16.8 %** | 559 (14 – 8,524) | 9 / 10 |
| full 384-d codebooks, distance ($d_m$ 768) | 22.8 % | 282 (8 – 5,779) | 6 / 8 |
| `174`'s learned $h^{1,2}$, dot | 0.35 % | 50,533 (2,617 – 335,067) | 148 / 102 |

PCA-64 keeps 51.4 % / 50.8 % of each half's variance.

**Gate (ADR-056 §5): fails,** 16.8 % < 50 %. So `175` does not run.

### Where the ceiling comes from (`ceiling_by_k.json`; same queries)

| | PCA-64 | full 384 |
|---|---|---|
| own cell ranked first among 2²⁰ | 1.1 % | 1.25 % |
| own row is the query's nearest centroid / own column likewise | 20.0 % / 17.1 % | 22.8 % / 20.0 % |
| own cell in the exact top 32 / 128 / 256 / 1,024 / 4,096 | 16.7 / 31.7 / 39.9 / 60.0 / 81.5 % | 22.8 / 40.2 / 48.4 / 68.5 / 86.7 % |
| product path, half_topk 32: $k_m$ 32 / 128 / 1,024 | 16.8 / 31.3 / 50.7 % | 22.8 / 42.7 / 72.8 % |
| product path, half_topk 64, $k_m$ 256 · half_topk 128, $k_m$ 1,024 | 40.9 % · 61.0 % | 50.1 % · 72.1 % |

## Interpretation

A cell's key is its pair of centroids, not its sentence. `040` puts 2²⁰ sentences into 2²⁰ cells, one each, so most sentences cannot have their nearest pair: only 23 % sit in their nearest row, and 25.4 % are outside their top 32 × 32 pairs altogether (`040`'s tiers). A query that is the sentence's exact embedding therefore lands among a few hundred cells whose centroid pairs are as close as its own: the median rank is 282 even with the full codebooks, and the own cell comes first 1.25 % of the time.

With half_topk 32 the product path can never reach more than about 74.6 % (the share placed inside the top 32 × 32), and it gets to 72.8 % only by reading all 1,024 candidates. At $k_m$ = 32 the ceiling is 17–23 % whatever the keys are made of. So the key side cannot be the remaining limit, and that holds for $d_m$ = 768 too, at a width cost of 704 → 448 (ADR-056). Against that ceiling, `174`'s achieved 0.19 % with learned keys (ceiling 0.35 %) is about half of what its own keys allowed.

By ADR-056 §5 the next step is the user's. ADR-056's "Not chosen" list already named the case: "a re-placement or a larger $k_m$ … a separate ADR if `044` says the ceiling is the placement's". It says so.

What is left for L8 is outside a cheap key fix. The options, none drafted:
- **Re-rank the candidates by the entries' own embeddings.** This needs a 768-d query and FLOPs, so it needs a width re-solve.
- **Re-place the table.** Fewer sentences per grid, or several per cell, rebuilding `040`.
- **Close L8 at `screen`.** H17 would then be tested by L8b's offline neighbours, which need no learned lookup.
