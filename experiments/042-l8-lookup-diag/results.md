# 042-l8-lookup-diag — L8's lookup on `173` (ADR-054): **the query carries no information about the article** (own-cell rank at chance at every iteration, in both halves), and **the keys separate cells only partly** (own-key oracle top-32 in 7.8 % of cases). The softmax is not uniform by the fixed scale reading, and gradient coherence is not resolvable at 8 windows

2026-11-04 09:16 UTC+1 · GPU 0, rig otherwise idle (no other compute process) · driver 615.71.09 · tree `8ab63ec` plus the new script · `python -m scripts.analysis.probe_l8_lookup --id 173-l8-screen --out 042-l8-lookup-diag` · checkpoint `173-l8-screen` step 1907, table `040` · 19.3 s, peak 3.03 GiB · full output in `diag.json`, inputs in `config.yaml`.

**Bears on:** I40, which asks why L8's lookup does not learn. It bears on no hypothesis directly; H17's verdict stays `173`'s. Rig series id `042`. Readings are ADR-054's, fixed before the run.

## Numbers

### 1. Does the query carry the article? (70,952 in-table first-sentence tokens per iteration)

| iteration | own-cell rank, median (chance 524,288) | own cell in top 32 | own row rank, median (chance 512) | own column rank, median |
|---|---|---|---|---|
| 1 | 527,168 | 7 × 10⁻⁵ | 525 | 510 |
| 2 | 525,618 | 7 × 10⁻⁵ | 510 | 510 |
| 3 | 515,568 | 6 × 10⁻⁵ | 508 | 503 |
| 4 | 513,808 | 1 × 10⁻⁵ | 506 | 510 |
| 5 | 515,086 | 3 × 10⁻⁵ | 501 | 510 |
| 6 | 513,361 | 3 × 10⁻⁵ | 507 | 506 |
| 7 | 515,266 | 1 × 10⁻⁵ | 507 | 510 |
| 8 | 520,821 | 8 × 10⁻⁵ | 503 | 515 |

Every median is within 2.1 % of chance, inside ADR-054's 20 % band. The 10th–90th percentile range at iteration 1 is 111 k–939 k, close to uniform. **Reading (1) fires: the query is not article-specific.** It does not even pick the article's row or column cluster, which is a much coarser target (1 of 1,024 per half).

### 2. Can the keys tell cells apart? (own key as the query, no model; 2,000 in-table cells)

| | value |
|---|---|
| own cell in top 32 | **7.8 %** (reading's line: 50 %) |
| own-cell rank | median 945, p10 47, p90 6,866 (of 2²⁰) |
| own row / column rank per half | median 4 / 25 (of 1,024) |
| sub-key norms $\lVert h^{1,2}(\mu)\rVert$ | 0.17 median in both halves |
| singular values | $W_m$: 2.18 … 0.35, effective rank 121 of 128 · $h^1$, $h^2$: 1.60 / 1.40 … 0.34, effective rank 61 of 64 · $W_v$: **3.09**, 1.50, 1.33 … 3 × 10⁻⁷, effective rank 173 of 768 |

**Reading (2) fires: the keys do not separate cells** at $k_m$ = 32. They are not collapsed: the key maps keep full rank, and a cell's own key puts it in the top 0.1 % at the median. But a perfect query would still miss its cell 92 % of the time. That is weaker in the column half (median 25) than in the row half (median 4). It is also an upper bound on what the dot-product lookup can do with these keys, read beside `040`'s +24 % placement distortion.

### 3. Is the softmax starved by scale? (same tokens)

| iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| top-1 − top-32 score, mean | 0.58 | 0.72 | 0.62 | 0.70 | 0.76 | 0.88 | 1.19 | 1.22 |
| std of the 32 retrieved scores | 0.15 | 0.18 | 0.16 | 0.19 | 0.22 | 0.25 | 0.35 | 0.35 |
| std of half-1 / half-2 scores over 1,024 | 0.19 / 0.20 | 0.25 / 0.19 | 0.31 / 0.18 | 0.42 / 0.19 | 0.29 / 0.18 | 0.44 / 0.21 | 0.56 / 0.24 | 0.40 / 0.28 |

Every mean gap is above ADR-054's 0.5 line, so **reading (3) does not fire**. The scores are small, though. With a 32-score std of 0.15–0.35, α stays near uniform, which matches `173`'s 99.6 % of ln 32. The gap grows with depth, from 0.58 to 1.22.

### 4. Does the lookup get a coherent training signal? (8 disjoint held-out windows of 2,048 tokens, train mode, no step)

| group | params | cross-batch cosine, mean | ‖mean g‖ / mean ‖g‖ (pure noise ≈ 1/√8 = 0.354) | ‖g‖ / ‖w‖ |
|---|---|---|---|---|
| $W_m$ (query) | 106,496 | +0.017 | 0.377 | 7.4 × 10⁻⁴ |
| $h^{1,2}$ (key maps) | 49,152 | −0.011 | 0.340 | 2.9 × 10⁻⁴ |
| $W_v$ (read-out) | 589,824 | +0.064 | 0.425 | 3.0 × 10⁻² |
| experts (reference) | 12,976,128 | +0.008 | 0.363 | 1.5 × 10⁻² |
| middle attention (reference) | 1,572,864 | +0.019 | 0.376 | 6.8 × 10⁻² |

**Reading (4) does not fire.** $W_m$'s cosine (0.017) is above the experts' (0.008); only the key maps sit at noise. The measurement has little power: every group is within a few percent of the 1/√8 noise ratio, the reference included, because one window per batch is too small. What it does show is that the gradient reaching the query and key maps is 20–50× smaller relative to their weights than the experts' gradient (7.4 × 10⁻⁴ and 2.9 × 10⁻⁴ against 1.5 × 10⁻²). Only the read-out $W_v$ gets a gradient of the experts' size.

### Branch output norm against the MoE output's (same windows, eval)

| iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| ‖Mem‖ | 1.29 | 1.32 | 1.33 | 1.35 | 1.30 | 1.30 | 1.29 | 1.28 |
| ‖MoE‖ | 2.48 | 4.58 | 5.39 | 6.60 | 4.03 | 5.48 | 6.79 | 4.20 |
| ratio | 0.52 | 0.29 | 0.25 | 0.20 | 0.32 | 0.24 | 0.19 | 0.30 |

## Interpretation

The lookup fails on the query side first. Under the model's own query, an article's cell ranks at chance among 2²⁰ at all eight iterations, and so do its row and column among 1,024 per half. The query does not even land in the right cluster of the encoder space, so nothing downstream can retrieve. The keys are a second, smaller limit. They have full rank and place a cell in the top 0.1 % when queried with its own key, but they reach the top 32 only 7.8 % of the time, so even a perfect query would mostly miss at $k_m$ = 32. The softmax is not flattened by a tiny score scale under ADR-054's fixed line (the gaps are 0.6–1.2), though its spread is small enough to keep α near uniform. Whether the lookup's gradient is noise cannot be told from 8 single windows. What can be told is that the query and key maps receive 20–50× less gradient, relative to their size, than the experts do, while the read-out gets the experts' share.

Together with the branch's output, these numbers are *consistent with* one picture, which I have not tested. The branch writes a vector about 0.2–0.5× the MoE's, almost constant across iterations (1.28–1.35). The read-out has one dominant direction (singular value 3.09 against 1.50 next), and swapping in random cells costs only 0.0025 nats (`173`). So the memory probably acts as a learned bias through $W_v$, the average of whatever 32 sentences come back, and the training signal that would teach the query to find a specific sentence is too weak to compete.

ADR-054's readings point at two variants. Reading (1) points at the query side: something has to make the hidden state predictive of the article's cell, for example an auxiliary retrieval loss on in-table text, or a query built in the encoder's space. Reading (2) points at the keys and placement: for example sub-keys taken from the codebooks without a learned map, a larger $d_m$, or a tighter placement than `040`'s. Each needs its own ADR, and the user chooses. Whether the gradient is noise (reading 4) would take a larger batch count to settle. It matters only if a variant still stays at chance.
