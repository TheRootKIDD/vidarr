# 050-g-recall-nope — ANN recall@32 over $\mathcal{G}$'s keys, NoPE (`204`) vs RoPE (`162`) (ADR-066 §6): +0.072 at 2 k (58 SE), +0.002 at 8 k (0.6 SE) → H22's second clause holds at the trained length only, **not met as written**

**Bears on:** **H22**, second clause ("ANN recall@$K$ over NoPE $\mathcal{G}$ keys is higher than over RoPE'd keys"), and Q9's motivation. ADR-066 §7 asks for recall(NoPE) − recall(RoPE) > 2 SE **at both lengths**. It is far above that at 2048 and inside it at 8192, so the clause is **not met as written**; the ADR sets no weakens band for it, so the reading is *silent, split by length*. H22's first clause is `204`'s (weakened at `screen`).

No training. `probe_g_recall` on the final checkpoints of `162-l5d-segmean-screen` (`rope_scope: all`) and `204-l5d-nope-screen` (`local`; the rerun of `182`), one card per lane with the rig idle at the end of `queue_screen_18b`, 2026-11-20 00:20–00:31, code `b64d4b6`, 4 × RTX 3060 (forward on GPU, faiss on CPU). Per iteration and KV head, the exact top-32 global keys by the model's own score against an HNSW index (M 32, efSearch 64, inner product) over that window's $\mathcal{G}$ keys, last segment of each window; SE is a bootstrap over windows. Outputs are in the two runs' directories (`probe_g_recall_ctx{2048,8192}.json`); wall-clock 275–353 s per lane.

| context | windows | `162` RoPE | `204` NoPE | Δ | SE of Δ | Δ / SE |
|---|---|---|---|---|---|---|
| 2048 (trained) | 256 | 0.8959 ± 0.0010 | **0.9679** ± 0.0007 | **+0.0720** | 0.0012 | 58 |
| 8192 (extrapolation) | 64 | 0.9071 ± 0.0017 | 0.9090 ± 0.0029 | +0.0019 | 0.0034 | 0.6 |

Per iteration $j$ = 1…8:

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| `162`, 2 k | 0.871 | 0.876 | 0.872 | 0.907 | 0.875 | 0.905 | 0.950 | 0.912 |
| `204`, 2 k | 0.956 | 0.970 | 0.959 | 0.961 | 0.969 | 0.971 | 0.975 | 0.983 |
| `162`, 8 k | 0.918 | 0.919 | 0.916 | 0.907 | 0.915 | 0.899 | 0.888 | 0.895 |
| `204`, 8 k | 0.889 | 0.915 | 0.893 | 0.888 | 0.903 | 0.916 | 0.927 | 0.941 |

## Interpretation

At the trained length, content-only keys index better: the same HNSW settings lose 3.2 % of the model's own top-32 on NoPE keys against 10.4 % on RoPE'd keys, at every iteration. That is the effect Q9's index was motivated by. At 8192 the difference is gone (+0.002, 0.6 SE): NoPE is ahead in iterations 6–8 and behind in 1, 3 and 4. Both models are far outside their training context there (L5d fails beyond 2 k in loss, `docs/04` 2026-10-10), so the 8 k row describes key geometry under extrapolation, as ADR-066 §6 warned, and it has a quarter of the windows. The clause needed both lengths and gets one. Against it stands `204`'s loss cost of +0.0152 nats (5.1σ): a 7-point recall gain at fixed HNSW settings can also be bought by raising efSearch, so this result does not by itself argue for ADR-063. A recall number at 8 k from a model trained at 8 k (the `158`/`159` configuration with `rope_scope: local`) would be the clean test; that is a new run and needs its own ADR.
