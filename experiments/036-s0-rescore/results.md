# 036-s0-rescore — S0 re-scored with the fixed cost per job (ADR-050 §5): **passes at $z$ = 16, 64, 256; misses at 1024** (throughput 1.46×, iteration p50 0.71×). The next term is the owner stages' efficiency at large batch

2026-11-03 · `python -m scripts.bench.s0_compare --measured 034-s0-ep-decode --id 036-s0-rescore` · CPU only (the simulator), seconds · tree `19effb0` (code clean; `git_dirty` in `result.json` is the untracked `035` folder) · scenario `local_3060.yaml` with the `software` section from `035` (sha256 in `result.json`). The measured side is `034`, unchanged (ADR-050 §5). Simulator: `batch_policy = round`, measured routing shares (`empirical`) and uniform, `context_tokens` 1792, $z \cdot 200$ tokens with $z \cdot 40$ warm-up, seed 0.

**Bears on:** no hypothesis directly. This is `02 §14`'s calibration point (a), which gates every S-result.
**Verdict: S0 not passed. It passes at three of four $z$.** The rule is ±20 % on throughput and iteration p50 at *every* $z$. Nothing was fitted to `034`.

| $z$ | measured tok/s | sim / meas | measured iter p50 | sim / meas | token p50 meas / sim | ±20 % |
|---|---|---|---|---|---|---|
| 16 | 802 | **1.132** | 1.927 ms | **0.929** | 19.6 / 17.6 ms | pass |
| 64 | 3,153 | **1.094** | 1.951 ms | **0.960** | 20.0 / 18.6 ms | pass |
| 256 | 9,832 | **1.168** | 2.515 ms | **0.879** | 25.9 / 22.3 ms | pass |
| 1024 | 18,618 | **1.464** | 5.045 ms | **0.712** | 54.8 / 37.6 ms | **miss** |

| without the `software` terms (same sim build) | 10.05× / 0.10× | 6.97× / 0.14× | 3.92× / 0.24× | 2.51× / 0.40× |
|---|---|---|---|---|

Uniform routing changes every ratio by ≤ 0.01.

## Interpretation

The fixed cost per job closes most of the gap. At $z$ ≤ 256 the sim now sits within 9–17 % of the measured throughput and 4–12 % of the iteration p50, from an error of 2.5–10× on throughput before. The remaining miss is at $z$ = 1024, which puts 256 sequences on each owner. `035` already showed, before this comparison, that $t_0$ + roofline under-predicts owner stages at that batch by 30–45 % (attention 3.34 against 2.49 ms; skeleton halves 6.07 / 6.55 against 4.22 / 4.50). That accounts for most of the 1024 miss, which comes to 17.2 ms of token latency:

- the skeleton halves are short by ≈ 4.6 ms per token (13.3 measured in `034`, 8.7 simulated);
- each of the 8 rounds is short by ≈ 1.45 ms. Of that, the owner stage is ≈ 1.06 ms (3.55 against 2.49) and the dispatch ≈ 0.4 ms (0.67 measured; the model's $2 t_{coll}$ + bytes/λ gives 0.27, against the bench's own 0.33).
 **The next term is the achieved efficiency of the owner's work at large batch.** ADR-043 decision 6 charges attention, KV reads and the skeleton GEMMs at peak φ and β_C. A sourced efficiency for these kernels (measured by a bench, as `003-gemm`'s grouped-GEMM table is for experts) would be the analogue. Under ADR-050 §5 that needs a new ADR. Nothing here was tuned to `034`.
