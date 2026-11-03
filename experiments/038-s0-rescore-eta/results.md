# 038-s0-rescore-eta — S0 re-scored with the owner stages' efficiency against batch (ADR-051 §5): **passes at every $z$**. Throughput is 1.09–1.14× of measured and iteration p50 0.88–0.96×

2026-11-03 · `python -m scripts.bench.s0_compare --measured 034-s0-ep-decode --id 038-s0-rescore-eta` · CPU only (the simulator), ≈ 2 min · code at tree `0b48da1`. `git_dirty` in `result.json` is the scenario's new `software` section from `037` and the untracked `037`/`038` folders. Their sha256 is in `result.json` (`a26183c6…`), and they are committed with this write-up. The measured side is `034`, unchanged (ADR-051 §5). Simulator: `batch_policy = round`, measured routing shares (`empirical`) and uniform, `context_tokens` 1792, $z \cdot 200$ tokens with $z \cdot 40$ warm-up, seed 0.

**Bears on:** no hypothesis directly. This is `02 §14`'s calibration point (a), which gates every S-result.
**Verdict: S0 passed** for `local_3060` with the named stack (PyTorch 2.13 eager plus the reference decode), up to 256 sequences per owner. The rule is ±20 % on throughput and iteration p50 at every $z$. Nothing was fitted to `034`: every scenario value comes from `scripts/bench/` (`037` for the `software` section).

| $z$ | measured tok/s | sim / meas | measured iter p50 | sim / meas | token p50 meas / sim | ±20 % |
|---|---|---|---|---|---|---|
| 16 | 802 | **1.128** | 1.927 ms | **0.932** | 19.6 / 17.7 ms | pass |
| 64 | 3,153 | **1.091** | 1.951 ms | **0.963** | 20.0 / 18.6 ms | pass |
| 256 | 9,832 | **1.141** | 2.515 ms | **0.901** | 25.9 / 22.8 ms | pass |
| 1024 | 18,618 | **1.135** | 5.045 ms | **0.884** | 54.8 / 48.5 ms | pass |

Same sim build, terms switched off (sim / meas, throughput / iteration p50):

| | $z$ 16 | 64 | 256 | 1024 |
|---|---|---|---|---|
| without η (ADR-050's model; repeats `036` with `037`'s $t_0$) | 1.13 / 0.93 | 1.09 / 0.96 | 1.17 / 0.88 | **1.46 / 0.71** |
| without the `software` section (ADR-049's model) | 10.05 / 0.10 | 6.97 / 0.14 | 3.92 / 0.24 | 2.51 / 0.40 |

Uniform routing changes every ratio by ≤ 0.01.

## Interpretation

η changes nothing at $z$ ≤ 64, where each owner holds ≤ 16 sequences and η = 1. It moves $z$ = 256 slightly (64 per owner) and closes the $z$ = 1024 miss, from 1.46× to 1.14× on throughput and from 0.71× to 0.88× on iteration p50.

The sim is still optimistic everywhere, by 9–14 % on throughput. At $z$ = 1024 the remaining 0.58 ms per round has two parts:
- the harness's owner phase is 3.55 ms, against 3.35 ms for attention timed alone in `037` (ADR-051 §6's bench-vs-harness check, ≈ 0.2 ms);
- the dispatch is 0.67 ms in `034` against the model's 0.27 ms. That gap is the host read-back the ADR left out, plus waiting for the slowest owner.

Both are inside the tolerance and are recorded, not modelled.

**What this calibration covers (ADR-051 §6).** The sim's owner stages on this rig are now measured inputs at the shapes the harness runs, so S0 confirms the composition, not the owner rooflines:
- the round barrier, the expert jobs, the collective terms and the routing imbalance;
- that stages timed alone add up to the harness within the tolerance.

It covers the round policy only: threshold + timeout firing is not calibrated (I35 stays open). It covers 16 to 1024 sequences in flight on four units at 1792 keys. **I38 closes.** Under `02 §14` no S-result is trusted until point (b) also holds: a published expert-parallel deployment within ±20 % (I1). That point is next.
