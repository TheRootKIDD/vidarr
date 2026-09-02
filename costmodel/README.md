# costmodel/ — FLOPs, bytes and memory as pure functions

Implements the accounting of `docs/01 §10`, the expert-step conditions of
`docs/02 §2`, and the anchor sizes of `docs/06 §3`. Every function is pure;
hardware values are arguments, never constants (CLAUDE.md).

```python
from costmodel import Model, Experts, Hardware, Numerics, solve_d_ff

sol = solve_d_ff(model, budget_flops_per_token=0.6e9, n_keys=512, hw=hw)
sol.d_ff, sol.at_floor, sol.relative_error
```

| Module | Contents |
|---|---|
| `config.py` | `Model`, `Attention`, `Experts`, `Memory`, `Hardware`, `Numerics` — notation per `docs/01 §1` |
| `params.py` | parameters per expert / block / model; the dense L0 baseline |
| `flops.py` | per-token FLOPs: prefill, decode, per iteration, per stream |
| `fabric.py` | fabric bytes per token per iteration; the `docs/02 §2` TP bandwidth condition |
| `memory.py` | persistent $\mathcal{G}$ vs transient per-iteration KV (ADR-021/022); experts resident per unit |
| `hardware.py` | $b_{min}$, tile conditions, `expert_step()`, the two-stream latency floor |
| `solve.py` | the inverse: FLOPs budget → $d_{ff}$, which `docs/06 §3` and `docs/01 §1` delegate here |

Tests: `tests/test_costmodel_worked_examples.py`, one test per worked example.

## Disagreements with the spec

CLAUDE.md: the doc wins until an ADR says otherwise, and disagreements are
reported rather than patched. The code implements what `docs/01 §10` says and
makes each contested convention an explicit argument, so either resolution is
one call away. **Five findings, in descending order of what they change.**

### A. `docs/06 §3` recurrent widths need two skeleton blocks, not four

`docs/06 §3` quotes $d_{ff}$ ≈ 1.5–1.8 k at ($N_e$=8, $k$=2) and ≈ 0.8 k at
($N_e$=128, $k$=4) for the 0.6 GFLOP/token budget. Both reproduce **exactly** —
1665 and 833 — but only with **two** skeleton blocks in total. `docs/01 §1` sets
$E$ = 2 and $F$ = 2, i.e. four, and `docs/06 §3` repeats "E = F = 2". At four
blocks the same budget gives **1153 and 577**, both outside the quoted ranges.

Two independent configurations agreeing at $E+F$ = 2 and disagreeing at
$E+F$ = 4 makes this a convention mismatch, not rounding. It matters because the
skeleton is the dominant fixed cost: four dense blocks at $d_{ff}$ = 4$d$ spend
75.5 MFLOP/token forward against the middle block's 39.5 MFLOP.

> **ADR-025 — Skeleton block count and width in the FLOPs budget** (proposed)
> Context: `docs/06 §3`'s recurrent widths are reproducible only at two skeleton
> blocks, while `docs/01 §1` specifies $E$ = $F$ = 2. The skeleton is ~65 % of
> the non-expert budget, so the two readings differ by 30–45 % in $d_{ff}$.
> Decision: pin (a) whether $E$ = $F$ = 2 means two blocks each or two in total,
> and (b) the skeleton's $d_{ff}$, which no document states — `costmodel/`
> currently assumes the conventional 4$d$. Then re-derive every width in
> `docs/06 §3` and `docs/01 §1` from `costmodel/`, which both documents already
> name as the source of truth.
> Consequences: every recurrent budget, and the `screen`/`small` per-GPU
> footprint. Bears on H4, H6, ADR-015, ADR-018.

### B. Two streams do not halve $d_{ff}$, and do not fit at `small`

`docs/06 §3`: "switching two streams on (L4, optional) halves every $d_{ff}$
above, down to the $U s_{min}$ floor of 512". The accounting does not do this.
Doubling the streams doubles the skeleton and per-iteration attention costs too,
not only the expert coefficient, so the width left over is **362**, not 833.
362 is below the 512 floor of `docs/01 §4.5`, so the floor binds and the
configuration **overspends its budget by 11 %**.

`docs/06 §3` requires an explicit trade of $k$ or $r$ in exactly this situation
("do not silently exceed the budget") one sentence after claiming the halving
works. Since ADR-020 already made two streams optional and moved L4 to the end
of the ladder, this costs nothing now — but L4 cannot be run as specified.

> **ADR-026 — L4 requires an explicit budget trade** (proposed)
> Context: at `small` with $r$ = 8 there is no $d_{ff}$ ≥ $U s_{min}$ that meets
> the 0.6 GFLOP budget with two streams; the tile floor overspends by 11 %.
> Decision: when L4 runs, trade $r$ (8 → 6) or $k$ (2 → 1) explicitly and log
> it, and strike the "halves every $d_{ff}$" sentence from `docs/06 §3`.
> Consequences: H12 is tested at a different depth or sparsity than the
> single-stream rungs, which must be stated when its result is reported.
> Bears on H12, ADR-020.

### C. `docs/06 §3` dense parameter counts are MHA figures under a GQA spec

The `small` row reads "12 layers, $d$ = 768, 12/4 heads, SwiGLU $d_{ff}$ = 2048
(≈ 85 M non-embedding)". With the stated GQA (12 query, 4 KV heads) the count is
**75.5 M** — 11.2 % low, which breaks the 10 % tolerance the same section sets.
Full MHA gives 84,934,656, i.e. the doc's figure exactly. `medium` shows the
same pattern: GQA 270.5 M, MHA 308.3 M, doc "≈ 300 M".

This is a straightforward correction rather than a decision — the head specs and
the parameter counts simply disagree, and the FLOPs anchors (0.6 / 2.0 GFLOP)
are consistent with **GQA**, so the head spec is the part to keep.

### D. The budget depends on an unstated attention-key convention

`docs/01 §10` gives scores as $2 \cdot 2 d (W + |\mathcal{G}_{visible}|)$.
Neither `docs/01 §10` nor `docs/06 §3` says what $|\mathcal{G}_{visible}|$ is
when budgeting: counting the local window alone gives $d_{ff}$ = 1665, while
un-indexed global attention over a 2048-token training context averages ~1024
more keys and gives 1452 — a **13 % swing**. The doc's own ranges are only
matched by the local-window-only reading.

> **ADR-027 — Key-count convention for FLOPs budgets** (proposed)
> Decision: fix whether budgeted attention-score FLOPs count $W$ alone, or
> $W + |\mathcal{G}_{visible}|$ at the training context, and state the context
> length the anchors assume. Bears on every anchor in `docs/06 §3`.

### E. `docs/01 §10` fabric-bytes line: per stream or both?

The section header reads "both streams unless noted"; the line
"Fabric bytes per token per iteration: $2 k d\, b_{act}$" does not note. The
physical reading is per stream — the 2 is dispatch + combine, each stream's
$d$-vector must reach each of its $k$ experts and return — which makes the
header's default wrong for this line. The two readings differ by exactly 2× on
every fabric number, and hence on S0/S10 and every `docs/02 §3` claim.

`fabric.py` implements the per-stream reading and exposes `per_stream=False` for
the other. Inert at the single-stream default; live as soon as L4 runs. Also
recorded in `experiments/002-nccl/results.md`.
