# model/ — reference implementation of `docs/01`

**Skeleton. Phase 1 work, not Phase 0.**

Readability over speed (CLAUDE.md). Simulated hardware quantisation (`U`,
`s_min`) is a config knob that sets `d_ff` and reported utilisation — no custom
kernels in Phases 0–2. Cross-GPU expert or tensor parallelism does **not** live
here; it exists only in the S0/S10 measurement scripts (ADR-015).

Unit tests owed by Phase 1: shapes, causal masks, stream visibility (ADR-002),
halting and ragged-depth semantics.
