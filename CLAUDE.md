# CLAUDE.md — bigmoe

Research project: an LLM architecture derived from the 19-point note "The big-DC MoE LLM design" (`docs/source/`). Everything from the note is a hypothesis until measured.

## Start of every session
1. Read `README.md` (map of the project), then the doc relevant to the task.
2. Check `docs/04-decision-log.md` — accepted ADRs are binding until a new ADR supersedes them.
3. Check `docs/03-research-plan.md` §5 for the current phase.

## Hard rules
- **Spec first.** Behaviour lives in `docs/01` (model) and `docs/02` (systems). Change the doc, add an ADR to `docs/04`, then change code. If code and doc disagree, the doc wins until an ADR says otherwise.
- **Ablation ladder is the order of work.** Never switch on two new ladder features in one experiment unless the ladder step says so.
- **Baselines travel with every experiment**: matched-FLOPs dense and matched-FLOPs fine-grained MoE at the same token budget.
- **Notation is fixed** in `docs/01 §1`. Use `d, U, k, N_e, r_t, c_t, p_t, E/M/F, W` etc. Don't invent parallel names.
- **Report units, not adjectives.** FLOPs/token, bytes/token/iteration on the fabric, latency/iteration, tokens in flight, utilisation. "Faster" is not a result.
- **Hypotheses by number** (H1–H19 map to the note's points P1–P19). Every `results.md` states which H it bears on and whether it supports, weakens, or is silent on it.
- **Experiments are append-only.** `experiments/<id>/` with `config.yaml` and `results.md` (numbers, seeds, hardware, wall-clock, one-paragraph interpretation). New run → new id.
- **Simulation is the source of truth for systems claims.** No hand-wavy network arguments; run `sim/`.
- **Unknown → question, not guess.** Add to `docs/04 §Open questions`.
- **Don't "fix" the note silently.** Disagreements go in `docs/00` or `docs/04`.
- **Hardware numbers are scenario inputs**, never constants in code. Each number in `sim/scenarios/*.yaml` carries a source line.

## Compute envelope (`docs/06`)
- One workstation: 4 × RTX 3060 12 GB (Ampere, SM 8.6), Threadripper PRO 3945WX (12 cores), 128 GB DDR4, no NVLink, no PCIe P2P — NCCL bounces through host memory.
- Sizes: `screen` (1 B tokens, 1 seed) → `small` (2.5 B, 2 seeds) → `medium` (7 B, 1 seed, FSDP). No config may exceed 10 GB per GPU. No run > 1 node-day without a finished `screen` of the same config.
- Training uses DDP with all experts replicated at `screen`/`small`; cross-GPU expert or tensor parallelism exists only in the measurement scripts for S0/S10.
- Ampere has no FP8/FP4: those are fake-quant, quality-only. 2:4 sparsity is real at inference shapes (`bench_sparse24`).
- Per-iteration external retrieval is not feasible in training here: L8 = GPU product-key memory; L8b = chunk-level retrieval with offline-precomputed neighbours.
- Use measured numbers from `scripts/bench/` (recorded in `sim/scenarios/local_3060.yaml`) for every budget and throughput claim; the nominal specs in `docs/06 §1` are placeholders.

## Code conventions
- Python ≥ 3.11, PyTorch. Type hints everywhere. `ruff` clean. `pytest` for `sim/`, `costmodel/`, and unit tests of `model/` blocks (shapes, causal masks, stream visibility, halting/ragged-depth semantics).
- `model/` is a *reference implementation*: readability over speed. Simulated hardware quantisation (`U`, `s_min`) is a config knob that sets `d_ff` and reported utilisation — no custom kernels in Phases 0–2.
- `sim/` is a discrete-event simulator (SimPy or hand-rolled event loop). All parameters in one dataclass; every scenario is a YAML in `sim/scenarios/`. If it becomes too slow, port the hot loop, not the model.
- `costmodel/` functions are pure and unit-tested against the worked examples in `docs/01 §10` and `docs/02 §2`.
- Commit prefixes: `docs:`, `model:`, `sim:`, `cost:`, `exp:`, `chore:`.

## Current phase
Phase 0 — spec hardening, literature refresh (`docs/05`, including the author's linked in-depth post), cost model, eval-suite and corpus choice, `scripts/bench/` → `sim/scenarios/local_3060.yaml`. No training runs yet except `bench_train_step`.
