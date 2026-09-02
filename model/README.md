# model/ — reference implementation of `docs/01`

Readability over speed (CLAUDE.md): explicit boolean masks, a Python loop over
experts, rotary as a function. No custom kernels; cross-GPU expert or tensor
parallelism lives only in the S0/S10 measurement scripts (ADR-015).

| File | Spec section | What |
|---|---|---|
| `config.py` | `01 §11` | dataclass mirror of the YAML schema; `d_ff` derived through `costmodel/` (ADR-025/027); `ladder(step)` presets for `docs/03 §2` |
| `layers.py` | `01 §4.1–4.5` | `DepthNorm` (RMSNorm + AdaLN modulation by $e_j$), rotary, `Attention` (GQA over an explicit key set: queries and K/V rows are separate arguments), `SwiGLU`, `Expert` ($L_e$ ∈ {1, 2}) |
| `router.py` | `01 §4.4` | top-$k$ with expert keys, selection bias (aux-loss-free load controller), small Switch-style aux loss, optional $e_j$ input (E-Q2) |
| `moe.py` | `01 §4.5` | dropless dispatch (gather / run / scatter-add); granularity $g$ → $N_e g$ sub-experts of width $d_{ff}/g$ |
| `block.py` | `01 §5`, L0–L3 | pre-norm block, sequential or parallel form, dense or MoE FF; streams: K/V from $c$ only, $p$ queries $c$ (ADR-002); prediction-only blocks |
| `middle.py` | `01 §4` | the shared block $M$: depth conditioning, local per-iteration K/V + global final-vector cache $\mathcal{G}$ (ADR-021), halting head, ragged depth with state copy and K/V frozen once (ADR-010/011) |
| `mtp.py` | `01 §7` | independent MTP heads with one auxiliary head per position per step (ADR-012) |
| `sparsity.py` | `01 §8` | fixed 2:4 magnitude masks on expert weights (ADR-019), quality-only |
| `model.py` | `01 §2` | `BigMoE`: layered (L0–L3) or recurrent (L5+); segment loop for `final_vector`; chunked tied-head cross-entropy; per-iteration activation checkpointing (`02 §13.4`) |

**Semantics pinned by `tests/test_model_semantics.py`** (CPU, toy size): every rung builds and steps; derived widths and parameter counts match `costmodel/` (< 0.5 %); causal masks and the sliding window; no future leakage for L0/L2/L5/L5d; the global cache reaches later segments only, with stop-gradient by default (ADR-023); the context stream never sees the prediction stream, and L4b's $p_t$ sees $c_{<t}$ only; halted tokens keep their state and a single frozen K/V entry; MTP subsampling; router gates, loads and the bias controller.

**Implementation decisions not spelled out in `docs/01`** (recorded as ADR-029):
1. The $\mathcal{G}$ entry is $W_k/W_v$ of the shared block applied to a dedicated RMSNorm of the final vector (no depth modulation); with `segment_memory_grad: stop` the *input* vector is detached so the norm and projections still train.
2. In segment mode the early/final skeleton blocks attend causally within the current segment only; only $M$ sees $\mathcal{G}$ ("the final vector is the only thing far tokens ever attend to").
3. `per_iteration` global source = plain causal attention over the whole context at each iteration (the L5 baseline); `final_vector` = window $W$ inside the segment plus $\mathcal{G}$, one softmax (ADR-008's gated alternative is not implemented).
4. A halted token's K/V is frozen at the first iteration after halting (with that iteration's $e_j$ modulation) and reused unchanged afterwards.
5. The halting head starts at a low probability (bias −2) and, in fixed-$r$ training, contributes a zero-weighted term so DDP sees every parameter.
6. Idle experts run one zero-weighted token per step for the same reason.
7. Training-time memory knobs (`checkpoint_iterations`, `ce_chunk_tokens`) do not change semantics.

Not implemented in Phase 1 and refused by `ModelCfg.validate()`: the index (L9), knowledge memory (L8), tenant adapters (L11), per-iteration attention LoRA (L5b), ADR-022 window crossing, L6b K/V sharing.

## Training

`scripts/train/train.py` — DDP over the four cards, 0.5 M-token batches by
accumulation, AdamW (6e-4, β 0.9/0.95, wd 0.1 on matrices only, no decay on
router keys — `02 §13.6`), 2 % warm-up, cosine to 10 %, bf16 autocast, grad
clip 1.0, checkpoints ≤ 30 min apart, held-out eval on a 2 M-token subset
every 100 steps and on the full 11.5 M at the end. Per-GPU memory at `small`
(`014-train-step-rungs`): micro-batch 4 for L0 and the recurrent rungs, 2 for
the 271 M-parameter layered MoE rungs (L1–L3).

    python -m scripts.train.train --rung L0 --id 100-l0-screen --tokens 1e9
