"""Parameter counts (`docs/01 §4.5`, §4.6, §10).

Pure functions. Every count is exact for the structure described in `docs/01`;
norms and biases are excluded unless a docstring says otherwise, because they
are below the resolution of the anchors in `docs/06 §3`.
"""

from __future__ import annotations

import math

from costmodel.config import Attention, Model


def attention_params(d: int, attn: Attention) -> int:
    """Q, K, V, O projections for one attention block.

    GQA: K and V project to $H_{kv} d_h$, not to $d$. Setting
    `kv_heads == heads` recovers plain MHA — the difference is ~11 % of a
    `small` model's non-embedding parameters, so it is never incidental.
    """
    q = d * attn.q_width
    k = d * attn.kv_width
    v = d * attn.kv_width
    o = attn.q_width * d
    return q + k + v + o


def swiglu_params(d: int, d_ff: int) -> int:
    """Gate, up and down projections of one SwiGLU feed-forward block."""
    return 3 * d * d_ff


def expert_params(model: Model, d_ff: int | None = None) -> int:
    """$L_e \\cdot 3 d\\, d_{ff}$ — parameters in one expert (`docs/01 §4.5`)."""
    width = _resolved_d_ff(model, d_ff)
    return model.experts.L_e * swiglu_params(model.d, width)


def experts_total_params(model: Model, d_ff: int | None = None) -> int:
    """All $N_e$ experts of the middle block."""
    return model.experts.n_e * expert_params(model, d_ff)


def router_params(model: Model) -> int:
    """Product-key router (`docs/01 §4.4`, ADR-004).

    A query projection from the (depth-conditioned) state plus two half-width
    key tables of $\\sqrt{N_e}$ entries each.
    """
    e = model.experts
    q_proj = (model.d + model.depth_embed) * e.d_r
    keys = 2 * math.isqrt(_ceil_square(e.n_e)) * (e.d_r // 2)
    return q_proj + keys


def memory_params(model: Model) -> int:
    """Product-key knowledge memory (`docs/01 §4.6`). Zero when disabled.

    Values dominate: $N_m d_v$. Keys are product-key factorised, so the key
    table is $\\sqrt{N_m} d_m$ rather than $N_m d_m$.
    """
    m = model.mem
    if not m.enabled:
        return 0
    q_proj = (model.d + model.depth_embed) * m.d_m
    keys = 2 * math.isqrt(_ceil_square(m.n_m)) * (m.d_m // 2)
    values = m.n_m * m.d_v
    out_proj = m.d_v * model.d
    return q_proj + keys + values + out_proj


def middle_block_params(model: Model, d_ff: int | None = None) -> int:
    """The one shared middle block $M$: attention + router + experts + memory."""
    return (
        attention_params(model.d, model.attn)
        + router_params(model)
        + experts_total_params(model, d_ff)
        + memory_params(model)
    )


def skeleton_params(model: Model, d_ff_skeleton: int | None = None) -> int:
    """The $E$ early and $F$ final blocks (unshared weights, `docs/01 §2`).

    Dense transformer blocks. `d_ff_skeleton` defaults to $4d$, the conventional
    ratio; ADR-006 (early/final blocks dense vs small local MoE) is pending, so
    this is the dense reading.
    """
    d_ff = d_ff_skeleton if d_ff_skeleton is not None else 4 * model.d
    per_block = attention_params(model.d, model.attn) + swiglu_params(model.d, d_ff)
    return (model.early_blocks + model.final_blocks) * per_block


def embedding_params(model: Model, tied: bool = True) -> int:
    """Token embedding, plus an untied output head when `tied` is False."""
    emb = model.vocab * model.d
    return emb if tied else 2 * emb


def total_params(
    model: Model,
    d_ff: int | None = None,
    tied: bool = True,
    d_ff_skeleton: int | None = None,
) -> dict[str, int]:
    """Full breakdown. Keys: `experts`, `middle_other`, `skeleton`, `embedding`,
    `non_embedding`, `total`."""
    experts = experts_total_params(model, d_ff)
    middle_other = middle_block_params(model, d_ff) - experts
    skeleton = skeleton_params(model, d_ff_skeleton)
    embedding = embedding_params(model, tied)
    non_embedding = experts + middle_other + skeleton
    return {
        "experts": experts,
        "middle_other": middle_other,
        "skeleton": skeleton,
        "embedding": embedding,
        "non_embedding": non_embedding,
        "total": non_embedding + embedding,
    }


def dense_baseline_params(
    layers: int,
    d: int,
    attn: Attention,
    d_ff: int,
    vocab: int,
    tied: bool = True,
) -> dict[str, int]:
    """A plain dense transformer — the L0 baseline of `docs/06 §3`.

    Separate from `total_params` because the baseline has no middle block, no
    router and no experts: it is `layers` identical attention+SwiGLU blocks.
    """
    per_layer = attention_params(d, attn) + swiglu_params(d, d_ff)
    non_embedding = layers * per_layer
    embedding = vocab * d * (1 if tied else 2)
    return {
        "non_embedding": non_embedding,
        "embedding": embedding,
        "total": non_embedding + embedding,
    }


def _resolved_d_ff(model: Model, override: int | None) -> int:
    if override is not None:
        return override
    if model.experts.d_ff is None:
        raise ValueError(
            "d_ff is not set on the config and no override was given; "
            "derive it with costmodel.solve.solve_d_ff() first"
        )
    return model.experts.d_ff


def _ceil_square(n: int) -> int:
    """Round `n` up to a perfect square so product keys factorise exactly."""
    root = math.isqrt(n)
    return n if root * root == n else (root + 1) ** 2
