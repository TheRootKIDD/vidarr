"""FLOPs accounting (`docs/01 §10`). Multiply-add counts as 2.

The section header of `docs/01 §10` reads "per token, per iteration, both
streams unless noted". Every function here takes the stream count explicitly and
returns a per-token figure, so the caller never has to guess which convention a
number is in. See `fabric.py` for the one line where the header and the formula
disagree.
"""

from __future__ import annotations

import math

from costmodel.config import Model


def attention_projection_flops(model: Model, n_streams: int | None = None) -> float:
    """Q and O on every stream; K and V on the context stream only.

    `docs/01 §10`: $2 \\cdot 2 \\cdot 2 d^2$ for Q and O over two streams, plus
    $2 \\cdot 2 d H_{kv} d_h$ for K and V on $c$ alone. Written out per stream so
    the single-stream default (ADR-020) falls out of the same expression.
    """
    s = model.n_streams if n_streams is None else n_streams
    a = model.attn
    qo = 2 * s * (model.d * a.q_width + a.q_width * model.d)
    kv = 2 * 2 * model.d * a.kv_width
    return float(qo + kv)


def attention_score_flops(
    model: Model,
    n_keys: int | None = None,
    n_streams: int | None = None,
) -> float:
    """$2 \\cdot 2 d (W + |\\mathcal{G}_{visible}|)$ per stream (`docs/01 §10`).

    The two factors of 2 are multiply-add and the score/value pair ($QK^\\top$
    then $AV$). `n_keys` overrides the visible key count; by default it is the
    local window $W$ plus the visible part of the global cache $\\mathcal{G}$,
    which is $K_{idx} B_{idx}$ once indexed (ADR-021).
    """
    s = model.n_streams if n_streams is None else n_streams
    a = model.attn
    if n_keys is None:
        g_visible = a.topk * a.block if a.indexed else 0
        n_keys = a.window + g_visible
    return float(2 * 2 * model.d * n_keys * s)


def moe_flops(model: Model, d_ff: int | None = None, n_streams: int | None = None) -> float:
    """$2 k \\cdot 3 d\\, d_{ff} L_e$ per stream, plus the router."""
    s = model.n_streams if n_streams is None else n_streams
    e = model.experts
    width = d_ff if d_ff is not None else e.d_ff
    if width is None:
        raise ValueError("d_ff unset; use costmodel.solve.solve_d_ff() first")
    experts = 2 * e.k * 3 * model.d * width * e.L_e
    return float((experts + router_flops(model)) * s)


def router_flops(model: Model) -> float:
    """$O(d_r \\sqrt{N_e})$ product-key scoring, plus the query projection."""
    e = model.experts
    q_proj = 2 * (model.d + model.depth_embed) * e.d_r
    scoring = 2 * e.d_r * math.isqrt(_ceil_square(e.n_e))
    return float(q_proj + scoring)


def memory_flops(model: Model, n_streams: int | None = None) -> float:
    """$O(d_m \\sqrt{N_m}) + 2 k_m d_v d$ per stream (`docs/01 §10`)."""
    s = model.n_streams if n_streams is None else n_streams
    m = model.mem
    if not m.enabled:
        return 0.0
    q_proj = 2 * (model.d + model.depth_embed) * m.d_m
    lookup = 2 * m.d_m * math.isqrt(_ceil_square(m.n_m))
    combine = 2 * m.k_m * m.d_v * model.d
    return float((q_proj + lookup + combine) * s)


def middle_iteration_flops(
    model: Model,
    d_ff: int | None = None,
    n_streams: int | None = None,
    n_keys: int | None = None,
) -> float:
    """One middle-block iteration for one token: attention + MoE + memory."""
    return (
        attention_projection_flops(model, n_streams)
        + attention_score_flops(model, n_keys, n_streams)
        + moe_flops(model, d_ff, n_streams)
        + memory_flops(model, n_streams)
    )


def skeleton_flops(
    model: Model,
    d_ff_skeleton: int | None = None,
    n_streams: int | None = None,
    n_keys: int | None = None,
) -> float:
    """The $E + F$ unshared blocks, per token. Dense blocks (ADR-006 pending)."""
    s = model.n_streams if n_streams is None else n_streams
    d_ff = d_ff_skeleton if d_ff_skeleton is not None else 4 * model.d
    per_block = (
        attention_projection_flops(model, s)
        + attention_score_flops(model, n_keys, s)
        + 2 * 3 * model.d * d_ff * s
    )
    return float((model.early_blocks + model.final_blocks) * per_block)


def output_head_flops(model: Model) -> float:
    """$2 V d$ — the vocabulary projection, once per token."""
    return float(2 * model.vocab * model.d)


def forward_flops_per_token(
    model: Model,
    d_ff: int | None = None,
    r: int | None = None,
    n_streams: int | None = None,
    n_keys: int | None = None,
    include_head: bool = False,
) -> float:
    """Decode-time forward FLOPs for one token: skeleton + $r$ middle iterations.

    `include_head` is False by default because the anchors of `docs/06 §3` are
    quoted as non-embedding costs; the output head is a further $2 V d$.
    """
    iters = model.r_max if r is None else r
    total = skeleton_flops(model, None, n_streams, n_keys) + iters * middle_iteration_flops(
        model, d_ff, n_streams, n_keys
    )
    return total + (output_head_flops(model) if include_head else 0.0)


def prefill_flops_per_token(
    model: Model,
    d_ff: int | None = None,
    r: int | None = None,
    n_keys: int | None = None,
    include_head: bool = False,
) -> float:
    """Prefill runs the context stream only (`docs/01 §10`).

    With two streams this is ≈ half of decode; with the single-stream default it
    equals decode, which is why ADR-020 removes the note's prefill/decode
    asymmetry along with the second stream.
    """
    return forward_flops_per_token(model, d_ff, r, 1, n_keys, include_head)


def training_flops_per_token(
    model: Model,
    d_ff: int | None = None,
    r: int | None = None,
    n_streams: int | None = None,
    n_keys: int | None = None,
    include_head: bool = False,
    backward_multiplier: float = 3.0,
) -> float:
    """Training cost per token = forward + backward ≈ 3 × forward (`docs/06 §3`)."""
    fwd = forward_flops_per_token(model, d_ff, r, n_streams, n_keys, include_head)
    return backward_multiplier * fwd


def dense_baseline_flops_per_token(
    layers: int,
    d: int,
    d_ff: int,
    q_width: int,
    kv_width: int,
    n_keys: int,
    vocab: int,
    include_head: bool = False,
) -> float:
    """Forward FLOPs per token for the plain dense L0 baseline of `docs/06 §3`.

    `n_keys` is the *average* number of visible keys, which for a causal
    sequence of length $L$ is $(L+1)/2$ — not $L$.
    """
    proj = 2 * (2 * d * q_width + 2 * d * kv_width)
    scores = 2 * 2 * d * n_keys
    ff = 2 * 3 * d * d_ff
    total = layers * (proj + scores + ff)
    return float(total + (2 * vocab * d if include_head else 0))


def _ceil_square(n: int) -> int:
    root = math.isqrt(n)
    return n if root * root == n else (root + 1) ** 2
