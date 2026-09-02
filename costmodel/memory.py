"""KV and unit memory (`docs/01 §10`, `docs/02 §7`, §8; ADR-017, ADR-021, ADR-022).

After v2 of the note the cache is split in two, and the split is the whole point
of P15/P16: a **persistent** global cache $\\mathcal{G}$ holding one entry per
completed token, and **transient** per-iteration caches that exist only for
tokens inside the local range $W$.
"""

from __future__ import annotations

from costmodel.config import Model, Numerics


def kv_bytes_per_entry(model: Model, num: Numerics) -> float:
    """$2 H_{kv} d_h b_{kv}$ — one K/V pair for one token at one iteration."""
    return 2.0 * model.attn.kv_width * num.b_kv


def persistent_kv_bytes_per_token(model: Model, num: Numerics) -> float:
    """One $\\mathcal{G}$ entry per token, from its final middle-layer vector.

    ADR-021: this is the only thing far tokens ever attend to, so it is the
    cache that grows with context. It does **not** scale with $r_{max}$ — that
    is the ~$r_{max}$× reduction v2 P7 claims.
    """
    return kv_bytes_per_entry(model, num)


def transient_kv_bytes_per_sequence(
    model: Model,
    num: Numerics,
    r: int | None = None,
) -> float:
    """$\\le W r_{max} \\cdot 2 H_{kv} d_h b_{kv}$ per in-flight sequence.

    Per-iteration caches for the tokens still in the local window. ADR-022 lets
    the window slide across a segment boundary, so the previous segment's caches
    are retained until they leave the window — the bound is $W$ tokens' worth
    regardless of where the boundary falls, which is why this does not need a
    segment-count term.
    """
    iters = model.r_max if r is None else r
    return model.attn.window * iters * kv_bytes_per_entry(model, num)


def kv_bytes_naive_per_token(model: Model, num: Numerics, r: int | None = None) -> float:
    """The pre-v2 accounting: a global cache entry per token *per iteration*.

    Kept to quantify what ADR-021 buys — the ratio against
    `persistent_kv_bytes_per_token` is the "almost the full 128×" of v2 P7.
    """
    iters = model.r_max if r is None else r
    return iters * kv_bytes_per_entry(model, num)


def kv_reduction_factor(model: Model, r: int | None = None) -> float:
    """How much smaller the persistent cache is than the pre-v2 one: $r_{max}$."""
    return float(model.r_max if r is None else r)


def expert_bytes(params_per_expert: int, num: Numerics) -> float:
    """Weight bytes for one expert at the configured numerics."""
    return params_per_expert * num.b_w


def experts_resident_per_unit(
    unit_memory_bytes: float,
    params_per_expert: int,
    num: Numerics,
    reserve_fraction: float = 0.2,
) -> int:
    """How many experts fit in a unit's memory, after a reserve for KV and
    activations. `docs/02 §8`; answers the capacity half of Q6."""
    usable = unit_memory_bytes * (1.0 - reserve_fraction)
    return int(usable // expert_bytes(params_per_expert, num))
