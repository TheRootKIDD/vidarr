"""Fabric traffic (`docs/01 §10`, `docs/02 §2`).

**Known spec ambiguity.** `docs/01 §10` gives fabric bytes per token per
iteration as $2 k d\\, b_{act}$ under a section header reading "both streams
unless noted", and that line does not note. Two readings:

1. *per stream* — the 2 is up (dispatch) + down (combine), and two streams
   double it. This is the physical reading: each stream's $d$-vector must reach
   each of its $k$ experts and come back.
2. *both streams* — the 2 is the stream count, and a single-stream model sends
   $k d\\, b_{act}$.

They differ by exactly 2× on every fabric number. This module implements
reading 1 and exposes the choice as `per_stream`, so a caller can price the
other reading without editing code. Recorded rather than silently resolved
(CLAUDE.md: the doc wins until an ADR says otherwise); see the open question
raised in `experiments/002-nccl/results.md`.
"""

from __future__ import annotations

from costmodel.config import Hardware, Model, Numerics


def fabric_bytes_per_token_iteration(
    model: Model,
    num: Numerics,
    n_streams: int | None = None,
    per_stream: bool = True,
) -> float:
    """$2 k d\\, b_{act}$ — dispatch plus combine, before multicast dedup.

    `docs/01 §10` notes this is the figure *before* multicast and reduction
    dedup; co-activation placement (`docs/02 §4`) reduces it, and by how much is
    a simulator result, not a closed form.
    """
    s = model.n_streams if n_streams is None else n_streams
    base = 2.0 * model.experts.k * model.d * num.b_act
    return base * s if per_stream else base


def unit_local_bytes_per_step(b: int, model: Model, num: Numerics) -> float:
    """$2 b d\\, b_{act}$ — broadcast in, reduce out, for a batch of `b` tokens.

    `docs/02 §2`: the unit-internal cost of running one tensor-parallel expert
    step over $U$ chips.
    """
    return 2.0 * b * model.d * num.b_act


def tp_bandwidth_required(model: Model, hw: Hardware, num: Numerics, d_ff: int) -> float:
    """The $\\lambda_C$ at which unit-internal ring time equals compute time.

    `docs/02 §2` requires $\\lambda_C \\gg \\phi\\, b_{act} / (3 (d_{ff}/U) L_e)$.
    This returns the equality point, so a caller compares its measured
    $\\lambda_C$ against it and reports the ratio. Independent of batch: both
    sides scale with $b$.
    """
    slice_width = d_ff / hw.U
    return hw.phi * num.b_act / (3.0 * slice_width * model.experts.L_e)


def tp_bandwidth_ratio(model: Model, hw: Hardware, num: Numerics, d_ff: int) -> float:
    """Measured $\\lambda_C$ over the requirement. < 1 means communication-bound."""
    return hw.lambda_c / tp_bandwidth_required(model, hw, num, d_ff)
