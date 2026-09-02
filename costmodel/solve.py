"""Derive $d_{ff}$ from a FLOPs budget (`docs/01 §1`, `docs/06 §3`).

`docs/06 §3` makes this module the source of truth for every width quoted there,
and `docs/01 §1` says $d_{ff}$ is *derived* from the budget rather than chosen.
The forward direction is in `flops.py`; this inverts it.

The relation is affine in $d_{ff}$ — the expert term is the only one that
depends on it — so the solve is exact, not iterative.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from costmodel.config import Hardware, Model
from costmodel.flops import (
    attention_projection_flops,
    attention_score_flops,
    memory_flops,
    output_head_flops,
    router_flops,
    skeleton_flops,
)


@dataclass(frozen=True)
class WidthSolution:
    """Result of solving a FLOPs budget for expert width."""

    d_ff_exact: float
    """The unrounded solution."""
    d_ff: int
    """Rounded to `multiple_of` and raised to the $U s_{min}$ floor if given."""
    at_floor: bool
    """True when the hardware floor bound the answer, not the budget."""
    achieved_flops: float
    """Training FLOPs/token actually delivered at `d_ff`."""
    budget_flops: float
    relative_error: float
    """(achieved - budget) / budget. `docs/06 §3` wants the anchors within 10 %."""


def solve_d_ff(
    model: Model,
    budget_flops_per_token: float,
    r: int | None = None,
    n_streams: int | None = None,
    n_keys: int | None = None,
    include_head: bool = False,
    backward_multiplier: float = 3.0,
    hw: Hardware | None = None,
    multiple_of: int = 1,
) -> WidthSolution:
    """Solve for the expert width that spends exactly `budget_flops_per_token`.

    The budget is a **training** cost per token (`docs/06 §3`:
    forward + backward ≈ 3 × forward). Pass `backward_multiplier=1.0` to treat
    it as a forward budget instead.

    `hw` applies the $d_{ff} \\ge U s_{min}$ sizing invariant of `docs/01 §4.5`;
    when the floor binds, `at_floor` is True and the achieved cost *exceeds* the
    budget — the caller must then trade $k$ or $r$ explicitly rather than
    silently overspend (`docs/06 §3`).
    """
    iters = model.r_max if r is None else r
    s = model.n_streams if n_streams is None else n_streams
    e = model.experts

    forward_budget = budget_flops_per_token / backward_multiplier

    # Everything that does not depend on d_ff.
    fixed = skeleton_flops(model, None, s, n_keys)
    if include_head:
        fixed += output_head_flops(model)
    per_iter_fixed = (
        attention_projection_flops(model, s)
        + attention_score_flops(model, n_keys, s)
        + router_flops(model) * s
        + memory_flops(model, s)
    )
    fixed += iters * per_iter_fixed

    # Coefficient of d_ff: 2*k*3*d*L_e per stream, once per iteration.
    coeff = iters * 2 * e.k * 3 * model.d * e.L_e * s
    if coeff <= 0:
        raise ValueError("expert term has zero coefficient; check k, L_e, r")

    d_ff_exact = (forward_budget - fixed) / coeff
    if d_ff_exact <= 0:
        raise ValueError(
            f"budget {budget_flops_per_token:.3g} FLOP/token is already spent by the "
            f"non-expert terms ({backward_multiplier * fixed:.3g}); "
            "reduce r, the window, or the skeleton"
        )

    d_ff = int(round(d_ff_exact / multiple_of) * multiple_of)
    at_floor = False
    if hw is not None:
        floor = hw.U * hw.s_min
        if d_ff < floor:
            d_ff, at_floor = floor, True

    achieved = backward_multiplier * (fixed + coeff * d_ff)
    return WidthSolution(
        d_ff_exact=d_ff_exact,
        d_ff=d_ff,
        at_floor=at_floor,
        achieved_flops=achieved,
        budget_flops=budget_flops_per_token,
        relative_error=(achieved - budget_flops_per_token) / budget_flops_per_token,
    )


def with_d_ff(model: Model, d_ff: int) -> Model:
    """Return a copy of `model` with the expert width pinned."""
    return replace(model, experts=replace(model.experts, d_ff=d_ff))
