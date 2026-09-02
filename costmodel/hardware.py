"""The expert step's hardware conditions (`docs/02 §2`, `docs/06 §1`).

`expert_step()` is the function the simulator's per-expert queue consults: it
answers "is this batch big enough to be compute-bound, and does the shape fill
a tile", and returns the step's time either way.
"""

from __future__ import annotations

from dataclasses import dataclass

from costmodel.config import Hardware, Model, Numerics


def b_min(hw: Hardware, num: Numerics) -> float:
    """$b_{min} = (\\phi/\\beta_C)(b_w/2)$ — the compute-bound batch threshold.

    Derivation (`docs/02 §2`): per chip per step the expert does
    $2 b \\cdot 3 d (d_{ff}/U) L_e$ FLOPs against $3 d (d_{ff}/U) L_e b_w$ weight
    bytes, so arithmetic intensity is $2b/b_w$ FLOP/byte independent of every
    shape parameter. Setting that equal to the ridge point gives $b_{min}$.
    """
    return hw.ridge * (num.b_w / 2.0)


def tile_ok(b: int, d_ff: int, hw: Hardware) -> bool:
    """The tile condition: $d_{ff}/U \\ge s_{min}$ **and** $b \\ge s_{min}$."""
    return (d_ff / hw.U) >= hw.s_min and b >= hw.s_min


def min_d_ff(hw: Hardware) -> int:
    """$U s_{min}$ — the hardware floor on expert width (`docs/01 §4.5`)."""
    return hw.U * hw.s_min


@dataclass(frozen=True)
class StepResult:
    """One expert step on one unit."""

    compute_s: float
    """Time if compute-bound."""
    weight_stream_s: float
    """Time to stream the expert's weights from chip memory."""
    local_traffic_s: float
    """Broadcast + reduce time on the unit-local fabric."""
    step_s: float
    """max(compute, weight streaming) + local traffic — the modelled step time."""
    compute_bound: bool
    tile_ok: bool
    utilisation: float
    """Achieved fraction of $\\phi$ over the step."""


def expert_step(
    b: int,
    d_ff: int,
    model: Model,
    hw: Hardware,
    num: Numerics,
) -> StepResult:
    """Time one expert step for a batch of `b` tokens (`docs/02 §2`).

    Weight streaming and compute overlap, so the larger of the two bounds the
    arithmetic; the unit-local broadcast/reduce is charged on top because it
    brackets the step rather than overlapping it in the two-step pipeline
    `docs/02 §2` describes.
    """
    slice_width = d_ff / hw.U
    flops = 2.0 * b * 3.0 * model.d * slice_width * model.experts.L_e
    weight_bytes = 3.0 * model.d * slice_width * model.experts.L_e * num.b_w

    compute_s = flops / hw.phi
    weight_stream_s = weight_bytes / hw.beta_c
    local_bytes = 2.0 * b * model.d * num.b_act
    local_traffic_s = local_bytes / hw.lambda_c if hw.lambda_c > 0 else float("inf")

    step_s = max(compute_s, weight_stream_s) + local_traffic_s
    return StepResult(
        compute_s=compute_s,
        weight_stream_s=weight_stream_s,
        local_traffic_s=local_traffic_s,
        step_s=step_s,
        compute_bound=b >= b_min(hw, num),
        tile_ok=tile_ok(b, d_ff, hw),
        utilisation=(flops / hw.phi) / step_s if step_s > 0 else 0.0,
    )


def latency_floor_per_iteration(
    b: int,
    d_ff: int,
    model: Model,
    hw: Hardware,
    num: Numerics,
) -> float:
    """Step time at the utilisation floor, for the two-stream latency claim.

    `docs/01 §3` (author's Q1, ADR-020): $b_{min}$ counts *rows*, and a token is
    two rows when two streams are on, so a unit at the floor holds $b_{min}/2$
    tokens and its step time halves. Throughput per unit is unchanged. Call this
    with and without `model.two_stream` to report both, which is what `docs/01
    §3` asks `costmodel/` to do.
    """
    return expert_step(b, d_ff, model, hw, num).step_s


def tokens_at_floor(hw: Hardware, num: Numerics, n_streams: int) -> float:
    """Tokens a unit holds at the utilisation floor: $b_{min} / n_{streams}$."""
    return b_min(hw, num) / n_streams
