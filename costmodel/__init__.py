"""Pure FLOPs / bytes / memory functions for `docs/01` and `docs/02`.

Unit-tested against the worked examples in `docs/01 §10`, `docs/02 §2` and the
anchor sizes of `docs/06 §3`. Hardware numbers are scenario inputs, never
constants here (CLAUDE.md).
"""

from costmodel.config import Attention, Experts, Hardware, Memory, Model, Numerics
from costmodel.fabric import (
    fabric_bytes_per_token_iteration,
    tp_bandwidth_ratio,
    tp_bandwidth_required,
    unit_local_bytes_per_step,
)
from costmodel.flops import (
    dense_baseline_flops_per_token,
    forward_flops_per_token,
    middle_iteration_flops,
    prefill_flops_per_token,
    training_flops_per_token,
)
from costmodel.hardware import (
    StepResult,
    b_min,
    expert_step,
    min_d_ff,
    tile_ok,
    tokens_at_floor,
)
from costmodel.memory import (
    experts_resident_per_unit,
    kv_bytes_naive_per_token,
    kv_bytes_per_entry,
    kv_reduction_factor,
    persistent_kv_bytes_per_token,
    transient_kv_bytes_per_sequence,
)
from costmodel.params import (
    dense_baseline_params,
    expert_params,
    experts_total_params,
    total_params,
)
from costmodel.solve import WidthSolution, budget_n_keys, solve_d_ff, with_d_ff

__all__ = [
    "Attention",
    "Experts",
    "Hardware",
    "Memory",
    "Model",
    "Numerics",
    "StepResult",
    "WidthSolution",
    "b_min",
    "dense_baseline_flops_per_token",
    "dense_baseline_params",
    "expert_params",
    "expert_step",
    "experts_resident_per_unit",
    "experts_total_params",
    "fabric_bytes_per_token_iteration",
    "forward_flops_per_token",
    "kv_bytes_naive_per_token",
    "kv_bytes_per_entry",
    "kv_reduction_factor",
    "middle_iteration_flops",
    "min_d_ff",
    "persistent_kv_bytes_per_token",
    "prefill_flops_per_token",
    "budget_n_keys",
    "solve_d_ff",
    "tile_ok",
    "tokens_at_floor",
    "total_params",
    "tp_bandwidth_ratio",
    "tp_bandwidth_required",
    "training_flops_per_token",
    "transient_kv_bytes_per_sequence",
    "unit_local_bytes_per_step",
    "with_d_ff",
]
