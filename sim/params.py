"""All simulator parameters in one dataclass (CLAUDE.md), and the loader that fills it.

Hardware numbers come from a scenario YAML in `sim/scenarios/` (every value there carries a
source line, which `check_sources` enforces); the model shape comes from a ladder rung of
`model.config` through `costmodel/`. Nothing here is a hardware constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from costmodel.config import Attention as CmAttention
from costmodel.config import Experts as CmExperts
from costmodel.config import Hardware, Model, Numerics
from costmodel.flops import (
    attention_projection_flops,
    attention_score_flops,
    output_head_flops,
    skeleton_flops,
)
from costmodel.hardware import b_min

SCENARIO_DIR = Path(__file__).parent / "scenarios"

BatchRule = Literal["floor", "roofline"]
Routing = Literal["uniform", "zipf"]


@dataclass(frozen=True)
class SimParams:
    """One simulation. Field groups: scenario (hardware), model shape, scheduler, workload."""

    # ---- scenario: hardware (docs/02 §1), from sim/scenarios/<name>.yaml ----------------
    scenario: str
    n_units: int
    U: int  # chips per unit
    phi: float  # FLOP/s per chip
    beta_c: float  # chip memory bandwidth, bytes/s
    m_c: float  # chip memory, bytes
    lambda_c: float  # unit-local fabric, bytes/s per chip
    s_min: int
    lambda_link: float  # inter-unit link, bytes/s per unit egress
    link_latency_s: float  # per-message latency floor
    b_floor_empirical: int  # occupancy floor of the expert GEMM (local_3060: 003-gemm)
    grouped_eff: tuple[tuple[int, float], ...]  # (tokens per expert, efficiency vs dense)

    # ---- model shape (docs/01 §1 notation) ------------------------------------------------
    rung: str
    d: int
    k: int
    n_e: int
    d_ff: int
    L_e: int
    r: int  # fixed depth in m1
    b_act: float
    b_w: float
    b_kv: float
    attn_proj_flops: float  # per token per middle iteration (Q, K, V, O)
    attn_proj_weight_bytes: float  # shared middle-block attention weights
    skel_in_flops: float  # early blocks, per token
    skel_out_flops: float  # final blocks + output head, per token
    skel_weight_bytes: float  # half of the E + F skeleton weights each way (in / out)
    head_weight_bytes: float  # tied output head
    kv_width: int  # H_kv d_h
    context_tokens: int  # keys visible to a decode token (decision 6)

    # ---- scheduler (docs/02 §5) -----------------------------------------------------------
    batch_rule: BatchRule = "floor"  # which threshold the expert queue fires at (I13/I18)
    w_max_s: float = 1e-3  # queue timeout; NOT calibrated (decision 9, I35)
    multicast: bool = False  # docs/02 §3 hook; m1 refuses True
    depth: Literal["fixed"] = "fixed"  # ragged depth hook; m1 refuses anything else

    # ---- workload --------------------------------------------------------------------------
    z: int = 256  # sequences in flight = decode tokens in flight (closed loop)
    tokens: int = 10_000  # decode tokens to admit in total
    warmup_tokens: int = 1_000  # completions before the measurement window opens
    routing: Routing = "uniform"
    zipf_s: float = 1.0
    seed: int = 0
    util_window_s: float = 1e-2  # bin width of the utilisation time series

    # ---- derived ---------------------------------------------------------------------------
    hardware: Hardware = field(init=False, repr=False)
    numerics: Numerics = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hardware",
            Hardware(
                phi=self.phi, beta_c=self.beta_c, m_c=self.m_c, lambda_c=self.lambda_c,
                U=self.U, s_min=self.s_min,
            ),
        )
        num = Numerics(b_w=self.b_w, b_act=self.b_act, b_kv=self.b_kv)
        object.__setattr__(self, "numerics", num)
        if self.multicast:
            raise NotImplementedError("multicast and in-network reduction: milestone 2 (02 §3)")
        if self.depth != "fixed":
            raise NotImplementedError("ragged depth is milestone 2 (docs/01 §4.7)")
        if not 1 <= self.k <= self.n_e:
            raise ValueError("need 1 <= k <= N_e")
        if self.z < 1 or self.tokens < self.z:
            raise ValueError("need z >= 1 and tokens >= z")

    @property
    def batch_threshold(self) -> int:
        """Tokens at which an expert queue fires (docs/02 §5.1; decision 4)."""
        roofline = math.ceil(b_min(self.hardware, self.numerics))
        return self.b_floor_empirical if self.batch_rule == "floor" else roofline

    @property
    def vector_bytes(self) -> float:
        """One token's stream vector on the fabric, $d\\, b_{act}$ (docs/02 §3)."""
        return self.d * self.b_act

    def expert_of_unit(self, expert: int) -> int:
        """Static placement: expert i lives on unit i mod n_units (decision 3)."""
        return expert % self.n_units

    def owner_of(self, seq: int) -> int:
        """Attention owner of a sequence, fixed for its lifetime (docs/02 §5.1)."""
        return seq % self.n_units


# ---- scenario files -------------------------------------------------------------------------
def check_sources(doc: Any, path: str = "", inherited: bool = False) -> None:
    """Every mapping holding a numeric value must carry `source` or `*_source` in itself or an
    ancestor (sim/scenarios/README.md: every value carries a source line)."""
    if not isinstance(doc, dict):
        return
    here = inherited or any(k == "source" or str(k).endswith("_source") for k in doc)
    for key, val in doc.items():
        sub = f"{path}.{key}" if path else str(key)
        numeric = isinstance(val, int | float) and not isinstance(val, bool)
        if numeric and not here:
            raise ValueError(f"scenario value {sub} has no source line")
        if isinstance(val, list) and not here and any(
            isinstance(v, int | float) and not isinstance(v, bool) for v in val
        ):
            raise ValueError(f"scenario value {sub} has no source line")
        check_sources(val, sub, here)


def load_scenario(name: str) -> dict[str, Any]:
    """Read and source-check `sim/scenarios/<name>.yaml`."""
    doc = yaml.safe_load((SCENARIO_DIR / f"{name}.yaml").read_text())
    check_sources(doc)
    return doc


def _scenario_fields(doc: dict[str, Any]) -> dict[str, Any]:
    chip, fab, unit = doc["chip"], doc["fabric"], doc["unit"]
    table = chip["grouped_gemm"]["efficiency_vs_dense_by_tokens_per_expert"]
    return {
        "scenario": doc["name"],
        "n_units": int(unit["n_units"]),
        "U": int(unit["U"]),
        "phi": float(chip["phi_flops"]),
        "beta_c": float(chip["beta_c_bytes_per_s"]),
        "m_c": float(chip["m_c_bytes"]),
        "lambda_c": float(fab["lambda_c_bytes_per_s"]),
        "s_min": int(chip["s_min"]),
        "lambda_link": float(fab["lambda_link_bytes_per_s"]),
        "link_latency_s": float(fab["latency_floor_s"]),
        "b_floor_empirical": int(chip["b_floor_empirical"]),
        "grouped_eff": tuple(sorted((int(b), float(e)) for b, e in table.items())),
    }


# ---- model shape -----------------------------------------------------------------------------
def _model_fields(rung: str, context_tokens: int | None) -> dict[str, Any]:
    from model.config import ladder  # deferred: model/ is only needed for presets

    cfg = ladder(rung)
    if cfg.arch != "recurrent":
        raise ValueError("the simulator models the recurrent design (L5+); layered rungs have no M")
    if cfg.streams.two_stream:
        raise NotImplementedError("two streams (L4) are milestone 2")
    if cfg.experts.granularity != 1:
        raise NotImplementedError("granularity g > 1 (L7) is milestone 2")
    a = cfg.attention
    cm = Model(
        d=cfg.d, vocab=cfg.vocab, early_blocks=cfg.early_blocks, final_blocks=cfg.final_blocks,
        r_max=cfg.middle.r_max, depth_embed=cfg.middle.depth_embed,
        attn=CmAttention(heads=a.heads, kv_heads=a.kv_heads, head_dim=a.head_dim, window=a.window),
        experts=CmExperts(
            n_e=cfg.router.n_experts, k=cfg.router.k, d_ff=cfg.expert_width(),
            L_e=cfg.experts.layers,
        ),
    )
    b = 2.0 if cfg.numerics.dtype == "bf16" else 4.0
    keys = cfg.context if context_tokens is None else context_tokens
    proj_w = (2 * cm.d * cm.attn.q_width + 2 * cm.d * cm.attn.kv_width) * b
    skel_ff_w = 3 * cm.d * cfg.skeleton_width * b
    blocks = cm.early_blocks + cm.final_blocks
    skel = skeleton_flops(cm, cfg.skeleton_width, 1, keys)
    return {
        "rung": rung,
        "d": cm.d,
        "k": cm.experts.k,
        "n_e": cm.experts.n_e,
        "d_ff": int(cm.experts.d_ff or 0),
        "L_e": cm.experts.L_e,
        "r": cfg.middle.r_max,
        "b_act": b,
        "b_w": b,
        "b_kv": b,
        "attn_proj_flops": attention_projection_flops(cm, 1),
        "attn_proj_weight_bytes": proj_w,
        "skel_in_flops": skel * cm.early_blocks / blocks,
        "skel_out_flops": skel * cm.final_blocks / blocks + output_head_flops(cm),
        "skel_weight_bytes": (proj_w + skel_ff_w) * blocks / 2,
        "head_weight_bytes": cm.vocab * cm.d * b,
        "kv_width": cm.attn.kv_width,
        "context_tokens": keys,
    }


def attention_score_flops_per_token(p: SimParams) -> float:
    """$2 \\cdot 2 d \\cdot n_{keys}$ at the decode position (docs/01 §10, decision 6)."""
    cm = Model(d=p.d)
    return attention_score_flops(cm, n_keys=p.context_tokens, n_streams=1)


def load_params(
    scenario: str = "local_3060",
    rung: str = "L5",
    context_tokens: int | None = None,
    **overrides: Any,
) -> SimParams:
    """Scenario + rung + overrides → `SimParams`. Overrides use the dataclass field names."""
    fields = {**_scenario_fields(load_scenario(scenario)), **_model_fields(rung, context_tokens)}
    params = SimParams(**fields)
    return replace(params, **overrides) if overrides else params
