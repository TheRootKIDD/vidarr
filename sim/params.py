"""All simulator parameters in one dataclass (CLAUDE.md), and the loader that fills it.

Hardware numbers come from a scenario YAML in `sim/scenarios/` (every value there carries a
source line, which `check_sources` enforces); the model shape comes from a ladder rung of
`model.config` through `costmodel/`. Nothing here is a hardware constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from dataclasses import fields as fields_of
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
BatchPolicy = Literal["queue", "round"]
Compose = Literal["max", "sum"]
Routing = Literal["uniform", "zipf", "empirical"]


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
    skel_in_weight_bytes: float  # weights read by the skel_in job (recurrent: half of E + F)
    skel_out_weight_bytes: float  # by skel_out, without the head (recurrent: the other half)
    head_weight_bytes: float  # tied output head
    kv_width: int  # H_kv d_h
    context_tokens: int  # keys visible to a decode token (decision 6)
    skel_in_layers: int = 2  # attention layers whose KV the skel_in job reads (E blocks)
    skel_out_layers: int = 2  # ... and the skel_out job (F blocks)
    # per cached key per attention layer: score FLOPs and KV bytes. None: the recurrent
    # model's 2 · 2 d and 2 H_kv d_h b_kv. DeepSeek-V3 (MLA, ADR-052 §3) sets both
    score_flops_per_key: float | None = None
    kv_bytes_per_key_layer: float | None = None
    # bytes per element of a dispatched / returned row; None: b_act (DeepSeek-V3: FP8 + scales
    # out, BF16 back)
    b_dispatch: float | None = None
    b_combine: float | None = None
    # scenario (decision 14): achieved rates of the owner's GEMMs and of the attention core
    # (None: phi), and the attention core's KV-read bandwidth (None: beta_c)
    phi_owner: float | None = None
    phi_attn: float | None = None
    beta_attn: float | None = None
    # two-tier fabric (decision 14): units in nodes of `node_size` (0: one tier). Rows within a
    # node cross lambda_intra, unless `intra_via_nic`, when every off-unit row uses lambda_link
    node_size: int = 0
    lambda_intra: float | None = None
    intra_via_nic: bool = False

    # ---- scheduler (docs/02 §5) -----------------------------------------------------------
    batch_rule: BatchRule = "floor"  # which threshold the expert queue fires at (I13/I18)
    w_max_s: float = 1e-3  # queue timeout; NOT calibrated (decision 9, I35)
    # "queue": threshold + timeout (02 §5.1). "round": S0's bulk-synchronous barrier (ADR-049 §5,
    # decision 11): each owner runs all its tokens as one job per stage, and each expert fires
    # once per iteration when every token's dispatch for that iteration has arrived
    batch_policy: BatchPolicy = "queue"
    multicast: bool = False  # docs/02 §3 hook; m1 refuses True
    depth: Literal["fixed"] = "fixed"  # ragged depth hook; m1 refuses anything else

    # software stack (ADR-050, decision 12): fixed cost per job kind, how it composes with the
    # roofline, and the latency of one collective (round policy). Absent = 0, not modelled
    t0_skel_in_s: float = 0.0
    t0_attn_s: float = 0.0
    t0_post_s: float = 0.0
    t0_skel_out_s: float = 0.0
    t0_expert_s: float = 0.0
    compose: Compose = "sum"
    t_coll_s: float | None = None  # None: the per-message latency floor stands in
    # owner-stage efficiency vs batch (ADR-051, decision 13): per owner kind, (sequences, eta)
    # points; the work term is roofline / eta. Absent kind or empty = 1 (peak)
    owner_eff: tuple[tuple[str, tuple[tuple[int, float], ...]], ...] = ()

    # ---- workload --------------------------------------------------------------------------
    z: int = 256  # sequences in flight = decode tokens in flight (closed loop)
    tokens: int = 10_000  # decode tokens to admit in total
    warmup_tokens: int = 1_000  # completions before the measurement window opens
    routing: Routing = "uniform"
    zipf_s: float = 1.0
    route_probs: tuple[float, ...] = ()  # routing = "empirical": per-expert selection weights
    # how the round policy draws routes: "sequential" (numpy choice per token, the S0 runs) or
    # "bulk" (one Gumbel-top-k draw per round: the same law, k distinct experts with
    # Plackett-Luce weights, at a fraction of the cost; ADR-052's runs)
    route_draw: Literal["sequential", "bulk"] = "sequential"
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
                phi=self.phi,
                beta_c=self.beta_c,
                m_c=self.m_c,
                lambda_c=self.lambda_c,
                U=self.U,
                s_min=self.s_min,
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
        if self.batch_policy == "round" and (self.tokens % self.z or self.warmup_tokens % self.z):
            raise ValueError("round policy: tokens and warmup_tokens must be multiples of z")
        if self.routing == "empirical" and len(self.route_probs) != self.n_e:
            raise ValueError("routing = empirical needs route_probs with N_e entries")
        if self.route_draw == "bulk" and self.batch_policy != "round":
            raise ValueError("route_draw = bulk is for the round policy")
        if self.node_size:
            if self.batch_policy != "round":
                raise NotImplementedError("the two-tier fabric is modelled under the round policy")
            if self.n_units % self.node_size:
                raise ValueError("n_units must be a multiple of node_size")
            if self.lambda_intra is None and not self.intra_via_nic:
                raise ValueError("node_size needs lambda_intra, or intra_via_nic")
        for kind, table in self.owner_eff:
            if kind not in OWNER_KINDS:
                raise ValueError(f"owner_eff: {kind} is not an owner kind {OWNER_KINDS}")
            if not table or any(not 0.0 < e <= 1.0 for _, e in table):
                raise ValueError(f"owner_eff[{kind}]: need a non-empty table with 0 < eta <= 1")

    def t0(self, kind: str) -> float:
        return float(getattr(self, f"t0_{kind}_s"))

    def job_time(self, kind: str, roofline_s: float) -> float:
        """compose(t0, roofline) for one job (decision 12)."""
        t0 = self.t0(kind)
        return max(t0, roofline_s) if self.compose == "max" else t0 + roofline_s

    def owner_efficiency(self, kind: str, n: int) -> float:
        """η of an owner kind at n sequences: the scenario's table, linear in log2(n), clamped
        at the ends; 1 where the scenario has none (ADR-051 §1)."""
        table = dict(self.owner_eff).get(kind)
        return log2_interp(table, n) if table else 1.0

    @property
    def collective_s(self) -> float:
        return self.link_latency_s if self.t_coll_s is None else self.t_coll_s

    @property
    def batch_threshold(self) -> int:
        """Tokens at which an expert queue fires (docs/02 §5.1; decision 4)."""
        roofline = math.ceil(b_min(self.hardware, self.numerics))
        return self.b_floor_empirical if self.batch_rule == "floor" else roofline

    @property
    def vector_bytes(self) -> float:
        """One token's stream vector on the fabric, $d\\, b_{act}$ (docs/02 §3)."""
        return self.d * self.b_act

    @property
    def dispatch_bytes(self) -> float:
        """One dispatched row (round policy): d b_dispatch, b_act when unset."""
        return self.d * (self.b_act if self.b_dispatch is None else self.b_dispatch)

    @property
    def return_bytes(self) -> float:
        return self.d * (self.b_act if self.b_combine is None else self.b_combine)

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
        if (
            isinstance(val, list)
            and not here
            and any(isinstance(v, int | float) and not isinstance(v, bool) for v in val)
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
        **_optional(chip, "phi_owner", "phi_owner_flops"),
        **_optional(chip, "phi_attn", "phi_attn_flops"),
        **_optional(chip, "beta_attn", "beta_attn_bytes_per_s"),
        **_optional(fab, "lambda_intra", "lambda_intra_bytes_per_s"),
        "node_size": int(fab.get("node_size", 0)),
        "intra_via_nic": bool(fab.get("intra_via_nic", False)),
        **_software_fields(doc.get("software")),
    }


def _optional(section: dict[str, Any], field_name: str, key: str) -> dict[str, Any]:
    return {field_name: float(section[key])} if key in section else {}


def _software_fields(sw: dict[str, Any] | None) -> dict[str, Any]:
    """The scenario's `software` section (ADR-050 §4); a scenario without one models none."""
    if not sw:
        return {}
    t0 = sw["t0_s"]
    out: dict[str, Any] = {f"t0_{k}_s": float(t0[k]) for k in SOFTWARE_KINDS}
    out["compose"] = str(sw["compose"])
    out["t_coll_s"] = float(sw["t_coll_s"])
    eff = sw.get("owner_efficiency") or {}
    out["owner_eff"] = tuple(
        (kind, tuple(sorted((int(n), float(e)) for n, e in eff[kind].items())))
        for kind in OWNER_KINDS
        if kind in eff
    )
    return out


SOFTWARE_KINDS = ("skel_in", "attn", "post", "skel_out", "expert")
OWNER_KINDS = ("skel_in", "attn", "post", "skel_out")


def log2_interp(table: tuple[tuple[int, float], ...], b: int) -> float:
    """A (batch, value) table read at b: linear in log2(b) between points, clamped at the ends."""
    if b <= table[0][0]:
        return table[0][1]
    if b >= table[-1][0]:
        return table[-1][1]
    for (b0, e0), (b1, e1) in zip(table, table[1:], strict=False):
        if b0 <= b <= b1:
            f = (math.log2(b) - math.log2(b0)) / (math.log2(b1) - math.log2(b0))
            return e0 + f * (e1 - e0)
    raise AssertionError("unreachable")


# ---- model shape -----------------------------------------------------------------------------
def _model_fields(rung: str, context_tokens: int | None) -> dict[str, Any]:
    if rung == "deepseek_v3":
        return _deepseek_v3_fields(context_tokens)
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
        d=cfg.d,
        vocab=cfg.vocab,
        early_blocks=cfg.early_blocks,
        final_blocks=cfg.final_blocks,
        r_max=cfg.middle.r_max,
        depth_embed=cfg.middle.depth_embed,
        attn=CmAttention(heads=a.heads, kv_heads=a.kv_heads, head_dim=a.head_dim, window=a.window),
        experts=CmExperts(
            n_e=cfg.router.n_experts,
            k=cfg.router.k,
            d_ff=cfg.expert_width(),
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
        "skel_in_weight_bytes": (proj_w + skel_ff_w) * blocks / 2,
        "skel_out_weight_bytes": (proj_w + skel_ff_w) * blocks / 2,
        "skel_in_layers": cm.early_blocks,
        "skel_out_layers": cm.final_blocks,
        "head_weight_bytes": cm.vocab * cm.d * b,
        "kv_width": cm.attn.kv_width,
        "context_tokens": keys,
    }


def _deepseek_v3_fields(context_tokens: int | None) -> dict[str, Any]:
    """DeepSeek-V3 as a layered round model (ADR-052 §3-4, decision 14). One round per MoE
    layer; the 3 dense layers are `skel_in`, the head `skel_out`; the shared expert and the
    router join each round's `attn` job (they run on the owner, DP attention). GEMM weights in
    FP8 (the released checkpoint), the absorbed kv_b, router and head in BF16, KV in BF16.
    `n_e` is the routed-expert count; a deployment with redundant replicas overrides it with
    its slot count (SGLang: 256 + 32)."""
    from costmodel import deepseek_v3 as ds

    if context_tokens is None:
        raise ValueError("deepseek_v3 needs context_tokens (the deployment's mean keys)")
    m = ds.DeepSeekV3()
    fp8, bf16 = 1.0, 2.0
    shared_w = m.n_shared * ds.swiglu(m.d, m.d_ff_expert)
    attn_w = ds.attention_weight_bytes(m, fp8, bf16)
    score = ds.attention_score_flops_per_key(m)
    dense_layer_flops = ds.attention_proj_flops(m) + ds.dense_ff_flops(m) + score * context_tokens
    return {
        "rung": "deepseek_v3",
        "d": m.d,
        "k": m.k,
        "n_e": m.n_routed,
        "d_ff": m.d_ff_expert,
        "L_e": 1,
        "r": m.moe_layers,
        "b_act": bf16,
        "b_w": fp8,
        "b_kv": bf16,
        "attn_proj_flops": ds.attention_proj_flops(m)
        + m.n_shared * ds.expert_flops(m)
        + 2.0 * ds.router_params(m),
        "attn_proj_weight_bytes": attn_w + shared_w * fp8 + ds.router_params(m) * bf16,
        "skel_in_flops": m.dense_layers * dense_layer_flops,
        "skel_out_flops": ds.head_flops(m),
        "skel_in_weight_bytes": m.dense_layers * (attn_w + ds.swiglu(m.d, m.d_ff_dense) * fp8),
        "skel_out_weight_bytes": 0.0,
        "head_weight_bytes": m.vocab * m.d * bf16,
        "kv_width": m.kv_entry // 2,
        "context_tokens": context_tokens,
        "skel_in_layers": m.dense_layers,
        "skel_out_layers": 0,
        "score_flops_per_key": score,
        "kv_bytes_per_key_layer": ds.kv_bytes_per_key(m, bf16),
        "b_dispatch": fp8 + 4.0 / 128,  # FP8 with one fp32 scale per 128 values (DeepEP LL)
        "b_combine": bf16,
    }


def attention_score_flops_per_token(p: SimParams) -> float:
    """$2 \\cdot 2 d \\cdot n_{keys}$ at the decode position (docs/01 §10, decision 6), or the
    model's own per-key figure (MLA)."""
    if p.score_flops_per_key is not None:
        return p.score_flops_per_key * p.context_tokens
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
    unknown = set(overrides) - {f.name for f in fields_of(SimParams) if f.init}
    if unknown:
        raise TypeError(f"unknown SimParams fields: {sorted(unknown)}")
    return SimParams(**{**fields, **overrides})  # validated once, with the overrides applied
