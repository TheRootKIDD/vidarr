"""Model configuration — a dataclass mirror of `docs/01 §11`.

Every knob of the spec is here, including ones the reference implementation
does not execute yet (index, memory, tenants): those are validated and refused
so a config cannot silently ask for a feature that is not implemented.

`d_ff` is *derived* from the FLOPs budget through `costmodel/` (ADR-025/027)
unless the config overrides it; the derivation is the sizing invariant of
`docs/01 §4.5`: d_ff = max(U * s_min, d_ff_quality), rounded to 64.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from costmodel.config import Attention as CmAttention
from costmodel.config import Experts as CmExperts
from costmodel.config import Hardware as CmHardware
from costmodel.config import Model as CmModel
from costmodel.solve import budget_n_keys, solve_d_ff

Arch = Literal["layered", "recurrent"]
BlockForm = Literal["sequential", "parallel"]
FFKind = Literal["dense", "moe"]
GlobalSource = Literal["per_iteration", "final_vector"]


@dataclass(frozen=True)
class AttentionCfg:
    heads: int = 12
    kv_heads: int = 4
    head_dim: int = 64
    window: int = 512  # W: local range, also the training segment length (v2 P7)
    global_source: GlobalSource = "per_iteration"  # L5 baseline; final_vector = L5d
    segment_memory_grad: Literal["stop", "through"] = "stop"  # ADR-023
    prev_segment_local: bool = False  # ADR-022 knob; only False is implemented
    kv_share_every: int = 1  # L6b; only 1 is implemented
    index_enabled: bool = False  # L9; not implemented in Phase 1
    rope_theta: float = 10_000.0


@dataclass(frozen=True)
class RouterCfg:
    k: int = 2
    n_experts: int = 8
    d_r: int = 128  # router query / key dim
    product_keys: bool = False  # only needed at N_e >= 256; plain keys otherwise
    hierarchical: bool = False
    balance: Literal["aux_free", "none"] = "aux_free"
    bias_update_rate: float = 1e-3  # aux-loss-free load controller step (DeepSeek-V3 style)
    aux_loss_coef: float = 1e-3  # small sequence-level auxiliary loss
    use_depth_embed: bool = True  # False = E-Q2 ablation (router without e_j)


@dataclass(frozen=True)
class ExpertsCfg:
    unit_width: int = 1  # U, simulated; training replicas are one GPU (ADR-025)
    tile_min: int = 256  # s_min, measured (003-gemm)
    d_ff: int | None = None  # derived unless overridden
    layers: int = 1  # L_e in {1, 2}
    granularity: int = 1  # g sub-experts per unit (L7): N_e*g experts of width d_ff/g


@dataclass(frozen=True)
class MiddleCfg:
    r_min: int = 2
    r_max: int = 8
    depth_embed: int = 64
    per_iter_lora_rank: int = 0  # L5b; not implemented in Phase 1


@dataclass(frozen=True)
class StreamsCfg:
    two_stream: bool = False  # L4 (ADR-020, ADR-026)
    p_sees_own_context: bool = True  # False = L4b


@dataclass(frozen=True)
class HaltingCfg:
    mode: Literal["fixed", "act"] = "fixed"
    ponder_cost: float = 0.0
    epsilon: float = 0.01
    output: Literal["last"] = "last"  # ADR-010
    halted_kv: Literal["once"] = "once"  # ADR-011


@dataclass(frozen=True)
class MTPCfg:
    m: int = 1  # 1 disables; 4 = L3
    style: Literal["independent"] = "independent"  # ADR-012
    subsample: float | Literal["auto"] = "auto"  # auto = 1/(m-1)
    lambdas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)  # per head, head 1 first


@dataclass(frozen=True)
class NumericsCfg:
    dtype: Literal["bf16", "fp32"] = "bf16"
    sparsity_2_4: bool = False  # L10a; mask fixed after 1 % of tokens (ADR-019)


@dataclass(frozen=True)
class ModelCfg:
    """`docs/01 §11`. `arch` selects the layered baselines (L0–L3) or the recurrent design (L5+)."""

    arch: Arch = "recurrent"
    d: int = 768
    vocab: int = 32_000
    context: int = 2048
    # layered (L0–L3)
    layers: int = 12
    block_form: BlockForm = "sequential"
    ff: FFKind = "dense"
    dense_d_ff: int = 2048
    # recurrent (L5+)
    early_blocks: int = 2
    final_blocks: int = 2
    pred_only_final: int = 1
    skeleton_d_ff: int | None = None  # ADR-006: 4d unless overridden
    budget_flops_per_token: float = 0.6e9  # training cost, docs/06 §3
    attention: AttentionCfg = field(default_factory=AttentionCfg)
    router: RouterCfg = field(default_factory=RouterCfg)
    experts: ExpertsCfg = field(default_factory=ExpertsCfg)
    middle: MiddleCfg = field(default_factory=MiddleCfg)
    streams: StreamsCfg = field(default_factory=StreamsCfg)
    halting: HaltingCfg = field(default_factory=HaltingCfg)
    mtp: MTPCfg = field(default_factory=MTPCfg)
    numerics: NumericsCfg = field(default_factory=NumericsCfg)
    memory_enabled: bool = False  # L8; not implemented in Phase 1
    tenant_lora_rank: int = 0  # L11; not implemented in Phase 1
    # training-time memory knobs (no effect on semantics): docs/02 §13.4, docs/06 §3
    checkpoint_iterations: bool = True  # recompute each middle-block iteration in backward
    ce_chunk_tokens: int = 2048  # chunked cross-entropy over the tied head

    # ---- derived quantities -------------------------------------------------
    @property
    def n_streams(self) -> int:
        return 2 if self.streams.two_stream else 1

    @property
    def skeleton_width(self) -> int:
        return self.skeleton_d_ff or 4 * self.d

    @property
    def tile_floor(self) -> int:
        return self.experts.unit_width * self.experts.tile_min

    def expert_width(self) -> int:
        """d_ff per expert (before the granularity split), derived or overridden."""
        if self.experts.d_ff is not None:
            return self.experts.d_ff
        if self.arch == "layered":
            # matched FLOPs to the dense block: k experts of width dense_d_ff / k
            return _round64(self.dense_d_ff // self.router.k)
        return self.derive_width().d_ff

    def derive_width(self) -> Any:
        """ADR-025/027: costmodel solve at the budget, 1024 keys, multiples of 64, tile floor."""
        cm = CmModel(
            d=self.d,
            vocab=self.vocab,
            early_blocks=self.early_blocks,
            final_blocks=self.final_blocks,
            r_max=self.middle.r_max,
            two_stream=self.streams.two_stream,
            depth_embed=self.middle.depth_embed,
            attn=CmAttention(
                heads=self.attention.heads,
                kv_heads=self.attention.kv_heads,
                head_dim=self.attention.head_dim,
                window=self.attention.window,
            ),
            experts=CmExperts(
                n_e=self.router.n_experts, k=self.router.k, d_ff=None, L_e=self.experts.layers
            ),
        )
        hw = CmHardware(
            phi=27.1e12,
            beta_c=342e9,
            m_c=12e9,
            lambda_c=3.59e9,
            U=self.experts.unit_width,
            s_min=self.experts.tile_min,
        )
        return solve_d_ff(
            cm,
            self.budget_flops_per_token,
            n_keys=budget_n_keys(self.context),
            multiple_of=64,
            hw=hw,
        )

    def validate(self) -> None:
        a, r, e = self.attention, self.router, self.experts
        if a.heads % a.kv_heads:
            raise ValueError("heads must be a multiple of kv_heads")
        if a.index_enabled or self.memory_enabled or self.tenant_lora_rank:
            raise NotImplementedError("index (L9), memory (L8) and tenants (L11) are Phase 3")
        if self.middle.per_iter_lora_rank:
            raise NotImplementedError("per-iteration attention LoRA (L5b) is not implemented yet")
        if a.prev_segment_local or a.kv_share_every != 1:
            raise NotImplementedError("ADR-022 crossing / L6b KV sharing are not implemented yet")
        if r.k > r.n_experts * e.granularity:
            raise ValueError("k exceeds the number of (sub-)experts")
        if e.layers not in (1, 2):
            raise ValueError("L_e must be 1 or 2")
        if self.mtp.m < 1:
            raise ValueError("mtp.m >= 1")
        final_vec = self.arch == "recurrent" and a.global_source == "final_vector"
        if final_vec and self.context % a.window:
            raise ValueError("context must be a multiple of the window (segment length)")
        if self.numerics.sparsity_2_4 and self.expert_width() % 4:
            raise ValueError("2:4 needs d_ff divisible by 4")

    # ---- serialisation -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["derived"] = {
            "expert_width": self.expert_width(),
            "sub_expert_width": self.expert_width() // self.experts.granularity,
            "n_sub_experts": self.router.n_experts * self.experts.granularity,
            "skeleton_width": self.skeleton_width,
            "tile_floor": self.tile_floor,
            "n_streams": self.n_streams,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelCfg:
        d = dict(d)
        d.pop("derived", None)
        sub = {
            "attention": AttentionCfg,
            "router": RouterCfg,
            "experts": ExpertsCfg,
            "middle": MiddleCfg,
            "streams": StreamsCfg,
            "halting": HaltingCfg,
            "mtp": MTPCfg,
            "numerics": NumericsCfg,
        }
        for key, typ in sub.items():
            if key in d and isinstance(d[key], dict):
                v = dict(d[key])
                if key == "mtp" and "lambdas" in v:
                    v["lambdas"] = tuple(v["lambdas"])
                d[key] = typ(**v)
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelCfg:
        doc = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(doc.get("model", doc))


def _round64(n: int) -> int:
    return max(64, (n // 64) * 64)


# ---- ladder presets (docs/03 §2) ------------------------------------------------
def ladder(step: str, **overrides: Any) -> ModelCfg:
    """The ablation ladder as configs. Each rung = previous rung + one feature."""
    base = ModelCfg(arch="layered", layers=12, block_form="sequential", ff="dense")
    moe = replace(base, ff="moe", router=RouterCfg(k=2, n_experts=8, use_depth_embed=False))
    rec = ModelCfg(arch="recurrent", router=RouterCfg(k=2, n_experts=8))
    presets: dict[str, ModelCfg] = {
        "L0": base,
        "L1": moe,
        "L2": replace(moe, block_form="parallel"),
        "L3": replace(moe, block_form="parallel", mtp=MTPCfg(m=4)),
        "L5": rec,
        "L5-ne1": replace(rec, router=RouterCfg(k=1, n_experts=1)),
        "L5-ne128": replace(rec, router=RouterCfg(k=4, n_experts=128)),
        "L5-noej": replace(rec, router=RouterCfg(k=2, n_experts=8, use_depth_embed=False)),
        "L5d": replace(rec, attention=AttentionCfg(global_source="final_vector")),
        "L6": replace(
            rec,
            attention=AttentionCfg(global_source="final_vector"),
            halting=HaltingCfg(mode="act", ponder_cost=1e-2),
        ),
        "L7-g4": replace(rec, experts=ExpertsCfg(granularity=4)),
        "L7b": replace(rec, experts=ExpertsCfg(layers=2)),
        "L10a": replace(rec, numerics=NumericsCfg(sparsity_2_4=True)),
        "L4": replace(
            rec, streams=StreamsCfg(two_stream=True), budget_flops_per_token=1.2e9
        ),  # ADR-026: 2x budget
    }
    if step not in presets:
        raise KeyError(f"unknown ladder step {step}; have {sorted(presets)}")
    cfg = replace(presets[step], **overrides) if overrides else presets[step]
    cfg.validate()
    return cfg
