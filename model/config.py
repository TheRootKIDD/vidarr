"""Model configuration — a dataclass mirror of `docs/01 §11`.

Every knob of the spec is here, including ones the reference implementation
does not execute yet (index, tenants): those are validated and refused
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
from costmodel.config import Memory as CmMemory
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
    # ADR-038 (I33): aux averaged over L5d's segments; "sum" reproduces runs before it
    segment_aux: Literal["mean", "sum"] = "mean"


@dataclass(frozen=True)
class ExpertsCfg:
    unit_width: int = 1  # U, simulated; training replicas are one GPU (ADR-025)
    tile_min: int = 256  # s_min, measured (003-gemm)
    d_ff: int | None = None  # derived unless overridden
    layers: int = 1  # L_e in {1, 2}
    granularity: int = 1  # g sub-experts per unit (L7): N_e*g of width d_ff/g, k*g active (ADR-047)


@dataclass(frozen=True)
class MiddleCfg:
    r_min: int = 2
    r_max: int = 8
    depth_embed: int = 64
    per_iter_lora_rank: int = 0  # 16 = L5b: per-iteration adapters on attention (ADR-032)
    shared_attention: bool = True  # False = L5c: one attention block per iteration (ADR-032)
    h_every: int = 0  # K; 8 = L5e: a dense H block after iteration j = K/2 (mod K) (ADR-034)
    h_d_ff: int = 2048  # the H block's SwiGLU width (L5e)

    @property
    def per_iteration_attention(self) -> bool:
        """Attention weights differ by iteration (L5b, L5c): halted K/V is recomputed (ADR-011)."""
        return self.per_iter_lora_rank > 0 or not self.shared_attention


@dataclass(frozen=True)
class StreamsCfg:
    two_stream: bool = False  # L4 (ADR-020, ADR-026)
    p_sees_own_context: bool = True  # False = L4b


@dataclass(frozen=True)
class HaltingCfg:
    mode: Literal["fixed", "act", "expert"] = "fixed"  # act = L6 (ADR-041), expert = L6c (ADR-042)
    ponder_cost: float = 0.0
    epsilon: float = 0.01
    output: Literal["last"] = "last"  # ADR-010
    halted_kv: Literal["once"] = "once"  # ADR-011
    gate: bool = True  # scale updates by the running probability (act) or the score (expert)
    survival: Literal["linear"] = "linear"  # L6c: s_j falls linearly from 1 at r_min ...
    survival_last: float = 0.3  # ... to this at r_max (ADR-042: mean depth 5.55 at r_max 8)
    tau_momentum: float = 0.99  # L6c: EMA of the per-iteration thresholds

    def survival_fraction(self, j: int, r_min: int, r_max: int) -> float:
        """L6c's s_j: the fraction of tokens still running at iteration j (ADR-042)."""
        if j <= r_min:
            return 1.0
        return 1.0 - (1.0 - self.survival_last) * (j - r_min) / (r_max - r_min)


@dataclass(frozen=True)
class MTPCfg:
    m: int = 1  # 1 disables; 4 = L3
    style: Literal["independent", "sequential"] = "independent"  # ADR-012; sequential = L3c
    subsample: float | Literal["auto"] = "auto"  # auto = 1/(m-1)
    lambdas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)  # per head, head 1 first
    curriculum: Literal["none", "forward"] = "none"  # forward = L3b: heads 2..i in phase i of m
    module_d_ff: int = 2048  # L3c (ADR-039): SwiGLU width of each module's dense block


@dataclass(frozen=True)
class QueryAuxCfg:
    """L8q (ADR-055, `docs/01 §4.6`): a training-only target for the memory's query. The frozen
    teacher encodes each window-aligned chunk; the next chunk's tokens are trained toward its PQ
    code (nearest codebook centroid per half). Inference is unchanged."""

    lambda_q: float = 0.1
    chunk: int = 64  # tokens per chunk, window-aligned
    teacher: str = "BAAI/bge-base-en-v1.5"  # the table's own encoder (ADR-053 §3)
    teacher_dir: str = "/mnt/nvme/memory/encoder/bge-base-en-v1.5"
    teacher_max_tokens: int = 128


@dataclass(frozen=True)
class RetrievalCfg:
    """L8b (ADR-057, `docs/01 §4.6`): exact chunk-level retrieval of the table's sentences.
    Every `stride` positions, the last <= `window` tokens of the current document (at least
    `min_tokens`) are encoded by the frozen encoder and the top `k` table rows by inner product
    feed the next `stride` positions."""

    stride: int = 16
    window: int = 32
    min_tokens: int = 8
    k: int = 8
    encoder_dir: str = "/mnt/nvme/memory/encoder/bge-base-en-v1.5"
    encoder_max_tokens: int = 128


@dataclass(frozen=True)
class MemoryCfg:
    """L8, the knowledge memory as built (ADR-053, `docs/01 §4.6`)."""

    enabled: bool = False
    grid: int = 1024  # sqrt(N_m): one entry per cell of a grid x grid product grid
    key_dim: int = 128  # d_m, two halves of d_m / 2
    topk: int = 32  # k_m values read (1 gives no gradient)
    half_topk: int = 32  # sub-keys kept per half; half_topk**2 candidate cells
    encoder_dim: int = 768  # d_enc of the frozen encoder
    encoder: str = "BAAI/bge-base-en-v1.5"
    source: str = "wiki-20231101-lead"
    table: str | None = None  # directory from build_wiki_memory.py; None = random toy (tests)
    query_aux: QueryAuxCfg | None = None  # L8q (ADR-055); None = off
    # L8k (ADR-056): "pca_fixed" = frozen per-half PCA sub-keys scored by distance, learned tau;
    # "learned" = the h^{1,2} key maps with dot scores (L8, L8q)
    keys: Literal["learned", "pca_fixed"] = "learned"
    # L8b (ADR-057): "retrieved" = neighbours from exact search outside the model, read by an
    # 8-way softmax; "product_key" = the learned in-loop lookup (L8, L8q, L8k; closed at screen)
    mode: Literal["product_key", "retrieved"] = "product_key"
    retrieval: RetrievalCfg | None = None

    @property
    def n_entries(self) -> int:
        return self.grid * self.grid


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
    memory: MemoryCfg = field(default_factory=MemoryCfg)  # L8 (ADR-053)
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
            mem=self.cm_memory(),
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

    def cm_memory(self) -> CmMemory:
        """The memory's costmodel view; inside the FLOPs budget at matched FLOPs (ADR-053 §6)."""
        mc = self.memory
        return CmMemory(
            enabled=mc.enabled,
            n_m=mc.n_entries,
            d_m=mc.key_dim,
            d_enc=mc.encoder_dim,
            k_m=mc.topk,
            fixed_keys=mc.keys == "pca_fixed",
            retrieved_k=0 if mc.retrieval is None or mc.mode != "retrieved" else mc.retrieval.k,
            retrieved_stride=1 if mc.retrieval is None else mc.retrieval.stride,
        )

    def validate(self) -> None:
        a, r, e, mc = self.attention, self.router, self.experts, self.memory
        if a.heads % a.kv_heads:
            raise ValueError("heads must be a multiple of kv_heads")
        if a.index_enabled or self.tenant_lora_rank:
            raise NotImplementedError("index (L9) and tenants (L11) are Phase 3")
        if mc.enabled:
            if self.arch != "recurrent":
                raise NotImplementedError("L8 is a middle-block branch (ADR-053)")
            if mc.key_dim % 2 or mc.encoder_dim % 2:
                raise ValueError("memory key_dim and encoder_dim split into two halves")
            if not (1 < mc.topk <= mc.half_topk**2 and mc.half_topk <= mc.grid):
                raise ValueError("memory needs 1 < topk <= half_topk**2 and half_topk <= grid")
            if mc.mode not in ("product_key", "retrieved"):
                raise ValueError("memory.mode must be product_key or retrieved (ADR-057)")
            if mc.mode == "retrieved":
                rc = mc.retrieval
                if rc is None:
                    raise ValueError("memory.mode retrieved (L8b) needs memory.retrieval")
                if mc.query_aux is not None:
                    raise ValueError("memory.query_aux (L8q) applies to the product-key lookup")
                if rc.stride < 1 or self.context % rc.stride or rc.k < 1:
                    raise ValueError("retrieval.stride must divide the context; k >= 1")
                if not 1 <= rc.min_tokens <= rc.window:
                    raise ValueError("retrieval needs 1 <= min_tokens <= window")
                if self.attention.global_source == "final_vector" and (
                    self.attention.window % rc.stride
                ):
                    raise ValueError("retrieval.stride must divide the segment length")
            if mc.keys not in ("learned", "pca_fixed"):
                raise ValueError("memory.keys must be learned or pca_fixed (ADR-056)")
            if mc.keys == "pca_fixed" and mc.key_dim > mc.encoder_dim:
                raise ValueError("memory.keys pca_fixed needs key_dim <= encoder_dim")
        qa = mc.query_aux
        if qa is not None:
            if not mc.enabled:
                raise ValueError("memory.query_aux (L8q) needs the memory enabled")
            if qa.chunk < 1 or self.context % qa.chunk or self.context // qa.chunk < 2:
                raise ValueError("query_aux.chunk must divide the context into >= 2 chunks")
            if qa.lambda_q < 0:
                raise ValueError("query_aux.lambda_q must be >= 0")
        if self.halting.mode == "expert" and not (
            self.arch == "recurrent" and self.middle.r_max > self.middle.r_min
        ):
            raise ValueError("L6c needs the recurrent arch and r_max > r_min (ADR-042)")
        if not 0.0 < self.halting.survival_last <= 1.0:
            raise ValueError("halting.survival_last must be in (0, 1]")
        if self.middle.per_iter_lora_rank and not self.middle.shared_attention:
            raise ValueError("L5b adapters on L5c's unshared attention: two features (ADR-032)")
        if self.middle.per_iteration_attention and a.global_source == "final_vector":
            raise NotImplementedError(
                "L5b/L5c over the final-vector cache: no rule yet for which iteration's "
                "weights make a G entry (ADR-032)"
            )
        if self.middle.h_every:
            if self.arch != "recurrent" or self.middle.h_every % 2:
                raise ValueError("L5e needs the recurrent arch and an even h_every (ADR-034)")
            if a.global_source == "final_vector" or self.middle.per_iteration_attention:
                raise NotImplementedError("L5e with L5d's G or with L5b/L5c: one feature (ADR-034)")
        if a.prev_segment_local or a.kv_share_every != 1:
            raise NotImplementedError("ADR-022 crossing / L6b KV sharing are not implemented yet")
        if r.k > r.n_experts:  # k*g active of N_e*g sub-experts (ADR-047)
            raise ValueError("k exceeds the number of experts")
        if e.granularity < 1 or self.expert_width() % e.granularity:
            raise ValueError("granularity must divide d_ff (ADR-047)")
        if e.layers not in (1, 2):
            raise ValueError("L_e must be 1 or 2")
        if self.mtp.m < 1:
            raise ValueError("mtp.m >= 1")
        if self.mtp.style == "sequential":
            if self.arch != "layered":
                raise NotImplementedError("L3c on the recurrent design is not specified (ADR-039)")
            if self.mtp.curriculum != "none":
                raise ValueError("L3c with L3b's curriculum: two features (ADR-039)")
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
            "active_sub_experts": self.router.k * self.experts.granularity,
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
            "memory": MemoryCfg,
        }
        if d.pop("memory_enabled", False):  # configs before ADR-053 carry this flag, always off
            raise ValueError("memory_enabled: true predates ADR-053; use memory.enabled")
        for key, typ in sub.items():
            if key in d and isinstance(d[key], dict):
                v = dict(d[key])
                if key == "mtp" and "lambdas" in v:
                    v["lambdas"] = tuple(v["lambdas"])
                if key == "memory" and isinstance(v.get("query_aux"), dict):
                    v["query_aux"] = QueryAuxCfg(**v["query_aux"])
                if key == "memory" and isinstance(v.get("retrieval"), dict):
                    v["retrieval"] = RetrievalCfg(**v["retrieval"])
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
        "L3b": replace(moe, block_form="parallel", mtp=MTPCfg(m=4, curriculum="forward")),
        # ADR-039: DeepSeek-V3-style sequential MTP modules in place of L3's heads
        "L3c": replace(moe, block_form="parallel", mtp=MTPCfg(m=4, style="sequential")),
        "L5": rec,
        "L5-ne1": replace(rec, router=RouterCfg(k=1, n_experts=1)),
        "L5-ne128": replace(rec, router=RouterCfg(k=4, n_experts=128)),
        # ADR-033: L5c's parameter count at L5's FLOPs (I31's control)
        "L5-ne13": replace(
            rec, router=RouterCfg(k=2, n_experts=13), experts=ExpertsCfg(d_ff=896)
        ),
        # ADR-035: I29's exactly parameter-matched H6 arm, L5-ne128 extended to L1's params
        "L5-pm": replace(
            rec, router=RouterCfg(k=4, n_experts=204), experts=ExpertsCfg(d_ff=448)
        ),
        "L5-noej": replace(rec, router=RouterCfg(k=2, n_experts=8, use_depth_embed=False)),
        # ADR-032: d_ff pinned at L5's 896 so per-iteration attention is the only change
        "L5b": replace(rec, middle=MiddleCfg(per_iter_lora_rank=16), experts=ExpertsCfg(d_ff=896)),
        "L5c": replace(rec, middle=MiddleCfg(shared_attention=False), experts=ExpertsCfg(d_ff=896)),
        # ADR-034: one dense H block with its own weights after iteration 4
        "L5e": replace(rec, middle=MiddleCfg(h_every=8), experts=ExpertsCfg(d_ff=896)),
        "L5d": replace(rec, attention=AttentionCfg(global_source="final_vector")),
        # ADR-041: gated ACT residual, ponder price below the measured marginal iteration value
        "L6": replace(
            rec,
            attention=AttentionCfg(global_source="final_vector"),
            halting=HaltingCfg(mode="act", ponder_cost=3e-3),
        ),
        # ADR-042: expert-choice depth by learned thresholds, mean depth 5.55
        "L6c": replace(
            rec,
            attention=AttentionCfg(global_source="final_vector"),
            halting=HaltingCfg(mode="expert"),
        ),
        # ADR-047: 32 sub-experts of width 224, 8 active -> L5's FLOPs, params and unit count
        "L7-g4": replace(rec, experts=ExpertsCfg(granularity=4)),
        "L7b": replace(rec, experts=ExpertsCfg(layers=2)),
        # ADR-053: the Wikipedia lead-sentence memory in every middle iteration, FLOPs matched
        "L8": replace(
            rec, memory=MemoryCfg(enabled=True, table="/mnt/nvme/memory/wiki-20231101-lead")
        ),
        # ADR-055: L8 plus the query-side retrieval loss from the frozen teacher
        "L8q": replace(
            rec,
            memory=MemoryCfg(
                enabled=True,
                table="/mnt/nvme/memory/wiki-20231101-lead",
                query_aux=QueryAuxCfg(),
            ),
        ),
        # ADR-056: L8q with fixed per-half PCA sub-keys scored by distance (same FLOPs and width)
        "L8k": replace(
            rec,
            memory=MemoryCfg(
                enabled=True,
                table="/mnt/nvme/memory/wiki-20231101-lead",
                query_aux=QueryAuxCfg(),
                keys="pca_fixed",
            ),
        ),
        # ADR-057: exact top-8 retrieval of the table's sentences every 16 tokens, read by the
        # memory branch at every iteration (width re-solves to 704)
        "L8b": replace(
            rec,
            memory=MemoryCfg(
                enabled=True,
                table="/mnt/nvme/memory/wiki-20231101-lead",
                mode="retrieved",
                retrieval=RetrievalCfg(),
            ),
        ),
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
