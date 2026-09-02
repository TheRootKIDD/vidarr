"""Configuration dataclasses for the cost model.

Notation follows `docs/01 §1` exactly (CLAUDE.md: notation is fixed there — no
parallel names). Field names are the ASCII spelling of the symbols: `d`, `U`,
`k`, `n_e` for $N_e$, `r_max`, `d_ff`, `L_e`, `W`, and so on.

Hardware numbers are **scenario inputs**, never constants (CLAUDE.md), so
`Hardware` carries no defaults for the measured quantities: a caller must supply
them from a `sim/scenarios/*.yaml` whose values cite a bench result id.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Attention:
    """`docs/01 §4.3` (v2 P7, ADR-021)."""

    heads: int = 12
    """$H$ — query heads."""
    kv_heads: int = 4
    """$H_{kv}$ — key/value heads (GQA). Equal to `heads` for plain MHA."""
    head_dim: int = 64
    """$d_h$."""
    window: int = 512
    """$W$ — local range: tokens still in flight / the current segment."""
    indexed: bool = False
    """Whether global attention over $\\mathcal{G}$ is indexed (`docs/01 §6`)."""
    block: int = 16
    """$B_{idx}$ — KV block size for the index."""
    topk: int = 32
    """$K_{idx}$ — blocks retrieved per query."""
    idx_dim: int = 32
    """$d_{idx}$ — indexer dimension."""

    @property
    def kv_width(self) -> int:
        """$H_{kv} d_h$ — width of one K (or V) projection's output."""
        return self.kv_heads * self.head_dim

    @property
    def q_width(self) -> int:
        """$H d_h$ — width of the Q (and O) projection."""
        return self.heads * self.head_dim


@dataclass(frozen=True)
class Experts:
    """`docs/01 §4.5`. $d_{ff}$ is derived unless pinned — see `costmodel.solve`."""

    n_e: int = 8
    """$N_e$ — experts in the middle block $M$. Default 8 per ADR-018."""
    k: int = 2
    """$k$ — experts active per token per iteration. 4 in the $N_e$=128 variant."""
    d_ff: int | None = None
    """$d_{ff}$ — expert intermediate width. `None` = derive from the budget."""
    L_e: int = 1
    """$L_e$ — FF layers inside one expert."""
    g: int = 1
    """$g$ — sub-experts per unit (granularity knob, H4/L7)."""
    d_r: int = 128
    """$d_r$ — router scoring dimension (product-key)."""


@dataclass(frozen=True)
class Memory:
    """`docs/01 §4.6` — GPU-resident product-key knowledge memory (L8)."""

    enabled: bool = False
    n_m: int = 1 << 20
    """$N_m$ — table entries."""
    d_m: int = 128
    """$d_m$ — key dimension."""
    d_v: int = 768
    """$d_v$ — value dimension."""
    k_m: int = 1
    """$k_m$ — values retrieved."""


@dataclass(frozen=True)
class Model:
    """One model configuration. Mirrors the YAML sketch of `docs/01 §11`."""

    d: int = 768
    """$d$ — model width, per stream."""
    vocab: int = 32_000
    """$V$."""
    early_blocks: int = 2
    """$E$."""
    final_blocks: int = 2
    """$F$."""
    r_max: int = 8
    """$r_{max}$ — middle-block iterations. `r_min` does not affect accounting."""
    two_stream: bool = False
    """ADR-020: single stream is the default; two streams are ladder step L4."""
    depth_embed: int = 64
    """$d_e$."""
    attn: Attention = field(default_factory=Attention)
    experts: Experts = field(default_factory=Experts)
    mem: Memory = field(default_factory=Memory)

    @property
    def n_streams(self) -> int:
        """Streams carried per token: 2 with the optional prediction stream."""
        return 2 if self.two_stream else 1


@dataclass(frozen=True)
class Hardware:
    """One chip's characteristics — a scenario input (`docs/02 §1`).

    No defaults: every value must come from a scenario file whose source line
    cites a measured result id (`docs/06 §6`), or from an explicitly labelled
    placeholder for `note64` / `gpu_today`.
    """

    phi: float
    """$\\phi$ — dense compute throughput, FLOP/s per chip."""
    beta_c: float
    """$\\beta_C$ — chip-local memory bandwidth, bytes/s."""
    m_c: float
    """$m_C$ — memory per chip, bytes."""
    lambda_c: float
    """$\\lambda_C$ — unit-local fabric bandwidth, bytes/s per chip."""
    U: int = 1
    """$U$ — chips per unit."""
    s_min: int = 128
    """$s_{min}$ — minimum per-chip tile width."""

    @property
    def ridge(self) -> float:
        """$\\phi / \\beta_C$ — the roofline ridge point, FLOP/byte."""
        return self.phi / self.beta_c


@dataclass(frozen=True)
class Numerics:
    """Bytes per element. `docs/01 §1`: $b_w$, $b_{act}$, $b_{kv}$."""

    b_w: float = 2.0
    """Bytes per weight (BF16 = 2, FP8 = 1, FP4 = 0.5)."""
    b_act: float = 2.0
    """Bytes per activation."""
    b_kv: float = 2.0
    """Bytes per KV element."""
