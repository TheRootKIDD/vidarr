"""Building blocks shared by the layered baselines and the recurrent design.

Everything here is written for readability (CLAUDE.md): masks are explicit
boolean tensors, GQA is a `repeat_interleave`, rotary is a plain function.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DepthNorm(nn.Module):
    """RMSNorm with optional AdaLN-style modulation by the depth embedding.

    `docs/01 §4`: u = LN(x) * (1 + gamma(e_j)) + beta(e_j). Without `e` it is a
    plain RMSNorm (skeleton blocks, layered baselines).
    """

    def __init__(self, d: int, depth_dim: int = 0) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.mod = nn.Linear(depth_dim, 2 * d, bias=True) if depth_dim else None
        if self.mod is not None:
            nn.init.zeros_(self.mod.weight)
            nn.init.zeros_(self.mod.bias)

    def forward(self, x: Tensor, e: Tensor | None = None) -> Tensor:
        h = self.norm(x)
        if self.mod is None or e is None:
            return h
        gamma, beta = self.mod(e).chunk(2, dim=-1)  # e: [d_e] -> broadcast over [..., d]
        return h * (1 + gamma) + beta


def rotary_cos_sin(positions: Tensor, head_dim: int, theta: float) -> tuple[Tensor, Tensor]:
    """cos/sin tables for absolute `positions` ([T] or [B, T]); returned as [..., T, head_dim]."""
    half = head_dim // 2
    freqs = theta ** (-torch.arange(0, half, device=positions.device, dtype=torch.float32) / half)
    ang = positions.to(torch.float32).unsqueeze(-1) * freqs  # [..., T, half]
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos(), ang.sin()


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: [B, H, T, dh]; cos/sin: [T, dh] or [B, T, dh] (broadcast over heads)."""
    if cos.dim() == 2:
        cos, sin = cos[None, None], sin[None, None]
    else:
        cos, sin = cos[:, None], sin[:, None]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rotated * sin).to(x.dtype)


def causal_mask(t_q: int, t_k: int, device: torch.device, offset: int = 0) -> Tensor:
    """[t_q, t_k] bool, True = may attend: query i (absolute i+offset) sees keys j <= i+offset."""
    q = torch.arange(t_q, device=device)[:, None] + offset
    k = torch.arange(t_k, device=device)[None, :]
    return k <= q


def local_window_mask(t: int, window: int, device: torch.device) -> Tensor:
    """Causal within a sliding window of `window` tokens (inclusive of self). ADR-022."""
    q = torch.arange(t, device=device)[:, None]
    k = torch.arange(t, device=device)[None, :]
    return (k <= q) & (k > q - window)


class Attention(nn.Module):
    """GQA with rotary positions over an explicit key set.

    `forward(q_in, kv_in, ...)` separates the rows that produce queries from the
    rows that produce keys/values — that is what makes stream visibility
    (`docs/01 §3`, ADR-002) and the global final-vector cache (`§4.3`) plain to
    express: pass `kv_in` = the context stream only, or prepend precomputed
    memory K/V through `mem_kv`.
    """

    def __init__(self, d: int, heads: int, kv_heads: int, head_dim: int, theta: float) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.theta = heads, kv_heads, head_dim, theta
        self.wq = nn.Linear(d, heads * head_dim, bias=False)
        self.wk = nn.Linear(d, kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d, kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(heads * head_dim, d, bias=False)

    def kv(self, x: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """K/V for rows `x` [B, T, d] at absolute `positions` [T] -> each [B, Hkv, T, dh]."""
        b, t, _ = x.shape
        k = self.wk(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = rotary_cos_sin(positions, self.head_dim, self.theta)
        return apply_rotary(k, cos, sin), v

    def forward(
        self,
        q_in: Tensor,
        positions: Tensor,
        kv: tuple[Tensor, Tensor],
        mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        """q_in [B, Tq, d]; kv = (K, V) [B, Hkv, Tk, dh]; mask [Tq, Tk] bool or None."""
        b, tq, _ = q_in.shape
        q = self.wq(q_in).view(b, tq, self.heads, self.head_dim).transpose(1, 2)
        cos, sin = rotary_cos_sin(positions, self.head_dim, self.theta)
        q = apply_rotary(q, cos, sin)
        k, v = kv
        rep = self.heads // self.kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        if is_causal:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(out.transpose(1, 2).reshape(b, tq, -1))


class SwiGLU(nn.Module):
    """W_down(SiLU(W_gate u) * W_up u)."""

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Expert(nn.Module):
    """`docs/01 §4.5`: a gated FF stack with L_e in {1, 2} layers (residual + norm between)."""

    def __init__(self, d: int, d_ff: int, layers: int) -> None:
        super().__init__()
        self.ffs = nn.ModuleList(SwiGLU(d, d_ff) for _ in range(layers))
        self.norms = nn.ModuleList(nn.RMSNorm(d) for _ in range(layers - 1))

    def forward(self, x: Tensor) -> Tensor:
        y = self.ffs[0](x)
        for norm, ff in zip(self.norms, self.ffs[1:], strict=True):
            y = y + ff(norm(y))
        return y
