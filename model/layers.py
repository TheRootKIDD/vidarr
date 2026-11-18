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


class IterLoRA(nn.Module):
    """Per-iteration low-rank delta for one projection (`docs/01 §4.2`, L5b, ADR-032).

    Iteration j (1-based) adds x A_j^T B_j^T. B starts at zero, so the adapted
    projection starts equal to the shared one.
    """

    def __init__(self, slots: int, d_in: int, d_out: int, rank: int) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.randn(slots, rank, d_in) / d_in**0.5)
        self.b = nn.Parameter(torch.zeros(slots, d_out, rank))

    def forward(self, x: Tensor, j: int) -> Tensor:
        return (x @ self.a[j - 1].T) @ self.b[j - 1].T


class Attention(nn.Module):
    """GQA with rotary positions over an explicit key set.

    `forward(q_in, kv_in, ...)` separates the rows that produce queries from the
    rows that produce keys/values — that is what makes stream visibility
    (`docs/01 §3`, ADR-002) and the global final-vector cache (`§4.3`) plain to
    express: pass `kv_in` = the context stream only, or prepend precomputed
    memory K/V through `mem_kv`.

    With `lora_rank` > 0 every projection carries `lora_slots` per-iteration
    adapters (L5b); callers pass the iteration `j` to use them, or None for the
    shared weights alone.
    """

    def __init__(
        self,
        d: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        theta: float,
        lora_rank: int = 0,
        lora_slots: int = 0,
        rope: bool = True,
        two_branch: bool = False,
        scoring: str = "softmax",
    ) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.theta = heads, kv_heads, head_dim, theta
        self.rope = rope  # False = NoPE (ADR-066, rope_scope = none)
        # ADR-068: the global range as its own branch, gated per head; scored by softmax or ReLU^2
        self.two_branch, self.scoring = two_branch, scoring
        self.g_gate = nn.Parameter(torch.zeros(heads)) if two_branch else None
        self.eval_cap = 0  # ADR-068 §3: keep only the top-K global keys by pre-ReLU score (eval)
        self.zero_stats: list[Tensor] = []  # (zero fraction, V-skip rate) per ReLU^2 call
        self.wq = nn.Linear(d, heads * head_dim, bias=False)
        self.wk = nn.Linear(d, kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d, kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(heads * head_dim, d, bias=False)
        self.lora: nn.ModuleDict | None = None
        if lora_rank:
            projections = {"q": self.wq, "k": self.wk, "v": self.wv, "o": self.wo}
            self.lora = nn.ModuleDict(
                {
                    name: IterLoRA(lora_slots, lin.in_features, lin.out_features, lora_rank)
                    for name, lin in projections.items()
                }
            )

    def _proj(self, name: str, lin: nn.Linear, x: Tensor, j: int | None) -> Tensor:
        y = lin(x)
        if self.lora is not None and j is not None:
            y = y + self.lora[name](x, j)
        return y

    def global_scores(
        self, q_in: Tensor, positions: Tensor, kg: Tensor, plain: bool, j: int | None = None
    ) -> Tensor:
        """ADR-070: pre-softmax scores of every query head against G's keys kg [B, Hkv, Ng, dh],
        summed over each KV group's query heads -> [B, Hkv, Tq, Ng]. `plain`: G is NoPE."""
        b, tq, _ = q_in.shape
        q = self._proj("q", self.wq, q_in, j).view(b, tq, self.heads, self.head_dim).transpose(1, 2)
        if self.rope and not plain:
            cos, sin = rotary_cos_sin(positions, self.head_dim, self.theta)
            q = apply_rotary(q, cos, sin)
        rep = self.heads // self.kv_heads
        q = q.view(b, self.kv_heads, rep, tq, self.head_dim)
        return torch.einsum("bgrtd,bgnd->bgtn", q, kg) * self.head_dim**-0.5

    def _two_branch(
        self, q: Tensor, q_g: Tensor, k: Tensor, v: Tensor, n_glob: int, mask: Tensor | None
    ) -> Tensor:
        """o = o_loc + sigmoid(gamma_h) o_glob; k, v already repeated to the query heads."""
        kg, vg, kl, vl = k[:, :, :n_glob], v[:, :, :n_glob], k[:, :, n_glob:], v[:, :, n_glob:]
        m_loc = None if mask is None else mask[:, n_glob:]
        o_loc = F.scaled_dot_product_attention(q, kl, vl, attn_mask=m_loc)
        if self.scoring == "relu2":
            a = _relu2_weights(q_g, kg, self.eval_cap)
            o_glob = a.to(vg.dtype) @ vg
            with torch.no_grad():
                zero = a == 0  # [B, H, Tq, Tg]
                grp = zero.view(a.shape[0], self.kv_heads, -1, *a.shape[2:]).all(dim=2)
                # per query head: zero fraction; per KV head: keys zero for the whole group
                self.zero_stats.append(
                    torch.cat([zero.float().mean((0, 2, 3)), grp.float().mean((0, 2, 3))])
                )
                del self.zero_stats[:-4096]  # forward-only callers never drain it
        else:
            o_glob = F.scaled_dot_product_attention(q_g, kg, vg)
        assert self.g_gate is not None
        return o_loc + torch.sigmoid(self.g_gate)[None, :, None, None].to(o_loc.dtype) * o_glob

    def kv(
        self, x: Tensor, positions: Tensor, j: int | None = None, rotate: bool = True
    ) -> tuple[Tensor, Tensor]:
        """K/V for rows `x` [B, T, d] at absolute `positions` [T] -> each [B, Hkv, T, dh].
        `rotate=False` stores the keys unrotated (ADR-066: NoPE global entries)."""
        b, t, _ = x.shape
        k = self._proj("k", self.wk, x, j).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self._proj("v", self.wv, x, j).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        if not (rotate and self.rope):
            return k, v
        cos, sin = rotary_cos_sin(positions, self.head_dim, self.theta)
        return apply_rotary(k, cos, sin), v

    def forward(
        self,
        q_in: Tensor,
        positions: Tensor,
        kv: tuple[Tensor, Tensor],
        mask: Tensor | None,
        is_causal: bool = False,
        j: int | None = None,
        n_plain: int = 0,
        n_glob: int = 0,
    ) -> Tensor:
        """q_in [B, Tq, d]; kv = (K, V) [B, Hkv, Tk, dh]; mask [Tq, Tk] bool or None.

        `n_plain` > 0 (ADR-066, rope_scope = local): the first `n_plain` keys are unrotated
        (NoPE global entries) and are scored against the unrotated query, the rest against the
        rotated one, all in one softmax. Done exactly by widening the head: q' = [rot(q) | q],
        k' = [rot(k) | 0] for local keys and [0 | k] for plain ones, with the 1/sqrt(dh) scale."""
        b, tq, _ = q_in.shape
        q = self._proj("q", self.wq, q_in, j).view(b, tq, self.heads, self.head_dim).transpose(1, 2)
        q_plain = q
        if self.rope:
            cos, sin = rotary_cos_sin(positions, self.head_dim, self.theta)
            q = apply_rotary(q, cos, sin)
        k, v = kv
        rep = self.heads // self.kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        if self.two_branch and n_glob:
            q_g = q_plain if n_plain else q  # NoPE G is scored by the unrotated query
            out = self._two_branch(q, q_g, k, v, n_glob, mask)
            return self._proj("o", self.wo, out.transpose(1, 2).reshape(b, tq, -1), j)
        if n_plain and self.rope:
            z = torch.zeros_like(k)
            k = torch.cat(
                [
                    torch.cat([z[:, :, :n_plain], k[:, :, :n_plain]], dim=-1),
                    torch.cat([k[:, :, n_plain:], z[:, :, n_plain:]], dim=-1),
                ],
                dim=2,
            )
            q = torch.cat([q, q_plain], dim=-1)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=self.head_dim**-0.5
            )
            return self._proj("o", self.wo, out.transpose(1, 2).reshape(b, tq, -1), j)
        if is_causal:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self._proj("o", self.wo, out.transpose(1, 2).reshape(b, tq, -1), j)


def _relu2_weights(q: Tensor, k: Tensor, cap: int) -> Tensor:
    """ADR-068: w = relu(q^ . k^ / sqrt(dh))^2 over RMS-normalised q, k, normalised by its sum;
    rows with no positive score are all zero. `cap` > 0 keeps the top-`cap` keys by score."""
    dh = q.shape[-1]
    s = F.rms_norm(q, (dh,)) @ F.rms_norm(k, (dh,)).transpose(-1, -2) / dh**0.5
    if cap and cap < s.shape[-1]:
        thr = s.topk(cap, dim=-1).values[..., -1:]
        s = s.masked_fill(s < thr, 0.0)
    w = F.relu(s).square()
    return w / (w.sum(-1, keepdim=True) + 1e-6)


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
