"""Incremental decode for the recurrent design with fixed r (`docs/01 §4.8`, ADR-049).

One K/V cache per early block, per middle-block iteration j = 1..r and per final
block, each over every past position. That is exactly what training's causal
attention sees at L5 (`global_source = per_iteration`): iteration j of token t
attends to iteration j of tokens <= t. `step` appends T positions (T = prompt
chunk for prefill, T = 1 for decode) and attends causally over the cache.

The middle iteration is also exposed in its two halves so that the expert-parallel
S0 harness (`scripts/bench/s0_ep_decode.py`) can run the experts on another card:
`owner_pre` (DepthNorm, attention, router) and `owner_post` (gate-weighted sum and
residual). `middle_step` composes them with the local experts.

Scope (ADR-049 §1): recurrent arch, fixed halting, L5's per-iteration global
attention, single stream, shared attention without adapters, no H block. Anything
else raises NotImplementedError.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from model.block import Block
from model.layers import Attention, apply_rotary, causal_mask, rotary_cos_sin
from model.model import BigMoE
from model.router import RouteResult


class KVBuffer:
    """Preallocated [B, Hkv, T_max, dh] keys and values; `length` positions are filled."""

    def __init__(
        self,
        b: int,
        kv_heads: int,
        t_max: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.k = torch.zeros(b, kv_heads, t_max, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write T new positions and return the filled view (old + new)."""
        t0, t1 = self.length, self.length + k.shape[2]
        if t1 > self.k.shape[2]:
            raise ValueError(f"cache full: {t1} > {self.k.shape[2]} positions")
        self.k[:, :, t0:t1] = k.to(self.k.dtype)
        self.v[:, :, t0:t1] = v.to(self.v.dtype)
        self.length = t1
        return self.k[:, :, :t1], self.v[:, :, :t1]

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def rows(self, b0: int, b1: int) -> KVBuffer:
        """A view of sequences b0..b1-1 sharing this buffer's storage (chunked prefill)."""
        view = KVBuffer.__new__(KVBuffer)
        view.k, view.v, view.length = self.k[b0:b1], self.v[b0:b1], self.length
        return view


@dataclass
class DecodeCache:
    early: list[KVBuffer]
    middle: list[KVBuffer]  # index j - 1
    final: list[KVBuffer]
    length: int = 0  # positions already processed

    def nbytes(self) -> int:
        return sum(c.nbytes() for c in [*self.early, *self.middle, *self.final])

    def buffers(self) -> list[KVBuffer]:
        return [*self.early, *self.middle, *self.final]

    def rows(self, b0: int, b1: int) -> DecodeCache:
        """Sequences b0..b1-1 as their own cache over shared storage. After prefilling every
        slice to the same length, call `set_length` on the parent."""
        return DecodeCache(
            [c.rows(b0, b1) for c in self.early],
            [c.rows(b0, b1) for c in self.middle],
            [c.rows(b0, b1) for c in self.final],
            self.length,
        )

    def set_length(self, n: int) -> None:
        self.length = n
        for c in self.buffers():
            c.length = n


def check_supported(model: BigMoE) -> None:
    cfg = model.cfg
    reasons = []
    if cfg.arch != "recurrent":
        reasons.append("layered arch")
    if cfg.halting.mode != "fixed":
        reasons.append(f"halting {cfg.halting.mode}")
    if cfg.attention.global_source != "per_iteration":
        reasons.append("global_source final_vector (L5d)")
    if cfg.streams.two_stream:
        reasons.append("two streams")
    if cfg.middle.per_iteration_attention:
        reasons.append("per-iteration attention (L5b/L5c)")
    if cfg.middle.h_every:
        reasons.append("H block (L5e)")
    if reasons:
        raise NotImplementedError("decode supports L5 only (ADR-049 §1): " + ", ".join(reasons))


def new_cache(
    model: BigMoE,
    batch: int,
    t_max: int,
    r: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> DecodeCache:
    check_supported(model)
    cfg = model.cfg
    a = cfg.attention
    r = cfg.middle.r_max if r is None else r
    dev = device or model.emb.weight.device
    dt = dtype or model.emb.weight.dtype

    def buf() -> KVBuffer:
        return KVBuffer(batch, a.kv_heads, t_max, a.head_dim, dev, dt)

    return DecodeCache(
        early=[buf() for _ in model.early],
        middle=[buf() for _ in range(r)],
        final=[buf() for _ in model.final],
    )


def _mask(t: int, length: int, device: torch.device) -> Tensor | None:
    """New rows at absolute positions length..length+t-1 over keys 0..length+t-1."""
    if t == 1:
        return None  # one query sees every cached key and itself
    return causal_mask(t, length + t, device, offset=length)


def attend(attn: Attention, u: Tensor, pos: Tensor, buf: KVBuffer, j: int | None = None) -> Tensor:
    """`Attention.forward` over the cache plus T new rows, with grouped K/V read in place
    (`enable_gqa`) instead of `repeat_interleave`: the same arithmetic, without copying the
    cache to every query head at each decode step."""
    b, t, _ = u.shape
    k, v = buf.append(*attn.kv(u, pos, j))
    q = attn._proj("q", attn.wq, u, j).view(b, t, attn.heads, attn.head_dim).transpose(1, 2)
    cos, sin = rotary_cos_sin(pos, attn.head_dim, attn.theta)
    q = apply_rotary(q, cos, sin)
    mask = _mask(t, buf.length - t, u.device)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    return attn._proj("o", attn.wo, out.transpose(1, 2).reshape(b, t, -1), j)


def block_step(blk: Block, c: Tensor, pos: Tensor, buf: KVBuffer) -> Tensor:
    """A skeleton block (single stream) over T new positions."""
    hc = blk.n1(c)
    a = attend(blk.attn, hc, pos, buf)
    if blk.form == "parallel":
        y, _ = blk._ff(hc)
        return c + a + y
    x = c + a
    y, _ = blk._ff(blk.n2(x))  # type: ignore[misc]
    return x + y


def depth_embed(model: BigMoE, j: int, device: torch.device) -> Tensor:
    return model.middle.depth(torch.tensor(j, device=device))


def owner_pre(
    model: BigMoE, j: int, c: Tensor, pos: Tensor, buf: KVBuffer
) -> tuple[Tensor, Tensor, RouteResult]:
    """Iteration j at the attention owner: (u [B,T,d], attention output, route over B*T rows)."""
    mid = model.middle
    e = depth_embed(model, j, c.device)
    u = mid.norm(c, e)
    a = attend(mid.attention_at(j), u, pos, buf, j)
    route = mid.moe.router(u.reshape(-1, u.shape[-1]), e)
    return u, a, route


def owner_post(c: Tensor, a: Tensor, expert_sum: Tensor) -> Tensor:
    """x^(j) = x + Attn + sum_i g_i E_i(u) (`01 §4.1`); expert_sum is already gate-weighted."""
    return c + a + expert_sum.view_as(c).to(c.dtype)


def local_experts(model: BigMoE, u: Tensor, route: RouteResult) -> Tensor:
    """Gate-weighted expert sum on this device, as `MoE.forward` computes it."""
    moe = model.middle.moe
    flat = u.reshape(-1, u.shape[-1])
    out = torch.zeros_like(flat)
    n, k = route.idx.shape
    token_of = torch.arange(n, device=u.device).repeat_interleave(k)
    expert_of, gate_of = route.idx.reshape(-1), route.gate.reshape(-1)
    for i, expert in enumerate(moe.experts):
        sel = (expert_of == i).nonzero(as_tuple=True)[0]
        if sel.numel():
            toks = token_of[sel]
            out.index_add_(0, toks, (expert(flat[toks]) * gate_of[sel, None]).to(out.dtype))
    return out


def middle_step(model: BigMoE, j: int, c: Tensor, pos: Tensor, buf: KVBuffer) -> Tensor:
    u, a, route = owner_pre(model, j, c, pos, buf)
    return owner_post(c, a, local_experts(model, u, route))


def positions(cache: DecodeCache, t: int, device: torch.device) -> Tensor:
    return torch.arange(cache.length, cache.length + t, device=device)


def early(model: BigMoE, tokens: Tensor, cache: DecodeCache) -> tuple[Tensor, Tensor]:
    """Embedding and early blocks for T new positions -> (c, positions)."""
    pos = positions(cache, tokens.shape[1], tokens.device)
    c, _ = model.embed(tokens)
    for blk, buf in zip(model.early, cache.early, strict=True):
        c = block_step(blk, c, pos, buf)
    return c, pos


def final(model: BigMoE, c: Tensor, pos: Tensor, cache: DecodeCache) -> Tensor:
    """Final blocks and the tied head -> logits [B, T, V]; advances the cache length."""
    for blk, buf in zip(model.final, cache.final, strict=True):
        c = block_step(blk, c, pos, buf)
    cache.length += c.shape[1]
    return model.head(c)


@torch.no_grad()
def step(model: BigMoE, tokens: Tensor, cache: DecodeCache) -> Tensor:
    """Append T positions (prefill chunk or one decode token); returns logits [B, T, V]."""
    c, pos = early(model, tokens, cache)
    for j, buf in enumerate(cache.middle, start=1):
        c = middle_step(model, j, c, pos, buf)
    return final(model, c, pos, cache)
