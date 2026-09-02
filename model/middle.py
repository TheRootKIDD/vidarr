"""The shared middle block M (`docs/01 §4`), applied r_t times.

    u      = DepthNorm(x, e_j)
    x^(j)  = x + Attn(u; local per-iteration K/V + global final-vector cache)
               + MoE(u, e_j)                                   (parallel branches, §4.1)

Depth conditioning through e_j (ADR-003); local range W over the current
segment's per-iteration caches and, with `global_source = final_vector`, the
global cache G of completed tokens' final vectors (ADR-021); ragged depth with
last-state output and K/V stored once for halted tokens (ADR-010/011).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from model.config import ModelCfg
from model.layers import Attention, DepthNorm, local_window_mask
from model.moe import MoE


@dataclass
class IterState:
    """Everything that persists across iterations of one segment."""

    c: Tensor  # [B, T, d]
    p: Tensor | None  # [B, T, d]
    active: Tensor  # [B, T] bool — still iterating
    r: Tensor  # [B, T] int — iterations executed so far
    cum_halt: Tensor  # [B, T] float — cumulative halting probability
    remainder: Tensor  # [B, T] float — ACT remainder at the halting step (for ponder cost)
    frozen_kv: tuple[Tensor, Tensor] | None  # [B, Hkv, T, dh] K/V of halted tokens (ADR-011)
    frozen: Tensor  # [B, T] bool — K/V frozen from this iteration on
    loads: list[Tensor]  # per-iteration expert load histograms (logging)


class MiddleBlock(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        a, m = cfg.attention, cfg.middle
        self.cfg = cfg
        self.depth = nn.Embedding(m.r_max + 1, m.depth_embed)  # e_j, j = 1..r_max (0 unused)
        self.norm = DepthNorm(cfg.d, m.depth_embed)
        self.attn = Attention(cfg.d, a.heads, a.kv_heads, a.head_dim, a.rope_theta)
        self.moe = MoE(cfg, depth_dim=m.depth_embed)
        # norm applied to final vectors before the G K/V; only exists when G is used (L5d+)
        self.g_norm = nn.RMSNorm(cfg.d) if a.global_source == "final_vector" else None
        self.halt = nn.Linear(cfg.d + m.depth_embed, 1)
        nn.init.zeros_(self.halt.weight)
        nn.init.constant_(self.halt.bias, -2.0)  # start with a low halting probability

    # ---- the global cache -----------------------------------------------------
    def global_entry(self, c_final: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """One K/V entry per completed token from its final middle-layer vector (§4.3)."""
        if self.g_norm is None:
            raise RuntimeError("global cache requested with global_source = per_iteration")
        if self.cfg.attention.segment_memory_grad == "stop":
            c_final = c_final.detach()  # ADR-023: no gradient into earlier segments' vectors
        return self.attn.kv(self.g_norm(c_final), positions)

    # ---- checkpoint-friendly wrapper -----------------------------------------------
    def iteration_flat(
        self,
        j: int,
        positions: Tensor,
        force_active: bool,
        c: Tensor,
        p: Tensor | None,
        active: Tensor,
        r: Tensor,
        cum: Tensor,
        rem: Tensor,
        fk: Tensor | None,
        fv: Tensor | None,
        frozen: Tensor,
        mem_k: Tensor | None,
        mem_v: Tensor | None,
    ) -> tuple[Tensor, ...]:
        """`iteration` over flat tensors so `torch.utils.checkpoint` can replay it."""
        st = IterState(
            c, p, active, r, cum, rem, None if fk is None else (fk, fv), frozen, []
        )
        mem = None if mem_k is None else (mem_k, mem_v)
        out, aux = self.iteration(st, j, positions, mem, force_active)
        fk2, fv2 = (None, None) if out.frozen_kv is None else out.frozen_kv
        load = out.loads[-1]
        return (
            out.c, out.p, out.active, out.r, out.cum_halt, out.remainder,
            fk2, fv2, out.frozen, aux, load,
        )

    # ---- one iteration ---------------------------------------------------------
    def iteration(
        self,
        st: IterState,
        j: int,
        positions: Tensor,
        mem_kv: tuple[Tensor, Tensor] | None,
        force_active: bool,
    ) -> tuple[IterState, Tensor]:
        """Apply M once (iteration j, 1-based). Returns the new state and the MoE aux loss."""
        cfg = self.cfg
        b, t, d = st.c.shape
        e = self.depth(torch.tensor(j, device=st.c.device))
        uc = self.norm(st.c, e)
        # per-iteration local K/V from the context stream; halted tokens keep their frozen entry
        fresh = self.attn.kv(uc, positions)
        if st.frozen_kv is not None:
            fm = st.frozen[:, None, :, None]
            k = torch.where(fm, st.frozen_kv[0], fresh[0])
            v = torch.where(fm, st.frozen_kv[1], fresh[1])
        else:
            k, v = fresh
        # tokens that halted before this iteration and are not yet frozen: freeze now (ADR-011)
        newly = ~st.active & ~st.frozen
        if newly.any():
            base_k, base_v = (k, v) if st.frozen_kv is None else st.frozen_kv
            nm = newly[:, None, :, None]
            frozen_kv = (torch.where(nm, k, base_k), torch.where(nm, v, base_v))
            frozen = st.frozen | newly
        else:
            frozen_kv, frozen = st.frozen_kv, st.frozen
        # key set = [global memory | local per-iteration], mask = [all | local window].
        # L5 baseline (`per_iteration`): the "local" range is the whole context, i.e. plain
        # causal attention over same-iteration caches; L5d: a window W within the segment.
        window = cfg.attention.window if cfg.attention.global_source == "final_vector" else t
        mask = local_window_mask(t, window, st.c.device)
        plain_causal = mem_kv is None and window >= t
        if mem_kv is not None:
            k = torch.cat([mem_kv[0], k], dim=2)
            v = torch.cat([mem_kv[1], v], dim=2)
            ones = torch.ones(t, mem_kv[0].shape[2], dtype=torch.bool, device=mask.device)
            mask = torch.cat([ones, mask], dim=1)
        streams = [(st.c, uc)]
        if st.p is not None:
            streams.append((st.p, self.norm(st.p, e)))
        new, aux = [], st.c.new_zeros(())
        for i, (x, u) in enumerate(streams):
            m_i = mask
            if i == 1 and not cfg.streams.p_sees_own_context:
                m_i = mask.clone()
                loc = m_i[:, -t:]
                loc &= ~torch.eye(t, dtype=torch.bool, device=mask.device)
                if mem_kv is None:
                    loc[0, 0] = True
            fast = plain_causal and (i == 0 or cfg.streams.p_sees_own_context)
            a = self.attn(u, positions, (k, v), None if fast else m_i, is_causal=fast)
            y, lm = self.moe(u, e)
            aux = aux + lm
            x_new = x + a + y
            # ragged depth: halted tokens keep their last computed state (ADR-010)
            new.append(torch.where(st.active[..., None], x_new, x))
        st.loads.append(self.moe.last_route.load if self.moe.last_route is not None else None)
        # halting head on the context stream (§4.7)
        h = torch.sigmoid(self.halt(torch.cat([uc, e.expand(b, t, -1)], dim=-1))).squeeze(-1)
        r = st.r + st.active.long()
        if force_active:
            active = st.active
            cum = st.cum_halt
            rem = st.remainder
            # fixed-r training: keep the (untrained) halting head in the graph so DDP's
            # reducer sees a gradient for every parameter (a zero one)
            aux = aux + 0.0 * h.sum()
        else:
            cum = st.cum_halt + h * st.active
            halts_now = st.active & (cum >= 1 - cfg.halting.epsilon) & (r >= cfg.middle.r_min)
            rem = torch.where(halts_now, 1 - (st.cum_halt), st.remainder)
            active = st.active & ~halts_now
        p_new = new[1] if len(new) > 1 else None
        out = IterState(new[0], p_new, active, r, cum, rem, frozen_kv, frozen, st.loads)
        return out, aux
