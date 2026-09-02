"""Dropless MoE: route, gather per expert, run, scatter-add gated outputs.

`docs/01 §4.5`. The granularity knob g (L7) turns N_e experts of width d_ff into
N_e*g sub-experts of width d_ff/g with the same k — per-chip batch 1/g, same
fabric bytes. The loop over experts is deliberate (readability; N_e <= 128).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from model.config import ModelCfg
from model.layers import Expert
from model.router import Router, RouteResult


class MoE(nn.Module):
    def __init__(self, cfg: ModelCfg, depth_dim: int = 0) -> None:
        super().__init__()
        g = cfg.experts.granularity
        self.n_experts = cfg.router.n_experts * g
        self.d_ff = cfg.expert_width() // g
        self.router = Router(cfg.d, cfg.router, self.n_experts, depth_dim)
        self.experts = nn.ModuleList(
            Expert(cfg.d, self.d_ff, cfg.experts.layers) for _ in range(self.n_experts)
        )
        self.last_route: RouteResult | None = None

    def forward(self, u: Tensor, e: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """u: [..., d] -> (output [..., d], aux loss). Stores the route for logging."""
        shape = u.shape
        flat = u.reshape(-1, shape[-1])
        route = self.router(flat, e)
        self.last_route = route
        out = torch.zeros_like(flat)
        n, k = route.idx.shape
        token_of = torch.arange(n, device=u.device).repeat_interleave(k)  # [N*k]
        expert_of = route.idx.reshape(-1)
        gate_of = route.gate.reshape(-1)
        for i, expert in enumerate(self.experts):
            sel = (expert_of == i).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                # no tokens this step: a zero-weighted pass keeps the expert's gradient
                # defined (all-zero) so DDP's reducer sees every parameter every step
                out = out + 0.0 * expert(flat[:1]).sum()
                continue
            toks = token_of[sel]
            y = expert(flat[toks]) * gate_of[sel, None]
            out.index_add_(0, toks, y.to(out.dtype))
        return out.view(shape), route.aux_loss
