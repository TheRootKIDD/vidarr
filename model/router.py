"""Top-k router with aux-loss-free balancing (`docs/01 §4.4`, ADR-004).

l_i = q(u, e_j)^T kappa_i + b_i + mu_tau,i ; S = top-k(l) ; g = softmax over S.
The bias b_i is *only* used for selection (DeepSeek-V3 practice): gates are the
softmax of the bias-free logits over the selected set. A load controller moves
b_i against the measured load after every step (`update_bias`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.config import RouterCfg


@dataclass
class RouteResult:
    idx: Tensor  # [N, k] expert indices
    gate: Tensor  # [N, k] gates, sum to 1 over k
    load: Tensor  # [n_experts] fraction of tokens routed to each expert (detached)
    aux_loss: Tensor  # scalar sequence-level balance loss (Switch-style), zero if disabled


class Router(nn.Module):
    def __init__(self, d: int, cfg: RouterCfg, n_experts: int, depth_dim: int = 0) -> None:
        super().__init__()
        self.cfg, self.n_experts = cfg, n_experts
        d_in = d + (depth_dim if cfg.use_depth_embed else 0)
        self.wq = nn.Linear(d_in, cfg.d_r, bias=False)
        self.keys = nn.Parameter(torch.randn(n_experts, cfg.d_r) / cfg.d_r**0.5)
        self.register_buffer("bias", torch.zeros(n_experts))
        self.use_depth = cfg.use_depth_embed and depth_dim > 0

    def logits(self, u: Tensor, e: Tensor | None) -> Tensor:
        if self.use_depth:
            if e is None:
                raise ValueError("router configured with depth embedding but none given")
            u = torch.cat([u, e.expand(*u.shape[:-1], e.shape[-1])], dim=-1)
        q = self.wq(u)
        return q @ self.keys.t()  # [N, n_experts]

    def forward(
        self, u: Tensor, e: Tensor | None = None, tenant_mask: Tensor | None = None
    ) -> RouteResult:
        """u: [N, d] flattened tokens. Returns top-k indices, gates, load, aux loss."""
        k = self.cfg.k
        raw = self.logits(u, e).float()
        select = raw + self.bias
        if tenant_mask is not None:
            select = select + tenant_mask
        idx = select.topk(k, dim=-1).indices  # [N, k]
        gate = F.softmax(raw.gather(-1, idx), dim=-1)  # bias-free gates
        load = torch.zeros(self.n_experts, device=u.device, dtype=torch.float32)
        load.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=u.device))
        load = load / max(1, idx.numel())
        if self.cfg.balance == "aux_free" and self.cfg.aux_loss_coef > 0:
            # Switch-style: n * sum_i f_i * P_i, f = fraction routed, P = mean softmax prob
            prob = F.softmax(raw, dim=-1).mean(0)
            aux = self.cfg.aux_loss_coef * self.n_experts * (load.detach() * prob).sum()
        else:
            aux = raw.new_zeros(())
        return RouteResult(idx=idx, gate=gate.to(u.dtype), load=load.detach(), aux_loss=aux)

    @torch.no_grad()
    def update_bias(self, load: Tensor) -> None:
        """Aux-loss-free controller: push overloaded experts down, underloaded up."""
        if self.cfg.balance != "aux_free":
            return
        target = 1.0 / self.n_experts
        self.bias -= self.cfg.bias_update_rate * torch.sign(load - target)
