"""Top-k router with aux-loss-free balancing (`docs/01 §4.4`, ADR-004).

l_i = q(u, e_j)^T kappa_i + b_i + mu_tau,i ; S = top-k(l) ; g = softmax over S.
The bias b_i is *only* used for selection (DeepSeek-V3 practice): gates are the
softmax of the bias-free logits over the selected set. A load controller moves
b_i against the measured load after every step (`update_bias`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
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
    def __init__(
        self,
        d: int,
        cfg: RouterCfg,
        n_experts: int,
        depth_dim: int = 0,
        k: int | None = None,
        iter_slots: int = 0,
    ) -> None:
        super().__init__()
        self.cfg, self.n_experts = cfg, n_experts
        self.k = cfg.k if k is None else k  # active (sub-)experts; k*g under L7 (ADR-047)
        d_in = d + (depth_dim if cfg.use_depth_embed else 0)
        self.wq = nn.Linear(d_in, cfg.d_r, bias=False)
        self.keys = nn.Parameter(torch.randn(n_experts, cfg.d_r) / cfg.d_r**0.5)
        self.register_buffer("bias", torch.zeros(n_experts))
        self.use_depth = cfg.use_depth_embed and depth_dim > 0
        # ADR-069 (EQB): per-iteration biases (row j, 1-based; row 0 unused) and the step's
        # unbiased logits, recorded only between the start of a training forward and its loss
        # (`eqb_record`), so checkpoint replays in backward do not record twice
        self.eqb = cfg.balance in ("eqb_iteration", "eqb_step")
        self.eqb_per_iter = cfg.balance == "eqb_iteration"
        if self.eqb_per_iter:
            if iter_slots < 1:
                raise ValueError("eqb_iteration needs iter_slots = r_max")
            self.register_buffer("bias_iter", torch.zeros(iter_slots + 1, n_experts))
        self.eqb_record = False
        self._eqb_buf: list[tuple[int, Tensor]] = []
        self.last_eqb_err: float | None = None

    def logits(self, u: Tensor, e: Tensor | None) -> Tensor:
        if self.use_depth:
            if e is None:
                raise ValueError("router configured with depth embedding but none given")
            u = torch.cat([u, e.expand(*u.shape[:-1], e.shape[-1])], dim=-1)
        q = self.wq(u)
        return q @ self.keys.t()  # [N, n_experts]

    def forward(
        self,
        u: Tensor,
        e: Tensor | None = None,
        tenant_mask: Tensor | None = None,
        j: int | None = None,
    ) -> RouteResult:
        """u: [N, d] flattened tokens; j: the middle-block iteration (EQB per iteration)."""
        k = self.k
        raw = self.logits(u, e).float()
        if self.eqb_per_iter:
            if j is None:
                raise ValueError("eqb_iteration needs the iteration index")
            select = raw + self.bias_iter[j]
        else:
            select = raw + self.bias
        if self.eqb and self.eqb_record and self.training and torch.is_grad_enabled():
            self._eqb_buf.append((j or 0, raw.detach().to(torch.bfloat16)))
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

    # ---- ADR-069: Exact Quantile Balancing ---------------------------------------------------
    @torch.no_grad()
    def eqb_update(self, rounds: int = 3, steps: int = 40) -> None:
        """Set the biases from this step's recorded logits (all micro-batches, all ranks): each
        expert's bias becomes the exact cutoff at which it is in the biased top-k of a k/N_e
        share of the pairs. Jacobi rounds over experts (each cutoff depends on the others'
        biases); per round, the cutoff of expert i is the (1 - k/N_e) quantile of
        l_ti - c_t,-i (c = the k-th largest biased logit among the other experts), found by
        bisection on all-reduced counts. Every rank calls this every step (collectives match)."""
        if not self.eqb:
            return
        buf, self._eqb_buf = self._eqb_buf, []
        keys = range(1, self.bias_iter.shape[0]) if self.eqb_per_iter else [0]
        errs = []
        for key in keys:
            rows = [x for jj, x in buf if (jj if self.eqb_per_iter else 0) == key]
            dev = self.bias.device
            lg = torch.cat(rows).float() if rows else torch.zeros(0, self.n_experts, device=dev)
            n = torch.tensor([float(lg.shape[0])], device=dev)
            _all_reduce(n)
            if n.item() == 0:
                continue
            b0 = self.bias_iter[key] if self.eqb_per_iter else self.bias
            b, err = _eqb_solve(lg, b0.clone(), self.k, float(n.item()), rounds, steps)
            b0.copy_(b)
            errs.append(err)
        self.last_eqb_err = max(errs) if errs else None

    @torch.no_grad()
    def update_bias(self, load: Tensor) -> None:
        """Aux-loss-free controller: push overloaded experts down, underloaded up."""
        if self.cfg.balance != "aux_free":
            return
        target = 1.0 / self.n_experts
        self.bias -= self.cfg.bias_update_rate * torch.sign(load - target)


def _all_reduce(x: Tensor, op: str = "sum") -> Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op={"sum": dist.ReduceOp.SUM, "min": dist.ReduceOp.MIN,
                               "max": dist.ReduceOp.MAX}[op])
    return x


def _eqb_margins(lg: Tensor, b: Tensor, k: int) -> Tensor:
    """m_ti = l_ti - c_ti, with c_ti the k-th largest *biased* logit among the experts other
    than i: expert i is in row t's biased top-k iff m_ti > -b_i. Stored as fp16 (chunked)."""
    out = torch.empty(lg.shape, dtype=torch.float16, device=lg.device)
    for i in range(0, lg.shape[0], 65536):
        x = lg[i : i + 65536]
        s = x + b
        top = s.topk(min(k + 1, s.shape[1]), dim=-1).values
        kth = top[:, k - 1 : k]
        k1 = top[:, k : k + 1] if top.shape[1] > k else torch.full_like(kth, -1e9)
        c = torch.where(s >= kth, k1, kth)  # i in the top-k: others' k-th = the (k+1)-th
        out[i : i + 65536] = (x - c).to(torch.float16)
    return out


def _eqb_solve(
    lg: Tensor, b: Tensor, k: int, n_total: float, rounds: int, steps: int
) -> tuple[Tensor, float]:
    """Jacobi rounds of exact per-expert cutoffs (see `Router.eqb_update`). Returns the biases
    and the realised share's largest relative error on these rows (all ranks)."""
    share = k / b.numel()
    for _ in range(rounds):
        m = _eqb_margins(lg, b, k) if lg.shape[0] else lg  # fp16, [n, N_e]
        lo = m.min(0).values.float() if lg.shape[0] else torch.full_like(b, float("inf"))
        hi = m.max(0).values.float() if lg.shape[0] else torch.full_like(b, -float("inf"))
        lo, hi = _all_reduce(lo.clone(), "min") - 1e-3, _all_reduce(hi.clone(), "max")
        for _ in range(steps):  # t_i: a `share` fraction of rows has m_ti > t_i; then b_i = -t_i
            mid = (lo + hi) / 2
            cnt = (m > mid.to(m.dtype)).sum(0).float() if lg.shape[0] else torch.zeros_like(b)
            frac = _all_reduce(cnt) / n_total
            lo = torch.where(frac > share, mid, lo)
            hi = torch.where(frac > share, hi, mid)
        b = -hi
    sel = (lg + b).topk(k, dim=-1).indices if lg.shape[0] else lg.new_zeros(0, k).long()
    cnt = torch.zeros_like(b).scatter_add_(0, sel.reshape(-1), torch.ones(sel.numel(),
                                                                       device=b.device))
    frac = _all_reduce(cnt) / n_total
    return b, float(((frac - share).abs() / share).max())

