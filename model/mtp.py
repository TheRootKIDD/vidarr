"""Multi-token prediction heads (`docs/01 §7`, ADR-012).

Head 1 is the main next-token head (tied embedding). Heads 2..m are independent:
RMSNorm -> linear d->d adapter -> tied embedding, predicting x_{t+j}. Each
position contributes to exactly one auxiliary head per step (subsampling), so
m = 4 costs one extra logits pass instead of three. Each head's loss is the mean
over its own positions — an unbiased estimate of that head's full-position
loss — so no rescaling is applied (ADR-012 as corrected).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.config import MTPCfg


class MTPHeads(nn.Module):
    def __init__(self, d: int, cfg: MTPCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.adapters = nn.ModuleList(
            nn.Sequential(nn.RMSNorm(d), nn.Linear(d, d, bias=False)) for _ in range(cfg.m - 1)
        )
        for a in self.adapters:
            nn.init.zeros_(a[1].weight)  # start as the main head

    def aux_losses(
        self, h: Tensor, tokens: Tensor, emb_weight: Tensor, generator: torch.Generator | None
    ) -> tuple[Tensor, dict[str, float]]:
        """h: [B, T, d] final hidden; tokens [B, T]. Returns (weighted aux loss, metrics)."""
        m = self.cfg.m
        if m == 1:
            return h.new_zeros(()), {}
        b, t, _ = h.shape
        n_aux = m - 1
        # each position t < T - m gets exactly one auxiliary head j in 2..m
        valid_t = t - m
        assign = torch.randint(0, n_aux, (b, valid_t), device=h.device, generator=generator)
        # Head 1's greedy prediction at every position, for the acceptance metric below.
        # Forward-only and chunked, so it costs one extra logits pass with no backward
        # and never holds [B*T, V] at once.
        pred1 = self._greedy_head1(h, emb_weight).reshape(b, t)
        total = h.new_zeros(())
        metrics: dict[str, float] = {}
        for i, adapter in enumerate(self.adapters):
            j = i + 2  # predicts x_{t+j}
            sel = (assign == i).nonzero(as_tuple=True)
            if sel[0].numel() == 0:
                continue
            hs = adapter(h[sel[0], sel[1]])  # [n, d]
            logits = hs @ emb_weight.t()
            target = tokens[sel[0], sel[1] + j]
            loss = F.cross_entropy(logits.float(), target)
            lam = self.cfg.lambdas[j - 1] if j - 1 < len(self.cfg.lambdas) else 0.5 ** (j - 1)
            total = total + lam * loss
            draft = logits.argmax(-1)
            # Two metrics that must never be confused (I27). `mtp_top1_h{j}` is top-1
            # accuracy against the ground-truth token. `mtp_accept_h{j}` is greedy
            # speculative-decode acceptance as ADR-012/ADR-013 define H13's threshold:
            # head j's draft of x_{t+j}, made at position t, is accepted when it agrees
            # with head 1's own prediction of x_{t+j}, made at position t+j-1.
            verify = pred1[sel[0], sel[1] + j - 1]
            metrics[f"mtp_loss_h{j}"] = float(loss.detach())
            metrics[f"mtp_top1_h{j}"] = float((draft == target).float().mean())
            metrics[f"mtp_accept_h{j}"] = float((draft == verify).float().mean())
        return total, metrics

    @staticmethod
    @torch.no_grad()
    def _greedy_head1(h: Tensor, emb_weight: Tensor, chunk: int = 4096) -> Tensor:
        """argmax of the tied main head over every position, in row chunks."""
        flat = h.detach().reshape(-1, h.shape[-1])
        out = torch.empty(flat.shape[0], dtype=torch.long, device=h.device)
        for i in range(0, flat.shape[0], chunk):
            out[i : i + chunk] = (flat[i : i + chunk] @ emb_weight.t()).argmax(-1)
        return out
