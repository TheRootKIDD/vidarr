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
            metrics[f"mtp_loss_h{j}"] = float(loss.detach())
            metrics[f"mtp_acc_h{j}"] = float((logits.argmax(-1) == target).float().mean())
        return total, metrics
