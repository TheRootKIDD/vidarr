"""Multi-token prediction heads (`docs/01 §7`, ADR-012).

Head 1 is the main next-token head (tied embedding). Heads 2..m are independent:
RMSNorm -> linear d->d adapter -> tied embedding, predicting x_{t+j}. Each
position contributes to exactly one auxiliary head per step (subsampling), so
m = 4 costs one extra logits pass instead of three. `subsample: 1.0` switches
that off and trains every head on every position (L3-full, I27 #3). Each head's loss is the mean
over its own positions — an unbiased estimate of that head's full-position
loss — so no rescaling is applied (ADR-012 as corrected).

L3b (ADR-036): with `curriculum: forward` the token budget is split into m
equal phases and phase i trains heads 2..i only (phase 1 is next-token only);
positions are subsampled among the active heads, inactive heads get a zero
gradient, and evaluation always runs every head.

L3c (ADR-039): `MTPModules`, DeepSeek-V3-style sequential modules. Module j
combines module j-1's state with the embedding of x_{t+j-1} (teacher-forced in
training; at evaluation the anchored greedy chain feeds each module the previous
draft at the same position) and runs one dense block over all positions. Loss
subsampling is ADR-012's, so the extra logits cost stays one pass.

Acceptance is measured against head 1 = the main head, output norm included
(`01 §7`, I34).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.block import Block
from model.config import ModelCfg, MTPCfg


class MTPHeads(nn.Module):
    def __init__(self, d: int, cfg: MTPCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.adapters = nn.ModuleList(
            nn.Sequential(nn.RMSNorm(d), nn.Linear(d, d, bias=False)) for _ in range(cfg.m - 1)
        )
        for a in self.adapters:
            nn.init.zeros_(a[1].weight)  # start as the main head
        self.active_heads = cfg.m - 1  # auxiliary heads trained now (L3b's curriculum)

    def set_progress(self, frac: float) -> None:
        """L3b (ADR-036): in phase i of m (frac = step / total steps) heads 2..i train."""
        if self.cfg.curriculum == "forward":
            self.active_heads = min(self.cfg.m - 1, int(frac * self.cfg.m))

    def aux_losses(
        self,
        h: Tensor,
        tokens: Tensor,
        emb_weight: Tensor,
        out_norm: nn.Module,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, dict[str, float]]:
        """h: [B, T, d] final hidden; tokens [B, T+1]; out_norm: the main head's norm.

        Returns (weighted aux loss, metrics)."""
        m = self.cfg.m
        if m == 1:
            return h.new_zeros(()), {}
        b, t, _ = h.shape
        n_aux = self.active_heads if self.training else m - 1
        if n_aux == 0:
            # L3b phase 1: next-token only; a zero gradient keeps every head in DDP's graph
            return self._zero_touch(h, 0), {}
        valid_t = t - m
        full = self.cfg.subsample == 1.0
        if full:
            # L3-full (I27 #3): every position trains every auxiliary head.
            assign = None
        else:
            # ADR-012 default: each position t < T - m trains exactly one head j in 2..m
            # (among the heads active in L3b's current phase)
            assign = torch.randint(0, n_aux, (b, valid_t), device=h.device, generator=generator)
        # Head 1's greedy prediction at every position, for the acceptance metric below.
        # Forward-only and chunked, so it costs one extra logits pass with no backward
        # and never holds [B*T, V] at once.
        with torch.no_grad():
            pred1 = greedy(out_norm(h), emb_weight)  # head 1 = the main head, norm included (I34)
        total = h.new_zeros(())
        metrics: dict[str, float] = {}
        for i, adapter in enumerate(self.adapters):
            j = i + 2  # predicts x_{t+j}
            if i >= n_aux:
                break  # inactive in this L3b phase; touched with a zero gradient below
            if assign is None:
                sel = torch.ones(b, valid_t, dtype=torch.bool, device=h.device).nonzero(
                    as_tuple=True
                )
            else:
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
        return total + self._zero_touch(h, n_aux), metrics

    def _zero_touch(self, h: Tensor, first: int) -> Tensor:
        """0 x the parameters of heads first+2..m, so DDP sees a (zero) gradient for each."""
        z = h.new_zeros(())
        for adapter in list(self.adapters)[first:]:
            z = z + 0.0 * sum(p.sum() for p in adapter.parameters())
        return z



@torch.no_grad()
def greedy(hn: Tensor, emb_weight: Tensor, chunk: int = 4096) -> Tensor:
    """argmax over the tied vocabulary of already-normalised states [..., d], in row chunks."""
    flat = hn.detach().reshape(-1, hn.shape[-1])
    out = torch.empty(flat.shape[0], dtype=torch.long, device=hn.device)
    for i in range(0, flat.shape[0], chunk):
        out[i : i + chunk] = (flat[i : i + chunk] @ emb_weight.t()).argmax(-1)
    return out.reshape(hn.shape[:-1])


class MTPModules(nn.Module):
    """L3c (ADR-039): sequential modules; module j (2..m) predicts x_{t+j} at position t."""

    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.cfg = cfg.mtp
        d, n = cfg.d, cfg.mtp.m - 1
        self.h_norms = nn.ModuleList(nn.RMSNorm(d) for _ in range(n))
        self.e_norms = nn.ModuleList(nn.RMSNorm(d) for _ in range(n))
        self.projs = nn.ModuleList(nn.Linear(2 * d, d, bias=False) for _ in range(n))
        self.blocks = nn.ModuleList(
            Block(cfg, "dense", cfg.mtp.module_d_ff, "parallel") for _ in range(n)
        )
        self.out_norms = nn.ModuleList(nn.RMSNorm(d) for _ in range(n))
        self.active_heads = n

    def set_progress(self, frac: float) -> None:
        """No curriculum in L3c (ADR-039)."""

    def module(self, i: int, h_prev: Tensor, e_in: Tensor, pos: Tensor) -> Tensor:
        """Module j = i + 2: [norm(h^{j-1}); norm(E x_in)] -> 2d->d -> dense block."""
        z = self.projs[i](torch.cat([self.h_norms[i](h_prev), self.e_norms[i](e_in)], dim=-1))
        out, _, _ = self.blocks[i](z, None, pos)
        return out

    def aux_losses(
        self,
        h: Tensor,
        tokens: Tensor,
        emb_weight: Tensor,
        out_norm: nn.Module,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, dict[str, float]]:
        """h: [B, T, d] trunk hidden; tokens [B, T+1]. Training is teacher-forced with one
        module's loss per valid position; evaluation runs the anchored greedy chain and
        scores every valid position."""
        m = self.cfg.m
        b, t, _ = h.shape
        valid_t = t - m
        dev = h.device
        pos = torch.arange(t, device=dev)
        with torch.no_grad():
            pred1 = greedy(out_norm(h), emb_weight)  # [B, T]: head 1 at s predicts x_{s+1}
        if self.training:
            assign = torch.randint(0, m - 1, (b, valid_t), device=dev, generator=generator)
        total = h.new_zeros(())
        metrics: dict[str, float] = {}
        h_prev = h
        x_in = pred1  # evaluation: the chain starts from head 1's draft of x_{t+1}
        for i in range(m - 1):
            j = i + 2  # predicts x_{t+j}
            if self.training:
                idx = (pos + j - 1).clamp(max=t)  # teacher forcing: x_{t+j-1}; tail unused
                x_in = tokens[:, idx]
            h_j = self.module(i, h_prev, F.embedding(x_in, emb_weight), pos)
            hn = self.out_norms[i](h_j)
            if self.training:
                sel = (assign == i).nonzero(as_tuple=True)
            else:
                sel = torch.ones(b, valid_t, dtype=torch.bool, device=dev).nonzero(as_tuple=True)
            if sel[0].numel():
                logits = hn[sel[0], sel[1]] @ emb_weight.t()
                target = tokens[sel[0], sel[1] + j]
                loss = F.cross_entropy(logits.float(), target)
                lam = self.cfg.lambdas[j - 1] if j - 1 < len(self.cfg.lambdas) else 0.5 ** (j - 1)
                total = total + lam * loss
                draft = logits.argmax(-1)
                verify = pred1[sel[0], sel[1] + j - 1]
                metrics[f"mtp_loss_h{j}"] = float(loss.detach())
                metrics[f"mtp_top1_h{j}"] = float((draft == target).float().mean())
                metrics[f"mtp_accept_h{j}"] = float((draft == verify).float().mean())
            else:  # no position drew this module: keep its parameters in DDP's graph
                total = total + 0.0 * sum(p.sum() for p in self._params(i))
            if not self.training:
                x_in = greedy(hn, emb_weight)  # the next module reads this module's draft
            h_prev = h_j
        return total, metrics

    def _params(self, i: int) -> list[Tensor]:
        mods = (self.h_norms[i], self.e_norms[i], self.projs[i], self.blocks[i], self.out_norms[i])
        return [p for mod in mods for p in mod.parameters()]
