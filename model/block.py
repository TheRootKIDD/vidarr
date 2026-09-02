"""Standard pre-norm block, sequential or parallel form, dense or MoE feed-forward.

Serves two roles: the layered baselines L0–L3 (`docs/03 §2`) and the early /
final skeleton blocks of the recurrent design (`docs/01 §5`, ADR-006: dense,
parallel form, width 4d). Streams: `forward(c, p)` — K/V come from `c` only;
`p` (when present) queries `c` (ADR-002); a prediction-only block updates `p`
alone.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from model.config import ModelCfg
from model.layers import Attention, DepthNorm, SwiGLU, causal_mask
from model.moe import MoE


class Block(nn.Module):
    def __init__(
        self, cfg: ModelCfg, ff: str, d_ff: int, form: str, pred_only: bool = False
    ) -> None:
        super().__init__()
        a = cfg.attention
        self.cfg, self.form, self.pred_only = cfg, form, pred_only
        self.n1 = DepthNorm(cfg.d)
        self.attn = Attention(cfg.d, a.heads, a.kv_heads, a.head_dim, a.rope_theta)
        self.n2 = DepthNorm(cfg.d) if self.form == "sequential" else None
        self.ff: nn.Module = MoE(cfg) if ff == "moe" else SwiGLU(cfg.d, d_ff)
        self.is_moe = ff == "moe"

    def _ff(self, h: Tensor) -> tuple[Tensor, Tensor]:
        if self.is_moe:
            return self.ff(h)
        return self.ff(h), h.new_zeros(())

    def forward(
        self, c: Tensor, p: Tensor | None, positions: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """c, p: [B, T, d] (p may be None). Returns (c', p', aux_loss)."""
        b, t, _ = c.shape
        hc = self.n1(c)
        kv = self.attn.kv(hc, positions)
        streams = [c] if p is None else [c, p]
        hs = [hc] if p is None else [hc, self.n1(p)]
        masks = _stream_masks(self.cfg, t, c.device, n=len(streams))
        outs, aux = [], c.new_zeros(())
        for i, (x, h) in enumerate(zip(streams, hs, strict=True)):
            if self.pred_only and i == 0 and p is not None:
                outs.append(x)  # prediction-only block: c passes through unchanged
                continue
            a = self.attn(h, positions, kv, masks[i], is_causal=masks[i] is None)
            if self.form == "parallel":
                y, lf = self._ff(h)
                x = x + a + y
            else:
                x = x + a
                y, lf = self._ff(self.n2(x))  # type: ignore[misc]
                x = x + y
            aux = aux + lf
            outs.append(x)
        return outs[0], (outs[1] if p is not None else None), aux


def _stream_masks(cfg: ModelCfg, t: int, device: torch.device, n: int) -> list[Tensor | None]:
    """Per-stream [T, T] masks: c is causal (fast path, None); p sees c_{<=t} or c_{<t} (L4b)."""
    masks: list[Tensor | None] = [None]
    if n == 2:
        if cfg.streams.p_sees_own_context:
            masks.append(None)
        else:
            m = causal_mask(t, t, device) & ~torch.eye(t, dtype=torch.bool, device=device)
            m[0, 0] = True  # position 0 has nothing earlier; let it see itself to avoid NaN
            masks.append(m)
    return masks
