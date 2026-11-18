"""L10b′: 4-bit weight fake-quant for quality measurement (ADR-073; Ampere has no FP4).

Weights of every linear in attention and in the SwiGLU FFNs (experts and skeleton) are rounded
per group of 64 consecutive input weights, with one bf16 absmax scale per group:
- `fp4_e2m1`: to the E2M1 grid {0, 0.5, 1, 1.5, 2, 3, 4, 6} x scale, scale = absmax / 6;
- `int4` (eval-only side reading): asymmetric, 16 levels between the group's min and max.
Embeddings, the tied head, norms and the router stay bf16. In training the rounding passes the
gradient straight through. Swapped in at build when `numerics.qat_format` is set; switched on by
`set_qat` (the trainer, from `qat_start_frac` of the steps; evaluation, always).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fake_quant(w: Tensor, fmt: str, group: int) -> Tensor:
    """Round-to-nearest of w [out, in] per `group` input weights; returns w's dtype and shape."""
    out_f, in_f = w.shape
    if in_f % group:
        raise ValueError(f"in_features {in_f} not a multiple of the group {group}")
    g = w.float().reshape(out_f, in_f // group, group)
    if fmt == "fp4_e2m1":
        scale = (g.abs().amax(-1, keepdim=True) / 6.0).to(torch.bfloat16).float()
        x = (g / scale.clamp_min(1e-12)).clamp(-6.0, 6.0)
        grid = torch.tensor(E2M1, device=w.device)
        mid = (grid[1:] + grid[:-1]) / 2  # round to nearest (a value on a midpoint goes down)
        q = grid[torch.bucketize(x.abs(), mid)] * x.sign()
        deq = q * scale
    elif fmt == "int4":
        lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        scale = ((hi - lo) / 15.0).to(torch.bfloat16).float().clamp_min(1e-12)
        zero = torch.round(-lo / scale)
        deq = (torch.clamp(torch.round(g / scale) + zero, 0, 15) - zero) * scale
    else:
        raise ValueError(f"unknown fake-quant format {fmt}")
    return deq.reshape(out_f, in_f).to(w.dtype)


class QATLinear(nn.Linear):
    """nn.Linear whose weight is fake-quantised when `qat_on` (straight-through gradient)."""

    qat_fmt: str = "fp4_e2m1"
    qat_group: int = 64
    qat_on: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self.qat_on:
            w = w + (fake_quant(w.detach(), self.qat_fmt, self.qat_group) - w).detach()
        return F.linear(x, w, self.bias)


def swap_linears(model: nn.Module, fmt: str, group: int) -> list[QATLinear]:
    """Turn the attention and SwiGLU linears into QATLinear in place (off until `set_qat`)."""
    from model.layers import Attention, SwiGLU

    out = []
    for mod in model.modules():
        if isinstance(mod, Attention | SwiGLU):
            for child in mod.children():
                if type(child) is nn.Linear:
                    child.__class__ = QATLinear
                    child.qat_fmt, child.qat_group, child.qat_on = fmt, group, False
                    out.append(child)
    return out


def set_qat(model: nn.Module, on: bool, fmt: str | None = None) -> int:
    n = 0
    for mod in model.modules():
        if isinstance(mod, QATLinear):
            mod.qat_on = on
            if fmt is not None:
                mod.qat_fmt = fmt
            n += 1
    return n
