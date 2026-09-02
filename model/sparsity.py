"""Weight-only 2:4 structured sparsity with a fixed mask (`docs/01 §8`, ADR-019).

After 1 % of the token budget, every expert weight matrix gets a magnitude
mask keeping the 2 largest of each 4 consecutive input weights; the mask is
then fixed for the rest of training. Quality-only on this rig (no sparse
training kernels, `docs/06 §2`): the mask is applied by a parametrization.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from model.layers import Expert


class FixedMask(nn.Module):
    def __init__(self, mask: Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, w: Tensor) -> Tensor:
        return w * self.mask


def mask_2_4(w: Tensor) -> Tensor:
    """[out, in] -> bool mask keeping the top-2 magnitudes in each group of 4 along `in`."""
    out, inn = w.shape
    assert inn % 4 == 0
    groups = w.detach().abs().reshape(out, inn // 4, 4)
    top = groups.topk(2, dim=-1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool).scatter_(-1, top, True)
    return mask.reshape(out, inn)


def apply_fixed_24_masks(model: nn.Module) -> int:
    """Mask every linear inside every `Expert`. Returns the number of masked matrices."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, Expert):
            for lin in (m for m in mod.modules() if isinstance(m, nn.Linear)):
                if parametrize.is_parametrized(lin, "weight"):
                    continue
                parametrize.register_parametrization(lin, "weight", FixedMask(mask_2_4(lin.weight)))
                n += 1
    return n


def sparsity_fraction(model: nn.Module) -> float:
    """Fraction of zero expert weights (sanity metric; 0.5 once masked)."""
    zeros = total = 0
    for mod in model.modules():
        if isinstance(mod, Expert):
            for lin in (m for m in mod.modules() if isinstance(m, nn.Linear)):
                w = lin.weight
                zeros += int((w == 0).sum())
                total += w.numel()
    return zeros / max(1, total)
