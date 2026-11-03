"""The knowledge memory (L8, ADR-053, `docs/01 §4.6`): a frozen sentence table on a product grid.

    q        = W_m [u; e_j]                              (d + d_e -> d_m, two halves)
    s^1, s^2 = q^1 . h^1(mu^1),  q^2 . h^2(mu^2)         (each half against its grid sub-keys)
    cells    = top-k_m of s^1_a + s^2_b over the top-half_topk x top-half_topk pairs
    Mem      = W_v sum_n alpha_n e_n,  alpha = softmax over the k_m cells

`e` is the frozen encoder table in cell order (cell (a, b) -> row a * grid + b), mu^{1,2} the
frozen k-means codebooks of its halves; W_m, h^{1,2} and W_v train, and W_v starts at zero so
the branch starts as a no-op. The table and codebooks are non-persistent buffers: they are
derived data (Wikipedia, CC BY-SA 3.0) loaded from `memory.table`, never saved in a checkpoint.
With `memory.table = None` a seeded random table stands in (unit tests only).

Probe knobs (ADR-053 §8; no effect on training): `random_cells` replaces the retrieved cells by
uniformly random ones (reliance), `record` keeps each call's cells and alpha (lookup accuracy,
table use, entropy).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.config import MemoryCfg


def load_table(path: str | Path) -> tuple[Tensor, Tensor, Tensor, dict]:
    """(table [grid^2, d_enc] in cell order, mu1, mu2 [grid, d_enc/2], manifest) from a build."""
    p = Path(path)
    manifest = json.loads((p / "manifest.json").read_text())
    table = torch.from_numpy(np.load(p / "table.npy", mmap_mode="r")[:].copy())
    mu1 = torch.from_numpy(np.load(p / "mu1.npy"))
    mu2 = torch.from_numpy(np.load(p / "mu2.npy"))
    return table, mu1, mu2, manifest


def toy_table(mc: MemoryCfg, seed: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    """A seeded random L2-normalised table and codebooks of the configured shape (tests)."""
    g = torch.Generator().manual_seed(seed)
    table = F.normalize(torch.randn(mc.n_entries, mc.encoder_dim, generator=g), dim=-1)
    half = mc.encoder_dim // 2
    mu1 = torch.randn(mc.grid, half, generator=g) / half**0.5
    mu2 = torch.randn(mc.grid, half, generator=g) / half**0.5
    return table, mu1, mu2


class KnowledgeMemory(nn.Module):
    def __init__(self, mc: MemoryCfg, d: int, depth_dim: int) -> None:
        super().__init__()
        self.mc = mc
        half_enc, half_key = mc.encoder_dim // 2, mc.key_dim // 2
        self.q = nn.Linear(d + depth_dim, mc.key_dim, bias=False)  # W_m
        self.h1 = nn.Linear(half_enc, half_key, bias=False)
        self.h2 = nn.Linear(half_enc, half_key, bias=False)
        self.out = nn.Linear(mc.encoder_dim, d, bias=False)  # W_v = W_o g_theta
        nn.init.zeros_(self.out.weight)
        self.manifest: dict = {}
        if mc.table is None:
            table, mu1, mu2 = toy_table(mc)
        else:
            table, mu1, mu2, self.manifest = load_table(mc.table)
        if table.shape != (mc.n_entries, mc.encoder_dim):
            raise ValueError(f"table {tuple(table.shape)} != ({mc.n_entries}, {mc.encoder_dim})")
        if mu1.shape != (mc.grid, half_enc) or mu2.shape != mu1.shape:
            raise ValueError("codebooks must be [grid, encoder_dim / 2]")
        # frozen data: bf16 table (1.5 GiB at 2^20 x 768), fp32 codebooks; never checkpointed
        self.register_buffer("table", table.to(torch.bfloat16), persistent=False)
        self.register_buffer("mu1", mu1.float(), persistent=False)
        self.register_buffer("mu2", mu2.float(), persistent=False)
        self.random_cells = False
        self.record = False
        self.records: list[tuple[Tensor, Tensor]] = []  # (cells, alpha) per call while recording

    def sub_keys(self) -> tuple[Tensor, Tensor]:
        """kappa^{1,2} = h^{1,2}(mu^{1,2}): the grid's learned row and column keys."""
        return self.h1(self.mu1), self.h2(self.mu2)

    def lookup(self, u: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        """Cells [..., k_m] (row-major grid index) and their scores [..., k_m]."""
        mc = self.mc
        q = self.q(torch.cat([u, e.expand(*u.shape[:-1], -1)], dim=-1))
        q1, q2 = q.float().chunk(2, dim=-1)
        k1, k2 = self.sub_keys()
        t1, i1 = (q1 @ k1.float().T).topk(mc.half_topk, dim=-1)
        t2, i2 = (q2 @ k2.float().T).topk(mc.half_topk, dim=-1)
        cand = (t1[..., :, None] + t2[..., None, :]).flatten(-2)  # [..., half_topk^2]
        score, ci = cand.topk(mc.topk, dim=-1)
        a = i1.gather(-1, ci // mc.half_topk)
        b = i2.gather(-1, ci % mc.half_topk)
        return a * mc.grid + b, score

    def forward(self, u: Tensor, e: Tensor) -> Tensor:
        cells, score = self.lookup(u, e)
        if self.random_cells:
            cells = torch.randint_like(cells, self.mc.n_entries)
        alpha = torch.softmax(score, dim=-1)
        if self.record:
            self.records.append((cells.detach(), alpha.detach()))
        k = self.mc.topk
        m = F.embedding_bag(
            cells.reshape(-1, k),
            self.table,
            per_sample_weights=alpha.reshape(-1, k).to(self.table.dtype),
            mode="sum",
        )  # sum_n alpha_n e_n without materialising the k_m gathered rows
        return self.out(m.view(*u.shape[:-1], -1).to(self.out.weight.dtype))
