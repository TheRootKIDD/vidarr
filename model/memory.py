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

L8q (ADR-055) adds `query_loss`, a training-only target for the query; `forward` is unchanged.

L8k (ADR-056, `memory.keys = "pca_fixed"`) replaces h^{1,2} by fixed keys and the dot score by a
distance score:
    k^h_i = P^h (mu^h_i - mean^h)        (P^h the top d_m/2 principal components of the table's
                                          h-th half-embeddings, mean^h their mean; frozen)
    s^h_i = tau (q^h . k^h_i - |k^h_i|^2 / 2) = -tau/2 |q^h - k^h_i|^2 + const(q)
with tau = exp(log_tau) learned. P and the mean come from `build_memory_pca.py` (`pca*.npy`,
`mean*.npy` beside the table); the toy table computes them in place.

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


def pca_halves(
    table: Tensor, dim: int, chunk: int = 1 << 16
) -> list[tuple[Tensor, Tensor, float]]:
    """ADR-056: per half of `table` [N, d_enc], (mean [d_enc/2], P [dim, d_enc/2], share of the
    variance kept). Covariance in fp64 over row chunks on the table's device; each component's
    sign is fixed so its largest-magnitude entry is positive (deterministic across runs)."""
    n, d_enc = table.shape
    half = d_enc // 2
    out = []
    for h in range(2):
        cols = slice(h * half, (h + 1) * half)
        s = torch.zeros(half, dtype=torch.float64, device=table.device)
        ss = torch.zeros(half, half, dtype=torch.float64, device=table.device)
        for i in range(0, n, chunk):
            x = table[i : i + chunk, cols].double()
            s += x.sum(0)
            ss += x.T @ x
        mean = s / n
        cov = ss / n - torch.outer(mean, mean)
        evals, evecs = torch.linalg.eigh(cov)  # ascending
        top = evecs[:, -dim:].flip(-1).T  # [dim, half], largest variance first
        idx = top.abs().argmax(-1, keepdim=True)
        top = top * torch.sign(top.gather(-1, idx))
        kept = float(evals[-dim:].sum() / evals.sum())
        out.append((mean.float().cpu(), top.float().cpu(), kept))
    return out


def load_pca(path: str | Path, dim: int) -> list[tuple[Tensor, Tensor]]:
    """(mean, P) per half from `build_memory_pca.py`'s output beside the table."""
    p = Path(path)
    out = []
    for h in (1, 2):
        mean = torch.from_numpy(np.load(p / f"mean{h}.npy"))
        P = torch.from_numpy(np.load(p / f"pca{h}.npy"))
        if P.shape[0] != dim:
            raise ValueError(f"pca{h}.npy has {P.shape[0]} components, key_dim / 2 = {dim}")
        out.append((mean.float(), P.float()))
    return out


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
        self.fixed = mc.keys == "pca_fixed"
        if not self.fixed:
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
        if self.fixed:  # ADR-056: k^h = P^h (mu^h - mean^h), frozen; tau starts at 1 / median gap
            if mc.table is None:
                pca = [(m, P) for m, P, _ in pca_halves(table.float(), half_key)]
            else:
                pca = load_pca(mc.table, half_key)
            keys = [(mu.float() - m) @ P.T for mu, (m, P) in zip((mu1, mu2), pca, strict=True)]
            self.register_buffer("k1", keys[0], persistent=False)
            self.register_buffer("k2", keys[1], persistent=False)
            gap = torch.cat([torch.cdist(k, k).pow(2).flatten() for k in keys])
            gap = gap[gap > 0].median()
            self.log_tau = nn.Parameter(-torch.log(gap).reshape(()).float())
        self.random_cells = False
        self.record = False
        self.records: list[tuple[Tensor, Tensor]] = []  # (cells, alpha) per call while recording

    def sub_keys(self) -> tuple[Tensor, Tensor]:
        """The grid's row and column keys: h^{1,2}(mu^{1,2}) learned, or the fixed PCA keys."""
        if self.fixed:
            return self.k1, self.k2
        return self.h1(self.mu1), self.h2(self.mu2)

    def score_halves(self, q1: Tensor, q2: Tensor) -> tuple[Tensor, Tensor]:
        """Each half's scores against its 1024 sub-keys: q . k (learned), or
        tau (q . k - |k|^2 / 2) (ADR-056), i.e. -tau/2 |q - k|^2 up to a per-query constant."""
        k1, k2 = (k.float() for k in self.sub_keys())
        s1, s2 = q1.float() @ k1.T, q2.float() @ k2.T
        if self.fixed:
            tau = self.log_tau.float().exp()
            s1 = tau * (s1 - 0.5 * (k1 * k1).sum(-1))
            s2 = tau * (s2 - 0.5 * (k2 * k2).sum(-1))
        return s1, s2

    def lookup(self, u: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        """Cells [..., k_m] (row-major grid index) and their scores [..., k_m]."""
        mc = self.mc
        q = self.q(torch.cat([u, e.expand(*u.shape[:-1], -1)], dim=-1))
        s1, s2 = self.score_halves(*q.float().chunk(2, dim=-1))
        t1, i1 = s1.topk(mc.half_topk, dim=-1)
        t2, i2 = s2.topk(mc.half_topk, dim=-1)
        cand = (t1[..., :, None] + t2[..., None, :]).flatten(-2)  # [..., half_topk^2]
        score, ci = cand.topk(mc.topk, dim=-1)
        a = i1.gather(-1, ci // mc.half_topk)
        b = i2.gather(-1, ci % mc.half_topk)
        return a * mc.grid + b, score

    def query_loss(self, u: Tensor, e: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """L8q (ADR-055): half-wise cross-entropy of the lookup's own sub-key scores toward the
        teacher's PQ code. `target` [..., 2] long (row a*, column b*; -1 = no target). The
        query's inputs are detached, so this trains W_m and h^{1,2} (or tau, L8k) only. Returns
        the summed loss over valid positions and their count (the caller averages)."""
        q = self.q(torch.cat([u.detach(), e.detach().expand(*u.shape[:-1], -1)], dim=-1))
        s1, s2 = self.score_halves(*q.float().chunk(2, dim=-1))
        s1, s2 = s1.flatten(0, -2), s2.flatten(0, -2)
        t = target.reshape(-1, 2)
        ce1 = F.cross_entropy(s1, t[:, 0], ignore_index=-1, reduction="sum")
        ce2 = F.cross_entropy(s2, t[:, 1], ignore_index=-1, reduction="sum")
        return 0.5 * (ce1 + ce2), (t[:, 0] >= 0).sum().to(ce1.dtype)

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
