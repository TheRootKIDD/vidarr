"""L8e: a hashed n-gram table read once per token (ADR-071, `docs/03 §2.1`).

    h_n(t)  = hash_n(x_{t-n+1}, ..., x_t) mod rows   (a seed per order n; collisions intended)
    e_t     = RMSNorm(sum_n T[h_n(t)])
    u_t     = W_e e_t                                     (W_e zero-initialised: step 0 is L5)

`u_t` is added to the middle block's *input* at every iteration (the block reads c + u), not
written into the residual stream, so it is not accumulated r times (ADR-071 §2). History before
the window's first token is a padding id, the same in training and evaluation.

Placement `gpu`: T is an ordinary embedding. Placement `host` (the L8e run): T lives in shared
host memory outside the module (`attach_host_table`); the forward gathers the rows on the CPU,
and in training keeps the gathered sum as a leaf whose gradient the trainer scatters back to the
host table after backward (lazy sparse Adam, `scripts/train/engram_host.py`).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from model.config import EngramCfg

_MIX1, _MIX2 = 0xBF58476D1CE4E5B9 - 2**64, 0x94D049BB133111EB - 2**64  # splitmix64, as int64
_PRIMES = (0x9E3779B97F4A7C15 - 2**64, 0xC2B2AE3D27D4EB4F - 2**64, 0x165667B19E3779F9)


def _mix(h: Tensor) -> Tensor:
    """splitmix64's finaliser on int64 (wrapping arithmetic; arithmetic shifts are masked)."""
    m = (1 << 63) - 1
    h = h ^ ((h >> 30) & (m >> 29))
    h = h * _MIX1
    h = h ^ ((h >> 27) & (m >> 26))
    h = h * _MIX2
    return h ^ ((h >> 31) & (m >> 30))


def ngram_ids(tokens: Tensor, cfg: EngramCfg, pad: int) -> Tensor:
    """tokens [B, T] -> table rows [B, T, len(ngrams)] (int64)."""
    b, t = tokens.shape
    x = tokens.long()
    padded = torch.cat([torch.full((b, max(cfg.ngrams) - 1), pad, dtype=x.dtype,
                                   device=x.device), x], dim=1)
    out = []
    for n in cfg.ngrams:
        h = torch.full_like(x, cfg.seed + 7919 * n)
        for off in range(n):  # x_{t-n+1+off}
            lo = padded.shape[1] - t - (n - 1) + off
            h = _mix(h ^ (padded[:, lo : lo + t] * _PRIMES[off % len(_PRIMES)]))
        out.append(h & (cfg.rows - 1))
    return torch.stack(out, dim=-1)


class Engram(nn.Module):
    def __init__(self, cfg: EngramCfg, d: int, vocab: int) -> None:
        super().__init__()
        self.cfg, self.d, self.pad = cfg, d, vocab
        self.table = nn.Embedding(cfg.rows, d) if cfg.placement == "gpu" else None
        if self.table is not None:
            nn.init.normal_(self.table.weight, std=cfg.init_std)
        self.norm = nn.RMSNorm(d)
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.host_table: Tensor | None = None  # [rows, d] bf16, shared host memory
        self.pending: list[tuple[Tensor, Tensor]] = []  # (ids, leaf rows-sum) per training forward

    def attach_host_table(self, table: Tensor) -> None:
        if self.cfg.placement != "host" or table.shape != (self.cfg.rows, self.d):
            raise ValueError("host table: placement host and shape [rows, d]")
        self.host_table = table

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens [B, T] -> u [B, T, d]."""
        ids = ngram_ids(tokens, self.cfg, self.pad)
        if self.table is not None:
            rows = self.table(ids).sum(2)
        else:
            if self.host_table is None:
                raise RuntimeError("L8e placement host: attach_host_table first")
            got = self.host_table[ids.cpu().reshape(-1)].view(*ids.shape, self.d)
            rows = got.to(tokens.device, non_blocking=True).float().sum(2)
            if self.training and torch.is_grad_enabled():
                rows.requires_grad_(True)
                self.pending.append((ids, rows))
        return self.proj(self.norm(rows.to(self.proj.weight.dtype)))
