"""L8e's host-resident table and its lazy sparse Adam (ADR-071 §3).

One copy for all DDP ranks, in shared memory (`/dev/shm`): bf16 values (what the forward reads),
fp32 master weights, Adam moments m, v. Per optimiser step:

1. every rank sums its training forwards' row gradients by row (GPU), divided by the world size
   so the update matches DDP's mean, and writes (rows, grads) to its own shared-memory file;
2. barrier; rank 0 merges the ranks' files, applies Adam to the touched rows only (moments of
   untouched rows are not decayed: lazy Adam; bias correction by the global step), and rewrites
   their bf16 values;
3. barrier. No rank gathers again before the update has landed, so there is no staleness.

The table's gradients are not in the trainer's global-norm clip (they never reach a parameter).
The trained values are saved beside the checkpoint (`engram_values.bf16`) for evaluation.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor

from model.config import EngramCfg
from model.engram import Engram

SHM = Path("/dev/shm")
VALUES_FILE = "engram_values.bf16"


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _shared(path: Path, numel: int, dtype: torch.dtype) -> Tensor:
    return torch.from_file(str(path), shared=True, size=numel, dtype=dtype)


def load_values(ckpt_dir: Path, cfg: EngramCfg, d: int) -> Tensor:
    """The trained table for evaluation: memory-mapped, read-only use."""
    return _shared(ckpt_dir / VALUES_FILE, cfg.rows * d, torch.bfloat16).view(cfg.rows, d)


class HostEngram:
    def __init__(
        self, cfg: EngramCfg, d: int, run_id: str, rank: int, world: int, seed: int
    ) -> None:
        self.cfg, self.d, self.rank, self.world = cfg, d, rank, world
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
        self.dir = SHM / f"bigmoe-engram-{tag}"
        n = cfg.rows * d
        if rank == 0:
            self.dir.mkdir(exist_ok=True)
            for name in ("values", "master", "m", "v"):
                (self.dir / name).unlink(missing_ok=True)
        _barrier()
        self.values = _shared(self.dir / "values", n, torch.bfloat16).view(cfg.rows, d)
        self.master = _shared(self.dir / "master", n, torch.float32).view(cfg.rows, d)
        self.m = _shared(self.dir / "m", n, torch.float32).view(cfg.rows, d)
        self.v = _shared(self.dir / "v", n, torch.float32).view(cfg.rows, d)
        if rank == 0:
            g = torch.Generator().manual_seed(seed + 0xE6)
            for i in range(0, cfg.rows, 1 << 16):  # chunked init: no 6 GB temporary
                blk = torch.randn(min(1 << 16, cfg.rows - i), d, generator=g) * cfg.init_std
                self.master[i : i + blk.shape[0]] = blk
                self.values[i : i + blk.shape[0]] = blk.to(torch.bfloat16)
            self.m.zero_()
            self.v.zero_()
        _barrier()
        self.t = 0
        self.last: dict[str, float] = {}

    @torch.no_grad()
    def step(self, module: Engram, lr: float, betas: tuple[float, float], eps: float) -> None:
        t0 = time.time()
        pending, module.pending = module.pending, []
        n_ord = len(self.cfg.ngrams)
        if pending:
            ids = torch.cat([i.reshape(-1, n_ord) for i, _ in pending])
            grads = torch.cat(
                [
                    (r.grad if r.grad is not None else torch.zeros_like(r)).reshape(-1, self.d)
                    for _, r in pending
                ]
            )
            flat = ids.reshape(-1)
            uid, inv = torch.unique(flat, return_inverse=True)
            acc = torch.zeros(uid.numel(), self.d, device=grads.device)
            acc.index_add_(0, inv, grads.float().repeat_interleave(n_ord, dim=0))
            acc /= self.world
            payload = (uid.cpu(), acc.cpu())
        else:
            payload = (torch.zeros(0, dtype=torch.long), torch.zeros(0, self.d))
        torch.save(payload, self.dir / f"grad.{self.rank}.pt")
        t1 = time.time()
        _barrier()
        if self.rank == 0:
            parts = [torch.load(self.dir / f"grad.{r}.pt") for r in range(self.world)]
            rows = torch.cat([p[0] for p in parts])
            gsum = torch.cat([p[1] for p in parts])
            u, inv = torch.unique(rows, return_inverse=True)
            g = torch.zeros(u.numel(), self.d).index_add_(0, inv, gsum)
            self.t += 1
            b1, b2 = betas
            m = self.m[u].mul_(b1).add_(g, alpha=1 - b1)
            v = self.v[u].mul_(b2).addcmul_(g, g, value=1 - b2)
            self.m[u], self.v[u] = m, v
            mhat = m / (1 - b1**self.t)
            vhat = v / (1 - b2**self.t)
            w = self.master[u] - lr * mhat / (vhat.sqrt() + eps)
            self.master[u] = w
            self.values[u] = w.to(torch.bfloat16)
            self.last = {"engram_rows": float(u.numel())}
        else:
            self.t += 1
        _barrier()
        self.last.update({"engram_scatter_s": t1 - t0, "engram_update_s": time.time() - t1})

    def save(self, ckpt_dir: Path) -> None:
        """rank 0: the bf16 values beside the checkpoint (evaluation reads them)."""
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_dir / (VALUES_FILE + ".tmp")
        _write(self.values, tmp)
        tmp.replace(ckpt_dir / VALUES_FILE)

    def close(self) -> None:
        _barrier()
        if self.rank == 0:
            for f in self.dir.iterdir():
                f.unlink()
            self.dir.rmdir()


def _write(t: Tensor, path: Path) -> None:
    out = _shared(path, t.numel(), t.dtype).view_as(t)
    out.copy_(t)
