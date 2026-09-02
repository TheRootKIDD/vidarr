"""Corpus access for training: the ADR-014 `uint16` memmap shards on the NVMe.

Train batches are random contiguous windows of `context + 1` tokens drawn from
the train shards in proportion to their length (so document boundaries, marked
by EOS = 2, fall anywhere in a window — the loader does not re-align to
documents; `docs/06 §7`). Validation walks `val.bin` sequentially in fixed
windows so every run scores the same tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class MemmapCorpus:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.shards = [
            np.memmap(self.root / s["file"], dtype=np.uint16, mode="r")
            for s in self.manifest["train_shards"]
        ]
        self.lengths = np.array([len(s) for s in self.shards], dtype=np.int64)
        self.val = np.memmap(self.root / "val.bin", dtype=np.uint16, mode="r")
        assert self.manifest["vocab_size"] < 2**16

    @property
    def train_tokens(self) -> int:
        return int(self.lengths.sum())

    def train_batch(self, rng: np.random.Generator, batch: int, context: int) -> torch.Tensor:
        """[batch, context + 1] int64 windows sampled uniformly over train tokens."""
        n = context + 1
        weights = (self.lengths - n).clip(min=0).astype(np.float64)
        weights /= weights.sum()
        out = np.empty((batch, n), dtype=np.int64)
        for i in range(batch):
            s = rng.choice(len(self.shards), p=weights)
            start = rng.integers(0, self.lengths[s] - n)
            out[i] = self.shards[s][start : start + n]
        return torch.from_numpy(out)

    def val_windows(self, context: int, max_tokens: int | None = None) -> list[torch.Tensor]:
        """Fixed sequential windows over val.bin, [1, context + 1] each."""
        n = context + 1
        total = len(self.val) if max_tokens is None else min(len(self.val), max_tokens)
        return [
            torch.from_numpy(np.array(self.val[i : i + n], dtype=np.int64))[None]
            for i in range(0, total - n, context)
        ]
