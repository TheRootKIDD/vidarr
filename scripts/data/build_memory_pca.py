"""ADR-056 (L8k): the fixed per-half PCA projection of the L8 table, written beside it.

Reads `table.npy` (cell order, `040`'s build), computes per half the mean and the top `--dim`
principal components (`model.memory.pca_halves`, fp64 covariance on GPU, sign-fixed), and
writes `mean{1,2}.npy`, `pca{1,2}.npy` and `pca.json` (dimension, variance kept per half,
sha256 of every input and output) into the table directory. `040`'s own files and manifest are
not touched. Derived data: never committed. The record goes to `experiments/<id>/pca.json`.

    python -m scripts.data.build_memory_pca --id 044-l8k-gate
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model.memory import pca_halves
from scripts.data.build_wiki_memory import sha256

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", required=True, help="experiment id that records the build")
    ap.add_argument("--table", default="/mnt/nvme/memory/wiki-20231101-lead")
    ap.add_argument("--dim", type=int, default=64, help="components per half (key_dim / 2)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    t0 = time.time()
    tdir = Path(a.table)
    table = torch.from_numpy(np.load(tdir / "table.npy")).to(a.device).float()
    halves = pca_halves(table, a.dim)
    rec: dict = {"dim": a.dim, "inputs": {}, "outputs": {}, "variance_kept": []}
    for f in ("table.npy", "mu1.npy", "mu2.npy"):
        rec["inputs"][f] = sha256(tdir / f)
    for h, (mean, P, kept) in enumerate(halves, start=1):
        np.save(tdir / f"mean{h}.npy", mean.numpy())
        np.save(tdir / f"pca{h}.npy", P.numpy())
        rec["variance_kept"].append(kept)
    for f in ("mean1.npy", "mean2.npy", "pca1.npy", "pca2.npy"):
        rec["outputs"][f] = sha256(tdir / f)
    rec["wall_s"] = time.time() - t0
    (tdir / "pca.json").write_text(json.dumps(rec, indent=2))
    exp = ROOT / "experiments" / a.id
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "pca.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
