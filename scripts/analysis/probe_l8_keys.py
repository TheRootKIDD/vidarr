"""ADR-056 §5 (rig `044`): the key ceiling, what a perfect query can retrieve through the lookup.

For each in-table wiki-lead article, the query is its first sentence's own encoder embedding
(the table row of its cell, fp16 from `table.npy`), mapped into each key space, and run through
the full product-key path: top `half_topk` per half, `half_topk`^2 candidate cells, top `k_m`.
The ceiling is the share whose own cell is among the `k_m` retrieved. Three key constructions:

  pca64      ADR-056: q^h = P^h (e^h - mean^h), keys P^h (mu^h - mean^h), distance score (gate);
  full384    q^h = e^h, keys mu^h, distance score (d_m = 768; for information);
  learned    a run's h^{1,2}: q^h = h^h(e^h), keys h^h(mu^h), dot score (for information).

Gate: pca64's ceiling >= 0.5. Also reported: the exact own-cell rank among all grid^2 and own
row / column ranks per half. No model is trained; run on an idle GPU:
  python -m scripts.analysis.probe_l8_keys --out 044-l8k-gate --learned-id 174-l8q-screen
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import Tensor

from model.memory import load_pca
from scripts.analysis.eval_wiki_lead import TOKENIZER, build_items
from scripts.analysis.probe_l8_lookup import pair_rank, stats
from scripts.analysis.probe_routing import RESULTS_DIR, load_run

GATE = 0.5


def retrieve(s1: Tensor, s2: Tensor, half_topk: int, k_m: int, grid: int) -> Tensor:
    """The lookup's product search on given half scores [N, grid] -> cells [N, k_m]."""
    t1, i1 = s1.topk(half_topk, dim=-1)
    t2, i2 = s2.topk(half_topk, dim=-1)
    cand = (t1[:, :, None] + t2[:, None, :]).flatten(1)
    _, ci = cand.topk(k_m, dim=-1)
    a = i1.gather(-1, ci // half_topk)
    b = i2.gather(-1, ci % half_topk)
    return a * grid + b


def dist_score(q: Tensor, k: Tensor) -> Tensor:
    """q . k - |k|^2 / 2: -|q - k|^2 / 2 up to a per-query constant (any tau > 0 ranks alike)."""
    return q @ k.T - 0.5 * (k * k).sum(-1)


def ceiling(
    e1: Tensor, e2: Tensor, cells: Tensor, score: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    half_topk: int, k_m: int, grid: int,
) -> dict[str, Any]:
    s1, s2 = score(e1, e2)
    got = retrieve(s1, s2, half_topk, k_m, grid)
    a, b = cells // grid, cells % grid
    return {
        "own_cell_retrieved_share": float((got == cells[:, None]).any(-1).float().mean()),
        "own_cell_rank": stats(pair_rank(s1, s2, a, b).double().cpu().numpy()),
        "own_row_rank": stats((s1 > s1.gather(-1, a[:, None])).sum(-1).double().cpu().numpy()),
        "own_col_rank": stats((s2 > s2.gather(-1, b[:, None])).sum(-1).double().cpu().numpy()),
    }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="044-l8k-gate")
    ap.add_argument("--table", default="/mnt/nvme/memory/wiki-20231101-lead")
    ap.add_argument("--learned-id", default="174-l8q-screen", help="run whose h^{1,2} to score")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--grid", type=int, default=1024)
    ap.add_argument("--half-topk", type=int, default=32)
    ap.add_argument("--k-m", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    t0 = time.time()
    tdir = Path(a.table)
    items = [x for x in build_items(Tokenizer.from_file(TOKENIZER)) if x["pool"] == "in_table"]
    cells = torch.tensor([x["cell"] for x in items], device=dev)
    table = np.load(tdir / "table.npy", mmap_mode="r")
    emb = torch.from_numpy(table[cells.cpu().numpy()].astype(np.float32)).to(dev)
    e1, e2 = emb.chunk(2, dim=-1)
    mu1 = torch.from_numpy(np.load(tdir / "mu1.npy")).float().to(dev)
    mu2 = torch.from_numpy(np.load(tdir / "mu2.npy")).float().to(dev)
    (m1, P1), (m2, P2) = ((m.to(dev), P.to(dev)) for m, P in load_pca(tdir, a.dim))
    k1, k2 = (mu1 - m1) @ P1.T, (mu2 - m2) @ P2.T
    kw = {"half_topk": a.half_topk, "k_m": a.k_m, "grid": a.grid}
    res: dict[str, Any] = {
        "in_table_articles": len(items),
        "gate_line": GATE,
        "pca": json.loads((tdir / "pca.json").read_text()),
    }
    res["pca64"] = ceiling(
        e1, e2, cells,
        lambda x1, x2: (dist_score((x1 - m1) @ P1.T, k1), dist_score((x2 - m2) @ P2.T, k2)),
        **kw,
    )
    res["full384"] = ceiling(
        e1, e2, cells, lambda x1, x2: (dist_score(x1, mu1), dist_score(x2, mu2)), **kw
    )
    if a.learned_id:
        model, _cfg, step = load_run(RESULTS_DIR / a.learned_id, dev)
        mem = model.middle.mem
        h1, h2 = mem.h1.float(), mem.h2.float()
        res["learned"] = {"run": a.learned_id, "step": step} | ceiling(
            e1, e2, cells,
            lambda x1, x2: (h1(x1) @ h1(mu1).T, h2(x2) @ h2(mu2).T),
            **kw,
        )
        del model
    res["gate_pass"] = res["pca64"]["own_cell_retrieved_share"] >= GATE
    res["wall_s"] = time.time() - t0
    out = RESULTS_DIR / a.out
    out.mkdir(exist_ok=True)
    (out / "ceiling.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
