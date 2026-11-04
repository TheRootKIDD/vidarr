"""ADR-054: a read-only diagnostic of L8's lookup on a finished run (I40).

Four measurements, each read against ADR-054's fixed readings:
  1. own-cell rank: on in-table wiki-lead first-sentence tokens, the exact rank of the article's
     own cell among all grid^2 cells under the model's query, and its row / column rank per half;
  2. own-key oracle: the rank of each in-table cell when the query is the cell's own key, plus
     the singular values of W_m, h^1, h^2, W_v;
  3. score scale: top-1 minus top-k_m retrieved score, the k_m scores' std, each half's spread;
  4. training signal: cross-batch cosine of the gradients and |g| / |w| per parameter group on
     held-out FineWeb-Edu windows, and the branch's output norm against the MoE's per iteration.

Hooks read the memory's inputs; the module is not changed. Run on an idle GPU:
  python -m scripts.analysis.probe_l8_lookup --id 173-l8-screen --out 042-l8-lookup-diag
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import Tensor

from model.memory import KnowledgeMemory
from scripts.analysis.eval_wiki_lead import CORPUS, EOS, TOKENIZER, build_items
from scripts.analysis.probe_routing import RESULTS_DIR, load_run
from scripts.train.data import MemmapCorpus

GROUPS = {  # parameter-name prefixes per group; experts and attention are the references
    "W_m": ("middle.mem.q.",),
    "h12": ("middle.mem.h1.", "middle.mem.h2."),
    "W_v": ("middle.mem.out.",),
    "experts": ("middle.moe.experts",),
    "attention": ("middle.attn.",),
}


def pair_rank(s1: Tensor, s2: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Number of grid cells (i, j) with s1_i + s2_j > s1_a + s2_b, per row: the exact rank of
    cell (a, b) among all grid^2 (0 = best). s1, s2 [N, grid]; a, b [N]."""
    target = s1.gather(-1, a[:, None]) + s2.gather(-1, b[:, None])  # [N, 1]
    s2s = s2.sort(dim=-1).values.contiguous()
    thr = (target - s1).contiguous()  # cell (i, j) beats the target iff s2_j > thr_i
    n_le = torch.searchsorted(s2s, thr, right=True)
    return (s2.shape[-1] - n_le).sum(-1)


def half_scores(mem: KnowledgeMemory, u: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
    q = mem.q(torch.cat([u, e.expand(*u.shape[:-1], -1)], dim=-1))
    q1, q2 = q.float().chunk(2, dim=-1)
    k1, k2 = mem.sub_keys()
    return q1 @ k1.float().T, q2 @ k2.float().T


def stats(x: list[float] | np.ndarray) -> dict[str, float]:
    v = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p10": float(np.quantile(v, 0.1)),
        "p90": float(np.quantile(v, 0.9)),
        "n": float(v.size),
    }


@torch.no_grad()
def own_cell(model: Any, items: list[dict[str, Any]], dev: torch.device) -> dict[str, Any]:
    """Measurements 1 and 3 on the in-table articles' first-sentence tokens."""
    mem: KnowledgeMemory = model.middle.mem
    grid, k = mem.mc.grid, mem.mc.topk
    per_j: dict[int, dict[str, list[Tensor]]] = {}
    ctx: dict[str, Any] = {}

    def hook(_m: Any, args: tuple[Tensor, Tensor], _out: Tensor) -> None:
        u, e = args
        j = ctx["j"]
        ctx["j"] += 1
        sel = ctx["first"]  # [B, T] positions whose next token is in the first sentence
        if not sel.any():
            return
        s1, s2 = half_scores(mem, u, e)
        cells = ctx["cell"][:, None].expand_as(sel)[sel]
        s1n, s2n = s1[sel], s2[sel]
        a, b = cells // grid, cells % grid
        _, score = mem.lookup(u, e)
        sc = score[sel].float()
        d = per_j.setdefault(j, {x: [] for x in ("rank", "ra", "rb", "gap", "sd", "s1sd", "s2sd")})
        d["rank"].append(pair_rank(s1n, s2n, a, b).cpu())
        d["ra"].append((s1n > s1n.gather(-1, a[:, None])).sum(-1).cpu())
        d["rb"].append((s2n > s2n.gather(-1, b[:, None])).sum(-1).cpu())
        d["gap"].append((sc[:, 0] - sc[:, k - 1]).cpu())
        d["sd"].append(sc.std(-1).cpu())
        d["s1sd"].append(s1n.std(-1).cpu())
        d["s2sd"].append(s2n.std(-1).cpu())

    h = mem.register_forward_hook(hook)
    try:
        for b0 in range(0, len(items), 16):
            chunk = items[b0 : b0 + 16]
            L = max(len(x["ids"]) for x in chunk)
            toks = torch.full((len(chunk), L - 1), EOS, dtype=torch.long)
            first = torch.zeros((len(chunk), L - 1), dtype=torch.bool)
            for r, x in enumerate(chunk):
                toks[r, : len(x["ids"]) - 1] = torch.tensor(x["ids"][:-1])
                first[r, : len(x["split"])] = torch.from_numpy(x["split"] == 1)
            ctx.update(
                j=0, first=first.to(dev), cell=torch.tensor([x["cell"] for x in chunk], device=dev)
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.forward_hidden(toks.to(dev))
    finally:
        h.remove()
    n_cells = grid * grid
    out: dict[str, Any] = {"chance_rank_median": n_cells / 2, "chance_half_rank_median": grid / 2}
    for j, d in sorted(per_j.items()):
        cat = {x: torch.cat(v).double().numpy() for x, v in d.items()}
        out[f"iter_{j + 1}"] = {
            "own_cell_rank": stats(cat["rank"]),
            "own_cell_in_top32_share": float((cat["rank"] < k).mean()),
            "own_row_rank": stats(cat["ra"]),
            "own_col_rank": stats(cat["rb"]),
            "top1_minus_top32": stats(cat["gap"]),
            "retrieved_score_std": stats(cat["sd"]),
            "half1_score_std": stats(cat["s1sd"]),
            "half2_score_std": stats(cat["s2sd"]),
        }
    return out


@torch.no_grad()
def oracle(model: Any, items: list[dict[str, Any]], dev: torch.device) -> dict[str, Any]:
    """Measurement 2: each cell queried by its own key; singular values of the maps."""
    mem: KnowledgeMemory = model.middle.mem
    grid, k = mem.mc.grid, mem.mc.topk
    cells = torch.tensor([x["cell"] for x in items], device=dev)
    a, b = cells // grid, cells % grid
    k1, k2 = (x.float() for x in mem.sub_keys())
    s1, s2 = k1[a] @ k1.T, k2[b] @ k2.T
    rank = pair_rank(s1, s2, a, b).double().cpu().numpy()
    ra = (s1 > s1.gather(-1, a[:, None])).sum(-1).double().cpu().numpy()
    rb = (s2 > s2.gather(-1, b[:, None])).sum(-1).double().cpu().numpy()

    def sv(w: Tensor) -> dict[str, Any]:
        s = torch.linalg.svdvals(w.float()).cpu().numpy()
        p = s / s.sum()
        return {
            "shape": list(w.shape),
            "top5": s[:5].tolist(),
            "min": float(s[-1]),
            "effective_rank": float(np.exp(-(p * np.log(np.clip(p, 1e-30, None))).sum())),
            "frobenius": float(np.sqrt((s**2).sum())),
        }

    return {
        "own_key_rank": stats(rank),
        "own_key_in_top32_share": float((rank < k).mean()),
        "own_key_row_rank": stats(ra),
        "own_key_col_rank": stats(rb),
        "subkey_norm_h1": stats(k1.norm(dim=-1).cpu().numpy()),
        "subkey_norm_h2": stats(k2.norm(dim=-1).cpu().numpy()),
        "singular_values": {
            "W_m": sv(mem.q.weight),
            "h1": sv(mem.h1.weight),
            "h2": sv(mem.h2.weight),
            "W_v": sv(mem.out.weight),
        },
    }


def signal(model: Any, dev: torch.device, n_batches: int) -> dict[str, Any]:
    """Measurement 4: gradient coherence across disjoint held-out windows; branch norms."""
    cfg = model.cfg
    windows = MemmapCorpus(Path(CORPUS)).val_windows(cfg.context, None)[:n_batches]
    named = {n.removeprefix("module."): p for n, p in model.named_parameters()}
    groups = {g: [p for n, p in named.items() if n.startswith(pre)] for g, pre in GROUPS.items()}
    grads: dict[str, list[Tensor]] = {g: [] for g in GROUPS}
    model.train()
    for w in windows:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.loss(w.to(dev), r=cfg.middle.r_max)
        out.loss.backward()
        for g, ps in groups.items():
            grads[g].append(torch.cat([p.grad.float().flatten() for p in ps]).cpu())
    model.zero_grad(set_to_none=True)
    model.eval()
    res: dict[str, Any] = {"batches": n_batches, "tokens_per_batch": int(windows[0].numel() - 1)}
    for g, gs in grads.items():
        G = torch.stack(gs)
        Gn = torch.nn.functional.normalize(G, dim=-1)
        cos = Gn @ Gn.T
        off = cos[~torch.eye(len(gs), dtype=torch.bool)]
        wn = float(torch.cat([p.detach().float().flatten() for p in groups[g]]).norm())
        res[g] = {
            "params": int(G.shape[1]),
            "cross_batch_cosine_mean": float(off.mean()),
            "cross_batch_cosine_min": float(off.min()),
            "grad_norm_mean": float(G.norm(dim=-1).mean()),
            "weight_norm": wn,
            "grad_over_weight": float(G.norm(dim=-1).mean()) / max(wn, 1e-30),
            "mean_grad_norm_over_grad_norm": float(G.mean(0).norm() / G.norm(dim=-1).mean()),
        }

    # branch output norm against the MoE output's, per iteration (eval, no grad)
    mem, moe = model.middle.mem, model.middle.moe
    acc: dict[str, list[list[float]]] = {"mem": [], "moe": []}
    ctx = {"jm": 0, "je": 0}

    def put(key: str, idx: str, y: Tensor) -> None:
        j = ctx[idx]
        ctx[idx] += 1
        while len(acc[key]) <= j:
            acc[key].append([])
        acc[key][j].append(float(y.float().norm(dim=-1).mean()))

    hm = mem.register_forward_hook(lambda _m, _a, o: put("mem", "jm", o))
    he = moe.register_forward_hook(lambda _m, _a, o: put("moe", "je", o[0]))
    try:
        with torch.no_grad():
            for w in windows:
                ctx.update(jm=0, je=0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model.forward_hidden(w[:, :-1].to(dev))
    finally:
        hm.remove()
        he.remove()
    res["branch_norm_by_iteration"] = [
        {
            "mem": float(np.mean(m)),
            "moe": float(np.mean(e)),
            "ratio": float(np.mean(m) / np.mean(e)),
        }
        for m, e in zip(acc["mem"], acc["moe"], strict=True)
    ]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", default="173-l8-screen")
    ap.add_argument("--out", default="042-l8-lookup-diag")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grad-batches", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device(a.device)
    t0 = time.time()
    model, cfg, step = load_run(RESULTS_DIR / a.id, dev)
    tok = Tokenizer.from_file(TOKENIZER)
    items = [x for x in build_items(tok) if x["pool"] == "in_table"]
    res: dict[str, Any] = {"run": a.id, "step": step, "in_table_articles": len(items)}
    res["own_cell"] = own_cell(model, items, dev)
    res["oracle"] = oracle(model, items, dev)
    res["signal"] = signal(model, dev, a.grad_batches)
    res["peak_gib"] = torch.cuda.max_memory_allocated(dev) / 2**30
    res["wall_s"] = time.time() - t0
    out = RESULTS_DIR / a.out
    out.mkdir(exist_ok=True)
    (out / "diag.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
