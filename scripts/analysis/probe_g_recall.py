"""ANN recall over the global cache's keys, NoPE vs RoPE (ADR-066 §6, rig `050`; H22's 2nd clause).

For held-out windows, a forward hook on the middle block's attention sees every iteration's
queries and the global keys G of the earlier segments. On the last segment of each window
(largest G), per iteration and KV head, it compares the exact top-K global keys by score
with an HNSW search (faiss, inner product, M 32, efSearch 64) over that window's G keys.
The query is the one the model scores G with: rotated under rope_scope = all (L5d), unrotated
under local (L5d-nope), so the ranking is the model's own. Reported: recall@K per iteration
and overall, with a bootstrap SE over windows.

Read-only on a checkpoint; GPU forward, faiss on CPU. Run with the GPUs idle:
`python -m scripts.analysis.probe_g_recall --id 182-l5d-nope-screen --context 2048`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from model.layers import apply_rotary, rotary_cos_sin
from scripts.analysis.eval_context import CORPUS
from scripts.analysis.probe_routing import RESULTS_DIR, load_run
from scripts.train.data import MemmapCorpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--queries", type=int, default=64, help="sampled queries per head")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device)
    run_dir = RESULTS_DIR / a.id
    model, cfg, step = load_run(run_dir, dev)
    if cfg.attention.global_source != "final_vector":
        raise SystemExit("needs a model with the global cache G (L5d family)")
    scope = cfg.attention.rope_scope
    seg = cfg.attention.window
    last0 = (a.context // seg - 1) * seg  # start of the last segment
    attn = model.middle.attn
    rep = attn.heads // attn.kv_heads
    gen = torch.Generator().manual_seed(0)
    per_window: list[list[float]] = []  # recall per (iteration, kv head) for the current window
    per_iter: dict[int, list[float]] = {}
    call = {"j": 0}

    def hook(_mod, args, kwargs):  # noqa: ANN001
        u, pos, (k, _v) = args[0], args[1], args[2]
        if int(pos[0]) != last0:
            return
        call["j"] += 1
        j = call["j"]
        t = u.shape[1]
        n_g = k.shape[2] - t
        if n_g < a.k:
            return
        q = attn.wq(u).view(1, t, attn.heads, attn.head_dim).transpose(1, 2)[0]  # [H, t, dh]
        if scope == "all":
            cos, sin = rotary_cos_sin(pos, attn.head_dim, attn.theta)
            q = apply_rotary(q[None], cos, sin)[0]
        keys = k[0, :, :n_g].float()  # [Hkv, n_g, dh]
        for h in range(attn.kv_heads):
            kh = np.ascontiguousarray(keys[h].cpu().numpy())
            index = faiss.IndexHNSWFlat(kh.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 64
            index.add(kh)
            qs = q[h * rep : (h + 1) * rep].reshape(-1, attn.head_dim).float()
            pick = torch.randperm(qs.shape[0], generator=gen)[: a.queries]
            qq = qs[pick.to(qs.device)]
            exact = (qq @ keys[h].T).topk(a.k, dim=-1).indices.cpu().numpy()
            _, approx = index.search(np.ascontiguousarray(qq.cpu().numpy()), a.k)
            rec = np.mean([len(set(e) & set(p)) / a.k for e, p in zip(exact, approx, strict=True)])
            per_window[-1].append(float(rec))
            per_iter.setdefault(j, []).append(float(rec))

    handle = attn.register_forward_pre_hook(hook, with_kwargs=True)
    windows = MemmapCorpus(Path(CORPUS)).val_windows(a.context, None)[: a.windows]
    t0 = time.time()
    with torch.no_grad():
        for w in windows:
            per_window.append([])
            call["j"] = 0
            model.forward_hidden(w[:, :-1].to(dev))
    handle.remove()
    means = np.array([np.mean(x) for x in per_window if x])
    boot = np.random.default_rng(0)
    se = float(np.std([boot.choice(means, means.size).mean() for _ in range(1000)]))
    res = {
        "id": a.id,
        "step": step,
        "rope_scope": scope,
        "context": a.context,
        "windows": int(means.size),
        "k": a.k,
        "hnsw": {"M": 32, "efSearch": 64, "metric": "inner_product"},
        "recall": float(means.mean()),
        "recall_se": se,
        "recall_per_iteration": {str(j): float(np.mean(v)) for j, v in sorted(per_iter.items())},
        "wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps(res, indent=2))
    out = run_dir / f"probe_g_recall_ctx{a.context}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
