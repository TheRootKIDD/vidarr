"""Build L8's knowledge table (ADR-053 §1–§4, §9a): Wikipedia lead sentences on a product grid.

Steps, each logged with its wall-clock:
  1. download `wikimedia/wikipedia` config `20231101.en` at the pinned revision (parquet);
  2. extract each article's first sentence (§2): the first paragraph cut at the first `.`, `!`
     or `?` followed by whitespace and an upper-case letter (a paragraph without such a boundary
     is one sentence); keep 40–300 characters; drop titles starting "List of" or containing
     "(disambiguation)" and sentences containing "may refer to";
  3. sample N_m = 2^20 eligible articles uniformly (seed 0); draw the two eval pools with
     seed 1 — in-table from the sampled articles, out-of-table from the eligible rest (§7);
  4. encode the sampled sentences with `BAAI/bge-base-en-v1.5` at the pinned revision (CLS
     pooling, L2-normalised; fp16 autocast, checked against fp32 on a subset) on every GPU;
  5. k-means, 1024 centroids per 384-d half (seed 0, 25 iterations, on GPU);
  6. greedy balanced placement (seed 0 order): each sentence takes the free cell (a, b) that
     minimises ||e^1 - mu^1_a||^2 + ||e^2 - mu^2_b||^2 over its top-32 x top-32 centroid pairs,
     else over top-128 x top-128, else over every free cell (the last tier is not in ADR-053;
     its count is reported);
  7. write, under `--out` (derived data, never committed — CC BY-SA 3.0 / GFDL):
     `table.npy` [2^20, 768] float16 in cell order (cell (a, b) -> row a * 1024 + b),
     `mu1.npy`, `mu2.npy` [1024, 384] float32, `cell_article_ids.npy` (article id per cell),
     `sentences.jsonl` (cell order), `eval_pools.jsonl`, and `manifest.json` (revisions,
     filters, counts, distortion, sha256 of every file);
  and into `experiments/<id>/`: `result.json` (the manifest) and `sampled_article_ids.npy`
  (ids only, no text).

Run on idle GPUs: `python -m scripts.data.build_wiki_memory --id 040-wiki-memory`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATASET, CONFIG = "wikimedia/wikipedia", "20231101.en"
DATA_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
ENCODER, ENC_REV = "BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
N_M, GRID, HALF = 1 << 20, 1024, 384
POOL = 2000
MIN_CH, MAX_CH = 40, 300
BOUNDARY = re.compile(r"[.!?](?=\s+[A-Z])")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- 1–2: download and extract ---------------------------------------------------------------
def first_paragraph(text: str) -> str:
    return text.strip().split("\n", 1)[0].strip()


def first_sentence(para: str) -> str:
    m = BOUNDARY.search(para)
    return para[: m.end()] if m else para


def extract(raw: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    files = sorted((raw / CONFIG).glob("train-*.parquet"))
    counts = {"articles": 0, "dropped_title": 0, "dropped_refer": 0, "dropped_length": 0}
    rows: list[dict[str, Any]] = []
    for f in files:
        t = pq.read_table(f, columns=["id", "title", "text"])
        for aid, title, text in zip(
            t.column("id").to_pylist(),
            t.column("title").to_pylist(),
            t.column("text").to_pylist(),
            strict=True,
        ):
            counts["articles"] += 1
            para = first_paragraph(text)
            sent = first_sentence(para)
            if title.startswith("List of") or "(disambiguation)" in title:
                counts["dropped_title"] += 1
            elif "may refer to" in sent:
                counts["dropped_refer"] += 1
            elif not MIN_CH <= len(sent) <= MAX_CH:
                counts["dropped_length"] += 1
            else:
                rows.append({"id": int(aid), "title": title, "sentence": sent, "lead": para})
        log(f"  {f.name}: {counts['articles']} articles, {len(rows)} eligible")
    counts["eligible"] = len(rows)
    return counts, rows


# ---- 4: encode --------------------------------------------------------------------------------
def encode(sentences: list[str], enc_dir: Path, batch: int = 512) -> np.ndarray:
    """CLS-pooled, L2-normalised BGE vectors [n, 768] float32, sharded over every GPU."""
    from tokenizers import Tokenizer
    from transformers import AutoModel

    tok = Tokenizer.from_file(str(enc_dir / "tokenizer.json"))
    tok.enable_truncation(512)
    ids = [e.ids for e in tok.encode_batch(sentences)]
    order = np.argsort([len(x) for x in ids], kind="stable")
    out = np.zeros((len(sentences), 768), dtype=np.float32)
    n_gpu = torch.cuda.device_count()
    chunks = np.array_split(order, n_gpu)

    def work(rank: int) -> None:
        dev = torch.device(f"cuda:{rank}")
        model = AutoModel.from_pretrained(enc_dir).to(dev).eval()
        idx = chunks[rank]
        for s in range(0, len(idx), batch):
            sel = idx[s : s + batch]
            L = max(len(ids[i]) for i in sel)
            x = torch.zeros(len(sel), L, dtype=torch.long)
            mask = torch.zeros(len(sel), L, dtype=torch.long)
            for r, i in enumerate(sel):
                x[r, : len(ids[i])] = torch.tensor(ids[i])
                mask[r, : len(ids[i])] = 1
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                h = model(input_ids=x.to(dev), attention_mask=mask.to(dev)).last_hidden_state
            out[sel] = F.normalize(h[:, 0].float(), dim=-1).cpu().numpy()

    threads = [threading.Thread(target=work, args=(r,)) for r in range(n_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def encode_fp32(sentences: list[str], enc_dir: Path) -> np.ndarray:
    from tokenizers import Tokenizer
    from transformers import AutoModel

    tok = Tokenizer.from_file(str(enc_dir / "tokenizer.json"))
    tok.enable_truncation(512)
    tok.enable_padding()
    enc = tok.encode_batch(sentences)
    x = torch.tensor([e.ids for e in enc], device="cuda:0")
    m = torch.tensor([e.attention_mask for e in enc], device="cuda:0")
    model = AutoModel.from_pretrained(enc_dir).to("cuda:0").eval()
    with torch.no_grad():
        h = model(input_ids=x, attention_mask=m).last_hidden_state
    return F.normalize(h[:, 0].float(), dim=-1).cpu().numpy()


# ---- 5: k-means -------------------------------------------------------------------------------
def sqdist(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return (x * x).sum(-1, keepdim=True) - 2 * x @ mu.T + (mu * mu).sum(-1)[None]


def kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> tuple[torch.Tensor, list[float]]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu = x[torch.randperm(x.shape[0], generator=g)[:k].to(x.device)].clone()
    hist = []
    for _ in range(iters):
        assign = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
        inertia = 0.0
        for s in range(0, x.shape[0], 1 << 16):
            d = sqdist(x[s : s + (1 << 16)], mu)
            v, a = d.min(-1)
            assign[s : s + (1 << 16)] = a
            inertia += float(v.clamp_min(0).sum())
        hist.append(inertia / x.shape[0])
        sums = torch.zeros_like(mu).index_add_(0, assign, x)
        cnt = torch.bincount(assign, minlength=k).float()
        empty = cnt == 0
        mu = torch.where(empty[:, None], mu, sums / cnt.clamp_min(1)[:, None])
    return mu, hist


def topk_centroids(x: torch.Tensor, mu: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    ds, ix = [], []
    for s in range(0, x.shape[0], 1 << 16):
        d = sqdist(x[s : s + (1 << 16)], mu).clamp_min(0)
        v, i = d.topk(k, dim=-1, largest=False)
        ds.append(v.cpu())
        ix.append(i.cpu())
    return torch.cat(ds).numpy(), torch.cat(ix).numpy().astype(np.int32)


# ---- 6: placement -----------------------------------------------------------------------------
def place(
    e1: torch.Tensor, e2: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """cell[n] for every entry; greedy in a seeded random order, tiers 32 / 128 / all free."""
    n = e1.shape[0]
    d1, i1 = topk_centroids(e1, mu1, 128)
    d2, i2 = topk_centroids(e2, mu2, 128)
    occupied = np.zeros(GRID * GRID, dtype=bool)
    cell = np.full(n, -1, dtype=np.int64)
    excess = np.zeros(n, dtype=np.float64)
    tiers = {"top32": 0, "top128": 0, "all_free": 0}
    order = np.random.default_rng(seed).permutation(n)
    t0 = time.time()
    for step, m in enumerate(order):
        best = d1[m, 0] + d2[m, 0]
        chosen = -1
        for name, k in (("top32", 32), ("top128", 128)):
            cells = (i1[m, :k, None].astype(np.int64) * GRID + i2[m, None, :k]).ravel()
            cost = (d1[m, :k, None] + d2[m, None, :k]).ravel()
            free = ~occupied[cells]
            if free.any():
                j = np.flatnonzero(free)[np.argmin(cost[free])]
                chosen, c = int(cells[j]), float(cost[j])
                tiers[name] += 1
                break
        if chosen < 0:  # every pair in the top-128 lists is taken: nearest free cell anywhere
            f1 = sqdist(e1[m : m + 1], mu1).clamp_min(0)[0]
            f2 = sqdist(e2[m : m + 1], mu2).clamp_min(0)[0]
            full = (f1[:, None] + f2[None, :]).flatten().cpu().numpy()
            full[occupied] = np.inf
            chosen = int(np.argmin(full))
            c = float(full[chosen])
            tiers["all_free"] += 1
        occupied[chosen] = True
        cell[m] = chosen
        excess[m] = c - best
        if step % (1 << 17) == 0:
            log(f"  placed {step} / {n} ({time.time() - t0:.0f} s) {tiers}")
    assert occupied.all() and len(np.unique(cell)) == n
    base = float(np.mean(d1[:, 0] + d2[:, 0]))
    stats = {
        "tiers": tiers,
        "share_outside_top32": 1 - tiers["top32"] / n,
        "mean_unconstrained_sqdist": base,
        "mean_excess_sqdist": float(excess.mean()),
        "p50_excess_sqdist": float(np.median(excess)),
        "p99_excess_sqdist": float(np.quantile(excess, 0.99)),
        "relative_distortion": float(excess.mean()) / base,
        "wall_s": time.time() - t0,
    }
    return cell, stats


# ---- main -------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", required=True)
    ap.add_argument("--raw", default="/mnt/nvme/memory/raw/wikipedia")
    ap.add_argument("--encoder-dir", default="/mnt/nvme/memory/encoder/bge-base-en-v1.5")
    ap.add_argument("--out", default="/mnt/nvme/memory/wiki-20231101-lead")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    t_all = time.time()
    raw, enc_dir, out = Path(args.raw), Path(args.encoder_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wall: dict[str, float] = {}

    t = time.time()
    have = (
        len(list((raw / CONFIG).glob("train-*.parquet"))) == 41
        and (enc_dir / "model.safetensors").exists()
    )
    if not have:  # fetched at the pinned revisions; the local_dir metadata records the commit
        snapshot_download(
            DATASET,
            repo_type="dataset",
            revision=DATA_REV,
            allow_patterns=[f"{CONFIG}/*"],
            local_dir=raw,
        )
        snapshot_download(
            ENCODER,
            revision=ENC_REV,
            local_dir=enc_dir,
            allow_patterns=["*.json", "model.safetensors", "vocab.txt"],
        )
    wall["download_s"] = time.time() - t
    log(f"downloaded ({wall['download_s']:.0f} s)")

    t = time.time()
    counts, rows = extract(raw)
    wall["extract_s"] = time.time() - t
    log(f"extracted: {counts} ({wall['extract_s']:.0f} s)")

    rng0 = np.random.default_rng(0)
    pick = np.sort(rng0.choice(len(rows), size=N_M, replace=False))
    table_rows = [rows[i] for i in pick]
    rest = np.setdiff1d(np.arange(len(rows)), pick)
    rng1 = np.random.default_rng(1)
    in_pool = rng1.choice(N_M, size=POOL, replace=False)
    out_pool = rng1.choice(rest, size=POOL, replace=False)

    t = time.time()
    emb = encode([r["sentence"] for r in table_rows], enc_dir)
    wall["encode_s"] = time.time() - t
    sub = np.random.default_rng(2).choice(N_M, size=256, replace=False)
    ref = encode_fp32([table_rows[i]["sentence"] for i in sub], enc_dir)
    fp16_cos = (ref * emb[sub]).sum(-1)
    enc_check = {
        "fp16_vs_fp32_min_cos": float(fp16_cos.min()),
        "fp16_vs_fp32_mean_cos": float(fp16_cos.mean()),
    }
    log(f"encoded ({wall['encode_s']:.0f} s); {enc_check}")

    t = time.time()
    x = torch.from_numpy(emb).to("cuda:0")
    e1, e2 = x[:, :HALF].contiguous(), x[:, HALF:].contiguous()
    mu1, h1 = kmeans(e1, GRID, 25, seed=0)
    mu2, h2 = kmeans(e2, GRID, 25, seed=0)
    wall["kmeans_s"] = time.time() - t
    log(
        f"k-means ({wall['kmeans_s']:.0f} s): inertia {h1[0]:.4f}->{h1[-1]:.4f}, "
        f"{h2[0]:.4f}->{h2[-1]:.4f}"
    )

    cell, pstats = place(e1, e2, mu1, mu2, seed=0)
    wall["place_s"] = pstats["wall_s"]
    log(f"placed: {pstats}")

    t = time.time()
    inv = np.empty(N_M, dtype=np.int64)
    inv[cell] = np.arange(N_M)  # row in cell order -> sampled entry
    np.save(out / "table.npy", emb[inv].astype(np.float16))
    np.save(out / "mu1.npy", mu1.cpu().numpy().astype(np.float32))
    np.save(out / "mu2.npy", mu2.cpu().numpy().astype(np.float32))
    ids_cell = np.array([table_rows[i]["id"] for i in inv], dtype=np.int64)
    np.save(out / "cell_article_ids.npy", ids_cell)
    with (out / "sentences.jsonl").open("w") as f:
        for c, i in enumerate(inv):
            f.write(
                json.dumps(
                    {
                        "cell": c,
                        "id": table_rows[i]["id"],
                        "title": table_rows[i]["title"],
                        "sentence": table_rows[i]["sentence"],
                    }
                )
                + "\n"
            )
    with (out / "eval_pools.jsonl").open("w") as f:
        for i in in_pool:
            r = table_rows[i]
            f.write(
                json.dumps(
                    {
                        "pool": "in_table",
                        "id": r["id"],
                        "title": r["title"],
                        "lead": r["lead"],
                        "sentence_chars": len(r["sentence"]),
                        "cell": int(cell[i]),
                    }
                )
                + "\n"
            )
        for i in out_pool:
            r = rows[i]
            f.write(
                json.dumps(
                    {
                        "pool": "out_of_table",
                        "id": r["id"],
                        "title": r["title"],
                        "lead": r["lead"],
                        "sentence_chars": len(r["sentence"]),
                        "cell": None,
                    }
                )
                + "\n"
            )
    wall["write_s"] = time.time() - t

    files = [
        "table.npy",
        "mu1.npy",
        "mu2.npy",
        "cell_article_ids.npy",
        "sentences.jsonl",
        "eval_pools.jsonl",
    ]
    manifest = {
        "build": "build_wiki_memory",
        "adr": "ADR-053",
        "id": args.id,
        "attribution": "text from English Wikipedia (CC BY-SA 3.0 / GFDL); derived data, "
        "not redistributed",
        "dataset": {"repo": DATASET, "config": CONFIG, "revision": DATA_REV},
        "encoder": {
            "repo": ENCODER,
            "revision": ENC_REV,
            "pooling": "cls",
            "normalise": "l2",
            "autocast": "fp16",
            **enc_check,
        },
        "filters": {
            "min_chars": MIN_CH,
            "max_chars": MAX_CH,
            "boundary": BOUNDARY.pattern,
            "drop_title_prefix": "List of",
            "drop_title_contains": "(disambiguation)",
            "drop_sentence_contains": "may refer to",
        },
        "counts": counts | {"n_m": N_M, "in_table_pool": POOL, "out_of_table_pool": POOL},
        "sample": {
            "table_seed": 0,
            "pool_seed": 1,
            "pools": "in-table from the sampled articles, out-of-table from the eligible rest",
        },
        "grid": GRID,
        "kmeans": {"k": GRID, "iters": 25, "seed": 0, "inertia_half1": h1, "inertia_half2": h2},
        "placement": pstats,
        "sentence_chars": {"mean": float(np.mean([len(r["sentence"]) for r in table_rows]))},
        "files": {f: {"bytes": (out / f).stat().st_size, "sha256": sha256(out / f)} for f in files},
        "wall_s": wall | {"total_s": time.time() - t_all},
        "out": str(out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    exp = ROOT / "experiments" / args.id
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "result.json").write_text(json.dumps(manifest, indent=2))
    np.save(exp / "sampled_article_ids.npy", np.sort(ids_cell))
    log(f"done ({manifest['wall_s']['total_s']:.0f} s)")


if __name__ == "__main__":
    main()
