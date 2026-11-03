"""`wiki-lead` (ADR-053 §7) and L8's usage probes (§8) for a finished run; `--compare` scores H17.

Each eval article is scored as "title, blank line, lead paragraph", tokenised with the corpus
tokenizer after one EOS (the corpus joins documents with EOS, no BOS), truncated to 256 tokens.
The loss is read on lead tokens only, split by character offset into:
  - `in_first`: in-table articles, tokens of the first sentence (the table holds it);
  - `in_rest`:  in-table articles, the rest of the lead (related knowledge, not copyable);
  - `out_lead`: out-of-table articles, the whole lead (the control).
Per-article loss sums and counts are written, so `--compare` can bootstrap over articles.

With `--probes` (L8 runs): the same evals and the FineWeb-Edu held-out loss with the retrieved
cells replaced by uniformly random ones (reliance); for `in_first` tokens the share whose own
article's cell is among the k_m retrieved, per iteration (lookup accuracy); the mean entropy of
alpha and the share of the 2^20 cells retrieved at least once on the held-out windows (use).

Run on an idle GPU:
  python -m scripts.analysis.eval_wiki_lead --id 110-l5-screen
  python -m scripts.analysis.eval_wiki_lead --id 173-l8-screen --probes
  python -m scripts.analysis.eval_wiki_lead --compare 173-l8-screen 110-l5-screen
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.model import BigMoE
from scripts.analysis.probe_routing import RESULTS_DIR, load_run
from scripts.train.data import MemmapCorpus

CORPUS = "/mnt/nvme/corpus/fineweb-edu/tok-mistral32k"
TOKENIZER = "/mnt/nvme/corpus/tokenizer/mistral-7b-v0.1/tokenizer.json"
POOLS = "/mnt/nvme/memory/wiki-20231101-lead/eval_pools.jsonl"
EOS, MAX_TOK = 2, 256
SPLITS = ("in_first", "in_rest", "out_lead")
SIGMA_SCREEN = 0.0030  # ADR-053 §7's floor: max(this, bootstrap SE), times 4
REGRESSION = 0.005  # H17's "<= 0.5 %" on the general held-out loss


def build_items(tok: Tokenizer) -> list[dict[str, Any]]:
    items = []
    for line in Path(POOLS).read_text().splitlines():
        a = json.loads(line)
        head = a["title"] + "\n\n"
        enc = tok.encode(head + a["lead"], add_special_tokens=False)
        ids, offs = enc.ids[:MAX_TOK], enc.offsets[:MAX_TOK]
        split = np.zeros(len(ids), dtype=np.int8)  # 0 = not scored (title)
        for i, (s, _) in enumerate(offs):
            if s < len(head):
                continue
            if a["pool"] == "out_of_table":
                split[i] = 3
            else:
                split[i] = 1 if s - len(head) < a["sentence_chars"] else 2
        items.append(
            {
                "id": a["id"],
                "pool": a["pool"],
                "cell": a["cell"],
                "ids": [EOS, *ids],
                "split": split,
            }
        )
    return items


@torch.no_grad()
def score_articles(
    model: BigMoE, items: list[dict[str, Any]], dev: torch.device, batch: int = 16
) -> tuple[dict[str, np.ndarray], list[list[float]]]:
    """Per-article loss sums / counts per split; with recording on, lookup hits per iteration."""
    sums = {s: np.zeros(len(items)) for s in SPLITS}
    cnts = {s: np.zeros(len(items)) for s in SPLITS}
    mem = model.middle.mem if model.cfg.memory.enabled else None
    hits: list[list[float]] = []  # per iteration: [hits, tokens]
    for b0 in range(0, len(items), batch):
        chunk = items[b0 : b0 + batch]
        L = max(len(x["ids"]) for x in chunk)
        toks = torch.full((len(chunk), L), EOS, dtype=torch.long)
        lab = torch.zeros((len(chunk), L - 1), dtype=torch.long)
        for r, x in enumerate(chunk):
            toks[r, : len(x["ids"])] = torch.tensor(x["ids"])
            lab[r, : len(x["split"])] = torch.from_numpy(x["split"].astype(np.int64))
        toks = toks.to(dev)
        if mem is not None:
            mem.records.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, _ = model.forward_hidden(toks[:, :-1])
            logits = model.head(h).float()
        loss = F.cross_entropy(logits.transpose(1, 2), toks[:, 1:], reduction="none").cpu()
        for r in range(len(chunk)):
            for k, s in enumerate(SPLITS, start=1):
                m = lab[r] == k
                sums[s][b0 + r] = float(loss[r][m].sum())
                cnts[s][b0 + r] = float(m.sum())
        if mem is not None and mem.record:
            own = torch.tensor([-1 if x["cell"] is None else x["cell"] for x in chunk], device=dev)
            first = (lab == 1).to(dev)  # positions whose next token is in the first sentence
            for j, (cells, _) in enumerate(mem.records):
                hit = (cells == own[:, None, None]).any(-1) & first & (own[:, None] >= 0)
                while len(hits) <= j:
                    hits.append([0.0, 0.0])
                hits[j][0] += float(hit.sum())
                hits[j][1] += float(first.sum())
            mem.records.clear()
    return {"sum_" + s: sums[s] for s in SPLITS} | {"cnt_" + s: cnts[s] for s in SPLITS}, hits


@torch.no_grad()
def heldout(model: BigMoE, dev: torch.device, max_tokens: int, use: bool) -> dict[str, float]:
    """FineWeb-Edu held-out loss on the trainer's windows; with `use`, alpha entropy and the
    share of cells retrieved at least once."""
    cfg = model.cfg
    windows = MemmapCorpus(Path(CORPUS)).val_windows(cfg.context, max_tokens)
    mem = model.middle.mem if cfg.memory.enabled else None
    seen = None if mem is None else torch.zeros(cfg.memory.n_entries, dtype=torch.bool, device=dev)
    tot, n, ent, ent_n = 0.0, 0, 0.0, 0
    for i in range(0, len(windows), 4):
        toks = torch.cat(windows[i : i + 4]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.loss(toks, r=None if cfg.halting.mode != "fixed" else cfg.middle.r_max)
        tot += float(out.main) * toks.shape[0]
        n += toks.shape[0]
        if mem is not None and use:
            for cells, alpha in mem.records:
                seen[cells.flatten()] = True
                a = alpha.float().clamp_min(1e-12)
                ent += float(-(a * a.log()).sum(-1).sum())
                ent_n += a[..., 0].numel()
            mem.records.clear()
    res = {"val_loss": tot / n, "val_windows": float(n)}
    if seen is not None and use:
        res |= {
            "cells_used_share": float(seen.float().mean()),
            "alpha_entropy_mean": ent / ent_n,
            "alpha_entropy_max": float(np.log(cfg.memory.topk)),
        }
    return res


def mean_loss(d: dict[str, Any], s: str) -> float:
    return float(np.sum(d["sum_" + s]) / np.sum(d["cnt_" + s]))


def run_eval(a: argparse.Namespace) -> None:
    dev = torch.device(a.device)
    model, cfg, step = load_run(RESULTS_DIR / a.id, dev)
    tok = Tokenizer.from_file(TOKENIZER)
    items = build_items(tok)
    t0 = time.time()
    res: dict[str, Any] = {
        "id": a.id,
        "step": step,
        "n_articles": len(items),
        "pools": POOLS,
        "max_tokens": MAX_TOK,
    }
    mem = model.middle.mem if cfg.memory.enabled else None
    if mem is not None:
        res["table_manifest_sha256"] = {k: v["sha256"] for k, v in mem.manifest["files"].items()}
        mem.record = a.probes
    per, hits = score_articles(model, items, dev)
    res["wiki_lead"] = {s: mean_loss(per, s) for s in SPLITS}
    res["per_article"] = {k: v.tolist() for k, v in per.items()} | {
        "pool": [x["pool"] for x in items],
        "id": [x["id"] for x in items],
    }
    if hits:
        res["lookup_accuracy_by_iteration"] = [h / max(c, 1) for h, c in hits]
    res["heldout"] = heldout(model, dev, a.max_tokens, use=a.probes)
    if mem is not None and a.probes:
        mem.record, mem.random_cells = False, True
        torch.manual_seed(0)
        per_r, _ = score_articles(model, items, dev)
        res["random_cells"] = {
            "wiki_lead": {s: mean_loss(per_r, s) for s in SPLITS},
            "heldout": heldout(model, dev, a.max_tokens, use=False),
        }
        mem.random_cells = False
    res["wall_s"] = time.time() - t0
    out = RESULTS_DIR / a.id / "wiki_lead.json"
    out.write_text(json.dumps(res))
    print(json.dumps({k: v for k, v in res.items() if k != "per_article"}, indent=1))
    print(out)


def compare(arm: str, base: str, n_boot: int = 2000) -> dict[str, Any]:
    """ADR-053 §10's verdict: arm minus base on each split, bootstrap SE over articles."""
    A = json.loads((RESULTS_DIR / arm / "wiki_lead.json").read_text())
    B = json.loads((RESULTS_DIR / base / "wiki_lead.json").read_text())
    pa, pb = A["per_article"], B["per_article"]
    assert pa["id"] == pb["id"], "the two runs were scored on different pools"
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {"arm": arm, "base": base, "splits": {}}
    pool = np.array(pa["pool"])
    for s in SPLITS:
        idx = np.flatnonzero(pool == ("out_of_table" if s == "out_lead" else "in_table"))
        sa, sb = np.array(pa["sum_" + s])[idx], np.array(pb["sum_" + s])[idx]
        c = np.array(pa["cnt_" + s])[idx]
        delta = (sa.sum() - sb.sum()) / c.sum()
        boots = []
        for _ in range(n_boot):
            k = rng.integers(0, len(idx), len(idx))
            boots.append((sa[k].sum() - sb[k].sum()) / c[k].sum())
        se = float(np.std(boots))
        floor = 4 * max(SIGMA_SCREEN, se)
        out["splits"][s] = {
            "arm": float(sa.sum() / c.sum()),
            "base": float(sb.sum() / c.sum()),
            "delta": float(delta),
            "bootstrap_se": se,
            "floor": floor,
            "beats_floor": bool(delta < -floor),
            "tokens": float(c.sum()),
        }
    va, vb = A["heldout"]["val_loss"], B["heldout"]["val_loss"]
    reg = (va - vb) / vb
    out["heldout"] = {"arm": va, "base": vb, "relative": reg, "within": reg <= REGRESSION}
    sp = out["splits"]
    both = sp["in_first"]["beats_floor"] and sp["in_rest"]["beats_floor"]
    neither = not sp["in_first"]["beats_floor"] and not sp["in_rest"]["beats_floor"]
    if both and reg <= REGRESSION:
        verdict = "supports"
    elif neither or reg > REGRESSION:
        verdict = "weakens"
    else:
        verdict = "silent"
    out["verdict_h17"] = verdict
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id")
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("ARM", "BASE"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    a = ap.parse_args()
    if a.compare:
        res = compare(*a.compare)
        (RESULTS_DIR / a.compare[0] / "wiki_lead_compare.json").write_text(
            json.dumps(res, indent=1)
        )
        print(json.dumps(res, indent=1))
    else:
        run_eval(a)


if __name__ == "__main__":
    main()
