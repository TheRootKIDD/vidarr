"""ADR-076 (rig `054`; I44 option (b)): L8b-pv's memory on facts no base can know.

The retrieval table is frozen data and the trained maps act on the encoder's space, so the table
is swapped at eval time; nothing is trained. Each of the pool's 2,000 articles
(`build_unknown_facts.py`) is scored as "title, blank line, first sentence" in two versions,
`real` and `renamed` (the title replaced by a pseudo-name, in the document's title too), under
  - `t0`:   build `040`'s table, which holds neither version;
  - `hold`: `040`'s table plus that version's 2,000 sentences (memory runs only).
Losses are read on first-sentence tokens with neighbours (`in_first_cov`, ADR-059's mask: it
depends on the boundaries alone, so it is the same under every table and for every run).

`--report` reads the per-run files and gives, with article-bootstrap SEs and ADR-055 §6's floors
(4 x max(0.0030, SE)):
  - the specific value S = loss under t0 - loss under hold (same model, same tokens), for real
    and renamed, at `small` (pair) and `medium`; S_ren - S_real; S_ren small - medium;
  - the memory run under `hold` against its L5 base, per version and size (reported, no verdict);
  - the premise: the base's loss on renamed minus on real, `medium` against `small`;
  - ADR-076 §4's reading.

Run on idle GPUs:
  python -m scripts.analysis.eval_unknown_facts --id 181-l8b-pv-medium --device cuda:2 \
      --out experiments/054-l8b-unknown-facts/uf_181.json
  python -m scripts.analysis.eval_unknown_facts --report experiments/054-l8b-unknown-facts \
      --small 178 179 --small-base 132 133 --medium 181 --medium-base 180
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

EOS, MAX_TOK = 2, 256
SIGMA = 0.0030  # ADR-055 §6's floor: 4 x max(this, bootstrap SE)
VERSIONS = ("real", "renamed")
SPLIT = "in_first_cov"
Term = tuple[float, np.ndarray, np.ndarray]  # sign, per-article loss sums, per-article counts


def build_items(tok: Any, pool: list[dict[str, Any]], version: str, n_table: int) -> list[dict]:
    """wiki-lead's item format (`eval_wiki_lead.build_items`) for one version of the pool. The
    own sentence's row under `hold` is `n_table + j`; under `t0` no row has that index."""
    items = []
    for a in pool:
        head = (a["title"] if version == "real" else a["name"]) + "\n\n"
        enc = tok.encode(head + a[version], add_special_tokens=False)
        ids, offs = enc.ids[:MAX_TOK], enc.offsets[:MAX_TOK]
        split = np.array([1 if s >= len(head) else 0 for s, _ in offs], dtype=np.int8)
        items.append({"id": a["id"], "pool": "in_table", "cell": n_table + a["j"],
                      "ids": [EOS, *ids], "split": split})  # fmt: skip
    return items


def score_conditions(
    model: Any, pipe: Any, items: dict[str, list[dict]], extras: dict[str, Any], dev: Any
) -> dict[str, Any]:
    """Every (version, table) condition for one run: `t0` always, `hold` (t0 plus that
    version's rows, for the pipeline and the model alike) if the run reads a retrieved memory.
    The run's own table is restored afterwards."""
    import torch

    from scripts.analysis.eval_wiki_lead import score_articles

    mem = model.middle.mem if model.cfg.memory.enabled else None
    retrieved = mem is not None and mem.retrieved
    t0 = pipe.table
    out: dict[str, Any] = {}
    for version in VERSIONS:
        for tab in ("t0", "hold") if retrieved else ("t0",):
            table = t0
            if tab == "hold":
                table = torch.cat([t0, extras[version].to(t0.device).to(t0.dtype)])
            pipe.table = table
            if retrieved:
                mem.table = table  # a non-persistent buffer: frozen data, never checkpointed
            per, _ = score_articles(model, items[version], dev, pipe=pipe)
            first = float(per["cnt_in_first"].sum())
            cov = float(per["cnt_" + SPLIT].sum())
            hit = float(per["cnt_in_first_hit"].sum())
            out[f"{version}_{tab}"] = {
                "loss_in_first": float(per["sum_in_first"].sum() / first),
                "loss_" + SPLIT: float(per["sum_" + SPLIT].sum() / max(cov, 1.0)),
                "in_first_tokens": first,
                "covered_tokens": cov,
                "coverage": cov / first,
                "own_retrieved_share_of_covered": hit / max(cov, 1.0),
                "per_article": {
                    k: per[k].tolist()
                    for k in ("sum_in_first", "cnt_in_first", "sum_" + SPLIT, "cnt_" + SPLIT,
                              "sum_in_first_hit", "cnt_in_first_hit")  # fmt: skip
                },
            }
            pipe.table = t0
            if retrieved:
                mem.table = t0
            del table
    return out


def run_eval(a: argparse.Namespace) -> None:
    import torch
    from tokenizers import Tokenizer

    from scripts.analysis.eval_wiki_lead import TOKENIZER, Pipeline
    from scripts.analysis.probe_routing import RESULTS_DIR, load_run

    dev = torch.device(a.device)
    facts = Path(a.facts)
    model, cfg, step = load_run(RESULTS_DIR / a.id, dev)
    mem = model.middle.mem if cfg.memory.enabled else None
    retrieved = mem is not None and mem.retrieved
    pipe = Pipeline(dev, model)
    t0 = pipe.table
    n = t0.shape[0]
    pool = [json.loads(x) for x in (facts / "pool.jsonl").read_text().splitlines()]
    tok = Tokenizer.from_file(TOKENIZER)
    res: dict[str, Any] = {
        "id": a.id,
        "step": step,
        "memory": retrieved,
        "facts": str(facts),
        "facts_manifest_sha256": {
            k: v["sha256"]
            for k, v in json.loads((facts / "manifest.json").read_text())["files"].items()
        },
        "n_articles": len(pool),
        "table_rows": n,
        "conditions": {},
    }
    start = time.time()
    items = {v: build_items(tok, pool, v, n) for v in VERSIONS}
    extras = {v: torch.from_numpy(np.load(facts / f"extra_{v}.npy")) for v in VERSIONS}
    res["conditions"] = score_conditions(model, pipe, items, extras, dev)
    res["wall_s"] = time.time() - start
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res))
    print(json.dumps(
        {k: {m: v for m, v in c.items() if m != "per_article"}
         for k, c in res["conditions"].items()}, indent=1))  # fmt: skip
    print(a.out)


# ---- report ------------------------------------------------------------------------------------
def boot(terms: list[Term], n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """sum_i sign_i * (sums_i / counts_i) over all articles, its SE under a paired bootstrap over
    articles (every term is indexed by the same pool), and ADR-055 §6's floor."""
    n = len(terms[0][1])
    rng = np.random.default_rng(seed)

    def est(k: np.ndarray | slice) -> float:
        return float(sum(sg * s[k].sum() / c[k].sum() for sg, s, c in terms))

    se = float(np.std([est(rng.integers(0, n, n)) for _ in range(n_boot)]))
    return {"value": est(slice(None)), "bootstrap_se": se, "floor": 4 * max(SIGMA, se)}


def side(files: list[dict[str, Any]], cond: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-article sums and counts of the covered split, added over a pair's seeds."""
    pa = [f["conditions"][cond]["per_article"] for f in files]
    s = np.sum([np.array(p["sum_" + SPLIT]) for p in pa], 0)
    c = np.sum([np.array(p["cnt_" + SPLIT]) for p in pa], 0)
    return s, c


def reading(premise: dict, s_ren_med: dict, gap_med: dict, drop: dict) -> str:
    """ADR-076 §4."""
    if premise["value"] <= premise["floor"]:
        return "silent: the pool does not separate known from unknown (premise fails)"
    if s_ren_med["value"] < s_ren_med["floor"]:
        return "explanation not sufficient: at medium the memory does not pay on unknown facts"
    if gap_med["value"] >= gap_med["floor"] and drop["value"] <= drop["floor"]:
        return "I44's explanation supported: at medium the memory still pays on unknown facts"
    return "mixed"


def report(
    arms: dict[str, list[dict]], bases: dict[str, list[dict]], n_boot: int = 2000
) -> dict[str, Any]:
    """`arms` / `bases`: size ('small' | 'medium') -> the per-run records of that side."""
    out: dict[str, Any] = {"split": SPLIT, "sizes": {}}
    S: dict[str, dict[str, list[Term]]] = {}
    D: dict[str, list[Term]] = {}
    for size in ("small", "medium"):
        A, B = arms[size], bases[size]
        S[size] = {}
        o: dict[str, Any] = {}
        for v in VERSIONS:
            t0s, t0c = side(A, f"{v}_t0")
            hs, hc = side(A, f"{v}_hold")
            bs, bc = side(B, f"{v}_t0")
            assert np.array_equal(t0c, hc), "covered masks differ between tables"
            assert np.array_equal(hc * len(B), bc * len(A)), "covered masks differ between runs"
            S[size][v] = [(1.0, t0s, t0c), (-1.0, hs, hc)]
            o["S_" + v] = boot(S[size][v], n_boot)
            o["arm_hold_minus_base_" + v] = boot([(1.0, hs, hc), (-1.0, bs, bc)], n_boot)
            o["arm_t0_minus_base_" + v] = boot([(1.0, t0s, t0c), (-1.0, bs, bc)], n_boot)
            o["base_loss_" + v] = float(bs.sum() / bc.sum())
            o["tokens_" + v] = float(hc.sum() / len(A))
            o["own_retrieved_share_" + v] = float(
                np.mean([f["conditions"][f"{v}_hold"]["own_retrieved_share_of_covered"] for f in A])
            )
        rs, rc = side(B, "renamed_t0")
        ls, lc = side(B, "real_t0")
        D[size] = [(1.0, rs, rc), (-1.0, ls, lc)]
        o["base_renamed_minus_real"] = boot(D[size], n_boot)
        o["S_renamed_minus_S_real"] = boot(
            S[size]["renamed"] + [(-sg, s, c) for sg, s, c in S[size]["real"]], n_boot
        )
        out["sizes"][size] = o
    neg = [(-sg, s, c) for sg, s, c in D["small"]]
    out["premise_medium_minus_small"] = boot(D["medium"] + neg, n_boot)
    neg = [(-sg, s, c) for sg, s, c in S["medium"]["renamed"]]
    out["S_renamed_small_minus_medium"] = boot(S["small"]["renamed"] + neg, n_boot)
    out["reading"] = reading(
        out["premise_medium_minus_small"],
        out["sizes"]["medium"]["S_renamed"],
        out["sizes"]["medium"]["S_renamed_minus_S_real"],
        out["S_renamed_small_minus_medium"],
    )
    return out


def run_report(a: argparse.Namespace) -> None:
    d = Path(a.report)

    def load(ns: list[str]) -> list[dict]:
        return [json.loads((d / f"uf_{n}.json").read_text()) for n in ns]

    res = report(
        {"small": load(a.small), "medium": load(a.medium)},
        {"small": load(a.small_base), "medium": load(a.medium_base)},
    )
    res["runs"] = {"small": a.small, "small_base": a.small_base, "medium": a.medium,
                   "medium_base": a.medium_base}  # fmt: skip
    (d / "report.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--facts", default="/mnt/nvme/memory/unknown-facts-054")
    ap.add_argument("--out")
    ap.add_argument("--report", help="directory holding uf_<n>.json; writes report.json there")
    ap.add_argument("--small", nargs="+", default=["178", "179"])
    ap.add_argument("--small-base", nargs="+", default=["132", "133"])
    ap.add_argument("--medium", nargs="+", default=["181"])
    ap.add_argument("--medium-base", nargs="+", default=["180"])
    a = ap.parse_args()
    if a.report:
        run_report(a)
    else:
        if not (a.id and a.out):
            ap.error("--id and --out are required for an eval")
        run_eval(a)


if __name__ == "__main__":
    main()
