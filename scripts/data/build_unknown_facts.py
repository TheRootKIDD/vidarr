"""Build ADR-076's unknown-facts pool and its table rows (rig `054`; I44 option (b)).

L8b's table is frozen data, so it can be swapped at eval time. This build draws 2,000 eligible
Wikipedia articles (seed 2) from outside build `040`'s table and outside its two eval pools,
keeps those whose first sentence begins with the title, and writes each in two versions:
  - `real`:    the first sentence as in the dump;
  - `renamed`: every occurrence of the title replaced by a generated two-word pseudo-name. The
               predicate is real; the binding to the name exists in no corpus.
A pseudo-name is rejected if either word is a title word of the dump or a whole word of the
corpus tokenizer's vocabulary. An article is skipped if its title, or any title word of four or
more letters, survives the renaming (a surname repeated alone, a name inside a longer word, a
run-together spelling such as a domain). Other cues to the entity (an abbreviation in
parentheses, a call sign's frequency) are not removed: ADR-076's premise check measures whether
the pool separates known from unknown.

Both versions are encoded exactly as `040` encoded the table (`build_wiki_memory.encode`). Only
the 2,000 extra rows per version are stored; the eval appends them to `040`'s table in memory
(T_real = T_0 + real rows, T_ren = T_0 + renamed rows), so `040`'s files are not touched.

Writes under `--out` (derived Wikipedia text, CC BY-SA 3.0 / GFDL, never committed):
`pool.jsonl` (j, id, title, name, real, renamed), `extra_real.npy`, `extra_renamed.npy`
([2000, 768] float16) and `manifest.json`; and the manifest (no text) to
`experiments/<id>/build.json`.

Run on idle GPUs: `python -m scripts.data.build_unknown_facts --id 054-l8b-unknown-facts`
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
N_POOL = 2000
POOL_SEED = 2  # ADR-076 §1
NAME_SEED = 2
MIN_TITLE = 4
ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "kr",
          "st", "tr", "vl", "sk", "th", "sh")  # fmt: skip
VOWELS = ("a", "e", "i", "o", "u", "ae", "ei", "ou")
CODAS = ("", "", "n", "r", "s", "l", "th", "k", "m", "rn", "sk")
WORD = re.compile(r"[^\W\d_]+")


def pseudo_word(rng: np.random.Generator) -> str:
    """A capitalised pronounceable non-word: 2-3 open syllables and an optional coda."""
    n = int(rng.integers(2, 4))
    body = "".join(ONSETS[rng.integers(len(ONSETS))] + VOWELS[rng.integers(len(VOWELS))]
                   for _ in range(n))  # fmt: skip
    return (body + CODAS[rng.integers(len(CODAS))]).capitalize()


def pseudo_name(rng: np.random.Generator, banned: set[str]) -> str:
    """Two distinct pseudo-words, neither (lower-cased) in `banned`."""
    while True:
        a, b = pseudo_word(rng), pseudo_word(rng)
        if a != b and a.lower() not in banned and b.lower() not in banned:
            return f"{a} {b}"


def starts_with_title(title: str, sentence: str) -> bool:
    """The first sentence begins with the title as a whole word (ADR-076 §1's restriction)."""
    if len(title) < MIN_TITLE or not sentence.startswith(title):
        return False
    return len(sentence) == len(title) or not sentence[len(title)].isalnum()


def rename(title: str, sentence: str, name: str) -> str:
    """Every whole-word occurrence of `title` replaced by `name`."""
    return re.sub(r"(?<!\w)" + re.escape(title) + r"(?!\w)", lambda _: name, sentence)


def leaks(title: str, renamed: str) -> bool:
    """True if the title, one of its words of four or more letters, or its letters and digits
    run together (a domain or product spelling, any case) is still in the text."""
    if title in renamed:
        return True
    joined = "".join(ch for ch in title.lower() if ch.isalnum())
    if joined in "".join(ch for ch in renamed.lower() if ch.isalnum() or ch.isspace()):
        return True
    words = set(WORD.findall(renamed))
    return any(len(w) >= MIN_TITLE and w in words for w in WORD.findall(title))


def words_of(titles: list[str]) -> set[str]:
    out: set[str] = set()
    for t in titles:
        out.update(w.lower() for w in WORD.findall(t))
    return out


def draw_pool(
    rows: list[dict[str, Any]],
    excluded: set[int],
    banned: set[str],
    n: int = N_POOL,
    pool_seed: int = POOL_SEED,
    name_seed: int = NAME_SEED,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Walk the candidates in a seeded order until `n` articles pass; returns pool and counts."""
    cand = [
        i for i, r in enumerate(rows)
        if r["id"] not in excluded and starts_with_title(r["title"], r["sentence"])
    ]  # fmt: skip
    order = np.random.default_rng(pool_seed).permutation(len(cand))
    rng = np.random.default_rng(name_seed)
    pool: list[dict[str, Any]] = []
    used: set[str] = set()
    counts = {"candidates": len(cand), "skipped_leak": 0, "walked": 0}
    for o in order:
        if len(pool) == n:
            break
        counts["walked"] += 1
        r = rows[cand[o]]
        name = pseudo_name(rng, banned | used)
        renamed = rename(r["title"], r["sentence"], name)
        if leaks(r["title"], renamed):
            counts["skipped_leak"] += 1
            continue
        used.update(w.lower() for w in name.split())
        pool.append({"j": len(pool), "id": r["id"], "title": r["title"], "name": name,
                     "real": r["sentence"], "renamed": renamed})  # fmt: skip
    if len(pool) < n:
        raise RuntimeError(f"only {len(pool)} of {n} articles pass the renaming rules")
    return pool, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", required=True)
    ap.add_argument("--raw", default="/mnt/nvme/memory/raw/wikipedia")
    ap.add_argument("--encoder-dir", default="/mnt/nvme/memory/encoder/bge-base-en-v1.5")
    ap.add_argument("--base", default="/mnt/nvme/memory/wiki-20231101-lead", help="build 040")
    ap.add_argument("--out", default="/mnt/nvme/memory/unknown-facts-054")
    ap.add_argument(
        "--tokenizer", default="/mnt/nvme/corpus/tokenizer/mistral-7b-v0.1/tokenizer.json"
    )
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    from scripts.data.build_wiki_memory import CONFIG, encode, extract, log, sha256

    t_all = time.time()
    raw, base, out = Path(a.raw), Path(a.base), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts, rows = extract(raw)
    titles: list[str] = []
    for f in sorted((raw / CONFIG).glob("train-*.parquet")):
        titles += pq.read_table(f, columns=["title"]).column("title").to_pylist()
    vocab = Tokenizer.from_file(a.tokenizer).get_vocab()
    banned = words_of(titles) | {w.lower() for v in vocab for w in WORD.findall(v)}
    log(f"extracted {counts}; {len(titles)} titles, {len(banned)} banned words")

    excluded = set(np.load(base / "cell_article_ids.npy").tolist())
    excluded |= {json.loads(x)["id"] for x in (base / "eval_pools.jsonl").read_text().splitlines()}
    pool, pcounts = draw_pool(rows, excluded, banned)
    log(f"pool: {pcounts}")

    files = ["pool.jsonl", "extra_real.npy", "extra_renamed.npy"]
    with (out / "pool.jsonl").open("w") as f:
        for p in pool:
            f.write(json.dumps(p) + "\n")
    for v in ("real", "renamed"):
        emb = encode([p[v] for p in pool], Path(a.encoder_dir))
        np.save(out / f"extra_{v}.npy", emb.astype(np.float16))
    base_manifest = json.loads((base / "manifest.json").read_text())
    manifest = {
        "build": "build_unknown_facts",
        "adr": "ADR-076",
        "id": a.id,
        "attribution": "text from English Wikipedia (CC BY-SA 3.0 / GFDL); derived data, "
        "not redistributed",
        "base": {"dir": str(base), "table_sha256": base_manifest["files"]["table.npy"]["sha256"]},
        "encoder": base_manifest["encoder"],
        "seeds": {"pool": POOL_SEED, "names": NAME_SEED},
        "counts": counts | pcounts | {"pool": len(pool), "excluded_ids": len(excluded),
                                      "banned_words": len(banned)},  # fmt: skip
        "chars": {v: float(np.mean([len(p[v]) for p in pool])) for v in ("real", "renamed")},
        "files": {f: {"bytes": (out / f).stat().st_size, "sha256": sha256(out / f)} for f in files},
        "wall_s": time.time() - t_all,
        "out": str(out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    exp = ROOT / "experiments" / a.id
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "build.json").write_text(json.dumps(manifest, indent=2))
    log(f"done ({manifest['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
