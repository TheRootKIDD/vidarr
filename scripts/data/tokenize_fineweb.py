"""Tokenise FineWeb-Edu `sample/10BT` once into `uint16` memmap shards (ADR-014).

Input:  the 14 parquet shards under `<root>/raw/sample/10BT/` (downloaded with
        `huggingface_hub.snapshot_download(allow_patterns=["sample/10BT/*"])`).
Output: `<root>/tok-mistral32k/train_NNNN.bin` (100 M tokens each, `uint16`,
        documents joined by one EOS = 2, no BOS), `val.bin` (the first 10 000
        documents of shard 000, never in train) and `manifest.json` (token and
        document counts per source file, the tokenizer's SHA-256, licences).

Run:    python -m scripts.data.tokenize_fineweb \\
            --root /mnt/nvme/corpus/fineweb-edu \\
            --tokenizer /mnt/nvme/corpus/tokenizer/mistral-7b-v0.1/tokenizer.json

Tokenisation uses the Rust `tokenizers` batch encoder, which parallelises over
all cores on its own; one process is enough to keep 12 cores busy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

EOS = 2
SHARD_TOKENS = 100_000_000
VAL_DOCS = 10_000
BATCH_DOCS = 4096


class ShardWriter:
    """Accumulates tokens and flushes fixed-size `uint16` shards."""

    def __init__(self, out_dir: Path, prefix: str, shard_tokens: int | None) -> None:
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.buf: list[np.ndarray] = []
        self.buffered = 0
        self.shards: list[dict[str, int | str]] = []

    def add(self, ids: np.ndarray) -> None:
        self.buf.append(ids)
        self.buffered += len(ids)
        if self.shard_tokens is not None and self.buffered >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, n: int | None) -> None:
        cat = np.concatenate(self.buf) if len(self.buf) > 1 else self.buf[0]
        if n is None or n >= len(cat):
            out, rest = cat, None
        else:
            out, rest = cat[:n], cat[n:]
        if self.shard_tokens:
            name = f"{self.prefix}_{len(self.shards):04d}.bin"
        else:
            name = f"{self.prefix}.bin"
        out.astype(np.uint16).tofile(self.out_dir / name)
        self.shards.append({"file": name, "tokens": int(len(out))})
        self.buf = [rest] if rest is not None and len(rest) else []
        self.buffered = int(len(rest)) if rest is not None else 0

    def close(self) -> None:
        if self.buffered:
            self._flush(None)


def encode_docs(tok: Tokenizer, texts: list[str]) -> np.ndarray:
    encs = tok.encode_batch(texts, add_special_tokens=False)
    parts = [np.asarray(e.ids + [EOS], dtype=np.uint16) for e in encs]
    return np.concatenate(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default <root>/tok-mistral32k")
    args = ap.parse_args()

    raw = sorted((args.root / "raw" / "sample" / "10BT").glob("*.parquet"))
    if not raw:
        raise SystemExit(f"no parquet shards under {args.root}/raw/sample/10BT")
    out = args.out or args.root / "tok-mistral32k"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.bin")):
        raise SystemExit(
            f"{out} already holds .bin shards — tokenise once (docs/06 §7); remove or pick --out"
        )

    tok = Tokenizer.from_file(str(args.tokenizer))
    assert tok.get_vocab_size() == 32000, tok.get_vocab_size()
    assert tok.get_vocab_size() < 2**16, "uint16 memmap needs V < 65536"
    tok_sha = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()

    train = ShardWriter(out, "train", SHARD_TOKENS)
    val = ShardWriter(out, "val", None)
    sources = []
    t_start = time.time()
    total_docs = total_tokens = 0
    val_taken = 0
    for i, path in enumerate(raw):
        pf = pq.ParquetFile(path)
        docs = tokens = 0
        for rb in pf.iter_batches(batch_size=BATCH_DOCS, columns=["text"]):
            texts = rb.column("text").to_pylist()
            # first VAL_DOCS documents of shard 000 -> val, never train
            if i == 0 and val_taken < VAL_DOCS:
                cut = min(len(texts), VAL_DOCS - val_taken)
                ids = encode_docs(tok, texts[:cut])
                val.add(ids)
                val_taken += cut
                docs += cut
                tokens += len(ids)
                texts = texts[cut:]
            if texts:
                ids = encode_docs(tok, texts)
                train.add(ids)
                docs += len(texts)
                tokens += len(ids)
        total_docs += docs
        total_tokens += tokens
        el = time.time() - t_start
        print(
            f"[{i + 1:2d}/{len(raw)}] {path.name}: {docs:,} docs, {tokens:,} tokens  "
            f"| total {total_tokens / 1e9:.2f} B tok, {total_tokens / el / 1e6:.1f} M tok/s",
            flush=True,
        )
        sources.append(
            {"file": path.name, "bytes": path.stat().st_size, "docs": docs, "tokens": tokens}
        )
    train.close()
    val.close()

    manifest = {
        "corpus": "HuggingFaceFW/fineweb-edu sample/10BT",
        "corpus_license": "ODC-By 1.0 (plus Common Crawl terms of use)",
        "tokenizer": "mistralai/Mistral-7B-v0.1 tokenizer.json (Apache 2.0)",
        "tokenizer_sha256": tok_sha,
        "vocab_size": tok.get_vocab_size(),
        "dtype": "uint16",
        "eos_id": EOS,
        "bos_in_stream": False,
        "val_docs": VAL_DOCS,
        "val_source": "first 10 000 documents of shard 000, excluded from train",
        "shard_tokens": SHARD_TOKENS,
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "train_tokens": sum(int(s["tokens"]) for s in train.shards),
        "val_tokens": sum(int(s["tokens"]) for s in val.shards),
        "train_shards": train.shards,
        "val_shards": val.shards,
        "sources": sources,
        "wall_s": round(time.time() - t_start, 1),
        "adr": "ADR-014",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {out}/manifest.json: {total_tokens:,} tokens, {total_docs:,} docs, "
        f"{manifest['wall_s']} s"
    )


if __name__ == "__main__":
    main()
