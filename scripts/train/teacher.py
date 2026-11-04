"""L8q's teacher (ADR-055, `docs/01 §4.6`): the frozen encoder's PQ code of the previous chunk.

Each training window's inputs (T tokens) are cut into window-aligned chunks of `chunk` tokens.
Chunk c's decoded text is encoded by the frozen encoder (BGE, as in build `040`: CLS pooling,
L2-normalised, fp16, no gradient), and its halves are assigned their nearest codebook centroid,
(a*_c, b*_c). Input positions p in [chunk (c+1), chunk (c+2)) take chunk c's code, so every
token the target depends on precedes p. The first chunk has no target, and a target is masked
(-1) if an EOS lies anywhere in [chunk c, chunk (c+2)), so it never crosses a document boundary.

`targets()` takes an `encode` callable, so the unit tests can use a toy encoder.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

EOS = 2
CORPUS_TOKENIZER = "/mnt/nvme/corpus/tokenizer/mistral-7b-v0.1/tokenizer.json"  # ADR-014


def chunk_plan(inp: Tensor, chunk: int, eos: int = EOS) -> Tensor:
    """[B, n_chunks - 1] bool: chunk c's code is a valid target for chunk c + 1 (no EOS in
    either). `inp` [B, T] are the model's input positions."""
    b, t = inp.shape
    n = t // chunk
    has_eos = (inp[:, : n * chunk].view(b, n, chunk) == eos).any(-1)  # [B, n]
    return ~(has_eos[:, :-1] | has_eos[:, 1:])


def spread(codes: Tensor, valid: Tensor, t: int, chunk: int) -> Tensor:
    """codes [B, n-1, 2] for chunks 0..n-2, valid [B, n-1] -> targets [B, T, 2] (-1 = none):
    the tokens of chunk c + 1 take chunk c's code."""
    b = codes.shape[0]
    out = torch.full((b, t, 2), -1, dtype=torch.long, device=codes.device)
    c = torch.where(valid[..., None], codes, torch.full_like(codes, -1))
    out[:, chunk : chunk * (c.shape[1] + 1)] = c.repeat_interleave(chunk, dim=1)
    return out


def nearest_code(emb: Tensor, mu1: Tensor, mu2: Tensor) -> Tensor:
    """[N, d_enc] L2-normalised embeddings -> [N, 2] nearest centroid per half (squared L2, as
    the k-means build assigns)."""
    e1, e2 = emb.float().chunk(2, dim=-1)

    def near(x: Tensor, mu: Tensor) -> Tensor:
        d = (x * x).sum(-1, keepdim=True) - 2 * x @ mu.float().T + (mu.float() ** 2).sum(-1)[None]
        return d.argmin(-1)

    return torch.stack([near(e1, mu1), near(e2, mu2)], dim=-1)


def targets(
    inp: Tensor,
    chunk: int,
    encode: Callable[[list[list[int]]], Tensor],
    mu1: Tensor,
    mu2: Tensor,
    eos: int = EOS,
) -> Tensor:
    """[B, T] input ids -> [B, T, 2] long PQ-code targets (-1 = none). `encode` maps a list of
    corpus-token id lists to [N, d_enc] normalised embeddings."""
    b, t = inp.shape
    n = t // chunk
    valid = chunk_plan(inp, chunk, eos)
    codes = torch.full((b, n - 1, 2), -1, dtype=torch.long, device=inp.device)
    idx = valid.nonzero(as_tuple=False)  # [M, 2] (row, chunk)
    if idx.numel():
        ids = inp.view(b, n, chunk)[idx[:, 0], idx[:, 1]].tolist()
        emb = encode(ids).to(inp.device)
        codes[idx[:, 0], idx[:, 1]] = nearest_code(emb, mu1, mu2)
    return spread(codes, valid, t, chunk)


class BgeTeacher:
    """The frozen encoder on one rank: corpus ids -> text -> BGE -> [N, 768], no gradient.
    Weights stay fp32 under fp16 autocast, as build `040` encoded the table."""

    def __init__(
        self, encoder_dir: str | Path, corpus_tokenizer: str | Path, device: torch.device,
        max_tokens: int = 128,
    ) -> None:
        from tokenizers import Tokenizer
        from transformers import AutoModel

        enc_dir = Path(encoder_dir)
        self.corpus_tok = Tokenizer.from_file(str(corpus_tokenizer))
        self.enc_tok = Tokenizer.from_file(str(enc_dir / "tokenizer.json"))
        self.enc_tok.enable_truncation(max_tokens)
        self.model = AutoModel.from_pretrained(enc_dir).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    def texts(self, ids: list[list[int]]) -> list[str]:
        return self.corpus_tok.decode_batch(ids, skip_special_tokens=True)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> Tensor:
        enc = self.enc_tok.encode_batch(texts)
        L = max(len(e.ids) for e in enc)
        x = torch.zeros(len(enc), L, dtype=torch.long)
        m = torch.zeros(len(enc), L, dtype=torch.long)
        for r, e in enumerate(enc):
            x[r, : len(e.ids)] = torch.tensor(e.ids)
            m[r, : len(e.ids)] = 1
        with torch.autocast("cuda", dtype=torch.float16):
            h = self.model(
                input_ids=x.to(self.device), attention_mask=m.to(self.device)
            ).last_hidden_state
        return F.normalize(h[:, 0].float(), dim=-1)

    def __call__(self, ids: list[list[int]]) -> Tensor:
        return self.encode_texts(self.texts(ids))
