"""Incremental decode equals the training forward (`docs/01 §4.8`, ADR-049 §1)."""

from __future__ import annotations

import pytest
import torch

from model import decode
from model.model import BigMoE
from tests.test_model_semantics import tiny


def _model(step: str = "L5") -> BigMoE:
    torch.manual_seed(0)
    m = BigMoE(tiny(step)).eval()
    with torch.no_grad():  # nonzero depth modulation, so e_j matters in the comparison
        for p in m.middle.norm.mod.parameters():
            p.normal_(0, 0.1)
    return m


def _full_logits(m: BigMoE, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        h, _ = m.forward_hidden(tokens)
        return m.head(h)


@pytest.mark.parametrize("prompt", [1, 10, 23])
def test_prefill_then_teacher_forced_decode_matches_full_forward(prompt: int) -> None:
    m = _model()
    torch.manual_seed(1)
    tokens = torch.randint(0, m.cfg.vocab, (3, 24))
    ref = _full_logits(m, tokens)
    cache = decode.new_cache(m, batch=3, t_max=24)
    got = [m.prefill(tokens[:, :prompt], cache)]
    for t in range(prompt, 24):
        got.append(m.decode_step(tokens[:, t : t + 1], cache)[:, None])
    out = torch.cat(got, dim=1)
    assert cache.length == 24
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_chunked_prefill_matches_one_chunk() -> None:
    m = _model()
    tokens = torch.randint(0, m.cfg.vocab, (2, 20))
    ref = _full_logits(m, tokens)
    cache = decode.new_cache(m, batch=2, t_max=20)
    out = torch.cat([m.prefill(tokens[:, :7], cache), m.prefill(tokens[:, 7:], cache)], dim=1)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_cache_has_one_buffer_per_block_and_iteration() -> None:
    m = _model()
    cache = decode.new_cache(m, batch=2, t_max=16)
    a = m.cfg.attention
    assert len(cache.early) == m.cfg.early_blocks
    assert len(cache.middle) == m.cfg.middle.r_max
    assert len(cache.final) == m.cfg.final_blocks
    assert cache.middle[0].k.shape == (2, a.kv_heads, 16, a.head_dim)
    with pytest.raises(ValueError, match="cache full"):
        m.prefill(torch.zeros(2, 17, dtype=torch.long), cache)


def test_split_iteration_equals_moe_forward() -> None:
    """owner_pre + local_experts + owner_post is the training iteration (the S0 harness split)."""
    m = _model()
    c = torch.randn(2, 5, m.cfg.d)
    pos = torch.arange(5)
    with torch.no_grad():
        u, a, route = decode.owner_pre(m, 2, c, pos, decode.new_cache(m, 2, 5).middle[1])
        split = decode.owner_post(c, a, decode.local_experts(m, u, route))
        y, _ = m.middle.moe(u, decode.depth_embed(m, 2, c.device))
        torch.testing.assert_close(split, c + a + y)


@pytest.mark.parametrize("step", ["L5d", "L5e", "L5c", "L6", "L0", "L4"])
def test_other_rungs_refuse(step: str) -> None:
    with pytest.raises(NotImplementedError):
        decode.new_cache(BigMoE(tiny(step)), batch=1, t_max=8)


def test_prefill_in_row_slices_matches_whole_batch() -> None:
    """The harness prefills sequences in slices to bound attention memory."""
    m = _model()
    tokens = torch.randint(0, m.cfg.vocab, (4, 12))
    ref = _full_logits(m, tokens)
    cache = decode.new_cache(m, batch=4, t_max=13)
    for b0 in (0, 2):
        m.prefill(tokens[b0 : b0 + 2, :11], cache.rows(b0, b0 + 2))
    cache.set_length(11)
    last = m.decode_step(tokens[:, 11:12], cache)
    torch.testing.assert_close(last, ref[:, 11], rtol=1e-4, atol=1e-4)
