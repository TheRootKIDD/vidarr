"""Unit tests of `model/` against `docs/01`: shapes, causal masks, stream visibility
(ADR-002), local/global ranges (ADR-021/022/023), halting and ragged-depth
semantics (ADR-010/011), MTP subsampling (ADR-012), router balance, widths and
parameter counts (ADR-025). Everything runs on CPU at toy size."""

from __future__ import annotations

import dataclasses as dc

import pytest
import torch

from costmodel.config import Attention as CmAttention
from costmodel.config import Experts as CmExperts
from costmodel.config import Model as CmModel
from costmodel.params import dense_baseline_params, total_params
from model.block import Block
from model.config import HaltingCfg, ModelCfg, StreamsCfg, ladder
from model.layers import Expert, causal_mask, local_window_mask
from model.middle import IterState
from model.model import BigMoE
from model.moe import MoE

STEPS = ["L0", "L1", "L2", "L3", "L5", "L5-ne1", "L5-noej", "L5d", "L6", "L7-g4", "L7b", "L4"]


def tiny(step: str, **kw) -> ModelCfg:
    c = ladder(step)
    c = dc.replace(
        c,
        d=64,
        vocab=97,
        context=64,
        layers=3,
        dense_d_ff=128,
        attention=dc.replace(c.attention, heads=4, kv_heads=2, head_dim=16, window=16),
        experts=dc.replace(c.experts, d_ff=32),
        middle=dc.replace(c.middle, r_max=3, depth_embed=8),
    )
    c = dc.replace(c, **kw)
    c.validate()
    return c


def r_for(cfg: ModelCfg) -> int | None:
    return None if cfg.halting.mode == "act" else cfg.middle.r_max


# ---- shapes and the ladder ------------------------------------------------------------
@pytest.mark.parametrize("step", STEPS)
def test_every_rung_builds_and_trains_one_step(step: str) -> None:
    torch.manual_seed(0)
    cfg = tiny(step)
    m = BigMoE(cfg)
    toks = torch.randint(0, cfg.vocab, (2, cfg.context + 1))
    out = m.loss(toks, r=r_for(cfg))
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    h, ex = m.forward_hidden(toks[:, :-1], r=r_for(cfg))
    assert h.shape == (2, cfg.context, cfg.d)
    if cfg.arch == "recurrent":
        assert ex.r_map.shape == (2, cfg.context)
        assert (ex.r_map <= cfg.middle.r_max).all()


def test_derived_widths_follow_adr_025_027() -> None:
    assert ladder("L5").expert_width() == 896
    assert ladder("L5-ne128").expert_width() == 448
    assert ladder("L5-ne1").expert_width() == 1792
    assert ladder("L7b").expert_width() == 448  # L_e = 2 halves the width at matched FLOPs
    d = ladder("L7-g4").to_dict()["derived"]
    assert d["sub_expert_width"] == 224 and d["n_sub_experts"] == 32
    assert ladder("L1").expert_width() == 1024  # dense 2048 / k = 2


def test_param_counts_match_costmodel() -> None:
    with torch.device("meta"):
        l5 = BigMoE(ladder("L5")).n_params()
        l0 = BigMoE(ladder("L0")).n_params()
    cm = CmModel(
        d=768,
        vocab=32000,
        early_blocks=2,
        final_blocks=2,
        r_max=8,
        attn=CmAttention(12, 4, 64),
        experts=CmExperts(n_e=8, k=2, d_ff=896, L_e=1),
    )
    ref = total_params(cm)
    assert abs(l5["total"] - ref["total"]) / ref["total"] < 0.005
    dense = dense_baseline_params(12, 768, CmAttention(12, 4, 64), 2048, 32000)
    assert abs(l0["total"] - dense["total"]) / dense["total"] < 0.005


# ---- masks --------------------------------------------------------------------------
def test_masks() -> None:
    m = causal_mask(4, 4, torch.device("cpu"))
    assert m.tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
    w = local_window_mask(5, 2, torch.device("cpu"))
    assert w.int().tolist() == [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ]
    off = causal_mask(2, 5, torch.device("cpu"), offset=3)  # queries at absolute 3, 4
    assert off.int().tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]


@pytest.mark.parametrize("step", ["L0", "L2", "L5", "L5d"])
def test_no_future_leakage(step: str) -> None:
    """Perturbing token t must not change hidden states at positions < t."""
    torch.manual_seed(0)
    cfg = tiny(step)
    m = BigMoE(cfg).eval()
    toks = torch.randint(0, cfg.vocab, (1, cfg.context))
    t = 40
    alt = toks.clone()
    alt[0, t] = (alt[0, t] + 1) % cfg.vocab
    with torch.no_grad():
        h1, _ = m.forward_hidden(toks, r=cfg.middle.r_max)
        h2, _ = m.forward_hidden(alt, r=cfg.middle.r_max)
    assert torch.allclose(h1[0, :t], h2[0, :t], atol=1e-5)
    assert not torch.allclose(h1[0, t:], h2[0, t:])


def test_final_vector_global_reaches_later_segments_only() -> None:
    """L5d: a token in segment 1 influences segment 2 through G; segment 2 never
    influences segment 1; and within a segment the local range is the window."""
    torch.manual_seed(0)
    cfg = tiny("L5d")  # window 16 -> 4 segments of 16
    m = BigMoE(cfg).eval()
    toks = torch.randint(0, cfg.vocab, (1, cfg.context))
    alt = toks.clone()
    alt[0, 3] = (alt[0, 3] + 1) % cfg.vocab  # segment 0
    with torch.no_grad():
        h1, _ = m.forward_hidden(toks, r=3)
        h2, _ = m.forward_hidden(alt, r=3)
    assert not torch.allclose(h1[0, 16:32], h2[0, 16:32])  # segment 1 sees it through G
    alt2 = toks.clone()
    alt2[0, 20] = (alt2[0, 20] + 1) % cfg.vocab  # segment 1
    with torch.no_grad():
        h3, _ = m.forward_hidden(alt2, r=3)
    assert torch.allclose(h1[0, :16], h3[0, :16], atol=1e-5)  # segment 0 unchanged


def test_segment_memory_gradient_knob() -> None:
    cfg = tiny("L5d")
    m = BigMoE(cfg)
    x = torch.randn(1, 16, cfg.d, requires_grad=True)
    pos = torch.arange(16)
    k, _ = m.middle.global_entry(x, pos)
    k.sum().backward()
    assert x.grad is None  # ADR-023 default: no gradient into the earlier segment's vectors
    assert m.middle.g_norm.weight.grad is not None  # but the cache's own parameters train
    cfg2 = tiny("L5d", attention=dc.replace(tiny("L5d").attention, segment_memory_grad="through"))
    m2 = BigMoE(cfg2)
    x2 = torch.randn(1, 16, cfg.d, requires_grad=True)
    k2, _ = m2.middle.global_entry(x2, pos)
    k2.sum().backward()
    assert x2.grad is not None and x2.grad.abs().sum() > 0


# ---- streams (ADR-002) ------------------------------------------------------------------
def test_context_stream_never_sees_the_prediction_stream() -> None:
    torch.manual_seed(0)
    cfg = tiny("L4")
    blk = Block(cfg, "dense", 128, "parallel").eval()
    c = torch.randn(1, 8, cfg.d)
    p1, p2 = torch.randn(1, 8, cfg.d), torch.randn(1, 8, cfg.d)
    pos = torch.arange(8)
    with torch.no_grad():
        c1, _, _ = blk(c, p1, pos)
        c2, _, _ = blk(c, p2, pos)
    assert torch.allclose(c1, c2)


@pytest.mark.parametrize("sees_own", [True, False])
def test_prediction_stream_visibility_of_own_context(sees_own: bool) -> None:
    torch.manual_seed(0)
    cfg = tiny("L4", streams=StreamsCfg(two_stream=True, p_sees_own_context=sees_own))
    blk = Block(cfg, "dense", 128, "parallel").eval()
    c = torch.randn(1, 8, cfg.d)
    p = torch.randn(1, 8, cfg.d)
    c_alt = c.clone()
    c_alt[0, 5] += 1.0
    pos = torch.arange(8)
    with torch.no_grad():
        _, p1, _ = blk(c, p, pos)
        _, p2, _ = blk(c_alt, p, pos)
    changed_at_5 = not torch.allclose(p1[0, 5], p2[0, 5])
    assert changed_at_5 == sees_own  # L4b: p_t sees c_{<t} only
    assert not torch.allclose(p1[0, 6:], p2[0, 6:])  # later positions see it either way
    assert torch.allclose(p1[0, :5], p2[0, :5])


# ---- ragged depth (ADR-010 / ADR-011) ----------------------------------------------------
def test_halted_tokens_keep_state_and_a_single_kv_entry() -> None:
    torch.manual_seed(0)
    base = tiny("L6", halting=HaltingCfg(mode="act", ponder_cost=0.0, epsilon=0.01))
    cfg = dc.replace(base, middle=dc.replace(base.middle, r_max=4))  # room for a 4th iteration
    m = BigMoE(cfg).eval()
    mid = m.middle
    b, t = 1, 16
    c = torch.randn(b, t, cfg.d)
    pos = torch.arange(t)
    # make the halting head fire: bias so that sigmoid ~ 1 for even positions, ~0 for odd
    with torch.no_grad():
        mid.halt.weight.zero_()
        mid.halt.bias.fill_(-20.0)
    st = IterState(
        c=c,
        p=None,
        active=torch.ones(b, t, dtype=torch.bool),
        r=torch.zeros(b, t, dtype=torch.long),
        cum_halt=torch.zeros(b, t),
        remainder=torch.zeros(b, t),
        frozen_kv=None,
        frozen=torch.zeros(b, t, dtype=torch.bool),
        loads=[],
    )
    # force even positions to halt after iteration 2 (= r_min) by seeding cum_halt
    st.cum_halt[0, ::2] = 1.0
    states, frozen_kvs = [], []
    with torch.no_grad():
        for j in range(1, 4):
            st, _ = mid.iteration(st, j, pos, None, force_active=False)
            states.append(st.c.clone())
            frozen_kvs.append(None if st.frozen_kv is None else st.frozen_kv[0].clone())
    even, odd = slice(0, None, 2), slice(1, None, 2)
    # even tokens halted at r = 2 (r_min): state after iteration 3 equals state after 2
    assert (st.r[0, even] == 2).all() and (st.r[0, odd] == 3).all()
    assert torch.allclose(states[2][0, even], states[1][0, even])
    assert not torch.allclose(states[2][0, odd], states[1][0, odd])
    # their K/V were frozen at iteration 3 (the first after halting) and are marked frozen
    assert st.frozen[0, even].all() and not st.frozen[0, odd].any()
    assert frozen_kvs[1] is None and frozen_kvs[2] is not None
    # a further iteration must reuse the frozen K/V unchanged
    with torch.no_grad():
        st4, _ = mid.iteration(st, 4, pos, None, force_active=False)
    assert torch.equal(st4.frozen_kv[0][:, :, even], frozen_kvs[2][:, :, even])
    assert torch.allclose(st4.c[0, even], states[1][0, even])


def test_fixed_mode_runs_exactly_r_iterations() -> None:
    cfg = tiny("L5")
    m = BigMoE(cfg).eval()
    toks = torch.randint(0, cfg.vocab, (2, cfg.context))
    with torch.no_grad():
        _, ex = m.forward_hidden(toks, r=2)
    assert (ex.r_map == 2).all()
    assert len(ex.loads) == 2


# ---- MTP (ADR-012) ------------------------------------------------------------------------
def test_mtp_subsampling_covers_every_position_once() -> None:
    torch.manual_seed(0)
    cfg = tiny("L3", mtp=dc.replace(tiny("L3").mtp, m=2))  # one auxiliary head -> all positions
    m = BigMoE(cfg).eval()
    toks = torch.randint(0, cfg.vocab, (2, cfg.context + 1))
    h, _ = m.forward_hidden(toks[:, :-1])
    aux, metrics = m.mtp.aux_losses(h, toks, m.emb.weight, None)
    # manual: head 2 predicts x_{t+2} for t < T - 2, over all such positions
    valid = cfg.context - 2
    hs = m.mtp.adapters[0](h[:, :valid])
    logits = hs @ m.emb.weight.t()
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab).float(), toks[:, 2 : 2 + valid].reshape(-1)
    )
    assert torch.allclose(aux, cfg.mtp.lambdas[1] * ref, atol=1e-5)
    # I27: both metrics logged under names that cannot be confused, and acceptance is
    # agreement with head 1's prediction of the same token from position t+1.
    assert "mtp_acc_h2" not in metrics
    draft = logits.argmax(-1)  # head 2's draft of x_{t+2}, made at t
    top1 = (draft == toks[:, 2 : 2 + valid]).float().mean()
    pred1 = (h @ m.emb.weight.t()).argmax(-1)  # head 1 at position s predicts x_{s+1}
    accept = (draft == pred1[:, 1 : 1 + valid]).float().mean()
    assert abs(metrics["mtp_top1_h2"] - float(top1)) < 1e-6
    assert abs(metrics["mtp_accept_h2"] - float(accept)) < 1e-6


def test_mtp_full_trains_every_head_on_every_position() -> None:
    """L3-full (I27 #3): subsample = 1.0 gives each head the full-position loss."""
    torch.manual_seed(0)
    cfg = tiny("L3", mtp=dc.replace(tiny("L3").mtp, m=3, subsample=1.0))
    m = BigMoE(cfg).eval()
    toks = torch.randint(0, cfg.vocab, (2, cfg.context + 1))
    h, _ = m.forward_hidden(toks[:, :-1])
    aux, metrics = m.mtp.aux_losses(h, toks, m.emb.weight, None)
    valid = cfg.context - 3
    ref = 0.0
    for j in (2, 3):
        logits = m.mtp.adapters[j - 2](h[:, :valid]) @ m.emb.weight.t()
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab).float(), toks[:, j : j + valid].reshape(-1)
        )
        assert abs(metrics[f"mtp_loss_h{j}"] - float(loss)) < 1e-5
        ref = ref + cfg.mtp.lambdas[j - 1] * loss
    assert torch.allclose(aux, ref, atol=1e-5)


# ---- router and MoE ---------------------------------------------------------------------
def test_router_gates_loads_and_bias_controller() -> None:
    torch.manual_seed(0)
    cfg = tiny("L1")
    moe = MoE(cfg)
    u = torch.randn(50, cfg.d)
    route = moe.router(u)
    assert route.idx.shape == (50, cfg.router.k)
    assert torch.allclose(route.gate.sum(-1), torch.ones(50), atol=1e-5)
    assert torch.allclose(route.load.sum(), torch.tensor(1.0))
    before = moe.router.bias.clone()
    load = torch.zeros(cfg.router.n_experts)
    load[0] = 1.0  # expert 0 takes everything
    moe.router.update_bias(load)
    assert moe.router.bias[0] < before[0] and (moe.router.bias[1:] > before[1:]).all()


def test_moe_with_one_expert_is_that_expert() -> None:
    torch.manual_seed(0)
    cfg = tiny("L5-ne1")
    moe = MoE(cfg)
    u = torch.randn(3, 5, cfg.d)
    y, _ = moe(u)
    assert torch.allclose(y, moe.experts[0](u), atol=1e-6)


def test_two_layer_expert_has_residual_and_norm() -> None:
    e = Expert(8, 16, layers=2)
    assert len(e.ffs) == 2 and len(e.norms) == 1
    x = torch.randn(4, 8)
    assert e(x).shape == (4, 8)
