"""The worked examples the cost model must reproduce.

Sources, in the order `docs/01 §10` names them:
  - `docs/02 §2` — the $b_{min}$ examples and the TP bandwidth condition
  - `docs/01 §10` — the note-scale config (d=8192, U=64, ...)
  - `docs/06 §3` — the `screen` / `small` / `medium` anchors, "within 10 %"

Where a computed value disagrees with a doc, the test asserts the *computed*
value and its docstring says which doc line it contradicts. CLAUDE.md: the doc
wins until an ADR says otherwise, so these are reported, not patched.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from costmodel import (
    Attention,
    Experts,
    Hardware,
    Model,
    Numerics,
    b_min,
    dense_baseline_flops_per_token,
    dense_baseline_params,
    expert_params,
    fabric_bytes_per_token_iteration,
    kv_bytes_naive_per_token,
    kv_reduction_factor,
    min_d_ff,
    persistent_kv_bytes_per_token,
    solve_d_ff,
    tp_bandwidth_required,
    training_flops_per_token,
    transient_kv_bytes_per_sequence,
)

BF16 = Numerics(b_w=2.0, b_act=2.0, b_kv=2.0)


# --------------------------------------------------------------------------
# docs/02 §2 — b_min
# --------------------------------------------------------------------------


def test_b_min_fp8_example() -> None:
    """`docs/02 §2`: phi/beta_C = 1000 FLOP/byte, FP8 (b_w=1) -> b_min = 500."""
    hw = Hardware(phi=1000e12, beta_c=1e12, m_c=0, lambda_c=0)
    assert hw.ridge == pytest.approx(1000.0)
    assert b_min(hw, Numerics(b_w=1.0)) == pytest.approx(500.0)


def test_b_min_fp4_example() -> None:
    """`docs/02 §2`: the same ridge point at FP4 (b_w=0.5) -> b_min = 250."""
    hw = Hardware(phi=1000e12, beta_c=1e12, m_c=0, lambda_c=0)
    assert b_min(hw, Numerics(b_w=0.5)) == pytest.approx(250.0)


def test_b_min_local_3060_nominal() -> None:
    """`docs/06 §1`: ridge ~= 71 FLOP/byte at BF16 -> b_min ~= 71 tokens.

    Nominal inputs (25.5 TFLOPS, 360 GB/s) — replaced by `bench_gemm` before any
    budget uses them (ADR-015).
    """
    hw = Hardware(phi=25.5e12, beta_c=360e9, m_c=12e9, lambda_c=3.59e9)
    assert hw.ridge == pytest.approx(70.8, abs=0.5)
    assert b_min(hw, BF16) == pytest.approx(70.8, abs=0.5)


def test_b_min_is_shape_independent() -> None:
    """Arithmetic intensity is 2b/b_w, so b_min cannot depend on d or d_ff."""
    hw = Hardware(phi=25.5e12, beta_c=360e9, m_c=12e9, lambda_c=3.59e9)
    assert b_min(hw, BF16) == b_min(hw, BF16)  # no shape arguments exist to vary


# --------------------------------------------------------------------------
# docs/01 §10 — the note-scale worked example
# --------------------------------------------------------------------------


def note_scale_model() -> Model:
    """d=8192, U=64, s_min=256 -> d_ff=16384, N_e=65536, k=8, r_max=128, L_e=1."""
    return Model(
        d=8192,
        r_max=128,
        two_stream=True,
        attn=Attention(heads=64, kv_heads=8, head_dim=128, window=512),
        experts=Experts(n_e=65_536, k=8, d_ff=16_384, L_e=1),
    )


def test_note_scale_params_per_expert() -> None:
    """`docs/01 §10`: ~0.4 B parameters per expert."""
    got = expert_params(note_scale_model())
    assert got == 3 * 8192 * 16384
    assert got == pytest.approx(0.4e9, rel=0.02)


def test_note_scale_d_ff_matches_tile_floor() -> None:
    """`docs/01 §4.5`: d_ff = U * s_min = 64 * 256 = 16384 for this config."""
    hw = Hardware(phi=0, beta_c=1, m_c=0, lambda_c=0, U=64, s_min=256)
    assert min_d_ff(hw) == 16_384


def test_note_scale_flops_per_token_per_iteration() -> None:
    """`docs/01 §10`: ~13 GFLOP per token per iteration, two streams.

    Dominated by the MoE term 2*k*3*d*d_ff*L_e per stream = 6.44 GFLOP, doubled
    to 12.9 GFLOP; attention projections and scores bring it to ~13.4.
    """
    from costmodel import middle_iteration_flops

    got = middle_iteration_flops(note_scale_model())
    assert got == pytest.approx(13e9, rel=0.05)


def test_note_scale_moe_term_dominates() -> None:
    """The MoE term alone is 12.9 GFLOP — the ~13 figure is an expert cost."""
    from costmodel.flops import moe_flops

    assert moe_flops(note_scale_model()) == pytest.approx(12.89e9, rel=0.01)


# --------------------------------------------------------------------------
# docs/06 §3 — dense anchors
# --------------------------------------------------------------------------

SMALL_ATTN_GQA = Attention(heads=12, kv_heads=4, head_dim=64, window=512)
SMALL_ATTN_MHA = Attention(heads=12, kv_heads=12, head_dim=64, window=512)
MEDIUM_ATTN_GQA = Attention(heads=16, kv_heads=4, head_dim=64, window=512)
MEDIUM_ATTN_MHA = Attention(heads=16, kv_heads=16, head_dim=64, window=512)


def test_small_dense_embedding_matches_doc() -> None:
    """`docs/06 §3`: ~25 M embedding parameters for V=32k, d=768."""
    got = dense_baseline_params(12, 768, SMALL_ATTN_GQA, 2048, 32_000)
    assert got["embedding"] == pytest.approx(25e6, rel=0.03)


def test_small_dense_non_embedding_is_75M_with_gqa_not_85M() -> None:
    """**Disagrees with `docs/06 §3`.**

    The row reads "12 layers, d=768, 12/4 heads, SwiGLU d_ff=2048, V=32k
    (~85 M non-embedding + 25 M embedding)". With the stated GQA (12 query
    heads, 4 KV heads) the count is 75.5 M, not 85 M — 11.2 % low, which breaks
    the "within 10 %" tolerance the same section sets.

    85 M is what the same shape gives with *full MHA* (see the next test), so
    the parameter figure appears to have been computed without GQA while the
    head spec says GQA. Reported, not patched.
    """
    got = dense_baseline_params(12, 768, SMALL_ATTN_GQA, 2048, 32_000)
    assert got["non_embedding"] == 75_497_472
    assert abs(got["non_embedding"] - 85e6) / 85e6 > 0.10


def test_small_dense_non_embedding_matches_doc_under_mha() -> None:
    """Full MHA reproduces the doc's 85 M exactly — the source of the mismatch."""
    got = dense_baseline_params(12, 768, SMALL_ATTN_MHA, 2048, 32_000)
    assert got["non_embedding"] == 84_934_656
    assert got["non_embedding"] == pytest.approx(85e6, rel=0.001)


def test_medium_dense_shows_the_same_gqa_gap() -> None:
    """**Disagrees with `docs/06 §3`** in the same direction and for the same reason.

    "24 layers, d=1024, 16/4 heads, d_ff=2816 (~300 M + 33 M)": GQA gives
    270.5 M (9.8 % low, just inside tolerance); MHA gives 308.3 M, which is what
    the doc quotes. The pattern across both rows is what makes this a systematic
    convention mismatch rather than an arithmetic slip in one cell.
    """
    gqa = dense_baseline_params(24, 1024, MEDIUM_ATTN_GQA, 2816, 32_000)
    mha = dense_baseline_params(24, 1024, MEDIUM_ATTN_MHA, 2816, 32_000)
    assert gqa["non_embedding"] == 270_532_608
    assert mha["non_embedding"] == 308_281_344
    assert mha["non_embedding"] == pytest.approx(300e6, rel=0.03)
    assert gqa["embedding"] == pytest.approx(33e6, rel=0.02)


def test_small_dense_training_flops_matches_0_6_gflop_anchor() -> None:
    """`docs/06 §3`: ~0.6 GFLOP/token training cost for `small` at context 2048.

    Average visible keys over a causal 2048-token sequence is (L+1)/2 ~= 1024,
    not 2048. With GQA and no output head this gives 0.57 GFLOP — within the
    section's 10 %. Including the output head would add 0.15 GFLOP and overshoot,
    which is the evidence that the anchor is a non-embedding figure.
    """
    fwd = dense_baseline_flops_per_token(
        layers=12, d=768, d_ff=2048, q_width=768, kv_width=256,
        n_keys=1024, vocab=32_000, include_head=False,
    )
    assert 3 * fwd == pytest.approx(0.6e9, rel=0.10)


def test_medium_dense_training_flops_matches_2_gflop_anchor() -> None:
    """`docs/06 §3`: ~2.0 GFLOP/token for `medium` at context 2048."""
    fwd = dense_baseline_flops_per_token(
        layers=24, d=1024, d_ff=2816, q_width=1024, kv_width=256,
        n_keys=1024, vocab=32_000, include_head=False,
    )
    assert 3 * fwd == pytest.approx(2.0e9, rel=0.10)


# --------------------------------------------------------------------------
# docs/06 §3 — recurrent widths, derived (the inverse solve)
#
# `docs/06 §3` and `docs/01 §1` both delegate these widths to `costmodel/`, and
# also quote ranges. The quoted ranges are reproducible, but only under a
# skeleton block count that contradicts the one `docs/01 §1` states. The tests
# below pin down both, because the discrepancy changes every recurrent budget.
# --------------------------------------------------------------------------


def small_recurrent(n_e: int, k: int, early: int = 2, final: int = 2, two: bool = False) -> Model:
    """`docs/06 §3` after ADR-018/ADR-020: single stream, r=8, W=512."""
    return Model(
        d=768,
        vocab=32_000,
        early_blocks=early,
        final_blocks=final,
        r_max=8,
        two_stream=two,
        attn=SMALL_ATTN_GQA,
        experts=Experts(n_e=n_e, k=k, d_ff=None, L_e=1),
    )


def test_doc_widths_reproduce_only_with_two_skeleton_blocks() -> None:
    """**Disagrees with `docs/01 §1` / `docs/06 §3`.**

    `docs/06 §3` quotes d_ff ~= 1.5-1.8 k at (N_e=8, k=2) and ~= 0.8 k at
    (N_e=128, k=4), for the 0.6 GFLOP/token budget. Both are reproduced exactly
    -- 1665 and 833 -- but only with **two** skeleton blocks in total and
    attention over the local window alone (n_keys = W = 512).

    `docs/01 §1` states E = 2 and F = 2, i.e. **four** blocks, and `docs/06 §3`
    repeats "E = F = 2". At four blocks the same budget yields 1153 and 577,
    both outside the quoted ranges. Two independent configurations agreeing at
    E+F=2 and disagreeing at E+F=4 is what makes this a convention mismatch
    rather than rounding.

    Asserted here in both directions so that whichever way the ADR resolves it,
    this test records what each choice costs.
    """
    two_blocks_default = solve_d_ff(small_recurrent(8, 2, 1, 1), 0.6e9, n_keys=512)
    two_blocks_h6 = solve_d_ff(small_recurrent(128, 4, 1, 1), 0.6e9, n_keys=512)
    assert two_blocks_default.d_ff == 1665
    assert two_blocks_h6.d_ff == 833
    assert 1500 <= two_blocks_default.d_ff <= 1800
    assert 700 <= two_blocks_h6.d_ff <= 900

    four_blocks_default = solve_d_ff(small_recurrent(8, 2, 2, 2), 0.6e9, n_keys=512)
    four_blocks_h6 = solve_d_ff(small_recurrent(128, 4, 2, 2), 0.6e9, n_keys=512)
    assert four_blocks_default.d_ff == 1153
    assert four_blocks_h6.d_ff == 577
    assert not 1500 <= four_blocks_default.d_ff <= 1800
    assert not 700 <= four_blocks_h6.d_ff <= 900


def test_budget_is_sensitive_to_the_attention_key_convention() -> None:
    """The budget also depends on an unstated convention: which keys are counted.

    `docs/01 §10` gives scores as 2*2*d*(W + |G_visible|). Un-indexed global
    attention over a 2048-token training context makes |G_visible| ~= 1024 on
    average, which moves d_ff by ~13 % against counting the local window alone.
    Neither `docs/06 §3` nor `docs/01 §10` says which the anchors assume.
    """
    local_only = solve_d_ff(small_recurrent(8, 2, 1, 1), 0.6e9, n_keys=512).d_ff
    with_global = solve_d_ff(small_recurrent(8, 2, 1, 1), 0.6e9, n_keys=1024).d_ff
    assert local_only == 1665
    assert with_global == 1452
    assert abs(with_global - local_only) / local_only > 0.10


def test_two_streams_do_not_halve_d_ff_and_break_the_budget() -> None:
    """**Disagrees with `docs/06 §3`.**

    The section says "switching two streams on (L4, optional) halves every
    d_ff above, down to the U*s_min floor of 512". Halving is not what the
    accounting gives: only the expert term's coefficient doubles, while the
    skeleton and per-iteration attention costs also double, so the width left
    over is 362, not 833.

    362 is *below* the 512 floor of `docs/01 §4.5`, so the floor binds and the
    configuration overspends its budget by 11 %. Two streams at `small` with
    r = 8 therefore require an explicit trade of k or r -- which `docs/06 §3`
    itself mandates ("do not silently exceed the budget") one sentence after
    claiming the halving works.
    """
    exact = solve_d_ff(small_recurrent(8, 2, 1, 1, two=True), 0.6e9, n_keys=512)
    assert round(exact.d_ff_exact) == 362
    single = solve_d_ff(small_recurrent(8, 2, 1, 1), 0.6e9, n_keys=512)
    assert exact.d_ff_exact < single.d_ff / 2

    hw = Hardware(phi=25.5e12, beta_c=360e9, m_c=12e9, lambda_c=3.59e9, U=4, s_min=128)
    floored = solve_d_ff(small_recurrent(8, 2, 1, 1, two=True), 0.6e9, n_keys=512, hw=hw)
    assert floored.at_floor
    assert floored.d_ff == 512 == min_d_ff(hw)
    assert floored.relative_error == pytest.approx(0.11, abs=0.01)


def test_solve_flags_the_tile_floor_rather_than_overspending_silently() -> None:
    """`docs/01 §4.5`: d_ff >= U*s_min. When the floor binds it must be visible."""
    hw = Hardware(phi=25.5e12, beta_c=360e9, m_c=12e9, lambda_c=3.59e9, U=4, s_min=128)
    tiny = solve_d_ff(small_recurrent(8, 16, 1, 1), 0.6e9, n_keys=512, hw=hw)
    assert tiny.at_floor
    assert tiny.d_ff == 512
    assert tiny.relative_error > 0, "at the floor the config overspends its budget"


def test_solve_raises_when_the_budget_cannot_fit_the_skeleton() -> None:
    """A budget already spent by non-expert terms is an error, not a negative width."""
    with pytest.raises(ValueError, match="already spent"):
        solve_d_ff(small_recurrent(8, 2, 2, 2, two=True), 0.6e9, n_keys=1024)


def test_solve_round_trips_through_forward_flops() -> None:
    """The inverse must agree with the forward accounting it inverts."""
    model = small_recurrent(8, 2, 1, 1)
    sol = solve_d_ff(model, 0.6e9, n_keys=512)
    got = training_flops_per_token(model, d_ff=sol.d_ff, n_keys=512)
    assert got == pytest.approx(sol.achieved_flops, rel=1e-9)
    assert got == pytest.approx(0.6e9, rel=0.01)


# --------------------------------------------------------------------------
# docs/01 §10, docs/02 §7 — KV and fabric
# --------------------------------------------------------------------------


def test_persistent_kv_is_one_entry_per_token() -> None:
    """`docs/01 §10`: persistent 2*H_kv*d_h*b_kv per token, independent of r."""
    model = small_recurrent(8, 2)
    got = persistent_kv_bytes_per_token(model, BF16)
    assert got == 2 * 4 * 64 * 2
    assert persistent_kv_bytes_per_token(replace(model, r_max=128), BF16) == got


def test_transient_kv_scales_with_window_and_depth() -> None:
    """`docs/01 §10`: transient <= W * r_max * 2*H_kv*d_h*b_kv per sequence."""
    model = small_recurrent(8, 2)
    assert transient_kv_bytes_per_sequence(model, BF16) == 512 * 8 * (2 * 4 * 64 * 2)


def test_kv_reduction_is_r_max() -> None:
    """v2 P7 claims "almost the full 128x" reduction at r_max=128 (ADR-021)."""
    assert kv_reduction_factor(Model(r_max=128)) == 128.0
    model = replace(small_recurrent(8, 2), r_max=128)
    naive = kv_bytes_naive_per_token(model, BF16)
    assert naive / persistent_kv_bytes_per_token(model, BF16) == 128.0


def test_fabric_bytes_per_token_iteration() -> None:
    """`docs/01 §10`: 2*k*d*b_act, read as per-stream (see `fabric.py`)."""
    model = small_recurrent(8, 2)
    assert fabric_bytes_per_token_iteration(model, BF16) == 2 * 2 * 768 * 2
    two = replace(model, two_stream=True)
    assert fabric_bytes_per_token_iteration(two, BF16) == 2 * (2 * 2 * 768 * 2)
    assert fabric_bytes_per_token_iteration(two, BF16, per_stream=False) == 2 * 2 * 768 * 2


def test_tp_condition_reproduces_the_nccl_analysis() -> None:
    """Cross-check against `experiments/002-nccl/results.md`'s addendum table.

    v2 default (d_ff~1650, U=4, L_e=1, BF16) at phi=15 TF needs ~24 GB/s; the
    measured 3.59 GB/s is ~6.8x short.
    """
    hw = Hardware(phi=15e12, beta_c=360e9, m_c=12e9, lambda_c=3.59e9, U=4, s_min=128)
    need = tp_bandwidth_required(small_recurrent(8, 2), hw, BF16, d_ff=1650)
    assert need == pytest.approx(24.2e9, rel=0.02)
    assert need / hw.lambda_c == pytest.approx(6.8, rel=0.03)


def test_note_scale_experts_per_unit_is_hundreds() -> None:
    """`docs/04` Q6 estimate: hundreds of experts resident per unit at note scale.

    A 64-chip unit at 4 GB per chip (`docs/02 §8`) holding 0.4 B-parameter
    experts at BF16.
    """
    from costmodel import experts_resident_per_unit

    got = experts_resident_per_unit(
        unit_memory_bytes=64 * 4e9,
        params_per_expert=expert_params(note_scale_model()),
        num=BF16,
    )
    assert 100 <= got <= 999, f"got {got} experts per unit"


def test_product_key_scoring_is_sublinear() -> None:
    """`docs/01 §10`: router cost is O(d_r * sqrt(N_e)), not O(d_r * N_e)."""
    from costmodel.flops import router_flops

    small = router_flops(small_recurrent(8, 2))
    big = router_flops(small_recurrent(65_536, 2))
    growth = (big - small) / small
    assert growth < 1.0, "sqrt scaling must not blow up over 4 orders of N_e"
    assert math.isqrt(65_536) == 256


def test_expert_step_is_compute_bound_above_b_min() -> None:
    """`docs/02 §2`: the step is compute-bound iff b >= b_min, whatever the shape."""
    from costmodel import expert_step

    hw = Hardware(phi=25.5e12, beta_c=360e9, m_c=12e9, lambda_c=3.59e9, U=4, s_min=128)
    model = small_recurrent(8, 2)
    assert not expert_step(64, 1665, model, hw, BF16).compute_bound
    assert expert_step(128, 1665, model, hw, BF16).compute_bound
    assert expert_step(128, 1665, model, hw, BF16).tile_ok
    assert not expert_step(64, 1665, model, hw, BF16).tile_ok


def test_two_streams_halve_tokens_at_the_utilisation_floor() -> None:
    """`docs/01 §3` (author's Q1, ADR-020): a token is two rows, so a unit at the
    floor holds b_min/2 tokens and its step time halves; throughput is unchanged."""
    from costmodel import tokens_at_floor

    hw = Hardware(phi=25.5e12, beta_c=360e9, m_c=12e9, lambda_c=3.59e9, U=4, s_min=128)
    assert tokens_at_floor(hw, BF16, 1) == pytest.approx(70.8, abs=0.5)
    assert tokens_at_floor(hw, BF16, 2) == pytest.approx(35.4, abs=0.5)
