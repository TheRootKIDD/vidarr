"""ADR-058 §3: the pair compare of `eval_wiki_lead` (seed merge and the H17 rule on pair means)."""

from __future__ import annotations

import numpy as np

from scripts.analysis.eval_wiki_lead import COVERED, SPLITS, merge_seeds, score


def record(
    rng: np.random.Generator, n: int, shift: float, shifts: dict[str, float] | None = None
) -> dict:
    """Token counts depend on the article only (fixed seed), as in real runs; losses vary.
    `shifts` overrides the loss shift per split (covered splits are added when given)."""
    shifts = shifts or {}
    names = SPLITS + tuple(k for k in COVERED if k in shifts)
    counts = np.random.default_rng(99).integers(5, 40, (len(names), n)).astype(float)
    pool = ["in_table"] * (n // 2) + ["out_of_table"] * (n - n // 2)
    pa: dict = {"id": list(range(n)), "pool": pool}
    for s, cnt in zip(names, counts, strict=True):
        d = shifts.get(s, shift)
        pa["cnt_" + s] = cnt.tolist()
        pa["sum_" + s] = (cnt * (3.0 + d + 0.1 * rng.standard_normal(n))).tolist()
    return {"per_article": pa, "heldout": {"val_loss": 3.0}}


def test_merge_is_token_weighted_pair_mean() -> None:
    rng = np.random.default_rng(0)
    a, b = record(rng, 200, 0.0), record(rng, 200, 0.1)
    m = merge_seeds([a, b], [3.01, 3.03])
    for s in SPLITS:
        want = (np.sum(a["per_article"]["sum_" + s]) + np.sum(b["per_article"]["sum_" + s])) / (
            np.sum(a["per_article"]["cnt_" + s]) + np.sum(b["per_article"]["cnt_" + s])
        )
        got = m["per_article"]["sum_" + s].sum() / m["per_article"]["cnt_" + s].sum()
        assert np.isclose(got, want)
    assert np.isclose(m["heldout"]["val_loss"], 3.02)


def test_rule_on_merged_pairs() -> None:
    rng = np.random.default_rng(1)
    base = merge_seeds([record(rng, 400, 0.0), record(rng, 400, 0.0)], [3.07, 3.08])
    same = merge_seeds([record(rng, 400, 0.0), record(rng, 400, 0.0)], [3.07, 3.08])
    better = merge_seeds([record(rng, 400, -0.1), record(rng, 400, -0.1)], [3.07, 3.08])
    worse_gen = merge_seeds([record(rng, 400, -0.1), record(rng, 400, -0.1)], [3.10, 3.10])
    assert score(same, base, "a", "b")["verdict_h17"] == "weakens"  # neither split beats
    assert score(better, base, "a", "b")["verdict_h17"] == "supports"
    assert score(worse_gen, base, "a", "b")["verdict_h17"] == "weakens"  # Δ 0.025 > 4σ


def test_covered_verdict() -> None:
    """ADR-059: covered first sentence beats and full misses -> covered supports, full silent."""
    rng = np.random.default_rng(2)
    z = {"in_first_cov": 0.0, "in_first_hit": 0.0}
    arm_s = {
        "in_first": 0.0,
        "in_rest": -0.1,
        "out_lead": 0.0,
        "in_first_cov": -0.1,
        "in_first_hit": -0.1,
    }
    base = merge_seeds([record(rng, 400, 0.0, z), record(rng, 400, 0.0, z)], [3.07, 3.08])
    arm = merge_seeds([record(rng, 400, 0.0, arm_s), record(rng, 400, 0.0, arm_s)], [3.07, 3.08])
    res = score(arm, base, "a", "b")
    assert res["verdict_h17"] == "silent"
    assert res["verdict_h17_covered"] == "supports"
    assert "verdict_h17_covered" not in score(
        record(rng, 400, 0.0), record(rng, 400, 0.0), "a", "b"
    )
