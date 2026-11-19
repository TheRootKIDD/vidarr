"""ADR-076 (rig 054): the renaming rules, the table extension, and the report's readings."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from model.memory import exact_topk
from model.model import BigMoE
from scripts.analysis.eval_unknown_facts import SPLIT, boot, report, score_conditions
from scripts.analysis.eval_wiki_lead import Pipeline
from scripts.data.build_unknown_facts import (
    draw_pool,
    leaks,
    pseudo_name,
    rename,
    starts_with_title,
)


def rows() -> list[dict]:
    out = []
    for i in range(400):
        title = f"Quorvale{i:03d}x" if i % 4 and i % 10 != 3 else f"Anna Quorvale{i:03d}x"
        sent = f"{title} is a commune in the Somme department in northern France."
        if i % 10 == 3:  # the surname alone again: must be skipped as a leak
            sent = f"{title} is a town; Quorvale{i:03d}x is also a river."
        if i % 10 == 7:  # does not begin with the title: never a candidate
            sent = f"The village of {title} lies in the hills of the north."
        out.append({"id": i, "title": title, "sentence": sent, "lead": sent})
    return out


def test_title_must_start_the_sentence_as_a_word() -> None:
    assert starts_with_title("Paris", "Paris is the capital of France.")
    assert not starts_with_title("Paris", "Parisian cafes are common in France.")
    assert not starts_with_title("Par", "Par is a short title for a place.")
    assert not starts_with_title("Paris", "The city of Paris is the capital.")


def test_rename_replaces_whole_words_only_and_leaks_are_seen() -> None:
    s = rename("Ham", "Ham is near Hamburg; Ham has a market.", "Zeku Brova")
    assert s == "Zeku Brova is near Hamburg; Zeku Brova has a market."
    assert leaks("Ham", s)  # the title survives inside a longer word
    assert leaks("Anna Smith", "Zeku Brova was born in Leeds; Smith moved to York.")
    assert not leaks("Anna Smith", "Zeku Brova was born in Leeds and moved to York.")
    assert leaks("Core FTP", "Zeku Brova is a client developed by CoreFTP.com.")


def test_pseudo_names_are_seeded_and_respect_the_ban() -> None:
    a = [pseudo_name(np.random.default_rng(5), set()) for _ in range(3)]
    assert len(set(a)) == 1
    rng = np.random.default_rng(5)
    first = pseudo_name(rng, set())
    banned = {w.lower() for w in first.split()}
    again = pseudo_name(np.random.default_rng(5), banned)
    assert not banned & {w.lower() for w in again.split()}


def test_pool_is_seeded_excludes_ids_and_no_title_survives() -> None:
    r = rows()
    excluded = set(range(0, 400, 9))
    pool, counts = draw_pool(r, excluded, {"zeku"}, n=100)
    pool2, _ = draw_pool(r, excluded, {"zeku"}, n=100)
    assert pool == pool2 and len(pool) == 100
    assert counts["skipped_leak"] > 0
    names = [p["name"] for p in pool]
    assert len(set(names)) == 100
    for p in pool:
        assert p["id"] not in excluded and p["id"] % 10 not in (3, 7)
        assert not leaks(p["title"], p["renamed"])
        assert p["renamed"].startswith(p["name"]) and p["real"].startswith(p["title"])
        assert p["renamed"][len(p["name"]):] == p["real"][len(p["title"]):]  # fmt: skip
        assert "zeku" not in p["name"].lower().split()
    assert [p["j"] for p in pool] == list(range(100))


def test_extra_rows_leave_the_base_tables_neighbours_in_order() -> None:
    """T_0 plus rows: dropping the new rows from the extended top-k gives a prefix of T_0's
    top-k, so a model under `hold` sees T_0's neighbours except where an added row displaces."""
    g = torch.Generator().manual_seed(0)
    t0 = F.normalize(torch.randn(500, 32, generator=g), dim=-1)
    extra = F.normalize(torch.randn(20, 32, generator=g), dim=-1)
    q = F.normalize(torch.cat([torch.randn(30, 32, generator=g), extra[:5]]), dim=-1)
    base = exact_topk(q, t0, 8, pre=500)
    ext = exact_topk(q, torch.cat([t0, extra]), 8, pre=520)
    for b, e in zip(base.tolist(), ext.tolist(), strict=True):
        kept = [x for x in e if x < 500]
        assert kept == b[: len(kept)]
    assert ext[30:, 0].tolist() == [500, 501, 502, 503, 504]  # an added sentence finds itself


def test_table_swap_on_a_toy_l8b_pv() -> None:
    """The eval path of rig 054 at toy size: under `hold` each document retrieves its own added
    row and the loss moves; under `t0` it cannot; the covered mask is the same; the model's own
    table is back afterwards; a run without a memory gets the `t0` conditions only."""
    from tests.test_model_semantics import tiny

    torch.manual_seed(0)
    model = BigMoE(tiny("L8b-pv")).eval()
    mem = model.middle.mem
    torch.nn.init.normal_(mem.out.weight, std=0.5)  # W_v is zero at init: make the read count
    dev = torch.device("cpu")
    pipe = Pipeline(dev, model)
    n = pipe.table.shape[0]
    rng = np.random.default_rng(0)
    items, own = {}, {}
    for v in ("real", "renamed"):
        items[v] = []
        for j in range(6):
            ids = [2, *rng.integers(3, 97, 40).tolist()]
            split = np.array([0] * 10 + [1] * 30, dtype=np.int8)
            items[v].append({"id": j, "pool": "in_table", "cell": n + j, "ids": ids,
                             "split": split})  # fmt: skip
        # the query at the first boundary (stride 8) is tokens [1, 8): add exactly that vector
        own[v] = mem.encoder([x["ids"][1:8] for x in items[v]])
    before = pipe.table.clone()
    res = score_conditions(model, pipe, items, own, dev)
    assert set(res) == {"real_t0", "real_hold", "renamed_t0", "renamed_hold"}
    for v in ("real", "renamed"):
        t0, hold = res[f"{v}_t0"], res[f"{v}_hold"]
        assert t0["own_retrieved_share_of_covered"] == 0.0
        assert hold["own_retrieved_share_of_covered"] > 0.0
        assert t0["per_article"]["cnt_" + SPLIT] == hold["per_article"]["cnt_" + SPLIT]
        assert t0["covered_tokens"] == 6 * 30
        assert abs(t0["loss_" + SPLIT] - hold["loss_" + SPLIT]) > 1e-4
    assert mem.table.shape[0] == n and torch.equal(mem.table, before)
    plain = BigMoE(tiny("L5")).eval()
    res = score_conditions(plain, pipe, items, own, dev)
    assert set(res) == {"real_t0", "renamed_t0"}


def record(n: int, loss: dict[str, float], seed: int) -> dict:
    cnt = np.random.default_rng(99).integers(10, 30, n).astype(float)
    rng = np.random.default_rng(seed)
    conds = {}
    for k, v in loss.items():
        conds[k] = {
            "per_article": {"sum_" + SPLIT: (cnt * (v + 0.05 * rng.standard_normal(n))).tolist(),
                            "cnt_" + SPLIT: cnt.tolist()},  # fmt: skip
            "own_retrieved_share_of_covered": 0.9,
        }
    return {"conditions": conds}


def arm(real_t0: float, real_hold: float, ren_t0: float, ren_hold: float, seed: int) -> dict:
    return record(300, {"real_t0": real_t0, "real_hold": real_hold, "renamed_t0": ren_t0,
                        "renamed_hold": ren_hold}, seed)  # fmt: skip


def base(real: float, ren: float, seed: int) -> dict:
    return record(300, {"real_t0": real, "renamed_t0": ren}, seed)


def test_boot_is_signed_and_floored() -> None:
    cnt = np.full(200, 20.0)
    a, b = cnt * 3.0, cnt * 2.9
    r = boot([(1.0, a, cnt), (-1.0, b, cnt)], n_boot=200)
    assert np.isclose(r["value"], 0.1) and r["bootstrap_se"] < 1e-9
    assert np.isclose(r["floor"], 0.012)


def test_report_readings() -> None:
    small_b = [base(3.0, 3.3, 1), base(3.0, 3.3, 2)]
    med_b = [base(2.5, 3.1, 3)]  # medium knows more of the real facts: premise holds
    small = [arm(2.95, 2.90, 3.25, 3.15, 4), arm(2.95, 2.90, 3.25, 3.15, 5)]
    # supported: at medium the value is gone on real facts and kept on renamed ones
    med = [arm(2.48, 2.48, 3.05, 2.95, 6)]
    r = report({"small": small, "medium": med}, {"small": small_b, "medium": med_b}, n_boot=300)
    assert r["reading"].startswith("I44's explanation supported")
    assert np.isclose(r["sizes"]["medium"]["S_renamed"]["value"], 0.10, atol=0.02)
    assert np.isclose(r["sizes"]["small"]["S_real"]["value"], 0.05, atol=0.02)
    # not sufficient: nothing on renamed either
    med = [arm(2.48, 2.48, 3.05, 3.05, 7)]
    r = report({"small": small, "medium": med}, {"small": small_b, "medium": med_b}, n_boot=300)
    assert r["reading"].startswith("explanation not sufficient")
    # mixed: pays on renamed, but no more than on real
    med = [arm(2.58, 2.48, 3.05, 2.95, 8)]
    r = report({"small": small, "medium": med}, {"small": small_b, "medium": med_b}, n_boot=300)
    assert r["reading"] == "mixed"
    # premise fails: the medium base knows no more of the real facts than the small one
    r = report({"small": small, "medium": med}, {"small": small_b, "medium": [base(2.8, 3.1, 9)]},
               n_boot=300)  # fmt: skip
    assert r["reading"].startswith("silent")
