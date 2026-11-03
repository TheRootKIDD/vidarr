"""Unit tests of `sim/` milestone 1 (ADR-043): determinism, Little's law, fabric bytes,
liveness, the compute bound, and the scenario source-line rule. Toy-sized; < 10 s on CPU."""

from __future__ import annotations

import dataclasses as dc

import numpy as np
import pytest

from sim.engine import Engine
from sim.params import check_sources, load_params, load_scenario
from sim.system import Simulation, simulate


def small(**kw):
    base = {"z": 16, "tokens": 300, "warmup_tokens": 50, "r": 2, "seed": 0}
    return load_params("local_3060", "L5", **{**base, **kw})


# ---- engine ----------------------------------------------------------------------------------
def test_engine_orders_ties_by_scheduling_order_and_detects_deadlock() -> None:
    eng, seen = Engine(), []
    eng.at(1.0, lambda: seen.append("a"))
    eng.at(1.0, lambda: seen.append("b"))
    eng.at(0.5, lambda: seen.append("c"))
    eng.run(lambda: len(seen) == 3)
    assert seen == ["c", "a", "b"]
    with pytest.raises(RuntimeError, match="deadlock"):
        eng.run(lambda: False)


# ---- determinism -----------------------------------------------------------------------------
def test_same_seed_same_run_different_seed_different_run() -> None:
    a, b, c = simulate(small()), simulate(small()), simulate(small(seed=1))
    assert a == b
    assert a != c


# ---- Little's law (docs/02 §5.5) -------------------------------------------------------------
def test_littles_law_holds_in_the_window() -> None:
    out = simulate(small(z=32, tokens=3000, warmup_tokens=500))
    z, xl = out["littles_law"]["z"], out["littles_law"]["x_times_l"]
    assert abs(xl - z) / z < 0.05
    assert out["tokens_in_flight_mean"] == pytest.approx(32, rel=1e-9)


# ---- fabric (docs/01 §10, docs/02 §3) --------------------------------------------------------
def test_fabric_bytes_per_token_iteration_are_2kd_bact_without_multicast() -> None:
    out = simulate(small())
    f = out["fabric_bytes_per_token_iteration"]
    p = small()
    assert f["formula_2kd_bact"] == 2 * p.k * p.d * p.b_act
    assert f["logical"] == pytest.approx(f["formula_2kd_bact"], rel=1e-12)
    assert 0 < f["wire"] < f["logical"]  # some experts sit on the owner's own unit
    one_unit = simulate(small(n_units=1))["fabric_bytes_per_token_iteration"]
    assert one_unit["wire"] == 0.0


def test_multicast_and_ragged_depth_are_refused_in_m1() -> None:
    with pytest.raises(NotImplementedError):
        small(multicast=True)


# ---- liveness and the w_max timeout (docs/02 §5.1, §5.5) -------------------------------------
def test_low_load_completes_through_timeouts() -> None:
    p = small(z=1, tokens=20, warmup_tokens=0)
    out = simulate(p)
    assert out["batches"]["fired_by_threshold"] == 0
    assert out["batches"]["fired_by_timeout"] == out["batches"]["count"] > 0
    assert out["token_latency_s"]["p50"] >= p.r * p.w_max_s  # every iteration waits w_max


def test_every_admitted_token_completes_and_nothing_is_left_queued() -> None:
    sim = Simulation(small(z=24, tokens=400))
    sim.run()  # raises if any unit or queue still holds work
    assert sim.started == sim.completed == 400


# ---- the compute bound (docs/02 §2) ----------------------------------------------------------
def test_saturated_single_expert_runs_compute_bound_batches() -> None:
    # the roofline alone (ADR-043 decision 1): the fixed cost per job is switched off here
    no_t0 = {f"t0_{kd}_s": 0.0 for kd in ("skel_in", "attn", "post", "skel_out", "expert")}
    p = small(n_e=1, k=1, n_units=1, r=1, z=2048, tokens=4096, warmup_tokens=1024, **no_t0)
    sim = Simulation(p)
    out = sim.run()
    assert max(sim.batch_sizes) == p.batch_threshold  # bursts fill the queue to the threshold
    dt, flops = sim.expert_step_s(p.batch_threshold)
    assert 0.95 <= flops / (p.phi * dt) <= 1.05  # the 003-gemm table tops out at 1.02
    assert out["batches"]["fired_by_threshold"] > 0


def test_batch_threshold_rule_and_grouped_efficiency_table() -> None:
    p = small()
    assert p.batch_threshold == 512  # 003-gemm occupancy floor
    assert small(batch_rule="roofline").batch_threshold == 80  # ceil(27.1e12 / 342e9 = 79.2)
    sim = Simulation(p)
    assert sim.grouped_efficiency(64) == pytest.approx(0.67)
    assert 0.67 < sim.grouped_efficiency(96) < 0.98
    assert sim.grouped_efficiency(10_000) == pytest.approx(1.02)


# ---- scenario files --------------------------------------------------------------------------
def test_every_value_in_local_3060_has_a_source_line() -> None:
    load_scenario("local_3060")  # raises if not


def test_a_value_without_a_source_line_is_rejected() -> None:
    check_sources({"chip": {"phi": 1.0, "phi_source": "experiments/003-gemm"}})
    check_sources({"fabric": {"source": "x", "inner": {"lambda": 3.59e9}}})  # inherited
    with pytest.raises(ValueError, match="chip.phi"):
        check_sources({"chip": {"phi": 1.0}})
    with pytest.raises(ValueError, match="per_card"):
        check_sources({"chip": {"per_card": [1.0, 2.0]}})


# ---- round policy (decision 11, ADR-049 §5) --------------------------------------------------
T0 = {
    "t0_skel_in_s": 3e-3,
    "t0_attn_s": 1e-3,
    "t0_post_s": 2e-4,
    "t0_skel_out_s": 4e-3,
    "t0_expert_s": 5e-4,
    "t_coll_s": 7e-5,
}


def _round_params(**kw):
    return load_params(
        "local_3060",
        "L5",
        z=4,
        tokens=8,
        warmup_tokens=0,
        r=2,
        k=1,
        n_e=2,
        n_units=2,
        batch_policy="round",
        routing="empirical",
        route_probs=(1.0, 0.0),
        **kw,
    )


@pytest.mark.parametrize("compose", ["max", "sum"])
def test_round_policy_reproduces_the_hand_timeline(compose: str) -> None:
    """Two units, z = 4, k = 1, every token routed to expert 0 on unit 0. Owner 0 holds
    sequences 0 and 2, owner 1 holds 1 and 3. First iteration, by hand (decisions 11-12):
    both owners run attention on 2 tokens (B); the dispatch collective takes 2 t_coll plus owner 1's
    2 vectors at λ (D); expert 0 runs once on all 4 rows (C); the return collective takes t_coll
    plus unit 0's 2 vectors to owner 1 (R); both owners run `post` on 2 tokens (P). Each job is
    compose(t0, roofline), with the rooflines taken from a t0 = 0 copy."""
    p = _round_params(compose=compose, **T0)
    zero = {f"t0_{kd}_s": 0.0 for kd in ("skel_in", "attn", "post", "skel_out", "expert")}
    bare = Simulation(_round_params(compose="sum", **zero))

    def job(kind: str, n: int) -> float:
        roof = bare.expert_step_s(n)[0] if kind == "expert" else bare.owner_step_s(kind, n)[0]
        t0 = T0[f"t0_{kind}_s"]
        return max(t0, roof) if compose == "max" else t0 + roof

    sim = Simulation(p)
    sim.run()
    v = 2 * p.dispatch_bytes / p.lambda_link
    a_s, b_s, c_s, p_s = job("skel_in", 2), job("attn", 2), job("expert", 4), job("post", 2)
    d_s, r_s = 2 * T0["t_coll_s"] + v, T0["t_coll_s"] + v
    it = b_s + d_s + c_s + r_s + p_s
    first = [x for _, x in sim.iter_lat[:2]]
    assert first == pytest.approx([it, it], rel=1e-9)
    assert sim.iter_lat[0][0] == pytest.approx(a_s + it, rel=1e-9)
    assert sim.fired == {"threshold": 0, "timeout": 0, "round": 2 * p.r}  # once per iteration
    assert sim.batch_sizes == [4] * (2 * p.r)
    if compose == "max":
        assert b_s == T0["t0_attn_s"] and c_s == T0["t0_expert_s"]  # t0 dominates at 2-4 rows


def test_software_section_is_read_and_absent_means_none() -> None:
    from sim.params import SimParams, _software_fields

    assert _software_fields(None) == {}
    defaults = {f.name: f.default for f in dc.fields(SimParams)}
    assert defaults["t0_expert_s"] == 0.0 and defaults["t_coll_s"] is None  # not modelled
    p = small()  # local_3060 carries the section (035, ADR-050 §4)
    assert p.t0("attn") > 0 and p.collective_s > 0 and p.compose in ("max", "sum")
    doc = load_scenario("local_3060")["software"]
    assert doc["source"].startswith("experiments/037")
    assert dict(p.owner_eff)["attn"][-1] == (256, pytest.approx(0.62, abs=0.01))


def test_round_policy_needs_whole_rounds_and_empirical_needs_probs() -> None:
    with pytest.raises(ValueError, match="multiples of z"):
        small(batch_policy="round", tokens=301)
    with pytest.raises(ValueError, match="route_probs"):
        small(routing="empirical")


def test_owner_efficiency_divides_the_owner_roofline() -> None:
    """Decision 13 (ADR-051): η = 1 (or no table) is ADR-050's owner time exactly; a table
    point is honoured, read in log2(n) between points and clamped at the ends."""
    zero = {f"t0_{kd}_s": 0.0 for kd in ("skel_in", "attn", "post", "skel_out", "expert")}
    bare = Simulation(small(owner_eff=(), **zero))
    ones = Simulation(small(owner_eff=(("attn", ((1, 1.0), (256, 1.0))),), **zero))
    for n in (1, 7, 64, 300):
        assert ones.owner_step_s("attn", n) == bare.owner_step_s("attn", n)
    table = (("attn", ((4, 1.0), (64, 0.5))), ("skel_in", ((1, 0.8),)))
    half = Simulation(small(owner_eff=table, **zero))
    assert half.owner_step_s("attn", 64)[0] == pytest.approx(2 * bare.owner_step_s("attn", 64)[0])
    assert half.owner_step_s("attn", 1000)[0] == pytest.approx(
        2 * bare.owner_step_s("attn", 1000)[0]
    )
    assert half.owner_step_s("attn", 2) == bare.owner_step_s("attn", 2)  # clamped at 1.0
    assert half.p.owner_efficiency("attn", 16) == pytest.approx(0.75)  # halfway in log2
    assert half.owner_step_s("skel_in", 9)[0] == pytest.approx(
        bare.owner_step_s("skel_in", 9)[0] / 0.8
    )
    assert half.owner_step_s("post", 9) == bare.owner_step_s("post", 9)  # no table: η = 1
    assert half.expert_step_s(64) == bare.expert_step_s(64)  # experts keep decision 1's table
    with_t0 = Simulation(small(owner_eff=table, compose="sum", t0_attn_s=1e-3))
    assert with_t0.owner_step_s("attn", 64)[0] == pytest.approx(
        1e-3 + 2 * bare.owner_step_s("attn", 64)[0]
    )
    with pytest.raises(ValueError, match="owner kind"):
        small(owner_eff=(("expert", ((1, 0.5),)),))
    with pytest.raises(ValueError, match="0 < eta"):
        small(owner_eff=(("attn", ((1, 1.2),)),))


# ---- layered round mode and two-tier fabric (decision 14, ADR-052) ----------------------------
@pytest.mark.parametrize("via_nic", [False, True])
def test_two_tier_round_reproduces_the_hand_timeline(via_nic: bool) -> None:
    """Four units in two nodes of two, z = 4, k = 1, every token routed to expert 2 (unit 2).
    Dispatch: units 0 and 1 send one row across nodes, unit 3 one row inside node 1, unit 2
    keeps its own. Return: unit 2 sends two rows across and one inside. Each leg costs the
    busiest sender's time, max(intra / lambda_intra, inter / lambda_link), or everything over
    the NIC when `intra_via_nic`."""
    lam, lam_in = 1e9, 4e9
    kw = {
        "z": 4,
        "tokens": 8,
        "warmup_tokens": 0,
        "r": 2,
        "k": 1,
        "n_e": 4,
        "n_units": 4,
        "batch_policy": "round",
        "routing": "empirical",
        "route_probs": (0.0, 0.0, 1.0, 0.0),
        "node_size": 2,
        "lambda_link": lam,
        "lambda_intra": lam_in,
        "intra_via_nic": via_nic,
        "compose": "sum",
        **T0,
    }
    p = load_params("local_3060", "L5", **kw)
    zero = {f"t0_{kd}_s": 0.0 for kd in ("skel_in", "attn", "post", "skel_out", "expert")}
    bare = Simulation(load_params("local_3060", "L5", **{**kw, **zero}))

    def job(kind: str, n: int) -> float:
        roof = bare.expert_step_s(n)[0] if kind == "expert" else bare.owner_step_s(kind, n)[0]
        return T0[f"t0_{kind}_s"] + roof

    sim = Simulation(p)
    sim.run()
    out, back = p.dispatch_bytes, p.return_bytes
    if via_nic:
        d_wire, r_wire = out / lam, 3 * back / lam
    else:
        d_wire, r_wire = max(out / lam, out / lam_in), max(2 * back / lam, back / lam_in)
    it = (
        job("attn", 1)
        + 2 * T0["t_coll_s"]
        + d_wire
        + job("expert", 4)
        + T0["t_coll_s"]
        + r_wire
        + job("post", 1)
    )
    assert [x for _, x in sim.iter_lat[:4]] == pytest.approx([it] * 4, rel=1e-9)
    assert sim.batch_sizes == [4] * (2 * p.r)


def test_two_tier_fabric_is_validated() -> None:
    with pytest.raises(NotImplementedError, match="round policy"):
        small(node_size=2, lambda_intra=1e9)
    with pytest.raises(ValueError, match="multiple of node_size"):
        small(node_size=3, lambda_intra=1e9, batch_policy="round", tokens=304, warmup_tokens=48)
    with pytest.raises(ValueError, match="lambda_intra"):
        small(node_size=2, batch_policy="round", tokens=304, warmup_tokens=48)
    with pytest.raises(TypeError, match="unknown SimParams fields"):
        small(no_such_field=1)


def test_bulk_route_draw_has_the_law_of_sequential_sampling() -> None:
    """Gumbel-top-k against numpy's sequential choice without replacement: k distinct experts
    per row, and the same per-expert inclusion rates under skewed weights."""
    p = small(batch_policy="round", tokens=304, warmup_tokens=48, routing="zipf", n_e=8, k=3)
    bulk = Simulation(dc.replace(p, route_draw="bulk"))
    rows = bulk._bulk_routes(40_000)
    assert all(len(set(r)) == 3 for r in rows[:1000])
    seq = Simulation(p)
    ref = np.array([seq.rng.choice(8, 3, replace=False, p=seq.route_p) for _ in range(40_000)])
    f_bulk = np.bincount(rows.ravel(), minlength=8) / rows.size
    f_seq = np.bincount(ref.ravel(), minlength=8) / ref.size
    assert f_bulk == pytest.approx(f_seq, abs=0.01)
    with pytest.raises(ValueError, match="round policy"):
        small(route_draw="bulk")


def test_deepseek_v3_shape_and_scenario() -> None:
    """ADR-052 §3: 671 B total / ≈ 37 B active; MLA's 576-value cache; the h100_sglang
    scenario loads with every value sourced, and the slot count overrides n_e."""
    from costmodel import deepseek_v3 as ds

    m = ds.DeepSeekV3()
    assert ds.total_params(m) == pytest.approx(671e9, rel=0.005)
    assert ds.active_params(m) == pytest.approx(37e9, rel=0.03)
    assert m.kv_entry == 576 and m.moe_layers == 58
    assert ds.attention_score_flops_per_key(m) == 2 * 128 * (576 + 512)
    p = load_params(
        "h100_sglang",
        "deepseek_v3",
        context_tokens=2050,
        n_e=288,
        z=72 * 4,
        tokens=72 * 12,
        warmup_tokens=72 * 4,
        batch_policy="round",
        route_draw="bulk",
    )
    assert (p.n_units, p.node_size, p.intra_via_nic, p.r, p.k) == (72, 8, True, 58, 8)
    assert p.dispatch_bytes == pytest.approx(7168 * (1 + 4 / 128))  # DeepEP LL: 7392 B per row
    sim = Simulation(p)
    assert sim.kv_bytes_per_layer == 2050 * 576 * 2
    out = sim.run()
    assert out["batches"]["fired_by_round"] > 0 and out["throughput_tok_s"] > 0
    with pytest.raises(ValueError, match="context_tokens"):
        load_params("h100_sglang", "deepseek_v3")
