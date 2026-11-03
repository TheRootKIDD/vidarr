"""Unit tests of `sim/` milestone 1 (ADR-043): determinism, Little's law, fabric bytes,
liveness, the compute bound, and the scenario source-line rule. Toy-sized; < 10 s on CPU."""

from __future__ import annotations

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
    p = small(n_e=1, k=1, n_units=1, r=1, z=2048, tokens=4096, warmup_tokens=1024)
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
def test_round_policy_reproduces_the_hand_timeline() -> None:
    """Two units, z = 4, k = 1, every token routed to expert 0 on unit 0. Owner 0 holds
    sequences 0 and 2, owner 1 holds 1 and 3. First iteration, by hand:
    both owners run attention on 2 tokens (B); owner 1 sends 2 vectors over its link (2 v/λ
    serialised, plus the latency floor ℓ); expert 0 fires once, on all 4 rows (C); unit 0 sends 2
    returns to owner 1 (2 v/λ + ℓ). Owner 0's iteration = B + 2v/λ + ℓ + C; owner 1's adds the
    return leg."""
    p = load_params(
        "local_3060", "L5", z=4, tokens=8, warmup_tokens=0, r=2, k=1, n_e=2,
        n_units=2, batch_policy="round", routing="empirical", route_probs=(1.0, 0.0),
    )
    sim = Simulation(p)
    sim.run()
    a_s, _ = sim.owner_step_s("skel_in", 2)
    b_s, _ = sim.owner_step_s("attn", 2)
    c_s, _ = sim.expert_step_s(4)
    leg = 2 * p.vector_bytes / p.lambda_link + p.link_latency_s
    first = sorted(x for _, x in sim.iter_lat[:2])
    assert first[0] == pytest.approx(b_s + leg + c_s, rel=1e-9)
    assert first[1] == pytest.approx(b_s + leg + c_s + leg, rel=1e-9)
    assert sim.fired == {"threshold": 0, "timeout": 0, "round": 2 * p.r}  # once per iteration
    assert sim.batch_sizes == [4] * (2 * p.r)
    assert sim.iter_lat[0][0] == pytest.approx(a_s + b_s + leg + c_s, rel=1e-9)


def test_round_policy_needs_whole_rounds_and_empirical_needs_probs() -> None:
    with pytest.raises(ValueError, match="multiples of z"):
        small(batch_policy="round", tokens=301)
    with pytest.raises(ValueError, match="route_probs"):
        small(routing="empirical")
