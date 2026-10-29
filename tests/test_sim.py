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
