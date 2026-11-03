"""Calibration point (b) (ADR-052, I1): `sim/` against SGLang's EP72 decode of DeepSeek-V3.

The scored number is the post's no-TBO decode throughput, 22,310 / 1.255 = 17,777 tok/s per
node, at 256 sequences per GPU on 9 x 8 H100 (scenario `h100_sglang`). Pass = within ±20 %
at the stated inputs AND across the §7 sweep, one unpublished input at a time; a flip inside
the sweep is "inconclusive", a miss across all of it is a miss. Nothing is fitted.

Reported beside it, not scored: the per-layer split from the step functions, and the sim at
DeepSeek's `profile-data` configuration (EP128, 128 per GPU, 4096 keys; 18,598 tok/s per node
measured *with* overlap, so the no-overlap sim should sit below it).

CPU only: `python -m scripts.bench.i1_sglang --id 039-i1-sglang`
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

from scripts.bench._common import write_result
from sim.params import load_params, load_scenario
from sim.system import Simulation

ROOT = Path(__file__).resolve().parents[2]
TOL = 0.20
GPUS_PER_NODE = 8
PCIE = 1513 / 1979  # H100 PCIe vs SXM dense tensor peak (datasheet), FP8 and BF16 alike

# (name, ADR-052 §7 input, overrides relative to the stated run)
SWEEP: list[tuple[str, str, dict[str, Any]]] = [
    ("context_2000", "keys = the post's KV length 2,000 (stated 2,050)", {"context_tokens": 2000}),
    ("nic_200g", "InfiniBand 200 Gb/s per GPU (placeholder 400; I39)", {"lambda_link": 20e9}),
    ("nvlink_intra", "rows inside a node over NVLink, not RDMA (I39)", {"intra_via_nic": False}),
    (
        "routing_zipf05",
        "residual imbalance: zipf s = 0.5 over the slots (I39)",
        {"routing": "zipf", "zipf_s": 0.5},
    ),
    (
        "attn_peak",
        "attention core at nominal BF16 / HBM peak, not FlashMLA's (I39)",
        {"phi_attn": 989.4e12, "beta_attn": 3.35e12},
    ),
    (
        "owner_m4096",
        "owner GEMMs at DeepGEMM's M = 4096 rate (stated M = 128)",
        {"phi_owner": 1220e12},
    ),
    (
        "expert_x114",
        "expert GEMMs x 1.14 (DeepGEMM's 2025-04 speed-up)",
        {"grouped_eff_scale": 1.14},
    ),
    ("pcie", "H100 PCIe instead of SXM (I39): rates x 0.765, HBM 2.0 TB/s", {"pcie": True}),
    ("seed_1", "routing seed 1 (noise, not an input)", {"seed": 1}),
]


def params(scenario: str, batch: int, n_units: int, slots: int, keys: int, **kw: Any) -> Any:
    """The round-policy run of one configuration; `pcie` and `grouped_eff_scale` are sweep arms
    expressed on the scenario's own values."""
    kw = dict(kw)
    chip = load_scenario(scenario)["chip"]
    if kw.pop("pcie", False):
        kw |= {
            "phi": float(chip["phi_flops"]) * PCIE,
            "phi_owner": float(chip["phi_owner_flops"]) * PCIE,
            "phi_attn": float(chip["phi_attn_flops"]) * PCIE,
            "beta_c": 2.0e12,
            "beta_attn": float(chip["beta_attn_bytes_per_s"]) * 2.0e12 / 3.35e12,
        }
    scale = kw.pop("grouped_eff_scale", None)
    if scale is not None:
        table = chip["grouped_gemm"]["efficiency_vs_dense_by_tokens_per_expert"]
        kw["grouped_eff"] = tuple(sorted((int(b), float(e) * scale) for b, e in table.items()))
    z = batch * n_units
    return load_params(
        scenario,
        "deepseek_v3",
        context_tokens=kw.pop("context_tokens", keys),
        n_e=slots,
        n_units=n_units,
        z=z,
        tokens=3 * z,
        warmup_tokens=z,
        batch_policy="round",
        route_draw="bulk",
        **kw,
    )


def run(p: Any) -> dict[str, Any]:
    t = time.time()
    s = Simulation(p)
    out = s.run()
    nodes = p.n_units / GPUS_PER_NODE
    return {
        "tok_s_per_node": out["throughput_tok_s"] / nodes,
        "round_p50_s": out["iteration_latency_s"]["p50"],
        "step_p50_s": out["token_latency_s"]["p50"],
        "mean_rows_per_slot": out["batches"]["mean_size"],
        "wire_bytes_per_token_iteration": out["fabric_bytes_per_token_iteration"]["wire"],
        "wall_s": time.time() - t,
    }


def split(p: Any, batch: int) -> dict[str, float]:
    """One round on one GPU from the step functions, uniform load (reported, not scored)."""
    s = Simulation(p)
    slots_per_gpu = p.n_e // p.n_units
    rows_per_slot = batch * p.n_units * p.k // p.n_e
    off = 1.0 - slots_per_gpu / p.n_e
    sent = batch * p.k * off
    return {
        "attn_s": s.owner_step_s("attn", batch)[0],
        "dispatch_s": 2 * p.collective_s + sent * p.dispatch_bytes / p.lambda_link,
        "experts_s": slots_per_gpu * s.expert_step_s(rows_per_slot)[0],
        "combine_s": p.collective_s + sent * p.return_bytes / p.lambda_link,
        "post_s": s.owner_step_s("post", batch)[0],
        "skel_in_per_step_s": s.owner_step_s("skel_in", batch)[0],
        "skel_out_per_step_s": s.owner_step_s("skel_out", batch)[0],
        "rows_per_slot": rows_per_slot,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", required=True)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    doc = load_scenario("h100_sglang")
    dep = doc["deployment"]
    target = float(dep["scored_tok_s_per_node"])
    batch, slots = int(dep["batch_per_gpu"]), int(dep["slots"])
    keys = int(dep["input_tokens"]) + int(dep["output_tokens"]) // 2  # P + N/2, as S0
    n_units = int(doc["unit"]["n_units"])

    stated_p = params("h100_sglang", batch, n_units, slots, keys)
    stated = run(stated_p) | {"ratio": 0.0}
    stated["ratio"] = stated["tok_s_per_node"] / target
    print(f"stated: {stated['tok_s_per_node']:.0f} tok/s/node, x{stated['ratio']:.3f}")
    sweep = []
    for name, what, kw in SWEEP:
        r = run(params("h100_sglang", batch, n_units, slots, keys, **kw))
        r |= {"name": name, "input": what, "ratio": r["tok_s_per_node"] / target}
        r["within"] = abs(r["ratio"] - 1) <= TOL
        sweep.append(r)
        print(f"{name:15s} {r['tok_s_per_node']:8.0f} x{r['ratio']:.3f} {r['wall_s']:.0f}s")
    within = abs(stated["ratio"] - 1) <= TOL
    inputs = [r for r in sweep if r["name"] != "seed_1"]
    if within and all(r["within"] for r in sweep):
        verdict = "pass"
    elif not within and not any(r["within"] for r in sweep):
        verdict = "miss"
    else:
        verdict = "inconclusive"
    flips = [r["name"] for r in inputs if r["within"] != within]
    # reported only: DeepSeek's profile-data configuration (EP128 on 16 nodes, 128 per GPU,
    # 4096 keys, 256 slots, balanced routing); measured 18,598 tok/s per node with overlap
    prof = run(params("h100_sglang", 128, 128, 256, 4096))
    payload = {
        "bench": "i1_sglang",
        "adr": "ADR-052",
        "scenario": "h100_sglang",
        "scenario_sha256": hashlib.sha256(
            (ROOT / "sim" / "scenarios" / "h100_sglang.yaml").read_bytes()
        ).hexdigest(),
        "target_tok_s_per_node": target,
        "tolerance": TOL,
        "keys": keys,
        "stated": stated,
        "split_stated": split(stated_p, batch),
        "sweep": sweep,
        "verdict": verdict,
        "flipping_inputs": flips,
        "reported_profile_data": prof | {"measured_with_overlap_tok_s_per_node": 18598},
        "reported_deepseek_production_tok_s_per_node": 14800,
    }
    print("I1 point (b):", verdict, flips)
    if not args.dry:
        print(write_result(args.id, payload))


if __name__ == "__main__":
    main()
