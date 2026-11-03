"""S0's comparison (`02 §14`, ADR-049 §5–6, ADR-050 §5): `sim/` under the round policy against a
measured `s0_ep_decode` result, at every z, with measured routing shares and with uniform routing.

Pass = throughput and per-iteration p50 within ±20 % at every z. Nothing is fitted: the scenario
is read as it stands (its `software` section included, if any), and `--no-software` reproduces the
comparison without the fixed per-job costs (ADR-049's original model). Beside the full model it
also reports the run without the owner efficiency table (ADR-051 §5: η = 1, ADR-050's model).

Run: `python -m scripts.bench.s0_compare --measured 034-s0-ep-decode --id 038-s0-rescore-eta`
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.bench._common import RESULTS_DIR, write_result
from sim.params import SOFTWARE_KINDS, load_params
from sim.system import simulate

TOL = 0.20


def compare(
    measured: dict[str, Any], scenario: str, software: bool, seed: int, eta: bool = True
) -> list[dict]:
    rows = []
    for r in measured["results"]:
        z = r["z"]
        keys = round(r["mean_keys_per_decode_token"])
        off = (
            {}
            if software
            else {f"t0_{k}_s": 0.0 for k in SOFTWARE_KINDS}
            | {
                "compose": "sum",
                "t_coll_s": None,
            }
        )
        if not (software and eta):
            off["owner_eff"] = ()
        row: dict[str, Any] = {
            "z": z,
            "context_tokens": keys,
            "measured_tok_s": r["throughput_tok_s"],
            "measured_iter_p50_s": r["iteration_s"]["p50"],
            "measured_token_p50_s": r["token_s"]["p50"],
        }
        for name, kw in (
            ("empirical", {"routing": "empirical", "route_probs": tuple(r["routing_share"])}),
            ("uniform", {"routing": "uniform"}),
        ):
            p = load_params(
                scenario,
                "L5",
                context_tokens=keys,
                z=z,
                tokens=z * 200,
                warmup_tokens=z * 40,
                batch_policy="round",
                seed=seed,
                **kw,
                **off,
            )
            s = simulate(p)
            row[name] = {
                "tok_s": s["throughput_tok_s"],
                "iter_p50_s": s["iteration_latency_s"]["p50"],
                "token_p50_s": s["token_latency_s"]["p50"],
                "tok_ratio": s["throughput_tok_s"] / r["throughput_tok_s"],
                "iter_ratio": s["iteration_latency_s"]["p50"] / r["iteration_s"]["p50"],
            }
        e = row["empirical"]
        row["pass"] = abs(e["tok_ratio"] - 1) <= TOL and abs(e["iter_ratio"] - 1) <= TOL
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--measured", required=True, help="an s0_ep_decode result id")
    ap.add_argument("--id", required=True)
    ap.add_argument("--scenario", default="local_3060")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    measured = json.loads((RESULTS_DIR / args.measured / "result.json").read_text())
    scen = Path(__file__).resolve().parents[2] / "sim" / "scenarios" / f"{args.scenario}.yaml"
    with_sw = compare(measured, args.scenario, True, args.seed)
    without = compare(measured, args.scenario, False, args.seed)
    no_eta = compare(measured, args.scenario, True, args.seed, eta=False)
    for row in with_sw:
        e = row["empirical"]
        print(
            f"z {row['z']:5d}: meas {row['measured_tok_s']:9.1f} tok/s, sim x{e['tok_ratio']:.3f}; "
            f"iter p50 meas {row['measured_iter_p50_s'] * 1e3:.3f} ms, sim x{e['iter_ratio']:.3f}; "
            f"{'PASS' if row['pass'] else 'miss'}"
        )
    payload = {
        "bench": "s0_compare",
        "adr": "ADR-050, ADR-051",
        "measured": args.measured,
        "measured_git_hash": measured["provenance"]["git_hash"],
        "scenario": args.scenario,
        "scenario_sha256": hashlib.sha256(scen.read_bytes()).hexdigest(),
        "tolerance": TOL,
        "with_software": with_sw,
        "without_owner_efficiency": no_eta,
        "without_software": without,
        "passed": all(r["pass"] for r in with_sw),
    }
    print("S0", "PASSED" if payload["passed"] else "NOT PASSED")
    if not args.dry:
        print(write_result(args.id, payload))


if __name__ == "__main__":
    main()
