"""`python -m sim.run --scenario local_3060 --rung L5 --z 256 --tokens 20000 --seed 0`

Prints a JSON summary (`docs/02 §14` subset; see `sim/README.md`).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from sim.params import load_params
from sim.system import simulate


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="local_3060")
    ap.add_argument("--rung", default="L5")
    ap.add_argument("--z", type=int, default=256, help="sequences (= decode tokens) in flight")
    ap.add_argument("--tokens", type=int, default=20_000)
    ap.add_argument("--warmup", type=int, default=None, help="default: tokens // 10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-rule", choices=["floor", "roofline"], default="floor")
    ap.add_argument("--w-max", type=float, default=1e-3, help="expert queue timeout, seconds")
    ap.add_argument("--routing", choices=["uniform", "zipf"], default="uniform")
    ap.add_argument("--zipf-s", type=float, default=1.0)
    ap.add_argument("--context", type=int, default=None, help="keys visible per decode token")
    ap.add_argument("--series", action="store_true", help="keep the utilisation time series")
    a = ap.parse_args(argv)
    p = load_params(
        a.scenario,
        a.rung,
        context_tokens=a.context,
        z=a.z,
        tokens=a.tokens,
        warmup_tokens=a.tokens // 10 if a.warmup is None else a.warmup,
        seed=a.seed,
        batch_rule=a.batch_rule,
        w_max_s=a.w_max,
        routing=a.routing,
        zipf_s=a.zipf_s,
    )
    out: dict[str, Any] = simulate(p)
    if not a.series:
        for unit in out["utilisation"].values():
            unit.pop("flop_utilisation_series")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
