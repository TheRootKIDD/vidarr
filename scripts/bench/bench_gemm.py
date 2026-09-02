"""BF16 GEMM throughput over the expert-step shape grid (`docs/06 §6`).

Measures the shapes `docs/02 §2` actually executes: an expert step is
$(b \\times d) @ (d \\times d_{ff}/U)$ per chip. Feeds three scenario inputs:

* $\\phi$ — peak achieved BF16 FLOP/s per card, replacing the nominal 25.5 TFLOPS
* $s_{min}$ — the tile floor: smallest $d_{ff}$ slice reaching 80 % of $\\phi$
* the $b_{min}$ *curve* — smallest batch reaching 80 % of $\\phi$, which is an
  occupancy floor and is **not** the roofline $b_{min} = (\\phi/\\beta_C)(b_w/2)$
  of `docs/02 §2`; both are reported so the scenario can carry the larger.

Runs the grid on every card separately: three of the four are GA104 and one is
GA106 (open question I11), so a single card's number must not be multiplied by
four without checking.

Run: `python -m scripts.bench.bench_gemm --id 003-gemm`
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from scripts.bench._common import write_result

# Expert-step shapes. `d` spans the `small` and `medium` anchors of `docs/06 §3`;
# `d_ff` spans the tile floor (512 = U*s_min) through the widths costmodel/
# derives (833, 1665) and the dense baselines (2048, 2816).
D_MODEL = [768, 1024]
D_FF = [128, 256, 512, 833, 1024, 1665, 2048, 2816, 4096]
BATCH = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _time_gemm(a: torch.Tensor, b: torch.Tensor, warmup: int, iters: int) -> float:
    """Median seconds per GEMM, timed with CUDA events."""
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        a @ b
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) / 1e3 for s, e in zip(starts, ends, strict=True))
    return times[len(times) // 2]


def sweep_card(device: int, dtype: torch.dtype) -> list[dict[str, Any]]:
    """The full (b, d, d_ff) grid on one card."""
    torch.cuda.set_device(device)
    rows: list[dict[str, Any]] = []
    for d in D_MODEL:
        for d_ff in D_FF:
            for b in BATCH:
                if b * d + d * d_ff + b * d_ff > 3e8:  # keep well inside 12 GB
                    continue
                x = torch.randn(b, d, device=f"cuda:{device}", dtype=dtype)
                w = torch.randn(d, d_ff, device=f"cuda:{device}", dtype=dtype)
                iters = 50 if b * d * d_ff < 1e9 else 10
                secs = _time_gemm(x, w, warmup=5, iters=iters)
                rows.append(
                    {
                        "device": device,
                        "b": b,
                        "d": d,
                        "d_ff": d_ff,
                        "seconds": secs,
                        "tflops": (2.0 * b * d * d_ff) / secs / 1e12,
                    }
                )
                del x, w
        torch.cuda.empty_cache()
    return rows


def grouped_gemm(device: int, dtype: torch.dtype, d: int, d_ff: int, total_b: int) -> list[dict]:
    """MoE dispatch efficiency: one big GEMM vs `n` expert-sized GEMMs.

    `docs/06 §6` wants "grouped-GEMM efficiency vs experts-per-batch". The
    baseline is a single $(total\\_b \\times d) @ (d \\times d_{ff})$; the grouped
    case splits the batch over `n` experts with distinct weights, which is what
    the MoE layer actually issues. The ratio is the MoE MFU assumption.
    """
    torch.cuda.set_device(device)
    dev = f"cuda:{device}"
    x = torch.randn(total_b, d, device=dev, dtype=dtype)
    w = torch.randn(d, d_ff, device=dev, dtype=dtype)
    dense_s = _time_gemm(x, w, 5, 20)
    dense_tflops = (2.0 * total_b * d * d_ff) / dense_s / 1e12
    del x, w
    torch.cuda.empty_cache()

    rows = []
    for n in (2, 4, 8, 16, 32, 64, 128):
        per = total_b // n
        if per < 1:
            continue
        xs = torch.randn(n, per, d, device=dev, dtype=dtype)
        ws = torch.randn(n, d, d_ff, device=dev, dtype=dtype)
        for _ in range(5):
            torch.bmm(xs, ws)
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(20):
            torch.bmm(xs, ws)
        ev1.record()
        torch.cuda.synchronize()
        secs = ev0.elapsed_time(ev1) / 1e3 / 20
        tflops = (2.0 * n * per * d * d_ff) / secs / 1e12
        rows.append(
            {
                "device": device,
                "n_experts": n,
                "tokens_per_expert": per,
                "d": d,
                "d_ff": d_ff,
                "tflops": tflops,
                "dense_tflops": dense_tflops,
                "efficiency_vs_dense": tflops / dense_tflops,
            }
        )
        del xs, ws
        torch.cuda.empty_cache()
    return rows


def derive(rows: list[dict[str, Any]], device: int, frac: float = 0.8) -> dict[str, Any]:
    """phi, s_min and the empirical batch floor for one card."""
    mine = [r for r in rows if r["device"] == device]
    phi = max(r["tflops"] for r in mine)
    target = frac * phi

    big_b = [r for r in mine if r["b"] >= 2048]
    s_min = min((r["d_ff"] for r in mine if r["tflops"] >= target), default=None)
    s_min_at_large_b = min((r["d_ff"] for r in big_b if r["tflops"] >= target), default=None)
    big_ff = [r for r in mine if r["d_ff"] >= 2048]
    b_floor = min((r["b"] for r in big_ff if r["tflops"] >= target), default=None)
    return {
        "device": device,
        "phi_tflops": phi,
        "threshold_tflops": target,
        "s_min": s_min_at_large_b if s_min_at_large_b is not None else s_min,
        "b_floor_empirical": b_floor,
    }


def alignment_probe(device: int, dtype: torch.dtype, d: int = 768, b: int = 8192) -> list[dict]:
    """Is the trough at d_ff = 833 / 1665 tensor-core misalignment?

    The widths `costmodel.solve` derives from a FLOPs budget are arbitrary
    integers, and 833 and 1665 are odd. BF16 tensor-core GEMMs want the reduction
    and output dimensions on a multiple of 8 (in practice 64+). This walks each
    derived width against its neighbours to separate alignment from noise.
    """
    torch.cuda.set_device(device)
    dev = f"cuda:{device}"
    widths = [832, 833, 834, 840, 896, 1660, 1664, 1665, 1666, 1672, 1728]
    rows = []
    for d_ff in widths:
        x = torch.randn(b, d, device=dev, dtype=dtype)
        w = torch.randn(d, d_ff, device=dev, dtype=dtype)
        secs = _time_gemm(x, w, 5, 30)
        rows.append(
            {
                "device": device,
                "b": b,
                "d": d,
                "d_ff": d_ff,
                "tflops": (2.0 * b * d * d_ff) / secs / 1e12,
                "multiple_of_8": d_ff % 8 == 0,
                "multiple_of_64": d_ff % 64 == 0,
            }
        )
        del x, w
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="003-gemm")
    ap.add_argument("--devices", type=int, default=None, help="how many cards (default: all)")
    ap.add_argument(
        "--alignment-probe",
        action="store_true",
        help="only walk d_ff around the costmodel-derived widths, on card 0",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA devices visible")
    n_dev = args.devices or torch.cuda.device_count()
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.alignment_probe:
        rows = alignment_probe(0, torch.bfloat16)
        path = write_result(args.id, {"bench": "gemm_alignment", "results": rows})
        print(f"wrote {path}\n")
        for r in rows:
            if r["multiple_of_64"]:
                flag = "aligned"
            else:
                flag = "mult8" if r["multiple_of_8"] else "ODD/unaligned"
            print(f"  d_ff={r['d_ff']:5d}  {r['tflops']:6.2f} TFLOPS   {flag}")
        return

    rows: list[dict[str, Any]] = []
    grouped: list[dict[str, Any]] = []
    for dev in range(n_dev):
        rows += sweep_card(dev, torch.bfloat16)
        grouped += grouped_gemm(dev, torch.bfloat16, d=768, d_ff=1665, total_b=8192)

    derived = [derive(rows, dev) for dev in range(n_dev)]
    payload = {
        "bench": "gemm",
        "dtype": "bfloat16",
        "grid": {"d": D_MODEL, "d_ff": D_FF, "b": BATCH},
        "results": rows,
        "grouped": grouped,
        "derived": derived,
        "note": (
            "b_floor_empirical is an occupancy/tile floor, not the roofline "
            "b_min = (phi/beta_C)(b_w/2) of docs/02 §2; the scenario carries both."
        ),
    }
    path = write_result(args.id, payload)
    print(f"wrote {path}\n")
    for d in derived:
        print(
            f"  gpu{d['device']}: phi = {d['phi_tflops']:5.2f} TFLOPS   "
            f"s_min = {d['s_min']}   b_floor = {d['b_floor_empirical']}"
        )
    spread = max(x["phi_tflops"] for x in derived) / min(x["phi_tflops"] for x in derived)
    print(f"\n  per-card phi spread (I11): {spread:.3f}x")


if __name__ == "__main__":
    main()
