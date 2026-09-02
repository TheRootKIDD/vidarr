"""2:4 structured sparsity vs dense GEMM at inference shapes (`docs/06 §6`).

Supplies the *speed* half of H3 (note P3). `docs/06 §2` is explicit that this is
an inference-time measurement: training applies the mask on dense kernels, so
nothing here bears on H3's quality half, which comes from L10a.

ADR-019 fixes the sparsity pattern this measures: weights only, magnitude mask
after 1 % of tokens, then fixed. So the shapes are expert weight matrices and
the sparse operand is always the weight.

Run: `python -m scripts.bench.bench_sparse24 --id 005-sparse24`
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any

import torch
from torch.sparse import to_sparse_semi_structured

from scripts.bench._common import write_result

# Expert weight shapes: (d_ff, d) @ (d, b). Widths follow 003/004-gemm — every
# one a multiple of 64, since odd widths cost 12-27 % on the dense side alone.
SHAPES = [
    (768, 832), (768, 1664), (768, 2048), (768, 2816),
    (1024, 1664), (1024, 2816), (1024, 4096),
]
BATCH = [256, 512, 1024, 2048, 4096, 8192]


def _magnitude_2_4_mask(w: torch.Tensor) -> torch.Tensor:
    """Keep the 2 largest-magnitude of every contiguous 4 along the last dim.

    This is ADR-019's rule (magnitude, then fixed). The mask is applied here only
    so the sparse kernel has a genuinely 2:4 tensor to compress; the values are
    irrelevant to timing.
    """
    n, m = w.shape
    groups = w.abs().reshape(n, m // 4, 4)
    keep = groups.argsort(dim=-1, descending=True)[..., :2]
    mask = torch.zeros_like(groups, dtype=torch.bool).scatter_(-1, keep, True)
    return mask.reshape(n, m)


def _time(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / 1e3 / iters


def measure(device: int, d: int, d_ff: int, b: int) -> dict[str, Any] | None:
    """Dense vs 2:4 for one expert weight at one batch size."""
    dev = f"cuda:{device}"
    w = torch.randn(d_ff, d, device=dev, dtype=torch.bfloat16)
    w = w * _magnitude_2_4_mask(w)
    x = torch.randn(d, b, device=dev, dtype=torch.bfloat16)

    try:
        w_sparse = to_sparse_semi_structured(w)
    except (RuntimeError, NotImplementedError) as exc:
        return {"device": device, "d": d, "d_ff": d_ff, "b": b, "error": str(exc)[:200]}

    iters = 50 if b <= 2048 else 20
    dense_s = _time(lambda: w @ x, 5, iters)
    sparse_s = _time(lambda: w_sparse @ x, 5, iters)
    flops = 2.0 * d_ff * d * b

    out = {
        "device": device,
        "d": d,
        "d_ff": d_ff,
        "b": b,
        "dense_s": dense_s,
        "sparse_s": sparse_s,
        "dense_tflops": flops / dense_s / 1e12,
        # Sparse TFLOPS counted at DENSE FLOPs: the useful work is the same, half
        # the multiplies are skipped. This is the convention that makes "2x" the
        # ceiling rather than "1x".
        "sparse_effective_tflops": flops / sparse_s / 1e12,
        "speedup": dense_s / sparse_s,
    }
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="005-sparse24")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=None,
        help="override the batch sweep (the 2:4 path needs large b to beat dense)",
    )
    ap.add_argument(
        "--shapes",
        type=int,
        nargs="+",
        default=None,
        help="flat d,d_ff pairs, e.g. --shapes 1024 1664 768 2816",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA devices visible")
    warnings.filterwarnings("ignore", message=".*SparseSemiStructuredTensor.*")

    batches = args.batches or BATCH
    shapes = (
        list(zip(args.shapes[::2], args.shapes[1::2], strict=True)) if args.shapes else SHAPES
    )
    rows = [
        r
        for d, d_ff in shapes
        for b in batches
        if (r := measure(args.device, d, d_ff, b)) is not None
    ]
    ok = [r for r in rows if "speedup" in r]
    best = max(ok, key=lambda r: r["speedup"]) if ok else None
    payload = {
        "bench": "sparse24",
        "dtype": "bfloat16",
        "note": (
            "ADR-019: weights only, magnitude 2:4, fixed mask. Inference-time "
            "speed only; H3 quality comes from L10a training runs."
        ),
        "results": rows,
        "n_failed": len(rows) - len(ok),
        "best": best,
        "median_speedup": sorted(r["speedup"] for r in ok)[len(ok) // 2] if ok else None,
    }
    path = write_result(args.id, payload)
    print(f"wrote {path}\n")
    print(f"{'d':>5} {'d_ff':>6} {'b':>6} {'dense TF':>9} {'sparse TF':>10} {'speedup':>8}")
    for r in ok:
        print(
            f"{r['d']:>5} {r['d_ff']:>6} {r['b']:>6} {r['dense_tflops']:>9.2f} "
            f"{r['sparse_effective_tflops']:>10.2f} {r['speedup']:>8.3f}"
        )
    if ok:
        print(f"\n  median speedup {payload['median_speedup']:.3f}x, best {best['speedup']:.3f}x")


if __name__ == "__main__":
    main()
