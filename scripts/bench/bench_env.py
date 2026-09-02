"""Environment capture and correctness gate for the local rig (`docs/06 §1`).

Records what the rig actually is (as opposed to the nominal table in `docs/06 §1`)
and gates the rest of the suite on two facts that every later benchmark assumes:
all four cards are on a PCIe x16 link, and BF16 autocast computes correctly.

Run: `python -m scripts.bench.bench_env --id 000-env`
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from typing import Any

import torch

from scripts.bench._common import write_result


def pcie_state() -> list[dict[str, Any]]:
    """Per-GPU PCIe link width and generation.

    `gen.current` idles below `gen.max` on Ampere (power state P8), so this is
    only meaningful alongside a loaded reading — see `pcie_under_load`.
    """
    fields = (
        "index,pci.bus_id,pcie.link.width.current,pcie.link.width.max,"
        "pcie.link.gen.current,pcie.link.gen.max"
    )
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        idx, bus, wc, wm, gc, gm = (f.strip() for f in line.split(","))
        rows.append(
            {
                "index": int(idx),
                "bus_id": bus,
                "link_width_current": int(wc),
                "link_width_max": int(wm),
                "link_gen_current": int(gc),
                "link_gen_max": int(gm),
            }
        )
    return rows


def pcie_under_load() -> list[dict[str, Any]]:
    """Re-read the PCIe link gen while host<->device traffic is in flight.

    The idle reading is misleading: the link downtrains to Gen 1 at P8 and
    retrains on demand. What matters for `bench_nccl` is the loaded value.
    """
    n = torch.cuda.device_count()
    host = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cpu").pin_memory()
    bufs = [torch.empty_like(host, device=f"cuda:{i}") for i in range(n)]
    for _ in range(3):
        for i, b in enumerate(bufs):
            with torch.cuda.device(i):
                b.copy_(host, non_blocking=True)
    state = pcie_state()
    torch.cuda.synchronize()
    del bufs, host
    torch.cuda.empty_cache()
    return state


def bf16_autocast_check() -> dict[str, Any]:
    """Confirm BF16 autocast runs on the tensor cores and stays numerically sane.

    Tolerance is set for BF16's 8-bit mantissa accumulated over a K=4096
    reduction; a wrong dtype or a silently-disabled autocast fails it loudly.
    """
    results = []
    for i in range(torch.cuda.device_count()):
        dev = torch.device(f"cuda:{i}")
        torch.manual_seed(0)
        a = torch.randn(1024, 4096, device=dev)
        b = torch.randn(4096, 1024, device=dev)
        ref = (a.double() @ b.double()).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            got = a @ b
        rel = ((got.float() - ref).norm() / ref.norm()).item()
        results.append(
            {
                "index": i,
                "autocast_dtype": str(got.dtype),
                "relative_error_vs_fp64": rel,
                "ok": got.dtype is torch.bfloat16 and rel < 5e-3,
            }
        )
    return {"per_gpu": results, "all_ok": all(r["ok"] for r in results)}


def host_info() -> dict[str, Any]:
    total, used, free = shutil.disk_usage(".")
    meminfo = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            meminfo[k] = v.strip()
    return {
        "cwd_disk_total_bytes": total,
        "cwd_disk_free_bytes": free,
        "mem_total": meminfo.get("MemTotal"),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in open("/proc/cpuinfo")
                if line.startswith("model name")
            ),
            None,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="000-env")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA devices visible")

    idle = pcie_state()
    loaded = pcie_under_load()
    bf16 = bf16_autocast_check()

    payload = {
        "bench": "env",
        "pcie_idle": idle,
        "pcie_under_load": loaded,
        "all_x16": all(r["link_width_current"] == 16 for r in loaded),
        "all_gen4_under_load": all(r["link_gen_current"] == 4 for r in loaded),
        "bf16_autocast": bf16,
        "host": host_info(),
    }
    path = write_result(args.id, payload)
    print(f"wrote {path}")
    print(f"  all four cards at x16: {payload['all_x16']}")
    print(f"  all four at PCIe gen4 under load: {payload['all_gen4_under_load']}")
    print(f"  bf16 autocast ok: {bf16['all_ok']}")


if __name__ == "__main__":
    main()
