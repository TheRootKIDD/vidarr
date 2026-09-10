"""Shared helpers for `scripts/bench/*`: provenance, result ids, JSON output.

Every bench result carries the provenance fields required by `docs/06 §6`
(git hash, driver/CUDA versions, date) so that a `sim/scenarios/*.yaml` value
can cite the result id that produced it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments"


def _sh(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_hash() -> str:
    return _sh(["git", "rev-parse", "HEAD"]) or "unknown"


def git_dirty() -> bool:
    return bool(_sh(["git", "status", "--porcelain"]))


def driver_version() -> str | None:
    out = _sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else None


def gpu_uuids() -> dict[int, str]:
    """Index -> GPU UUID. The index and the bus id are properties of the *slot*,
    so neither survives a re-slotting; the UUID is the only stable handle on a
    physical card, and GeForce parts report no serial (`docs/04`, 2026-09-10
    night: tracking one card across a move is what `021` had to reconstruct by
    hand)."""
    out = _sh(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"])
    uuids = {}
    for line in (out or "").strip().splitlines():
        idx, _, uuid = line.partition(",")
        if uuid.strip():
            uuids[int(idx)] = uuid.strip()
    return uuids


def gpu_inventory() -> list[dict[str, Any]]:
    """Per-device identity. Kept per-device on purpose: the four cards are not
    guaranteed to be the same die (see `docs/04` I8)."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    uuids = gpu_uuids()
    inv = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        inv.append(
            {
                "index": i,
                "name": p.name,
                "uuid": uuids.get(i),
                "sm": f"{p.major}.{p.minor}",
                "multi_processor_count": p.multi_processor_count,
                "total_memory_bytes": p.total_memory,
            }
        )
    return inv


def provenance() -> dict[str, Any]:
    try:
        import torch

        torch_version, cuda_version = torch.__version__, torch.version.cuda
    except ImportError:
        torch_version, cuda_version = None, None
    return {
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "git_dirty": git_dirty(),
        "driver_version": driver_version(),
        "torch_version": torch_version,
        "torch_cuda_version": cuda_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpus": gpu_inventory(),
    }


def write_result(result_id: str, payload: dict[str, Any]) -> Path:
    """Write `experiments/<result_id>/result.json`. Append-only: refuses to
    overwrite an existing result (CLAUDE.md — new run, new id)."""
    out_dir = RESULTS_DIR / result_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "result.json"
    if path.exists():
        sys.exit(f"refusing to overwrite {path} — experiments are append-only, pick a new id")
    doc = {"result_id": result_id, "provenance": provenance(), **payload}
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    return path
