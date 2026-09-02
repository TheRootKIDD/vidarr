# experiments/

**Append-only.** A directory here is a record of something that was run. It is
never overwritten and never edited to change a number. A rerun — different code,
different config, different hardware state, or just a second attempt — gets a
**new id**. If a result turns out to be wrong, the correction is a new id plus a
note in the old `results.md` saying which id supersedes it.

## Id convention

```
NNN-short-slug/
```

`NNN` is a zero-padded three-digit serial, allocated in order and never reused.
The slug is lowercase kebab-case. Ids are allocated in these bands:

| Band | Contents |
|---|---|
| `000`–`099` | rig characterisation: `scripts/bench/` results, environment capture |
| `100`–`199` | ladder training runs (L0–L11) at `screen` |
| `200`–`299` | ladder training runs at `small` |
| `300`–`399` | ladder training runs at `medium` |
| `400`–`499` | simulator experiments (S0–S10) |

## Contents of a directory

| File | When | What |
|---|---|---|
| `config.yaml` | training / simulator runs | the exact config the run consumed |
| `result.json` | `scripts/bench/` results | machine-readable numbers plus provenance (`docs/06 §6`: git hash, driver/CUDA versions, date) — written by `scripts/bench/_common.py:write_result`, which refuses to overwrite |
| `results.md` | every run | numbers, seeds, hardware, wall-clock, and a one-paragraph interpretation |

`results.md` states **which hypothesis (H1–H19) the run bears on, and whether it
supports, weakens, or is silent on it** (CLAUDE.md). A bench result that feeds a
scenario value says so, because `sim/scenarios/*.yaml` cites result ids as its
source lines.

Heavy outputs (checkpoints, logs, tokenised data) are gitignored and stay local;
`config.yaml`, `result.json` and `results.md` are tracked.
