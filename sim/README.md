# sim/ — discrete-event simulator of `docs/02`

Milestone 1 (ADR-043). It is enough to attempt S0 once a measured S0 run exists. No
S-result may be quoted from it until both calibration points of `02 §14` pass (S0 and I1).

| file | role |
|---|---|
| `params.py` | `SimParams`, the one parameter dataclass; `load_params(scenario, rung, **overrides)`; `check_sources` (every scenario value has a source line) |
| `engine.py` | hand-rolled heap event loop; ties broken by scheduling order; an empty queue before the stop condition is a deadlock error (`02 §5.5`) |
| `system.py` | units, attention owners, expert queues, links, decode tokens; `simulate(params) -> dict` |
| `run.py` | CLI: `python -m sim.run --scenario local_3060 --rung L5 --z 256 --tokens 20000 --seed 0` |
| `scenarios/` | one YAML per hardware scenario, each value with its source |

## What milestone 1 models

- **Units** (`02 §1`): `n_units` units of $U$ chips each, every unit playing every role
  (managed aggregation, `02 §6`). A unit owns sequences (seq $s$ → unit $s \bmod n$) and hosts
  experts (expert $i$ → unit $i \bmod n$).
- **Decode tokens in a closed loop**: `z` sequences each generate one token at a time; a
  completed token immediately starts that sequence's next one until `tokens` have been
  admitted. A token runs: early blocks → $r$ × (attention at the owner → dispatch to $k$
  distinct experts → expert batch → return → sum) → final blocks and head. Fixed $r$.
- **Expert queues** (`02 §5.1`): fire at `batch_threshold` tokens or when the oldest token
  has waited `w_max_s`; the step time is `costmodel.expert_step` with the scenario's
  grouped-GEMM efficiency on the compute term.
- **Links**: one egress serialiser per unit at $\lambda_{link}$, plus a latency floor per
  message. $k$ copies out and $k$ back, one vector each; no multicast or in-network reduction.
- **Outputs** (subset of `02 §14`): throughput, per-token and per-iteration latency
  (mean / p50 / p99), mean tokens in flight and Little's-law check, expert queue wait,
  batch counts by firing reason, fabric bytes per token per iteration (logical = $2kd\,b_{act}$
  and on-the-wire), per-unit busy fraction and FLOP utilisation with an optional time series.

## Not yet (later milestones)

Multicast and in-network reduction (`02 §3`; hook `multicast`, refused); the placement
controller and co-activation (`02 §4`); ragged depth (ACT/L6, L6c thresholds; hook `depth`,
refused); prefill and the prefill/decode mix, TTFT (`02 §6`, S5); KV tiers and the cold
index (`02 §7`, S6); memory budget (`02 §8`, S7); knowledge memory, tenants, failures,
geography (`02 §9–12`); two streams; granularity $g > 1$ and tensor parallelism inside a unit
beyond the `costmodel` local-traffic term; expert-choice balancing in time (`02 §5.3`).

## Implementation decisions (ADR-043)

1. **Expert step** = `costmodel.expert_step`: max(compute / grouped-GEMM efficiency,
   weight streaming), plus the unit-internal broadcast/reduce **only when $U$ > 1**. With
   $U$ = 1 there is no ring inside the unit, and `costmodel` would otherwise charge
   $2bd\,b_{act}/\lambda_C$ for nothing. Efficiency is the scenario table, linear in
   $\log_2 b$ between points and clamped at the ends.
2. **One FIFO compute server per unit**, no concurrency between jobs (no CUDA streams). The
   oldest ready job runs next; ties go to expert jobs. Owner work of one kind that is waiting
   when the server frees runs as one batched job.
3. **Static placement**: expert $i$ on unit $i \bmod n_{units}$; the owner of sequence $s$
   is unit $s \bmod n_{units}$. The placement controller is milestone 2.
4. **Batch threshold** = the scenario's empirical occupancy floor (`b_floor_empirical`,
   512 on `local_3060`) by default; `batch_rule = roofline` gives
   $\lceil b_{min} \rceil$ from `costmodel.b_min`. `02 §2` names $b_{min}$, the scenario note
   names the floor, and I13/I18 are open; S2 sweeps both.
5. **Links**: a per-unit egress serialiser at $\lambda_{link}$ plus a latency floor per
   message, which is propagation and does not occupy the link. Ingress contention is not
   modelled; $\lambda_{link}$ on `local_3060` is per-rank wire bandwidth with every rank
   sending at once (`002-nccl`), which is what an egress-only model needs. Messages are one
   vector each (no aggregation per destination). Same-unit deliveries are free and count as
   logical bytes but not wire bytes.
6. **Owner work is a roofline per batched job**: time = max(FLOPs / $\phi$, bytes /
   $\beta_C$). FLOPs come from `costmodel.flops`; bytes are the weights read once per batch
   plus each token's KV reads. Every decode token sees `context_tokens` keys per attention
   layer (default: the rung's training context, 2048), for L5 and L5d alike. Attention FLOPs
   use `attention_projection_flops` + `attention_score_flops`. The skeleton is split by block
   count (early → `skel_in`, final + output head → `skel_out`); each half reads half the
   skeleton weights.
7. **Routing**: $k$ distinct experts per token per iteration, drawn without replacement from a
   fixed distribution, uniform or Zipf($s$) over expert index; no co-activation structure and no
   dependence on $j$ in m1.
8. **The owner's $g$-weighted sum of returns is free**: $k d$ FLOPs per token per iteration,
   negligible against the expert step.
9. **`w_max_s` = 1 ms by default, uncalibrated.** `02 §5.1` gives no value. S0's measured
   side must fix the same policy, so the value is an open question (I35), not a result.
10. **Measurement window**: from the `warmup_tokens`-th completion to the last admission.
    There the number of tokens in flight is exactly `z`, so $Z = XL$ is a check on the event
    accounting, not an assumption.

## S0: the calibration still to do

S0 (`docs/03`, `06 §6`) compares this simulator on `local_3060` with a **measured**
expert-parallel inference run of the `small` model on the four cards. That run does not
exist yet. Its script must, at fixed $z$ ∈ {16, 64, 256, 1024} decode sequences in flight
and a fixed $r$:

- place experts and owners exactly as decisions 2–3 (or record its own placement, and the
  simulator is then run with that placement);
- use the same batching policy: a threshold and a timeout, logged as numbers;
- record throughput (tokens/s, after warm-up), per-iteration latency (p50/p99) measured at the
  owner from attention start to the last expert return, per-token latency, expert batch-size
  distribution, and per-GPU busy fraction, all on a rig with zero `n_thermal` rows;
- record the git hash, driver and the scenario file's hash, under its own `0NN` rig id.

Pass rule (`02 §14`): throughput and per-iteration p50 within ±20 % at every $z$.

## Running

Tests (`tests/test_sim.py`, toy-sized) run only on an idle machine or inside a queue gate, never
beside a DDP run (CLAUDE.md):

    .venv/bin/python -m pytest -q tests/test_sim.py
    .venv/bin/python -m sim.run --scenario local_3060 --rung L5 --z 256 --tokens 20000 --seed 0
