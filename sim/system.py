"""Milestone 1 of the `docs/02` system: owners, expert queues, units, links, decode tokens.

One unit plays every role (managed aggregation, `02 §6`): it owns sequences (attention,
skeleton, the sum of expert returns) and hosts experts. Each unit is a single FIFO compute
server. Owner work of one kind that is waiting when the server frees is run as one batched
job; an expert queue fires a batch at `batch_threshold` tokens or when its oldest token has
waited `w_max_s` (`02 §5.1`). Messages cross per-unit egress links. Numbered decisions refer
to `sim/README.md` (ADR-043). `batch_policy = "round"` replaces both batching rules with
S0's barrier (decision 11, ADR-049 §5).
"""

from __future__ import annotations

import gc
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from costmodel.config import Experts as CmExperts
from costmodel.config import Model
from costmodel.hardware import expert_step
from sim.engine import Engine
from sim.params import SimParams, attention_score_flops_per_token, log2_interp

OwnerKind = Literal["skel_in", "attn", "post", "skel_out"]


@dataclass
class Token:
    seq: int
    owner: int
    start_t: float
    j: int = 0  # current middle-block iteration, 1-based once started
    iter_start_t: float = 0.0
    pending: int = 0  # expert returns still outstanding this iteration


@dataclass
class ExpertJob:
    expert: int
    batch: list[tuple[float, Token]]  # (arrival time at the queue, token)
    ready_t: float


@dataclass
class Unit:
    uid: int
    busy: bool = False
    egress_free_t: float = 0.0
    expert_jobs: deque[ExpertJob] = field(default_factory=deque)
    owner_items: dict[str, list[tuple[float, Token]]] = field(
        default_factory=lambda: {"skel_in": [], "attn": [], "post": [], "skel_out": []}
    )
    log: list[tuple[float, float, float]] = field(default_factory=list)  # (start, end, flops)


@dataclass
class ExpertQueue:
    items: list[tuple[float, Token]] = field(default_factory=list)
    gen: int = 0  # bumps on every fire, invalidating the pending w_max timer


class Simulation:
    def __init__(self, p: SimParams) -> None:
        self.p = p
        self.eng = Engine()
        self.rng = np.random.default_rng(p.seed)
        self.units = [Unit(u) for u in range(p.n_units)]
        self.queues = [ExpertQueue() for _ in range(p.n_e)]
        self.route_p = self._routing_distribution()
        self.cm = Model(d=p.d, experts=CmExperts(n_e=p.n_e, k=p.k, d_ff=p.d_ff, L_e=p.L_e))
        self.score_flops = attention_score_flops_per_token(p)
        per_key = (
            2 * p.kv_width * p.b_kv
            if p.kv_bytes_per_key_layer is None
            else p.kv_bytes_per_key_layer
        )
        self.kv_bytes_per_layer = p.context_tokens * per_key
        # counters and samples
        self.started = 0
        self.completed = 0
        self.last_admit_t = math.inf
        self.warm_t = 0.0 if p.warmup_tokens == 0 else math.inf
        self.token_lat: list[tuple[float, float]] = []  # (completion time, latency)
        self.iter_lat: list[tuple[float, float]] = []
        self.queue_wait: list[tuple[float, float]] = []  # (batch start, wait)
        self.batch_sizes: list[int] = []
        self.fired = {"threshold": 0, "timeout": 0, "round": 0}
        # round policy (decision 11): owned-token counts, this iteration's dispatches and
        # per-expert targets, and per-owner iteration starts / returns for the latency sample
        self.round = p.batch_policy == "round"
        self.n_owned = [sum(1 for s in range(p.z) if p.owner_of(s) == u) for u in range(p.n_units)]
        self.round_rows: list[tuple[Token, list[int]]] = []  # this iteration's dispatches
        self.round_jobs_left = 0  # expert jobs of this iteration still to finish
        self.round_returns: list[tuple[int, int, Token]] = []  # (expert unit, owner, token)
        self.owner_iter_start = [0.0] * p.n_units
        self.bytes_logical = 0.0
        self.bytes_wire = 0.0
        self.token_iterations = 0
        self.in_flight: list[tuple[float, int]] = [(0.0, 0)]  # step function (time, count)

    # ---- workload ----------------------------------------------------------------------
    def _routing_distribution(self) -> np.ndarray:
        """Per-expert selection weights (decision 7): uniform, Zipf over expert index, or the
        measured shares of an S0 run (ADR-049 §5)."""
        p = self.p
        if p.routing == "empirical":
            w = np.asarray(p.route_probs, dtype=float)
        elif p.routing == "zipf":
            w = 1.0 / np.arange(1, p.n_e + 1) ** p.zipf_s
        else:
            w = np.ones(p.n_e)
        return w / w.sum()

    def _start_token(self, seq: int) -> None:
        tok = Token(seq=seq, owner=self.p.owner_of(seq), start_t=self.eng.now)
        self.started += 1
        self._track_in_flight(+1)
        if self.started == self.p.tokens:
            self.last_admit_t = self.eng.now
        self._owner_enqueue(tok, "skel_in")

    def _track_in_flight(self, delta: int) -> None:
        self.in_flight.append((self.eng.now, self.in_flight[-1][1] + delta))

    # ---- owner side ----------------------------------------------------------------------
    def _owner_enqueue(self, tok: Token, kind: OwnerKind) -> None:
        self.units[tok.owner].owner_items[kind].append((self.eng.now, tok))
        self._kick(tok.owner)

    def _owner_done(self, kind: str, batch: list[tuple[float, Token]]) -> None:
        if kind == "post" and self.round and batch:
            # one sample per owner per iteration, attention start to the end of `post` (harness);
            # taken first, because the loop below may start the owner's next attention job
            u = batch[0][1].owner
            self.iter_lat.append((self.eng.now, self.eng.now - self.owner_iter_start[u]))
        for _, tok in batch:
            if kind == "skel_in":
                self._begin_iteration(tok)
            elif kind == "attn":
                self._dispatch(tok)
            elif kind == "post":
                self._end_iteration(tok)
            else:
                self._complete(tok)

    def _begin_iteration(self, tok: Token) -> None:
        tok.j += 1
        tok.iter_start_t = self.eng.now
        self._owner_enqueue(tok, "attn")

    def _dispatch(self, tok: Token) -> None:
        """Route to k distinct experts and send one vector to each (`02 §3`; no multicast)."""
        tok.pending = self.p.k
        if self.p.route_draw == "bulk":
            self._round_dispatch(tok, None)  # drawn with the whole round
            return
        experts = self.rng.choice(self.p.n_e, size=self.p.k, replace=False, p=self.route_p)
        if self.round:
            self._round_dispatch(tok, [int(e) for e in experts])
            return
        for e in experts:
            dest = self.p.expert_of_unit(int(e))
            self._send(tok.owner, dest, lambda e=int(e), tok=tok: self._expert_arrive(e, tok))

    def _expert_return(self, tok: Token) -> None:
        tok.pending -= 1
        if tok.pending:
            return
        self._owner_enqueue(tok, "post")  # the g-weighted sum is the `post` job (decision 12)

    def _end_iteration(self, tok: Token) -> None:
        self.token_iterations += 1
        if not self.round:
            self.iter_lat.append((self.eng.now, self.eng.now - tok.iter_start_t))
        if tok.j < self.p.r:
            self._begin_iteration(tok)
        else:
            self._owner_enqueue(tok, "skel_out")

    def _complete(self, tok: Token) -> None:
        self.completed += 1
        self._track_in_flight(-1)
        self.token_lat.append((self.eng.now, self.eng.now - tok.start_t))
        if self.completed == self.p.warmup_tokens:
            self.warm_t = self.eng.now
        if self.started < self.p.tokens:
            self._start_token(tok.seq)  # closed loop: the sequence's next decode token

    # ---- fabric -----------------------------------------------------------------------------
    def _send(self, src: int, dst: int, deliver: Any) -> None:
        """One vector, src → dst over src's egress link (decision 5)."""
        nbytes = self.p.vector_bytes
        self.bytes_logical += nbytes
        if src == dst:
            self.eng.after(0.0, deliver)
            return
        self.bytes_wire += nbytes
        unit = self.units[src]
        start = max(self.eng.now, unit.egress_free_t)
        unit.egress_free_t = start + nbytes / self.p.lambda_link
        self.eng.at(unit.egress_free_t + self.p.link_latency_s, deliver)

    # ---- expert side -------------------------------------------------------------------------
    def _bulk_routes(self, n: int) -> np.ndarray:
        """n rows of k distinct experts, drawn without replacement with weights `route_p`:
        Gumbel-top-k, which has the law of sequential sampling without replacement."""
        k = self.p.k
        with np.errstate(divide="ignore"):
            keys = np.log(self.route_p)[None, :] + self.rng.gumbel(size=(n, self.p.n_e))
        return np.argpartition(-keys, k - 1, axis=1)[:, :k]

    def _round_dispatch(self, tok: Token, experts: list[int] | None) -> None:
        """Round policy (decisions 11-12): the dispatch is a collective. It completes when the
        last owner has joined, after two collective latencies (counts, rows) plus the busiest
        unit's off-unit bytes at lambda_link; then every expert fires once on all its rows."""
        self.round_rows.append((tok, experts))
        if len(self.round_rows) < self.p.z:
            return
        rows, self.round_rows = self.round_rows, []
        if self.p.route_draw == "bulk":
            drawn = self._bulk_routes(len(rows))
            rows = [(t, [int(e) for e in drawn[i]]) for i, (t, _) in enumerate(rows)]
        k = self.p.k
        src = np.repeat(np.fromiter((t.owner for t, _ in rows), dtype=np.int64, count=len(rows)), k)
        dst = np.fromiter((e for _, es in rows for e in es), dtype=np.int64, count=len(rows) * k)
        dt = 2 * self.p.collective_s + self._leg_s(src, dst % self.p.n_units, self.p.dispatch_bytes)

        def deliver() -> None:
            for t, es in rows:
                for e in es:
                    self.queues[e].items.append((self.eng.now, t))
            fire = [e for e in range(self.p.n_e) if self.queues[e].items]
            self.round_jobs_left = len(fire)
            for e in fire:
                self._fire(e, "round")

        self.eng.after(dt, deliver)

    def _round_expert_done(self, uid: int, job: ExpertJob) -> None:
        """The return leg: one collective once every expert job of the iteration has finished."""
        self.round_returns.extend((uid, tok.owner, tok) for _, tok in job.batch)
        self.round_jobs_left -= 1
        if self.round_jobs_left:
            return
        rets, self.round_returns = self.round_returns, []
        src = np.fromiter((u for u, _, _ in rets), dtype=np.int64, count=len(rets))
        dst = np.fromiter((o for _, o, _ in rets), dtype=np.int64, count=len(rets))
        dt = self.p.collective_s + self._leg_s(src, dst, self.p.return_bytes)

        def deliver() -> None:
            for _, _, tok in rets:
                self._expert_return(tok)

        self.eng.after(dt, deliver)

    def _leg_s(self, src: np.ndarray, dst: np.ndarray, row_bytes: float) -> float:
        """Wire time of one collective leg (decisions 11, 14): the busiest sender's bytes. One
        tier: off-unit rows at lambda_link. Two tiers: rows within a node at lambda_intra and
        across nodes at lambda_link, in parallel; `intra_via_nic` sends both over the NIC."""
        p = self.p
        off = src != dst
        self.bytes_logical += row_bytes * len(src)
        self.bytes_wire += row_bytes * int(off.sum())
        if p.node_size and not p.intra_via_nic:
            near = off & (src // p.node_size == dst // p.node_size)
        else:
            near = np.zeros_like(off)
        far = off & ~near
        intra = np.bincount(src[near], minlength=p.n_units) * row_bytes
        inter = np.bincount(src[far], minlength=p.n_units) * row_bytes
        lam_in = p.lambda_intra or p.lambda_link
        return float(np.max(np.maximum(intra / lam_in, inter / p.lambda_link)))

    def _expert_arrive(self, e: int, tok: Token) -> None:
        q = self.queues[e]
        q.items.append((self.eng.now, tok))
        if len(q.items) >= self.p.batch_threshold:
            self._fire(e, "threshold")
        elif len(q.items) == 1:
            gen = q.gen
            self.eng.after(self.p.w_max_s, lambda: self._timeout(e, gen))

    def _timeout(self, e: int, gen: int) -> None:
        q = self.queues[e]
        if q.gen == gen and q.items:
            self._fire(e, "timeout")

    def _fire(self, e: int, reason: str) -> None:
        q = self.queues[e]
        batch, q.items = q.items, []
        q.gen += 1
        self.fired[reason] += 1
        unit = self.units[self.p.expert_of_unit(e)]
        unit.expert_jobs.append(ExpertJob(e, batch, self.eng.now))
        self._kick(unit.uid)

    # ---- the unit's compute server --------------------------------------------------------------
    def _kick(self, uid: int) -> None:
        """Start the oldest ready job if the unit is idle (FIFO across roles, decision 2)."""
        unit = self.units[uid]
        if unit.busy:
            return
        best_kind, best_t = None, math.inf
        if unit.expert_jobs:
            best_kind, best_t = "expert", unit.expert_jobs[0].ready_t
        for kind, items in unit.owner_items.items():
            if self.round and len(items) < self.n_owned[uid]:
                continue  # round policy: an owner stage waits for all its tokens
            if items and items[0][0] < best_t:
                best_kind, best_t = kind, items[0][0]
        if best_kind is None:
            return
        unit.busy = True
        now = self.eng.now
        if best_kind == "expert":
            job = unit.expert_jobs.popleft()
            dt, flops = self.expert_step_s(len(job.batch))
            self.batch_sizes.append(len(job.batch))
            for arrived, _ in job.batch:
                self.queue_wait.append((now, now - arrived))
            done = lambda: self._expert_job_done(unit, job)  # noqa: E731
        else:
            batch, unit.owner_items[best_kind] = unit.owner_items[best_kind], []
            if self.round and best_kind == "attn":
                self.owner_iter_start[uid] = now
            dt, flops = self.owner_step_s(best_kind, len(batch))
            done = lambda: self._owner_job_done(unit, best_kind, batch)  # noqa: E731
        unit.log.append((now, now + dt, flops))
        self.eng.after(dt, done)

    def _expert_job_done(self, unit: Unit, job: ExpertJob) -> None:
        unit.busy = False
        if self.round:
            self._round_expert_done(unit.uid, job)
            self._kick(unit.uid)
            return
        for _, tok in job.batch:
            self._send(unit.uid, tok.owner, lambda tok=tok: self._expert_return(tok))
        self._kick(unit.uid)

    def _owner_job_done(self, unit: Unit, kind: str, batch: list[tuple[float, Token]]) -> None:
        unit.busy = False
        self._owner_done(kind, batch)
        self._kick(unit.uid)

    # ---- step times (rooflines; decisions 1, 6) -------------------------------------------------
    def grouped_efficiency(self, b: int) -> float:
        """Grouped-GEMM efficiency vs dense at `b` tokens per expert: the scenario's table,
        linear in log2(b) between points, clamped at the ends (decision 1)."""
        return log2_interp(self.p.grouped_eff, b)

    def expert_step_s(self, b: int) -> tuple[float, float]:
        """`costmodel.expert_step` with the grouped-GEMM efficiency on its compute term; the
        unit-internal broadcast/reduce is charged only when U > 1 (decision 1)."""
        p = self.p
        r = expert_step(b, p.d_ff, self.cm, p.hardware, p.numerics)
        local = r.local_traffic_s if p.U > 1 else 0.0
        flops = 2.0 * b * 3.0 * p.d * p.d_ff * p.L_e
        roof = max(r.compute_s / self.grouped_efficiency(b), r.weight_stream_s) + local
        return p.job_time("expert", roof), flops

    def owner_step_s(self, kind: str, n: int) -> tuple[float, float]:
        """Batched owner work: max(FLOPs / φ, (weights + KV reads) / β_C) (decision 6), divided
        by the scenario's efficiency at n sequences (decision 13; 1 when absent)."""
        p = self.p
        phi_o = p.phi if p.phi_owner is None else p.phi_owner
        phi_a = p.phi if p.phi_attn is None else p.phi_attn
        beta_a = p.beta_c if p.beta_attn is None else p.beta_attn
        kv = self.kv_bytes_per_layer
        if kind == "attn":
            flops = n * (p.attn_proj_flops + self.score_flops)
            compute = n * p.attn_proj_flops / phi_o + n * self.score_flops / phi_a
            mem = p.attn_proj_weight_bytes / p.beta_c + n * kv / beta_a
        elif kind == "post":
            # g-weighted sum of k returns plus the residual: k d FLOPs, (k + 2) d reads per token
            flops = n * 2.0 * p.k * p.d
            compute = flops / p.phi
            mem = n * (p.k + 2) * p.d * p.b_act / p.beta_c
        elif kind == "skel_in":
            flops = n * p.skel_in_flops
            compute = flops / phi_o
            mem = p.skel_in_weight_bytes / p.beta_c + n * p.skel_in_layers * kv / beta_a
        else:
            flops = n * p.skel_out_flops
            compute = flops / phi_o
            w = p.skel_out_weight_bytes + p.head_weight_bytes
            mem = w / p.beta_c + n * p.skel_out_layers * kv / beta_a
        roof = max(compute, mem) / p.owner_efficiency(kind, n)
        return p.job_time(kind, roof), flops

    # ---- run and report ----------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        for seq in range(self.p.z):
            self.eng.at(0.0, lambda seq=seq: self._start_token(seq))
        # the run allocates millions of acyclic sample tuples; the cyclic collector would
        # rescan them all on every generation-2 pass (quadratic in the run's length)
        enabled = gc.isenabled()
        gc.disable()
        try:
            self.eng.run(lambda: self.completed >= self.p.tokens)
        finally:
            if enabled:
                gc.enable()
        for unit in self.units:
            if unit.expert_jobs or any(unit.owner_items.values()):
                raise RuntimeError(f"unit {unit.uid} has work left after the last token")
        if any(q.items for q in self.queues):
            raise RuntimeError("expert queues hold tokens after the last token")
        return self.summary()

    def window(self) -> tuple[float, float]:
        """Measurement window: from the warm-up completion to the last admission, where the
        number of tokens in flight is exactly z (decision 10)."""
        t0 = 0.0 if self.p.warmup_tokens == 0 else self.warm_t
        t1 = self.last_admit_t
        if not t1 > t0:
            raise ValueError("empty measurement window: raise tokens or lower warmup_tokens")
        return t0, t1

    def summary(self) -> dict[str, Any]:
        p = self.p
        t0, t1 = self.window()
        span = t1 - t0
        lat = np.array([x for t, x in self.token_lat if t0 < t <= t1])
        it = np.array([x for t, x in self.iter_lat if t0 < t <= t1])
        qw = np.array([x for t, x in self.queue_wait if t0 < t <= t1])
        x = len(lat) / span
        return {
            "scenario": p.scenario,
            "rung": p.rung,
            "z": p.z,
            "tokens": p.tokens,
            "seed": p.seed,
            "batch_policy": p.batch_policy,
            "routing": p.routing,
            "batch_threshold": p.batch_threshold,
            "w_max_s": p.w_max_s,
            "window_s": [t0, t1],
            "events": self.eng.events,
            "throughput_tok_s": x,
            "token_latency_s": _dist(lat),
            "iteration_latency_s": _dist(it),
            "tokens_in_flight_mean": self._mean_in_flight(t0, t1),
            "littles_law": {"z": p.z, "x_times_l": float(x * lat.mean()) if len(lat) else 0.0},
            "expert_queue_wait_s": _dist(qw),
            "batches": {
                "count": len(self.batch_sizes),
                "mean_size": float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0,
                "fired_by_threshold": self.fired["threshold"],
                "fired_by_timeout": self.fired["timeout"],
                "fired_by_round": self.fired["round"],
            },
            "fabric_bytes_per_token_iteration": {
                "logical": self.bytes_logical / max(self.token_iterations, 1),
                "wire": self.bytes_wire / max(self.token_iterations, 1),
                "formula_2kd_bact": 2.0 * p.k * p.d * p.b_act,
            },
            "utilisation": self.utilisation(t0, t1),
        }

    def _mean_in_flight(self, t0: float, t1: float) -> float:
        area = 0.0
        pts = self.in_flight + [(math.inf, 0)]
        for (ta, n), (tb, _) in zip(pts, pts[1:], strict=False):
            lo, hi = max(ta, t0), min(tb, t1)
            if hi > lo:
                area += n * (hi - lo)
        return area / (t1 - t0)

    def utilisation(self, t0: float, t1: float) -> dict[str, Any]:
        """Per unit: busy fraction and FLOP utilisation over the window, plus a time series in
        bins of `util_window_s` (`02 §14`)."""
        p = self.p
        n_bins = max(1, math.ceil((t1 - t0) / p.util_window_s))
        out: dict[str, Any] = {}
        for unit in self.units:
            busy, flops = 0.0, 0.0
            series = np.zeros(n_bins)
            for s, e, f in unit.log:
                lo, hi = max(s, t0), min(e, t1)
                if hi <= lo:
                    continue
                share = (hi - lo) / (e - s)
                busy += hi - lo
                flops += f * share
                b0 = int((lo - t0) / p.util_window_s)
                b1 = min(n_bins - 1, int((hi - t0) / p.util_window_s))
                for b in range(b0, b1 + 1):
                    blo = t0 + b * p.util_window_s
                    ov = min(hi, blo + p.util_window_s) - max(lo, blo)
                    if ov > 0:
                        series[b] += f * share * ov / (hi - lo) / (p.phi * p.util_window_s)
            out[f"unit_{unit.uid}"] = {
                "busy_fraction": busy / (t1 - t0),
                "flop_utilisation": flops / (p.phi * (t1 - t0)),
                "flop_utilisation_series": [round(float(v), 4) for v in series],
            }
        return out


def _dist(xs: np.ndarray) -> dict[str, float]:
    if len(xs) == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p99": 0.0}
    return {
        "n": len(xs),
        "mean": float(xs.mean()),
        "p50": float(np.percentile(xs, 50)),
        "p99": float(np.percentile(xs, 99)),
    }


def simulate(p: SimParams) -> dict[str, Any]:
    return Simulation(p).run()
