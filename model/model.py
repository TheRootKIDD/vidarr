"""The reference model (`docs/01`): layered baselines and the recurrent design in one class.

    forward_hidden(tokens, r) -> final hidden state h [B, T, d] and Extras
    loss(tokens, r)           -> total loss and its parts

Recurrent, `global_source = per_iteration` (L5): one segment = the whole
context; each iteration attends causally over all tokens' same-iteration K/V.
Recurrent, `global_source = final_vector` (L5d, v2 P7): the context is
processed as segments of W tokens in order; each segment attends locally to
its own per-iteration K/V and globally to the final vectors of the earlier
segments (the cache G), with stop-gradient into G by default (ADR-023).
L5e (ADR-034): a dense H block with its own weights runs after iteration
j = K/2 (mod K) and writes the tokens still active. The router aux is summed
within a segment and averaged over segments (ADR-038, I33).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from model.block import Block
from model.config import ModelCfg
from model.middle import IterState, MiddleBlock
from model.mtp import MTPHeads, MTPModules


@dataclass
class Extras:
    router_aux: Tensor
    ponder: Tensor  # ACT ponder cost (mean over tokens), zero in fixed mode
    r_map: Tensor | None = None  # [B, T] iterations executed per token (recurrent)
    loads: list[Tensor] = field(default_factory=list)  # expert load per iteration / layer
    q_loss: Tensor | None = None  # L8q's L_q, mean over valid positions and iterations (ADR-055)


@dataclass
class LossOut:
    loss: Tensor
    main: Tensor
    mtp: Tensor
    router_aux: Tensor
    ponder: Tensor
    metrics: dict[str, float]


class BigMoE(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.stream_tag = nn.Parameter(torch.zeros(cfg.d)) if cfg.streams.two_stream else None
        self.h_block: Block | None = None
        if cfg.arch == "layered":
            self.blocks = nn.ModuleList(
                Block(cfg, cfg.ff, cfg.dense_d_ff, cfg.block_form) for _ in range(cfg.layers)
            )
        else:
            w = cfg.skeleton_width
            self.early = nn.ModuleList(
                Block(cfg, "dense", w, "parallel") for _ in range(cfg.early_blocks)
            )
            self.middle = MiddleBlock(cfg)
            if cfg.middle.h_every:  # L5e: the slow dense H block, own weights (ADR-034)
                self.h_block = Block(cfg, "dense", cfg.middle.h_d_ff, "parallel")
            n_pred = cfg.pred_only_final
            self.final = nn.ModuleList(
                Block(cfg, "dense", w, "parallel", pred_only=i >= cfg.final_blocks - n_pred)
                for i in range(cfg.final_blocks)
            )
        self.out_norm = nn.RMSNorm(cfg.d)
        self.mtp: MTPHeads | MTPModules | None = None
        if cfg.mtp.m > 1 and cfg.mtp.style == "sequential":
            self.mtp = MTPModules(cfg)  # L3c (ADR-039)
        elif cfg.mtp.m > 1:
            self.mtp = MTPHeads(cfg.d, cfg.mtp)

    # ---- embedding and streams ---------------------------------------------------
    def embed(self, tokens: Tensor) -> tuple[Tensor, Tensor | None]:
        c = self.emb(tokens)
        p = c + self.stream_tag if self.stream_tag is not None else None
        return c, p

    def head(self, h: Tensor) -> Tensor:
        return F.linear(self.out_norm(h), self.emb.weight)  # tied

    # ---- forward -------------------------------------------------------------------
    def forward_hidden(
        self, tokens: Tensor, r: int | None = None, q_target: Tensor | None = None
    ) -> tuple[Tensor, Extras]:
        """`q_target` [B, T, 2]: L8q's teacher PQ code per input position (-1 = none)."""
        if self.cfg.arch == "layered":
            return self._forward_layered(tokens)
        return self._forward_recurrent(tokens, r, q_target)

    def _forward_layered(self, tokens: Tensor) -> tuple[Tensor, Extras]:
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        c, p = self.embed(tokens)
        aux = c.new_zeros(())
        loads = []
        for blk in self.blocks:
            c, p, a = blk(c, p, pos)
            aux = aux + a
            if blk.is_moe and blk.ff.last_route is not None:
                loads.append(blk.ff.last_route.load)
        h = p if p is not None else c
        return h, Extras(router_aux=aux, ponder=c.new_zeros(()), loads=loads)

    def _forward_recurrent(
        self, tokens: Tensor, r: int | None, q_target: Tensor | None = None
    ) -> tuple[Tensor, Extras]:
        cfg = self.cfg
        b, t = tokens.shape
        dev = tokens.device
        final_vec = cfg.attention.global_source == "final_vector"
        seg = cfg.attention.window if final_vec else t
        n_iter = cfg.middle.r_max if r is None else r
        fixed = cfg.halting.mode == "fixed"
        expert = cfg.halting.mode == "expert" and self.training
        if expert:
            self.middle.apply_pending_tau()  # L6c: last step's quantiles (ADR-042)
        mem: tuple[Tensor, Tensor] | None = None
        hs, r_maps, ponders, loads = [], [], [], []
        aux = torch.zeros((), device=dev)
        q_sum: Tensor | None = None  # L8q: summed (L_q, count) over segments and iterations
        for s0 in range(0, t, seg):
            pos = torch.arange(s0, s0 + seg, device=dev)
            c, p = self.embed(tokens[:, s0 : s0 + seg])
            for blk in self.early:
                c, p, a = blk(c, p, pos)
                aux = aux + a
            st = IterState(
                c=c,
                p=p,
                active=torch.ones(b, seg, dtype=torch.bool, device=dev),
                r=torch.zeros(b, seg, dtype=torch.long, device=dev),
                cum_halt=torch.zeros(b, seg, device=dev),
                remainder=torch.zeros(b, seg, device=dev),
                frozen_kv=None,
                frozen=torch.zeros(b, seg, dtype=torch.bool, device=dev),
                loads=[],
                q_tgt=None if q_target is None else q_target[:, s0 : s0 + seg],
            )
            for j in range(1, n_iter + 1):
                prev_active = st.active
                st, a = self._iterate(st, j, pos, mem, fixed)
                if expert and j > cfg.middle.r_min:
                    self.middle.record_tau(j, st.cum_halt, prev_active)
                aux = aux + a
                k_h = cfg.middle.h_every
                if self.h_block is not None and j % k_h == k_h // 2:
                    st = self._apply_h(st, pos)
                if not fixed and not st.active.any():
                    break
            loads.extend(st.loads)
            if st.q_loss is not None:
                q_sum = st.q_loss if q_sum is None else q_sum + st.q_loss
            c, p = st.c, st.p
            for blk in self.final:
                c, p, a = blk(c, p, pos)
                aux = aux + a
            hs.append(p if p is not None else c)
            r_maps.append(st.r)
            ponders.append(st.r.float() + st.remainder)
            if final_vec:
                g = self.middle.global_entry(st.c, pos)
                if mem is None:
                    mem = g
                else:
                    mem = (torch.cat([mem[0], g[0]], 2), torch.cat([mem[1], g[1]], 2))
        if cfg.router.segment_aux == "mean":
            aux = aux / len(hs)  # ADR-038 (I33): averaged over segments; one segment is unchanged
        h = torch.cat(hs, dim=1)
        r_map = torch.cat(r_maps, dim=1)
        ponder = torch.cat(ponders, dim=1).mean() if not fixed else torch.zeros((), device=dev)
        # L_q (ADR-055 §3): the mean over valid positions and iterations; every iteration sees
        # the same targets, so this equals the mean over iterations of the per-iteration means
        q_loss = None if q_sum is None else q_sum[0] / q_sum[1].clamp_min(1.0)
        return h, Extras(router_aux=aux, ponder=ponder, r_map=r_map, loads=loads, q_loss=q_loss)

    def _apply_h(self, st: IterState, pos: Tensor) -> IterState:
        """L5e (ADR-034): H reads every token's current state and writes active tokens only."""
        assert self.h_block is not None
        c, p, _ = self.h_block(st.c, st.p, pos)
        keep = st.active[..., None]
        c = torch.where(keep, c, st.c)
        p = None if p is None or st.p is None else torch.where(keep, p, st.p)
        return replace(st, c=c, p=p)

    def _iterate(
        self,
        st: IterState,
        j: int,
        pos: Tensor,
        mem: tuple[Tensor, Tensor] | None,
        fixed: bool,
    ) -> tuple[IterState, Tensor]:
        """One middle-block iteration, recomputed in backward when configured (02 §13.4)."""
        if not (self.training and self.cfg.checkpoint_iterations and torch.is_grad_enabled()):
            return self.middle.iteration(st, j, pos, mem, force_active=fixed)
        fk, fv = (None, None) if st.frozen_kv is None else st.frozen_kv
        mk, mv = (None, None) if mem is None else mem
        outs = checkpoint(
            self.middle.iteration_flat,
            j, pos, fixed, st.c, st.p, st.active, st.r, st.cum_halt, st.remainder,
            fk, fv, st.frozen, mk, mv, st.q_tgt, st.q_loss,
            use_reentrant=False,
        )
        c, p, active, r, cum, rem, fk2, fv2, frozen, aux, load, q_loss = outs
        new = IterState(
            c, p, active, r, cum, rem, None if fk2 is None else (fk2, fv2), frozen,
            [*st.loads, load], st.q_tgt, q_loss,
        )
        return new, aux

    # ---- decode (ADR-049; `model/decode.py`) -------------------------------------------
    def prefill(self, tokens: Tensor, cache: object) -> Tensor:
        """Fill `cache` (a `decode.DecodeCache`) with prompt `tokens` [B, T]; logits [B, T, V]."""
        from model import decode

        assert isinstance(cache, decode.DecodeCache)
        return decode.step(self, tokens, cache)

    def decode_step(self, tokens: Tensor, cache: object) -> Tensor:
        """One position per sequence: tokens [B, 1] -> logits [B, V]."""
        from model import decode

        assert isinstance(cache, decode.DecodeCache)
        return decode.step(self, tokens, cache)[:, -1]

    def set_progress(self, frac: float) -> None:
        """Training progress in [0, 1), for schedules inside the model (L3b, ADR-036)."""
        if self.mtp is not None:
            self.mtp.set_progress(frac)

    # ---- loss ------------------------------------------------------------------------
    def forward(
        self,
        tokens: Tensor,
        r: int | None = None,
        generator: torch.Generator | None = None,
        q_target: Tensor | None = None,
    ) -> LossOut:
        """The training forward is the loss, so DDP's hooks wrap the whole computation."""
        return self.loss(tokens, r, generator, q_target)

    def loss(
        self,
        tokens: Tensor,
        r: int | None = None,
        generator: torch.Generator | None = None,
        q_target: Tensor | None = None,
    ) -> LossOut:
        """tokens [B, T+1]: positions 0..T-1 are inputs, the last is the final target.
        `q_target` [B, T, 2] (L8q, ADR-055): the teacher's PQ code per input position."""
        inp = tokens[:, :-1]
        h, ex = self.forward_hidden(inp, r, q_target)
        main = self._chunked_ce(h, tokens[:, 1:])
        metrics: dict[str, float] = {}
        if self.mtp is not None:
            # heads 2..m read the same hidden state; targets x_{t+j}, j >= 2, from `tokens`
            mtp, metrics = self.mtp.aux_losses(h, tokens, self.emb.weight, self.out_norm, generator)
        else:
            mtp = main.new_zeros(())
        ponder = self.cfg.halting.ponder_cost * ex.ponder
        total = main + mtp + ex.router_aux + ponder
        if ex.q_loss is not None:  # L8q (ADR-055): lambda_q L_q, reported apart from the main loss
            qa = self.cfg.memory.query_aux
            assert qa is not None, "q_target given but memory.query_aux is off"
            total = total + qa.lambda_q * ex.q_loss
            metrics["q_loss"] = float(ex.q_loss.detach())
        if ex.r_map is not None:
            metrics["r_mean"] = float(ex.r_map.float().mean())
        return LossOut(total, main, mtp, ex.router_aux, ponder, metrics)

    def _chunked_ce(self, h: Tensor, targets: Tensor) -> Tensor:
        """Mean cross-entropy over the tied head without holding [B*T, V] logits for backward."""
        flat_h = h.reshape(-1, h.shape[-1])
        flat_t = targets.reshape(-1)
        n = flat_h.shape[0]
        chunk = self.cfg.ce_chunk_tokens

        def part(hc: Tensor, tc: Tensor) -> Tensor:
            return F.cross_entropy(self.head(hc).float(), tc, reduction="sum")

        total = flat_h.new_zeros((), dtype=torch.float32)
        for i in range(0, n, chunk):
            hc, tc = flat_h[i : i + chunk], flat_t[i : i + chunk]
            if self.training and torch.is_grad_enabled():
                total = total + checkpoint(part, hc, tc, use_reentrant=False)
            else:
                total = total + part(hc, tc)
        return total / n

    # ---- bookkeeping -----------------------------------------------------------------
    def routers(self) -> list[nn.Module]:
        mods = []
        for m in self.modules():
            if hasattr(m, "update_bias"):
                mods.append(m)
        return mods

    def n_params(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        emb = self.emb.weight.numel()
        return {"total": total, "embedding": emb, "non_embedding": total - emb}
