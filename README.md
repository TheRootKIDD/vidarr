# bigmoe — spec for a big-DC MoE LLM architecture (research project)

**Status:** DRAFT v0.3 — 2026-09-02 (author's answers to Q1–Q10 and v2 of the note incorporated; see `docs/04 §Author's answers` and `§v2 diff`). Working title `bigmoe`; rename freely.
**Source:** "The big-DC MoE LLM design" (19-point design note), v1 and v2 in `docs/source/` (with extracted text `v1.txt`, `v2.txt`); v2 is current. The author's linked in-depth document, "Designing AI Chip Hardware and Software" (2026, 154 pp., a public Google Doc), is beside it as `Designing_AI_chip_software_and_hardware.{pdf,txt}`; its summary against P1–P19 is in `docs/05`.
**Tooling:** Claude Code. `CLAUDE.md` holds the session rules.
**Authors:** Rasmus Søe Holt Christensen and Annemette Brok Pirchert.
**Licence:** free to use, **credit required**. Code under [Apache-2.0](LICENSE); documentation, specifications and experiment records under [CC BY 4.0](LICENSE-docs). Any derived work — code, models, specs, results or writing — must credit both authors: Apache §4(d) carries the [`NOTICE`](NOTICE) file into every redistribution, and CC BY §3(a) requires attribution outright. Cite via [`CITATION.cff`](CITATION.cff).

## Goal

Turn the 19-point note into (1) a precise architecture + systems specification, (2) a set of numbered, falsifiable hypotheses, and (3) an experiment programme that tests what can be tested at research scale and *simulates* what can only be validated at datacenter scale.

The note is a hardware/model co-design. Most of its individual ideas have precedents; the novelty is the combination and the organizing rule behind it (expert size = one hardware unit). The project's job is to find out which combinations pay off, in what order, and where the note's claims break.

## Documents

| File | What it is | Read when |
|---|---|---|
| `CLAUDE.md` | Working rules for Claude Code sessions | Every session |
| `docs/00-paper-review.md` | Point-by-point critique of the note: pros, cons, issues, pitfalls, claim-status table | Before touching the spec |
| `docs/01-architecture-spec.md` | The model: two streams, repeated middle block, router, experts, indexed attention, knowledge memory, MTP, numerics, adapters. **Notation lives here.** | Implementing `model/` |
| `docs/02-systems-spec.md` | Fabric (N×U), unit-internal execution, placement, continuous scheduler, prefill/decode mixing, KV tiers, memory budget, multi-tenancy, failures, the training gap | Implementing `sim/` and `costmodel/` |
| `docs/03-research-plan.md` | Hypotheses H1–H19, ablation ladder, simulator experiments, metrics, milestones, risks | Planning any experiment |
| `docs/04-decision-log.md` | ADRs (decisions made) + open questions, including questions for the note's author | Before deciding anything |
| `docs/05-reading-list.md` | Prior work per design point; what to re-check in the Phase-0 literature refresh | Phase 0 |
| `docs/07-author-hardware-doc.md` | The author's 154-page hardware/software document read against P1–P19: unit model, fabric ratios, numbers for `note64.yaml`, disagreements with `docs/02` | Before any `note64` scenario or S-experiment |
| `docs/08-hrm-brief.md` | Response to the HRM / HRM-Text / FlexMoE brief: source verification, spec diff for all 19 points, and the six experiments worth running | Before acting on HRM-derived ideas |
| `docs/09-findings-summary.md` | Findings so far, written for a co-author: recipe, architectures per rung, results table, what they say about the note, what is next | Catching up on results |
| `docs/06-compute-envelope.md` | The local rig (4 × RTX 3060 12 GB, TR PRO 3945WX, 128 GB): what it can measure vs only simulate, anchor sizes `screen`/`small`/`medium`, time budgets, microbenchmarks, operational rules | Before sizing or launching any run |

## Point → spec map

Tier A = testable in small-scale training experiments on the local rig (`docs/06`). Tier S = testable only by simulation / cost model, except S0 and S10 which are measured locally. O = opinion or out of scope; recorded, not tested.

| # | Note's point (short) | Spec section | Tier |
|---|---|---|---|
| 1 | Systolic arrays, not LPUs | 02 §1 | S / O |
| 2 | Parallel attention + FF | 01 §4.1 | A |
| 3 | 2:4 structured sparsity from the start; FP4 | 01 §8 | A |
| 4 | Expert size = smallest FF that fills a U-chip unit | 01 §4.5, 02 §2 | A + S |
| 5 | N × U fabric (rail-optimized clos + fat local unit) | 02 §3 | S |
| 6 | Few early/final layers + one repeated middle layer holding all experts | 01 §2, §4 | A |
| 7 | Variable iteration count per token; RL for compute allocation; **v2:** local per-iteration + global final-vector attention | 01 §4.3, §4.7 | A |
| 8 | Experts with 2+ internal layers | 01 §4.5 | A + S |
| 9 | Multicast + co-activation placement | 02 §3–4 | S |
| 10 | One combined installation per DC | 02 §5.6 | S / O |
| 11 | Continuous (non-lockstep) execution | 02 §5 | S |
| 12 | Two latent streams per token (context / prediction) | 01 §3 | A (optional — author suggests leaving it out) |
| 13 | Multi-token prediction trained from the start | 01 §7 | A |
| 14 | Managed prefill/decode aggregation, no disaggregation | 02 §6 | S |
| 15 | Indexed (~O(log n)) attention over the final-vector cache, cold KV on SSD/storage | 01 §6, 02 §7 | A + S |
| 16 | ~4 GB HBM per chip, or no HBM | 02 §8 | S / O |
| 17 | Per-iteration knowledge retrieval from a vector DB | 01 §4.6, 02 §9 | A |
| 18 | Per-vector tiny QLoRAs; bespoke experts; shared spine | 01 §9, 02 §10 | A + S |
| 19 | One giant DC; no multi-DC inference | 02 §12 | O |

## Repo layout (to create)

```
bigmoe/
  CLAUDE.md
  README.md
  docs/                 # this spec; source PDF in docs/source/; ADRs in docs/04
  model/                # PyTorch reference implementation of docs/01 (Tier A)
  sim/                  # discrete-event simulator of docs/02 (Tier S); scenarios in sim/scenarios/
  costmodel/            # FLOPs / bytes / latency / $ formulas as pure functions + tests
  experiments/<id>/     # config.yaml, run notes, results.md — append-only, never overwritten
  scripts/              # bench/ (rig microbenchmarks -> experiments/0NN), data/ (corpus tokenisation, ADR-014), train/ (DDP trainer -> experiments/1NN+)
```

## Principles

1. **One change at a time.** The ablation ladder in `docs/03` is the order features get switched on.
2. **Every claim gets a number** or an explicit "not testable at our scale" label.
3. **`U` is a parameter, not 64.** The reference model treats the hardware unit width abstractly; 64 is one setting.
4. **Spec before code.** Change `docs/01`/`docs/02`, log an ADR in `docs/04`, then implement.
5. **Two baselines always:** matched-FLOPs dense, and matched-FLOPs fine-grained MoE (DeepSeek-style).
6. **Budget before run.** Every experiment fits the envelope in `docs/06`: per-GPU footprint ≤ 10 GB, `screen` before `small` before `medium`, measured throughput numbers, not nominal ones.
