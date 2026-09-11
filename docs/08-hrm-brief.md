# 08 — Response to the HRM / FlexMoE brief (2026-09-11)

Answers the ten questions in the brief Annemette raised, against the spec as it stands. Read `00`
for the note's own critique, `03 §2` for the ladder, `04` for what is already decided, and `05` for
what the literature actually says — several of the brief's claims are already in `05` **with caveats
the brief omits**, and those caveats change the answers.

**Summary in one paragraph.** Two of the brief's ten questions are already closed by decisions on
record and should not be reopened without an ADR (Q6 objective, Q8 split latent). Two rest on sources
this project has not verified (Q7, and the FlexMoE half of Q1 #3). The remaining six are live, and one
of them — Q2's H/L split — is the most interesting idea in the brief and **fits the existing FLOPs
budget without trading anything**, which I did not expect before doing the arithmetic. The brief's
own framing of its strongest evidence is, however, wrong in a way that matters: HRM-Text cannot
settle H6 for us, for a reason already recorded in `05`.

---

## 0. Verification status of the sources — read this first

CLAUDE.md: *unknown → question, not guess*. `05` was verified entry by entry on 2026-09-02. Against
that:

| Source | Status |
|---|---|
| HRM (arXiv:2506.21734) | **Verified**, `05` line 92 |
| HRM-Text (arXiv:2605.20613) | **Verified**, `05` line 113, with caveats below |
| DFM Mimir v1 (arXiv:2608.13517) | **Verified**, same entry |
| Mixture-of-Recursions (arXiv:2507.10524) | **Verified**, `05` line 102 |
| TRM (arXiv:2510.04871) | **Verified**, `05` line 92 |
| Engram (arXiv:2601.07372) | **Verified**, `05` line 256 |
| Ouro / Huginn | **Verified**, in `05` |
| ARC Prize HRM ablation | **Verified**, `05` line 419 |
| **FlexMoRE (arXiv:2602.08818)** | **NOT in `05`. Unverified.** |
| **FlexMoE (arXiv:2606.27866)** | **NOT in `05`. Unverified**, and dated after my knowledge cutoff |
| **HRM-MoE (HF `Xiaoye08/HRM-MoE`)** | A community checkpoint, not a paper. The brief says so. **Unverified and not peer-reviewed.** |

I am not treating the bottom three as established. Q7 rests entirely on FlexMoRE and the FlexMoE half
of Q1 #3 rests entirely on FlexMoE, so both answers below are conditional. **Verifying them is a
`docs/05` task and should happen before anything is built on them** — the reading list's own format
requires a quantitative takeaway per entry, which is exactly the discipline that would catch a
number that does not survive contact with the paper.

### The caveat that changes the brief's conclusion

The brief's central architectural claim is *"two separate weight sets beat one shared loop"*, sourced
to HRM-Text, and it drives Q2. `05` line 419 already records why that cannot be read that way here:

> Neither HRM-Text nor Mimir reports a held-out LM loss, so they are silent on H6 as ADR-013 defines
> it.

HRM-Text's wins are **benchmark** wins — MMLU 60.7 vs 53.2, GSM8K 84.5 vs 75.1. ADR-013 defines every
Tier-A verdict as a held-out **loss** delta against a matched-FLOPs baseline with a measured-σ band,
and ADR-014 fixes that held-out set. A benchmark win at 1B on instruction data is not evidence about
loss on FineWeb-Edu at 80 M, and the two can disagree — `05` records exactly that pattern for HRM-MoE,
where extra expert parameters bought benchmark gains with **MMLU flat**.

Second caveat, also already recorded (`05` line 92): the ARC Prize re-run found **a same-size plain
transformer within ~5 points of HRM**, concluding *"the outer refinement loop, not the hierarchy,
carries it"*. So the evidence that the **H/L split specifically** is the active ingredient is weak at
puzzle scale and absent at LM scale. That does not make Q2 a bad experiment — it makes it a genuinely
open one, which is a better reason to run it.

---

## 1. Spec diff (Q1)

| # | Idea | Status |
|---|---|---|
| 1 | Systolic arrays not LPUs | Out of scope (Tier S/O, `02 §1`) |
| 2 | Parallel attention + FF | **In spec**, `01 §4.1`; **measured** at L2: −1.73 % vs L1, where H2 budgeted ≤ +1 % (`107`) |
| 3 | 2:4 from the start, FP4 | **In spec**, `01 §8`, ADR-019 (weight-only, fixed early mask); L10a/L10b. FlexMoE half **unverified** |
| 4 | Expert = one hardware unit | **In spec** as the organizing rule; ADR-018, `02 §2`. DC-scale for its systems half |
| 5 | N × U fabric | Out of scope locally; `02 §3`, S3 |
| 6 | One repeated middle layer | **In spec and being built**: L5, H6. **Q2 proposes a change to it — see below** |
| 7 | Variable iteration count | **In spec**, `01 §4.7`, L6, H7. **I24 now open** on token-choice vs expert-choice depth |
| 8 | Multi-layer experts | **In spec**, L7b, H8 |
| 9 | Multicast + co-activation placement | Out of scope locally; S3 |
| 10 | One installation per DC | Out of scope; S/O |
| 11 | Continuous execution | Out of scope locally; ADR-016 fixes training as lockstep. **Interacts with I24** |
| 12 | Two latent streams | **Closed by the author and ADR-020** — see Q8 |
| 13 | Native MTP | **In spec and running now**: L3, H13, ADR-012. `108` in flight |
| 14 | No prefill/decode disaggregation | Out of scope locally; S5 |
| 15 | Indexed attention | **In spec**, `01 §6`, L9, H15b |
| 16 | ~4 GB HBM per chip | Out of scope; S/O |
| 17 | Per-iteration retrieval | **In spec with a local adaptation**: CLAUDE.md says per-iteration external retrieval is not feasible here, so L8 = GPU product-key memory and L8b = offline-precomputed neighbours |
| 18 | Per-vector QLoRA, bespoke experts | **In spec**, L11, H18. Q7's FlexMoRE framing **unverified** |
| 19 | One giant DC | Out of scope; O |

**Contradictions with the spec, flagged explicitly as Q1 asks:**

1. **Q6 (objective) contradicts ADR-014.** See below — this is the serious one.
2. **Q8 (#12) contradicts ADR-020 and the author's own answer.**
3. **Q2's H/L split contradicts the note's P6** ("a *single* middle layer"), though not fatally — see
   below. It also sits against ADR-025/026/027, which derived every width from the single-block
   budget.
4. **Q5's "route to different experts on different iterations" is already the spec's behaviour**, not
   a proposal — `01 §4.4` conditions the router on the depth embedding $e_j$, and E-Q2 is the
   ablation that turns it off.

---

## 2. Q6 — the objective. **Do not adopt it.**

This is the one that would do real damage if taken at face value, so it goes before the rest.

The brief observes, correctly and importantly, that roughly half of HRM-Text's gain came from its
training objective (response-only loss + PrefixLM on 40 B instruction tokens) rather than its
architecture, and asks whether we should adopt it.

**No, and the reason is not conservatism.**

- **ADR-014 fixes the corpus, the tokenizer and the held-out set**, and says in terms that a change is
  a new ADR *and a full re-tokenisation*. The held-out FineWeb-Edu loss is the metric every ADR-013
  band is expressed in.
- **It would invalidate every number the ladder has produced.** $L_{ref}$ = 3.3360 from `100`, and
  `106`/`107`/`108` are all measured against it. An objective change makes them incomparable — not
  worse, *incomparable*, which is more expensive.
- **HRM-Text reports no held-out LM loss at all.** Adopting its objective would move us onto the
  metric space where its own evidence lives, and that space is benchmark accuracy, which ADR-013 does
  not define verdicts in and which an 80 M model largely cannot resolve. ADR-014 chose a loss-based
  suite precisely so that a model this size scores above chance.
- **The note says nothing about objectives**, so this is not a point we would be dropping. It is a
  new axis.

**The cost of saying no**, stated as Q6 asks: if HRM-Text's objective really carries half the gain,
then any architecture comparison we run is measuring the smaller half of what is available. That is
a real limitation and it should be written down — but it is a limitation of *scope*, not an error.
The project's question is which of the note's 19 points pay off, not how to build the best 1 B model.

**If the team wants it anyway**, the honest form is a separate ADR and a separate programme, with its
own corpus, its own baselines retrained, and no cross-comparison to the current ladder. That is
weeks, not days.

---

## 3. Q8 — the split latent. **Already closed; don't re-open without new evidence.**

ADR-020 is accepted, on the author's own answer to Q1: *"nice idea, not standard, easier to avoid."*
CLAUDE.md is explicit that the author's answers settle **intent**. The brief's own cross-reference
table agrees the idea has no precedent in the reviewed papers.

Answering the sanity-check anyway, since it was asked:

- **"Halves params at same FLOPs"** — ADR-026 found the opposite at our scale. Two streams *overspend*
  the budget with **zero experts**: the non-expert terms alone exceed it. That is why L4 was moved to
  the end of the ladder and given a 2× budget against a √2·$d$ comparator.
- **"Is vector 2 just a second residual stream with a stop-gradient from attention?"** Not stop-grad —
  ADR-002 (stream visibility) governs this. It is a second stream that attention does not *read*;
  gradients still flow to it through the shared weights.
- **What breaks with #13** — nothing; MTP heads read the single stream by default under ADR-020, and
  read $p$ only if two streams are on.
- **What breaks with #7's global-final-only attention** — this is the sharp one the brief is right to
  ask about. Under ADR-021/022 the global range is the final-vector cache $\mathcal{G}$, one entry per
  token. With two streams there are two candidate final vectors per token and the spec does not say
  which enters $\mathcal{G}$. **That is a genuine gap**, and it is dormant only because ADR-020 made
  two streams optional. If L4 is ever run it must be answered first.

---

## 4. Q2 — the H/L split. **The best idea in the brief, and it fits the budget.**

Proposed design, keeping the note's "one middle layer holds all the experts":

- **L (fast)** = the middle block exactly as specified today — attention + MoE, depth-conditioned,
  run every iteration. All experts live here, so P4's "expert = one unit" rule is untouched.
- **H (slow)** = a small **dense** block with its own weights, run once every $K$ iterations, reading
  and writing the same residual. No experts, so it does not disturb expert sizing or placement.
- Depth conditioning $e_j$ stays on both; ADR-010/011 ragged semantics are unaffected.

**Cost, computed with `costmodel/` rather than estimated** (`small`, $d$ = 768, $r_{max}$ = 8,
$d_{ff}$ = 896, budget 0.6 GFLOP/token; L5 today is 5.43e8, so there is ≈ 10 % headroom):

| $K$ | $d_{ff}$(H) | extra params | extra train FLOPs | total | fits 0.6e9? |
|---|---|---|---|---|---|
| 8 | 1536 | +5.1 M | +5.6 % | 5.74e8 | **yes** |
| 8 | 2048 | +6.3 M | +7.0 % | 5.81e8 | **yes** |
| 8 | 3072 | +8.7 M | +9.6 % | 5.95e8 | **yes, just** |
| 4 | 1536 | +5.1 M | +11.3 % | 6.04e8 | no — trade $r$ or $k$ |
| 2 | 2048 | +6.3 M | +27.8 % | 6.94e8 | no |

**So H-every-8 fits the existing budget with nothing traded**, at +6.3 M parameters on a 77.4 M model.
$K$ = 4 needs an explicit trade and `06 §3` says to make such trades explicit rather than silently
exceed the budget. HRM-Text's own ratio is H2×L3, i.e. H runs every 3 L steps — between our $K$ = 2
and $K$ = 4, both of which cost us real budget. **That mismatch is itself worth recording**: their
ratio is affordable at 1 B where the skeleton is proportionally cheaper, and ours is not.

For context on where the FLOPs go today: the middle block is **58 %** of forward FLOPs across its 8
iterations, the 4 skeleton blocks are the other 42 %.

**Recommended as L5e**, beside L5 rather than after it, testing the one thing that is actually open:
does a slow/fast split beat one shared loop *on held-out loss at matched FLOPs*, which is the question
HRM-Text did not answer. Run at $K$ = 8 first, since it is free.

---

## 5. Q3 — iteration control. Minimal ablation.

The spec already stages this (`01 §4.7`): fixed → ACT → RL. The brief's evidence adds a reason to
question the middle step, and I24 (opened today) adds another.

Recommended order, cheapest first:

1. **Fixed $r$ = 8** — L5, already running as the ladder's next rung. Establishes the recurrent model.
2. **Fixed $r$ sweep** at $r$ ∈ {4, 8, 12} — nearly free, and it is the control H7 needs. Without it,
   "learned depth saves 30 % FLOPs" has no matched-FLOPs comparator, because the honest comparison is
   against a *fixed* $r$ at the same mean depth, not against $r_{max}$.
3. **Confidence threshold** (no learned parameters): halt when the output distribution's entropy or
   top-1 margin stabilises between iterations. HRM-Text dropped ACT "to keep the design simpler" and
   this is the simpler thing that is not nothing.
4. **ACT halting** — L6 as specified.
5. **Expert-choice depth router** — L6c, per I24.

Steps 2 and 3 are new; both are cheap and both are controls rather than features. **H7 should not be
scored without step 2.**

---

## 6. Q4 — the KV scheme

Attention pattern under ADR-021/022, which is v2's answer and already accepted:

- Iterations $1..r-1$: **local only**, a sliding window $W$ = 512 over that iteration's own transient
  cache.
- Final iteration: **global** over $\mathcal{G}$, the final-vector cache, one entry per token.

Cache size, per token, against the alternative the brief cites: HRM-Text keeps a **separate KV cache
per recurrent step** — for a 32-layer-equivalent that is ~128 layers' worth. Our scheme stores
$r_{max}$ transient windows of $W$ entries plus **one** persistent $\mathcal{G}$ entry per token. The
persistent term is what scales with context, and it is $1/r$ of a per-iteration scheme.
`costmodel/memory.py` has `kv_reduction_factor` for the exact figure per config; use it rather than
the sketch here.

**Where the serialization bites** — the brief asks precisely, so precisely:

- **Training**: it does not. ADR-016 fixes training as lockstep, and segment-recurrent training with
  stop-gradient memory (ADR-023) means $\mathcal{G}$ for segment $s$ is already complete when segment
  $s+1$ starts.
- **Prefill**: it does, and this is the real cost. A token's $\mathcal{G}$ entry exists only after its
  *final* iteration, so a later token's global attention cannot see an earlier token until that
  earlier token has finished all $r$ iterations. ADR-024 (still proposed) exists for exactly this and
  offers two modes.
- **Speculative decode (#13)**: it bites here too, and the spec does not currently say how. Drafted
  tokens have no $\mathcal{G}$ entry until verified, so a draft cannot attend globally to its
  predecessors within the same draft window. **Gap — worth an open question.**
- **The "different batch" argument** holds only under continuous execution (P11), where other
  sequences fill the bubble. It does not help single-stream latency, and it is a throughput argument
  dressed as a latency one.

---

## 7. Q5 — MoE inside recurrence

Mostly already decided, and the brief's framing of one part is backwards:

- **Load balancing**: ADR-004 and the implementation use **aux-loss-free bias** (DeepSeek-V3 style)
  with a small Switch-style auxiliary loss behind it. Measured working at L1: every router 7.99–8.00
  of 8 effective experts, 0 dead (`106/routing.json`). This is settled and does not need HRM-MoE's
  recipe.
- **Expert count / top-k / width**: fixed by ADR-018 and ADR-025 from the *budget*, not from taste —
  $N_e$ = 8, $k$ = 2, $d_{ff}$ = 896 at `small`, with the parameter-matched $N_e$ = 128, $k$ = 4
  variant at $d_{ff}$ = 448 for H6. HRM-MoE's 64×top-8×512 is a different budget; adopting its shape
  would break matched-FLOPs comparison, which is the whole method.
- **"Is routing to different experts on different iterations desirable?"** — **this is already the
  spec's behaviour**, not an open choice. `01 §4.4` conditions the router on the depth embedding
  $e_j$, so a token can and does route differently per iteration. The ablation that *ties* routing
  across iterations is already scheduled as **E-Q2** (router without $e_j$). Nothing to add.

---

## 8. Q7 — rank-heterogeneous experts. **Conditional on verifying FlexMoRE.**

If FlexMoRE's result holds, then as Q7 suggests it is two things at once: a memory-fitting device, and
an implementation of P18.

**Where it conflicts with the note**, which Q7 asks directly: P4 is the note's *organizing rule* —
expert size = the smallest FF that fills one hardware unit, so all experts are the same width, so
placement, multicast and the N×U fabric all have a uniform object to schedule. **Rank-heterogeneous
experts break that uniformity by construction.** That is not a detail; P4 is the point from which
P5, P9 and P16 hang.

So the honest framing is not "adopt FlexMoRE" but: *does heterogeneous rank buy enough quality to be
worth giving up the uniform expert?* That is a real question and it belongs in `00` as a disagreement
with the note, with the systems cost priced in `sim/` before the quality half is run. **Not a Phase-1
item.**

---

## 9. Q9 — prioritised experiments

Ordered by expected information per GPU-hour. Wall-clock from the **measured** rates in `018`/`026`
(effective 30.3 TFLOPS aggregate; a `screen` rung at 0.6 GFLOP/token is ≈ 5.5 h).

| # | Experiment | Hypothesis | Baseline / control | Metric | Est. |
|---|---|---|---|---|---|
| 1 | **Fixed-$r$ sweep**, $r$ ∈ {4, 8, 12} | H7's missing control: learned depth must beat *fixed depth at the same mean*, not $r_{max}$ | L5 at $r$ = 8 | held-out loss at matched mean depth | ≈ 16 h (3 runs) |
| 2 | **L5e: H/L split**, $K$ = 8, $d_{ff}$(H) = 2048 | Two weight sets beat one shared loop **on loss** — the question HRM-Text left open | L5, matched FLOPs (fits budget) | held-out loss | ≈ 6 h |
| 3 | **L6c: expert-choice depth router** | I24: expert-choice > token-choice by 2.6 pts in MoR | L6 (ACT), same budget | loss + mean depth | ≈ 6 h |
| 4 | **L3 at `small`** | ADR-030: is MTP budget-starved or broken? | L2 at `small`, 2 seeds | acceptance + main-loss Δ | ≈ 17 h |
| 5 | **E-Q2: router without $e_j$** | Does per-iteration routing matter? | L5 | held-out loss | ≈ 6 h |
| 6 | **L7b: $L_e$ = 2** | H8, already scheduled; also the recovery variant for H4 | L5 | loss + fabric bytes/FLOP | ≈ 7 h |

**#1 is first and it is not a new idea — it is a control the ladder currently lacks.** Everything else
about H7 is uninterpretable without it, and it is the cheapest thing on the list.

Deliberately excluded: anything requiring the objective change (Q6), anything requiring unverified
sources (Q7), and #12 (ADR-020).

---

## 10. Q10 — the author's meta-request

The note asks that anyone in the area have a good reason for not doing each of these things. That
discipline is already the repo's: `00` holds the point-by-point critique with a claim-status table,
and `04` holds the ADRs. Points we are *not* testing are recorded there with their reason — Tier S/O
for the DC-scale ones, the author's own answer for #12.

What this brief adds to that record: **#12's reason is now doubly documented** (author's answer *and*
ADR-026's budget arithmetic), and **#6 gains a live challenger** in L5e that the note itself did not
consider.

---

## What I recommend

1. **Verify FlexMoRE and FlexMoE into `05`**, or drop Q7 and the FlexMoE half of Q1 #3 until someone
   has. One reading session.
2. **Add the fixed-$r$ sweep** (#1 above) to the ladder as a control. It is cheap and H7 needs it.
3. **Add L5e** (H/L split at $K$ = 8) as a variant beside L5. It fits the budget, it tests the brief's
   best idea, and it asks the question HRM-Text did not answer.
4. **Do not adopt the HRM-Text objective**, and record the scope limitation that follows.
5. **Leave #12 closed.**

Items 2 and 3 are ADR-shaped and should be written as one ADR when L5 has run, not before — L5's own
result changes how interesting L5e is.
