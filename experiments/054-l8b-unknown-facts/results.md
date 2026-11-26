# 054-l8b-unknown-facts — rig for ADR-076 (I44's cause): with a table that holds facts no base can know, L8b-pv's memory is worth **0.0177 nats at `medium` and 0.0215 at `small`** on the covered first sentence, about twice its value on real facts at both sizes → ADR-076 §4's reading is **mixed**: the memory's fact-specific value does not shrink with base size, but it is small, and "the larger base already knows the table" does not account for `181`

Eval only, no training. `scripts.data.build_unknown_facts` then `scripts.analysis.eval_unknown_facts`
on six existing checkpoints, run by `queue_screen_26` on 2026-11-26 20:34–20:40 (build 296 s on
four RTX 3060; scoring 24–75 s per checkpoint, one card each), code at the queue's gate commit
`0f1e85a`. Commands and pool in `config.yaml`; per-checkpoint outputs `uf_*.json`, the scored
reading in `report.json`. Seeds: pool and pseudo-names seed 2; checkpoints are `178`/`179`
(L8b-pv `small`, seeds 0/1), `181` (L8b-pv `medium`), bases `132`/`133` and `180`.

**Bears on:** **H17**, without changing its verdict (ADR-075: supported at `small`, silent at
`medium`; ADR-076 §5). It answers **I44** (why the memory's value shrank at `medium`).

## What was measured

2,000 Wikipedia articles from outside the retrieval table, each as a title plus first sentence,
in two versions: *real*, and *renamed* (the title replaced by a generated pseudo-name, so the
predicate is real and the binding to the name is new). Each L8b-pv checkpoint is scored on the
covered first-sentence tokens under the trained table $T_0$ and under a table that also holds
the document's own sentence. $S$ = loss under $T_0$ − loss under the holding table: what having
the fact in the memory is worth to the same model on the same tokens.

Pipeline, no model: coverage 74.7 % of first-sentence tokens (real) and 82.3 % (renamed); of
covered tokens, 98.0 % and 95.9 % retrieve their own sentence when the table holds it.

Floor: 4 × max(0.0030, article-bootstrap SE), ADR-055 §6; 0.012 where the SE is small.

| | `small` (`178`/`179` vs `132`/`133`) | `medium` (`181` vs `180`) | floor |
|---|---|---|---|
| base loss, real | 3.0198 | 2.5470 | |
| base loss, renamed | 3.0861 | 2.7136 | |
| base: renamed − real (what the base knows) | +0.0664 | +0.1666 | 0.037 / 0.035 |
| **$S_{real}$** | 0.0108 | 0.0081 | 0.012 |
| **$S_{ren}$** | **0.0215** | **0.0177** | 0.012 |
| $S_{ren} − S_{real}$ | 0.0107 | 0.0096 | 0.012 |
| L8b-pv under $T_0$ − base, real | −0.0375 | −0.0096 | 0.012 / 0.015 |
| L8b-pv under the holding table − base, real | −0.0482 | −0.0177 | 0.012 / 0.015 |
| L8b-pv under $T_0$ − base, renamed | −0.0133 | +0.0115 | 0.012 / 0.015 |
| L8b-pv under the holding table − base, renamed | −0.0348 | −0.0062 | 0.012 / 0.015 |

Bootstrap SEs of the $S$ rows are 0.0003–0.0004; of the arm − base rows 0.0027–0.0037.

## Reading (ADR-076 §4, set before the run)

| rule | measured | met? |
|---|---|---|
| *premise*: `180`'s (renamed − real) exceeds `132`/`133`'s past the floor | 0.1666 − 0.0664 = +0.1002, floor 0.0155 | **holds** |
| supported (i): at `medium`, $S_{ren}$ ≥ floor | 0.0177 ≥ 0.012 | yes |
| supported (ii): at `medium`, $S_{ren} − S_{real}$ ≥ floor | 0.0096 < 0.012 | **no** (21 SE from zero, 0.0024 short of the floor) |
| supported (iii): $S_{ren}$ at `medium` not below `small`'s by more than the floor | 0.0215 − 0.0177 = 0.0038 < 0.012 | yes |
| not sufficient: at `medium`, $S_{ren}$ < floor | 0.0177 | no |

**Reading: mixed**, as `report.json` records. Written up as measured:

- **The `medium` model does read a fact it cannot know.** $S_{ren}$ is above the floor at
  `medium` and within 0.004 of `small`'s. ADR-076's "not sufficient" branch (training taught
  the larger model to ignore the memory) is ruled out.
- **Unknown facts are worth about twice known ones, at both sizes** (0.0215 vs 0.0108; 0.0177
  vs 0.0081). The difference is far from zero statistically and 0.002 short of the 0.012
  floor, which is why the "supported" branch is not met.
- **The fact-specific value hardly depends on base size.** $S_{real}$ falls from 0.0108 to
  0.0081 between `small` and `medium`. That is 0.003 of a gap that shrank by 0.030.
- **What shrank at `medium` is the part that is not the fact.** Under the trained table, which
  holds none of these sentences, L8b-pv beats its base by 0.0375 at `small` and by 0.0096 at
  `medium` on real documents. That gain comes from retrieving related sentences, and it is the
  one a larger base replaces. On renamed documents it turns into a cost at `medium` (+0.0115,
  inside the floor).
- **The memory recovers little of what the base lacks.** Renaming costs the `medium` base
  0.167 nats; holding the fact gives back 0.018, about a tenth. Values are one sentence vector
  (ADR-057 §8), so the name and exact tokens are not available to the reader.

**I44, answered:** `181`'s shrunken gain is not explained by the larger base already knowing
the table's facts. The memory's fact-specific contribution was small at both sizes; most of the
`small` pair's first-sentence gain was related-content retrieval, and that is what a larger base
makes redundant. H17's verdict stays ADR-075's.

## Limits (ADR-076 §6)

The models were trained reading a table of learnable facts; this measures what they read as
trained. The real and renamed scores are over different token sets (51,424 and 60,149 covered
tokens; the renamed sentence contains the pseudo-name), so "renamed − real" for a base mixes
not knowing the fact with the cost of the name itself; the premise is read on the difference
between sizes, which shares that term. `medium` is one seed. No clause of ADR-055 §6 or
ADR-059 is re-read on this pool.

## Interpretation

Swapping the retrieval table at eval time shows that L8b-pv, at both sizes, uses a retrieved
fact it could not have learned: about 0.02 nats on the covered first sentence, twice what a
known fact is worth, and nearly the same at `medium` as at `small`. So the reader was not
trained out of using its memory, and its fact-specific value does not shrink with the base.
But that value is a tenth of what the base is missing, and it was never the bulk of the
`small` pair's gain: most of that came from related sentences under the ordinary table, which
is worth 0.037 at `small` and 0.010 at `medium`. By ADR-076's pre-set rules the result is
mixed (one of three "supported" conditions misses its floor by 0.002). It bears on H17 as an
explanation of `181` and changes no verdict. If the retrieval arm were ever reopened, the
lever this points at is the value format (a sentence vector cannot hand over a name), not the
table's contents or the base's size.
