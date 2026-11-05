# 046-l5-small-wiki-lead — wiki-lead with the covered masks for the L5 `small` pair (ADR-058 §2)

**Bears on:** H17, as the matched arm of ADR-058's L8b `small` pair; no verdict of its own (*silent*). It also bears on the wiki-lead rule itself (I42) and on I41.

No training. `python -m scripts.analysis.eval_wiki_lead --id <run> --covered --out experiments/046-l5-small-wiki-lead/wiki_lead_<n>.json`, at `0c32ba0`. `132` ran on `cuda:0` and `133` on `cuda:1`, with the rig otherwise idle. 82 s each. The pools and retrieval masks are ADR-057's.

## Numbers

| split | tokens | `132` (s0) | `133` (s1) | pair mean | 133 − 132 | bootstrap SE | floor | beats? |
|---|---|---|---|---|---|---|---|---|
| in-table, first sentence | 70,952 | 2.8422 | 2.7767 | 2.8095 | **−0.0656** | 0.0037 | 0.0148 | **yes** |
| in-table, rest of lead | 77,911 | 3.1383 | 3.1251 | 3.1317 | **−0.0132** | 0.0032 | 0.0129 | **yes** |
| out-of-table lead (control) | 148,298 | 2.9866 | 2.9442 | 2.9654 | **−0.0424** | 0.0025 | 0.012 | **yes** |
| first sentence, covered | 55,516 | 2.8950 | 2.8929 | 2.8940 | −0.0020 | 0.0041 | 0.0163 | no |
| first sentence, own retrieved | 54,597 | 2.8833 | 2.8817 | 2.8825 | −0.0017 | 0.0042 | 0.0166 | no |
| first sentence, not covered (derived) | 15,436 | 2.6527 | 2.3585 | 2.5056 | **−0.2942** | | | |
| held-out (eval's 976 windows) | | 3.0732 | 3.0669 | | −0.0063 | | | |
| held-out (final, full, `summary.json`) | | 3.0778 | 3.0718 | 3.0748 | −0.0060 | | | |

The pipeline reproduces `045`: coverage 78.2 %, accuracy 98.3 %. `seed_spread_133_vs_132.json` holds the full `score(133, 132)`.

## Interpretation

These are the two seeds of **one** config. Seed 1 beats seed 0 past ADR-055 §6's floor on all three full wiki-lead splits, and the rule's verdict between them is **"supports"**. So the full-split floors, 4 × max(0.0030, article-bootstrap SE), do not cover seed-to-seed variation on wiki-lead. The bootstrap resamples articles, not training runs.

The variation is concentrated:
- **Tokens before the first retrieval boundary** (the title and the lead's opening words, 15,436 tokens) differ by 0.29 nats between the seeds.
- **Covered first-sentence tokens** differ by 0.002, a ninth of their floor.

This bears on earlier results:
- **`175`'s first-sentence half.** It was read as robust because its covered tokens alone beat `110` by −0.037. That stands: −0.037 is 18× this covered seed spread.
- **`175`'s rest-of-lead miss** (−0.0105) and its control gain (−0.0276) are both within the seed spread seen here (−0.013, −0.042).
- **I41.** The ≈ 0.1-nat edge of `174`/`175` over `110` on uncovered tokens is a third of the uncovered seed spread here. So I41's likely answer is seed noise, not an L8-family effect, though this is at `small`, not `screen`.
- **`173`/`174`'s *weakens*.** Neither in-table split beat its floor, so they are unaffected in direction. Their full-split deltas are also inside this noise.

ADR-058's pair rule averages two seeds per side, but its floor still has no seed term. Whether the `small` verdict should use the covered splits, or add a seed term, is opened as **I42** for the user. `176`/`177` run unchanged in the meantime: the rule only reads their output, and either answer needs the same runs.
