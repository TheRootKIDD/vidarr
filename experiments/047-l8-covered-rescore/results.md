# 047-l8-covered-rescore — `173` (L8) on wiki-lead with the covered masks (ADR-059 §4)

**Bears on:** H17. The re-read of `173`'s verdict under ADR-059 is **weakens, unchanged**.

No training. `173`'s checkpoint was scored with `--covered` on `cuda:0` with the rig idle after `queue_small_11`, at code `a6c429a`. `173`'s full splits reproduce its own `wiki_lead.json` exactly (3.1328 / 3.4320 / 3.2748). It was compared against `045`'s covered re-score of `110` with the eval script's `score()`, and the result is in `wiki_lead_compare_173_vs_110.json`. The file is here because `173`'s directory is closed.

| split | `110` | `173` | Δ (floor) | beats? |
|---|---|---|---|---|
| in-table first sentence, covered (ADR-059's clause) | 3.1963 | 3.2244 | **+0.0281** (0.0147) | no |
| in-table first sentence, own retrieved | 3.1856 | 3.2122 | +0.0266 (0.0149) | no |
| in-table first sentence, full | 3.1307 | 3.1328 | +0.0021 (0.0134) | no |
| in-table rest of lead | 3.4080 | 3.4320 | +0.0240 (0.0121) | no |
| out-of-table lead | 3.2613 | 3.2748 | +0.0135 (0.012) | no |

Neither in-table clause beats its floor, so the verdict is *weakens*, on both the full-split rule and ADR-059's (`verdict_h17_covered`). The covered tokens are worse than `110`, as `174`'s were (+0.0187). With ADR-059 §4's re-reads of `174` and `175`, every earlier H17 verdict stands.

**Recorded:** a mistyped `--compare 173-l8-screen …` during this re-score wrote a stray file `x` into `173`'s closed directory. It was deleted at once, uncommitted, and nothing of `173`'s committed files changed.
