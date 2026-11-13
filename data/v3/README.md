# Dataset v3: v2 plus same-level targets

Dataset v3 is [dataset v2](../v2/README.md) with one change to the target rule. For base and syntax, a pair whose simplification stays in the original's READ-IT level gets that level as its target, instead of `any`. The pairs, splits, columns, lexical targets and test sets are v2's.

| | |
|---|---|
| Source | dataset v2 (`data/v2/training_set_v2.zip`), itself built from the Wikipedia part of IMPaCTS, revision `489c78e` |
| Training pairs | **89,460**, the same as v2: 88,599 `train` + 861 `validation` |
| Changed | `tgt_base` in 10,767 pairs, `tgt_syntax` in 4,302 pairs (and `ctrl_tokens`, `ctrl_tokens_with_source`) |
| Test sets | the v1 test sets, byte for byte (`*_test_set_v3.csv`) |
| Built by | `python scripts/build_dataset_v3.py` (about 20 seconds, after `build_dataset_v2.py`) |
| Explore | [data explorer](https://michelepapucci.github.io/simplification-journal/#data=v3), dataset "v3" (the two "same level" slices show the relabelled pairs) |

## Why

In v1 and v2, a dimension gets a target only if the simplification falls to a lower level and the score drops by at least READ-IT's noise margin. Otherwise its target is `any`, so `any` means two things:
- "free": target dropout at training time, and the free dimensions at test time;
- "stayed where it was".

The models never see a token that asks for a light edit, and they over-simplify mild requests. At base 70–90 and syntax 70–90, v1 and v2 outputs land at about 53–68 ([results](../v1/experiments/README.md#results)).

The CLiC-it 2026 data has these tokens: 18.2% of its base tokens and 11.0% of its syntax tokens are at the original's own level. The CLiC-it 0.6B is the only model that follows mild base requests, and it does so on test originals it never saw as well:

| Base requests at 60–80, on test originals it never saw | RMSE | ρ within original |
|---|---|---|
| CLiC-it 2026 0.6B | 19.0 | 0.50 |
| v1 · targets | 22.3 | −0.13 |

Its other advantages come from having trained on 93–98% of the test originals, and from its lexical token being a base token. v3 does not copy either of them.

## The rule

| How the dimension moved in the pair | v1 / v2 | **v3** (base, syntax) | CLiC-it 2026 |
|---|---|---|---|
| Dropped to a lower level, by at least the margin | that level | that level | that level |
| **Stayed in the original's level** | `any` | **the original's level** | that level |
| Stayed in the original's level, but rose by at least the margin | `any` | `any` | pair excluded |
| Crossed into a lower level within the margin | `any` | `any` | that level |
| Rose to a higher level | `any` | `any` | pair excluded |

The margins are v1's: base 5.7, lexical 3.8, syntax 17.6. The same-level pairs are light edits in READ-IT terms: the median change is −1.4 points for base and −1.9 for syntax. Unlike CLiC-it's same-level tokens, none of them are copies, because the release has no identity pairs. In CLiC-it, copies make up 20% of the base same-level tokens and 33% of the syntax ones.

**Lexical keeps v2's targets.** The same rule would relabel 53% of all pairs as "keep lexical at its level", mostly at 30. That would be a different experiment. It could also help lexical, by giving pairs where base drops and lexical stays.

## What changes

Training targets by level (training split):

| Level | Base v2 | Base v3 | Syntax v2 | Syntax v3 | Base CLiC-it | Syntax CLiC-it |
|---|---|---|---|---|---|---|
| 50 | 4,889 | 6,378 | 8,534 | 8,697 | 4,077 | 5,769 |
| 60 | 2,163 | 2,990 | 3,130 | 3,771 | 2,187 | 5,266 |
| 70 | 717 | 1,177 | 511 | 1,605 | 1,149 | 4,112 |
| 80 | 90 | 272 | 0 | 629 | 549 | 1,659 |
| 90 | 0 | 26 | 0 | 109 | 226 | 32 |

Most of the new targets are at levels 20–40, where most originals sit. At the top levels, v3 roughly matches CLiC-it's base coverage at 70, has half of it at 80, and has syntax targets at 80–90 for the first time.

**Side effect on the training mix.** Target dropout (p = 0.3 per set target, at least one kept) is unchanged. Per epoch, examples with a single target fall from 58% to 50%, and lexical-only examples from 5.4% to 3.7%, because many lexical pairs now also carry a base or syntax target. 20,850 training pairs have all three targets set, against 17,296 in v2.

## Files

| File | Content |
|---|---|
| `training_set_v3.zip` | v2's columns, with the v3 `tgt_base`, `tgt_syntax`, `ctrl_tokens` and `ctrl_tokens_with_source` |
| `{base,lexical,syntax}_test_set_v3.csv` | identical to `data/v2/*_test_set_v2.csv` and `data/v1/*_test_set_v1.csv`. Their target columns still follow the v1 rule; the evaluation requests do not use them. |
| `construction_v3.json` | the rule, the margins, the counts, the targets by level and the per-epoch composition |

## Experiment

`v3_targets` is Qwen3-0.6B-Base trained on v3 with exactly the recipe and evaluation of `v2_targets`: three target tokens, target dropout 0.3, LoRA r 32, effective batch 64 (4 × 16), 3 epochs (4,155 steps). Since only the labels differ, `v2_targets` is its direct comparison.

- Run: `bash data/v1/experiments/run_v3_targets.sh`.
- Results: [`data/v1/experiments/README.md`](../v1/experiments/README.md#dataset-v3-same-level-targets).

**Result: mild base requests improved; syntax did not.**
- Base requests at 70–80 land at 66.9 instead of v2's 62.3 (+4.6, 95% interval +3.0 to +6.1).
- At base 60–80, RMSE goes from 20.8 to 19.8 (CLiC-it: 18.9); within-original ρ goes from 0.68 to 0.73.
- On test originals CLiC-it never saw, base RMSE at 60–80 is 20.4, against 19.0 for CLiC-it and 22.3 for v1 and v2. Copies stay at 1.0%.
- Base 80–90 did not move (68.6). v3 has only 26 targets at 90 and 90 pairs that go from 90 to 80.
- Syntax is unchanged (requests at 70–80: 56.6, v2 57.0). Same-level targets teach "stay here", but the evaluation asks for one level below. For syntax, the v1 rule labels almost no one-level drops at the top, because the margin (17.6) is wider than a level: originals at 80 have 77 labelled drops to 70 and 1,406 left `any`.
- Lexical and the joint evaluation are unchanged.

The predictions below were right on base at 70–80, base 60–80, ρ, levels 0–50, copies and joint. They were wrong on base 80–90 and syntax 70–80.

### Predicted before the run (2026-11-11, while training)

| Main evaluation | v2 · targets | CLiC-it 2026 0.6B | Prediction for v3 |
|---|---|---|---|
| Base: mean achieved at requests 70–80 / 80–90 | 62.3 / 67.8 | 71.7 / 79.7 | **65–69 / 70–76** |
| Base 60–80: RMSE / share inside | 20.8 / 12.1% | 18.9 / 13.0% | 19–20 / 12–16% |
| Base: ρ within original | 0.68 | 0.76 | 0.70–0.74 |
| Syntax: mean achieved at requests 70–80 / 80–90 (46 requests) | 57.0 / 53.4 | 68.1 / 81.1 | **60–64 / 58–70** |
| Base and syntax at requests 0–50 | | | within 1.5 points of v2 (inside) |
| Copies, base requests | 0.8% | 3.9% | 1–3% |
| Lexical: inside / ρ within | 11.6% / 0.06 | 14.4% / 0.13 | unchanged or slightly worse (10–12% / ≤ 0.08) |
| Joint: all three inside | 3.3% | – | 3–4.5% |

- **Falsified if** base outputs at requests 70–80 stay at 64 or below *and* syntax at 70–80 stays at 59 or below. That would mean same-level targets do not explain CLiC-it's top-level advantage. The remaining candidates would be its identity pairs, its 1.75× more training examples, or one token per example.
- **Risk:** more light edits or near-copies on the highest request for each original, which is one level below it. Watch the copy rate and the share inside at that request.
- **Confidence:** about 60% that base at 70–80 rises by at least 3 points. The v2 prediction was wrong on its main claim, so treat this one cautiously.
