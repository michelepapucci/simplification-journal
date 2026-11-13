# Dataset v4: v3 plus syntax crossings

Dataset v4 is [dataset v3](../v3/README.md) with one change to the syntax targets. A pair whose syntax falls into a lower READ-IT level by less than the noise margin gets that level as target, instead of `any`. Syntax then has a target in every pair except those where it rose. The pairs, splits, base and lexical targets and test sets are v3's.

| | |
|---|---|
| Source | dataset v3 (`data/v3/training_set_v3.zip`), itself built from v2 and the Wikipedia part of IMPaCTS, revision `489c78e` |
| Training pairs | **89,460**, as in v2 and v3: 88,599 `train` + 861 `validation` |
| Changed | `tgt_syntax` in 9,833 pairs (and `ctrl_tokens`, `ctrl_tokens_with_source`) |
| Test sets | the v1 test sets, byte for byte (`*_test_set_v4.csv`) |
| Built by | `python scripts/build_dataset_v4.py` (about 20 seconds, after `build_dataset_v3.py`) |
| Explore | [data explorer](https://michelepapucci.github.io/simplification-journal/#data=v4), dataset "v4" (the slice "Syntax, crossed within the margin" shows the relabelled pairs) |

## Why

v3 gave base and syntax a target at the original's own level when the simplification stays in it. Mild base requests improved: requests at 70–80 land at 66.9 instead of 62.3. Syntax did not change: requests at 70–80 land at 56.6, against 68.1 for the CLiC-it 2026 0.6B ([v3 results](../v1/experiments/README.md#dataset-v3-same-level-targets)).

The cause is the noise margin. Since v1, a drop gets a target only if it is at least the READ-IT noise margin. For syntax that margin is 17.6 points, wider than a level. So a simplification that goes one syntax level down usually does so by less than the margin and is labelled `any`. At the top of the scale the model almost never sees "one level below":

| Training pairs, original's syntax level | One level below, labelled in v3 | One level below, in v4 | Same level (v3 and v4) |
|---|---|---|---|
| 60 | 102 | 1,202 | 641 |
| 70 | 145 | 2,389 | 1,094 |
| 80 | 77 | 1,483 | 629 |
| 90 | 0 | 417 | 109 |

The evaluation's syntax requests at 70–80 come from originals at 80 (813 of 859 requests), and those at 80–90 from originals at 90. The CLiC-it 2026 data labels these crossings with the lower level. It has 4,112 syntax tokens at 70 and 1,659 at 80.

For base, the margin (5.7) is narrower than a level, and about 90% of one-level drops from originals at 70–90 are already labelled. Base is left as in v3.

## The rule

| How syntax moved in the pair | v1 / v2 | v3 | **v4** | CLiC-it 2026 |
|---|---|---|---|---|
| Dropped to a lower level, by at least the margin | that level | that level | that level | that level |
| **Dropped to a lower level, by less than the margin** | `any` | `any` | **that level** | that level |
| Stayed in the original's level | `any` | the original's level | the original's level | that level |
| Rose to a higher level | `any` | `any` | `any` | pair excluded |

The relabelled pairs are small drops: the median syntax change is −11.8 points. 7,404 of them drop one level and 2,429 two levels. Only 724 pairs keep `any` for syntax, all of them rises.

## What changes

Syntax training targets by level (training split):

| Level | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 7,776 | 13,465 | 14,460 | 14,448 | 13,188 | 8,697 | 3,771 | 1,605 | 629 | 109 |
| **v4** | 8,764 | 14,194 | 14,950 | 14,767 | 13,637 | 10,587 | 6,598 | 3,230 | 1,046 | 109 |
| CLiC-it 2026 | | | | | | 5,769 | 5,266 | 4,112 | 1,659 | 32 |

**Side effect on the training mix.** Target dropout is unchanged (p = 0.3 per set target, at least one kept). Because syntax is now almost always set, the examples per epoch change:
- base-only examples fall from 23.9% to 18.6%;
- lexical-only examples fall from 3.7% to 2.6%;
- syntax-only examples rise from 22.7% to 24.6%.

25,059 training pairs have all three targets set, against 20,850 in v3.

## Files

| File | Content |
|---|---|
| `training_set_v4.zip` | v3's columns, with the v4 `tgt_syntax`, `ctrl_tokens` and `ctrl_tokens_with_source` |
| `{base,lexical,syntax}_test_set_v4.csv` | identical to the v1, v2 and v3 test sets. Their target columns still follow the v1 rule; the evaluation requests do not use them. |
| `construction_v4.json` | the rule, the margins, the counts, the targets by level, the one-level-below counts and the per-epoch composition |

## Experiment

`v4_targets` is Qwen3-0.6B-Base trained on v4 with exactly the recipe and evaluation of `v3_targets`: three target tokens, target dropout 0.3, LoRA r 32, effective batch 64 (4 × 16), 3 epochs (4,155 steps). Only the syntax labels differ, so `v3_targets` is its direct comparison.

- Run: `bash data/v1/experiments/run_v4_targets.sh`.
- Results: [`data/v1/experiments/README.md`](../v1/experiments/README.md#dataset-v4-syntax-crossings).

**Result: syntax at the top did not move, and base lost v3's gain.**
- **Syntax at the top:** requests at 70–80 land at 57.3 (v3: 56.6). This falsifies the prediction below: one-level labels at the top do not explain the CLiC-it model's advantage there.
- **Why:** in this data, simplifications of originals at syntax 80 are equally heavy edits, whether their score lands at 80, 70, 60 or 50 (median character similarity 0.65–0.73). So the new labels carry no distinct behaviour to learn.
- **Whole syntax test set:** three of the four goal conditions are met (RMSE 26.0, inside 17.9%, ρ within 0.504, copies 1.6%). RMSE at 60–80 is 25.1 against the CLiC-it model's 23.2.
- **Base:** requests at 70–80 fell to 60.6 (v3: 66.9). In v3, syntax `any` marked the pairs whose syntax barely changed; v4 labels them, so a syntax `any` request now yields more syntactic simplification.

### Goal

Syntax at or above the CLiC-it 2026 0.6B, without copying the original more. On the whole syntax test set, v3 is already slightly ahead on RMSE (26.6 against 26.9) and on the share inside (17.7% against 15.5%). It is behind on within-original ρ (0.48 against 0.50) and at the top of the scale (RMSE at requests 60–80: 25.3 against 23.2). v4 reaches the goal if, on syntax requests:
- RMSE at 60–80 is at most 23.2;
- within-original ρ is at least 0.50;
- the share inside stays at least 15.5% overall;
- copies stay below CLiC-it's 5.7%.

### Predicted before the run (2026-11-11, while training)

| Main evaluation, syntax requests | v3 · targets | CLiC-it 2026 0.6B | Prediction for v4 |
|---|---|---|---|
| Mean achieved at requests 60–70 / 70–80 / 80–90 (46 requests) | 54.1 / 56.6 / 60.0 | 56.2 / 68.1 / 81.1 | **56–60 / 62–68 / 62–80** |
| RMSE / share inside at 60–80 | 25.3 / 13.9% | 23.2 / 12.2% | **22.5–24.5 / 14–19%** |
| ρ within original | 0.48 | 0.50 | 0.50–0.56 |
| All requests: RMSE / share inside | 26.6 / 17.7% | 26.9 / 15.5% | 25.0–26.5 / 17–20% |
| Share inside at requests 0–50 | 19.0% | 16.5% | within 2 points of v3 |
| Copies | 1.3% | 5.7% | 1.5–4% |

| Other dimensions | v3 · targets | Prediction for v4 |
|---|---|---|
| Base: mean achieved at requests 70–80 / RMSE at 60–80 | 66.9 / 19.8 | within noise of v3: 64–69 / 19.3–20.5 |
| Lexical: inside / ρ within | 12.8% / 0.05 | 11–13.5% / ≤ 0.08 |
| Joint: all three inside / syntax inside | 3.6% / 21.7% | 3–4.5% / 21–25% |

- **Falsified if** syntax at requests 70–80 stays at 59.6 or below (a rise of less than 3 points). Then one-level labels at the top would not explain CLiC-it's advantage. The remaining candidates would be its copies (33% of its same-level syntax tokens are identity pairs) and its source distribution.
- **Risks:**
  - Labels within the margin are partly noise, so the syntax token could become less precise at mid levels. Watch ρ within and the share inside at 0–50.
  - Fewer base-only examples (−5 points per epoch) could cost some base control. Watch base at 70–80.
- **Confidence:**
  - about 65% that syntax at 70–80 rises by at least 3 points;
  - about 40% that all four goal conditions are met.
