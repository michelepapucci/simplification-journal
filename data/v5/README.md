# Dataset v5: v3 plus identity pairs

Dataset v5 is [dataset v3](../v3/README.md) plus 989 identity pairs: training originals given as their own simplification, with a syntax target at the original's level. They give the top of the syntax scale the same share of identical pairs as the CLiC-it 2026 data. Everything in v3 (pairs, splits, labels, test sets) is unchanged.

| | |
|---|---|
| Source | dataset v3 (`data/v3/training_set_v3.zip`); shares from the CLiC-it 2026 data (`data/clicit2026/`) |
| Training pairs | **90,449**: v3's 89,460 + 989 identity pairs, all in `train` (89,588 `train` + 861 `validation`) |
| Identity pairs | original → original; target syntax = the original's level, base and lexical `any`; one per original, drawn at random (seed 0) |
| Test sets | the v1 test sets, byte for byte (`*_test_set_v5.csv`) |
| Built by | `python scripts/build_dataset_v5.py` (about 20 seconds, after `build_dataset_v3.py`) |
| Explore | [data explorer](https://michelepapucci.github.io/simplification-journal/#data=v5), dataset "v5" (the identity pairs are in the "All pairs" and "Syntax" slices) |

## Why

[v4](../v4/README.md) labelled every syntax crossing, and syntax at the top did not move: requests at 70–80 land at 57.3, against 68.1 for the CLiC-it 2026 0.6B ([v4 results](../v1/experiments/README.md#dataset-v4-syntax-crossings)). In this data, simplifications of originals at syntax 80 are equally heavy edits whether their score lands at 80, 70, 60 or 50. No label can then teach "change little" for a high target.

The CLiC-it data has another source for it: 1,598 identity pairs (3.5% of its pairs). It shows each pair under one token at a time, and a copy keeps the original's level, so among its syntax tokens the identical pairs sit at the top:

| Syntax token | 60 | 70 | 80 | 90 |
|---|---|---|---|---|
| CLiC-it 2026: syntax tokens | 5,266 | 4,112 | 1,659 | 32 |
| of which identical pairs | 261 (5.0%) | 765 (18.6%) | 555 (33.5%) | 16 (50%) |

Its model behaves accordingly. At its highest syntax request, 22% of its outputs are copies and 48% near-copies, and its mean at requests 70–80 is 68.1. Our training data has no identical pairs.

## The rule

For each syntax level L from 60 to 90, v5 adds as many identity pairs as give v3's syntax targets at L the CLiC-it share:

| Syntax level L | 60 | 70 | 80 | 90 |
|---|---|---|---|---|
| v3 syntax targets at L (train) | 3,771 | 1,605 | 629 | 109 |
| Identity pairs added | 197 | 367 | 316 | 109 |
| Identity share at L in v5 | 5.0% | 18.6% | 33.4% | 50% |
| Training originals at L (candidates) | 2,251 | 5,319 | 3,431 | 953 |

- **Only a syntax target.** Base and lexical are `any`. The CLiC-it data has each identity pair under its base token too. Here only the syntax row is reproduced, so that the base and lexical tokens keep v3's meaning and base keeps v3's gain.
- **Training only.** The identity pairs are always shown with their syntax token: it is their only target, so target dropout never removes it. The validation split is v3's, unchanged.
- **Per epoch:** syntax-only examples rise from 22.7% to 23.5%, base-only fall from 23.9% to 23.6%.

The test originals are not among the identity pairs: they are not in the training data.

## Files

| File | Content |
|---|---|
| `training_set_v5.zip` | v3's columns and rows, plus the identity pairs (idx from 1 above v3's largest; `simplification` equals `original_text`, simplification scores and features equal the original's) |
| `{base,lexical,syntax}_test_set_v5.csv` | identical to the v1 to v4 test sets |
| `construction_v5.json` | the CLiC-it shares, the counts per level, the targets by level and the per-epoch composition |

## Experiment

`v5_targets` is Qwen3-0.6B-Base trained on v5 with exactly the recipe and evaluation of `v3_targets`: three target tokens, target dropout 0.3, LoRA r 32, effective batch 64 (4 × 16), 3 epochs (4,200 steps). Only the identity pairs differ, so `v3_targets` is its direct comparison.

- Run: `bash data/v1/experiments/run_v5_targets.sh`.
- Results: [`data/v1/experiments/README.md`](../v1/experiments/README.md#dataset-v5-identity-pairs).

**Result: syntax at the top moved past the CLiC-it model, but with twice its copies.**
- **Syntax at the top:** requests at 70–80 land at 73.7 (v3: 56.6; CLiC-it: 68.1).
- **Goal:** three of the four syntax conditions are met: RMSE at 60–80 is 22.4, ρ within 0.59 and the share inside 16.6%. Copies are not: 8.7% of all syntax outputs, and 46.8% at the highest request per original (CLiC-it: 5.7% and 22.1%).
- **Why so many copies:** the identity pairs have only a syntax target, so they always appear in the syntax-only format of the evaluation, while v3's pairs appear in it only when dropout removes their other targets. In that format the identity share was 48% at token 70 and 70% at 80, against the CLiC-it data's 19% and 33%.
  - v6 showed that this explanation was wrong. Dosed to match the CLiC-it data in that format, it copied a quarter as often as the CLiC-it model. The share that fits both runs is among all examples that show the syntax token, where v5 was slightly above the CLiC-it data (40.6% against 33.5% at 80).
- **Base:** requests at 70–80 land at 64.0 (v3: 66.9).
- **Follow-up:** [dataset v6](../v6/README.md) has 240 identity pairs, dosed on syntax-only examples. It barely copies, and syntax at the top did not move.

### Goal

The goal is v4's, with base added. On syntax requests:
- RMSE at 60–80 is at most 23.2 (CLiC-it 2026 0.6B);
- within-original ρ is at least 0.50;
- the share inside is at least 15.5% overall;
- copies stay below CLiC-it's 5.7%.

On base, requests at 70–80 stay within noise of v3 (at least 64).

### Predicted before the run (2026-11-12, while training)

| Main evaluation, syntax requests | v3 · targets | v4 · targets | CLiC-it 2026 0.6B | Prediction for v5 |
|---|---|---|---|---|
| Mean achieved at requests 60–70 / 70–80 / 80–90 (46 requests) | 54.1 / 56.6 / 60.0 | 53.6 / 57.3 / 63.2 | 56.2 / 68.1 / 81.1 | **54–58 / 60–66 / 60–78** |
| RMSE / share inside at 60–80 | 25.3 / 13.9% | 25.1 / 14.6% | 23.2 / 12.2% | **23–25 / 12–16%** |
| ρ within original | 0.48 | 0.50 | 0.50 | 0.48–0.53 |
| All requests: RMSE / share inside | 26.6 / 17.7% | 26.0 / 17.9% | 26.9 / 15.5% | 25.8–27.0 / 16.5–19% |
| Share inside at requests 0–50 | 19.0% | 19.0% | 16.5% | within 2 points of v3 |
| Copies: all requests / highest request per original | 1.3% / 2.6% | 1.6% / 3.1% | 5.7% / 22.1% | **2.5–5.5% / 8–20%** |

| Other dimensions | v3 · targets | Prediction for v5 |
|---|---|---|
| Base: mean achieved at requests 70–80 / RMSE at 60–80 / copies | 66.9 / 19.8 / 1.0% | within noise of v3: 64–69 / 19.3–20.5 / ≤ 2% |
| Lexical: inside / ρ within | 12.8% / 0.05 | 11–14% / ≤ 0.08 |
| Joint: all three inside | 3.6% | 3–4.5% |

- **Falsified if** syntax at requests 70–80 stays at 59.6 or below (a rise of less than 3 points over v3). If copies at the highest request also stay below 6%, the model did not learn to copy from these pairs. If it copies and the mean still does not rise, copying is not what lifts the CLiC-it model's means.
- **Risks:**
  - The model sees the token, not the original's level. Identity pairs at token 70 come from originals at 70, while the evaluation's 70–80 requests come from originals at 80. The model must generalise "target near the sentence's own level, change little" from the sentence alone. The CLiC-it model had the same setup.
  - Copies could spread to low requests. Watch copies and the share inside at 0–50.
- **Confidence:**
  - about 60% that syntax at 70–80 rises by at least 3 points;
  - about 35% that all four syntax goal conditions are met;
  - about 75% that base stays within noise of v3.
