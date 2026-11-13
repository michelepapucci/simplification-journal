# Dataset v7: v3 in the CLiC-it format, plus combined examples and identity pairs

Dataset v7 is [dataset v3](../v3/README.md) shown to the model the way the CLiC-it 2026 data is:
- one training example per target, with that token alone;
- plus one combined example per pair, with all its targets, so the model still learns to take several tokens at once.

It adds 1,031 identity pairs (training originals given as their own simplification), dosed so that the top of the syntax scale has the CLiC-it data's share of copies. Target dropout is off. v3's pairs, labels, validation split and test sets are unchanged.

| | |
|---|---|
| Source | dataset v3 (`data/v3/training_set_v3.zip`); shares from the CLiC-it 2026 data (`data/clicit2026/`) |
| Training examples | **267,390** from 89,630 pairs (v3's 88,599 + 1,031 identity pairs): 83,023 base alone, 26,733 lexical alone, 79,179 syntax alone, 78,455 combined |
| Validation | v3's 861 pairs, one example each with all their targets (as for v3 to v6) |
| Identity pairs | original → original; syntax and base targets at the original's levels, lexical `any`; 908 keep their base target |
| Test sets | the v1 test sets, byte for byte (`*_test_set_v7.csv`) |
| Built by | `python scripts/build_dataset_v7.py` (about 50 seconds, after `build_dataset_v3.py`; it reuses `build_dataset_v5.py`) |

## Why

v5 and v6 added identity pairs with a syntax target only, in our training format: each pair is one example per epoch, carrying all its targets, and each target is dropped with probability 0.3.
- **v5 (989 pairs):** syntax at the top rose past the CLiC-it 2026 0.6B, but with twice its copies.
- **v6 (240 pairs):** the model barely copied, and syntax at the top did not move.

Copies followed the identity share among all examples that show the syntax token ([v6 results](../v1/experiments/README.md#dataset-v6-identity-pairs-dosed-for-syntax-only-requests)).

The CLiC-it model was trained differently. Its data shows every pair three times per epoch, once under each token, one token at a time. Per epoch:

| | CLiC-it 2026 data | v3 | **v7** |
|---|---|---|---|
| Pairs | 45,209 | 88,599 | 89,630 |
| Examples | 135,627 (3 per pair) | 88,599 (1 per pair) | **267,390** (3.0 per pair) |
| Token alone: base / lexical / syntax | 45,209 each (lexical is a copy of base) | 21,131 / 3,252 / 20,085 | **83,023 / 26,733 / 79,179** |
| Two or three tokens at once | none | 44,131 | **78,455** |
| Examples showing a syntax token at 60–90 | 11,069 | 4,636 | **13,678** |
| …of which identity pairs | 1,597 | 0 | **1,939** |
| Steps for 3 epochs at batch 64 | 6,351 | 4,155 | **12,534** |

Our model sees the high syntax tokens 2.4 times less often than the CLiC-it model, and each token alone far less often. v7 tests whether giving it CLiC-it's format, and with it CLiC-it's exposure at the top of the scale, closes the gap at the top of the syntax scale.

## The rule

**Examples.** Each training pair gives:
- one example per set target, with that token alone and the others `any`;
- if two or three targets are set, one more example with all of them.

A pair with one target gives one example. No target is dropped during training (`train.py --target-dropout 0`).

**Identity pairs.**
- **Targets:** syntax and base at the original's own levels, lexical `any`. The CLiC-it data's lexical token is a copy of its base token, so there is no lexical share to match, and lexical stays as in v3.
- **Expansion:** expanded like any pair: base alone, syntax alone, and combined.

| Syntax level L | 60 | 70 | 80 | 90 |
|---|---|---|---|---|
| Examples showing syntax token L | 7,498 | 3,854 | 1,890 | 436 |
| Identity share among them (CLiC-it data) | 5.0% (5.0%) | 18.6% (18.6%) | 33.4% (33.5%) | 50.0% (50.0%) |
| Identity pairs | 192 | 370 | 336 | 133 |
| …with a base target | 180 | 347 | 296 | 85 |

- **Syntax:** per syntax level, the identity pairs give the CLiC-it share among all examples that show the token, alone or combined. This is the measure that fitted v5 and v6. Each is drawn from a different training original at that level, at random (seed 0).
- **Base:** identity originals drawn by syntax level are mostly high on base too. A base target is kept on at most as many of them as give the CLiC-it share among the examples showing base token B.
  - That thins base 80 and 90: 16.4% and 23.5% (CLiC-it: 16.4% and 23.9%).
  - Lower base levels stay below CLiC-it's share; at 70, for example, 11.5% against 14.4%.
- **Syntax only:** a pair without a base target shows syntax once instead of twice, so extra identity pairs, with a syntax target only, restore the syntax share.

## Files

| File | Content |
|---|---|
| `training_set_v7.zip` | one row per training example: v3's columns, plus `row` (unique), `example` (`base`, `lexical` or `syntax` for a token alone, `combined`, or `pair` for validation) and `pair_row` (one row per pair with all its targets: the pair-level view). `idx` identifies the pair. |
| `{base,lexical,syntax}_test_set_v7.csv` | identical to the v1 to v6 test sets |
| `construction_v7.json` | the rules, the counts, and the identity shares by syntax, base and lexical level |

## Experiment

`v7_targets` is Qwen3-0.6B-Base trained on v7 with the recipe of `v3_targets`, except:
- target dropout is off, since the data already has single-token examples;
- 3 epochs are 12,534 steps instead of about 4,200, because every pair gives about three examples.

The evaluation is the same. It differs from v5 in two ways at once: the format (which brings 3 times the training steps) and the identity pairs (base targets, and the dose). If it works, this run cannot say which of the two did it.

- Run: `bash data/v1/experiments/run_v7_targets.sh` (about 7.2 hours of training at 2.07 s/step).
- Results: [`data/v1/experiments/README.md`](../v1/experiments/README.md#dataset-v7-the-clic-it-format-plus-combined-examples).

**Result: the model copies like the CLiC-it model and syntax at the top rose, but syntax RMSE at 60–80 is still behind.**
- **Copies:** 21.0% at the highest syntax request per original (CLiC-it: 22.1%), 4.2% over all syntax requests (CLiC-it: 5.7%). This is inside the 8–40% band, so the prediction holds. In this format too, the identity share among all examples that show the syntax token sets copying.
- **Syntax at the top:** requests at 70–80 land at 65.7, against v3's 56.6 (+9.1, interval +7.2 to +11.0) and CLiC-it's 68.1 (−2.4, interval −4.2 to −0.7).
- **Goal:** three of the four syntax conditions are met: ρ within 0.52, share inside 16.5%, copies 4.2%. RMSE at 60–80 is 24.4 (CLiC-it: 23.2; +1.2, interval +0.3 to +2.2).
  - The gap is in the outputs that are not copies. At request 70, v7 copies as often as the CLiC-it model, but its other outputs land at 59.8 against 63.0.
  - At request 60, it copies half as often (5.9% against 10.7%).
- **Base:** the goal is met: requests at 70–80 land at 68.3 (v3: 66.9; CLiC-it: 71.7) and at 80–90 at 75.5 (v3: 68.6). Requests at 50–70 land lower than v3's (51.7 and 59.2 against 57.3 and 61.4).
- **Lexical moved, although its labels did not change:** the share inside is 14.9% (v3: 12.8%; CLiC-it: 14.4%). Lexical requests now also move base and syntax as much as the CLiC-it model's do. Within an original, lexical still barely follows the request (ρ 0.10).
- **Overfitting:**
  - Validation loss was lowest at step 2,500 (0.708) and rose to 0.815 by the end, while training loss fell to 0.38 (v3: 0.58).
  - Meaning scores are slightly lower than v3's (syntax BERTScore 0.844 against 0.849).
  - The outputs are not degenerate.

### Goal

As for v5 and v6. On syntax requests:
- RMSE at 60–80 is at most 23.2 (CLiC-it 2026 0.6B);
- within-original ρ is at least 0.50;
- the share inside is at least 15.5% overall;
- copies stay below CLiC-it's 5.7%.

On base, requests at 70–80 stay at 64 or above.

### Predicted before the run (2026-11-12, while training)

Points of reference:
- **v5:** identity share among examples showing the syntax token, 1.2 times the CLiC-it data's. Copies at the highest request are 2.1 times the CLiC-it model's.
- **v6:** 0.4 times the CLiC-it data's share; copies 0.3 times.
- **v7:** 1.0 times by construction. If copies follow that share, v7 copies about as often as the CLiC-it model.

| Main evaluation, syntax requests | v3 · targets | v5 · targets | v6 · targets | CLiC-it 2026 0.6B | Prediction for v7 |
|---|---|---|---|---|---|
| Mean achieved at requests 60–70 / 70–80 / 80–90 (46 requests) | 54.1 / 56.6 / 60.0 | 58.8 / 73.7 / 84.1 | 53.1 / 54.8 / 65.5 | 56.2 / 68.1 / 81.1 | **55–60 / 63–72 / 65–85** |
| RMSE / share inside at 60–80 | 25.3 / 13.9% | 22.4 / 9.5% | 27.5 / 11.8% | 23.2 / 12.2% | **22.5–25 / 10–14%** |
| ρ within original | 0.48 | 0.59 | 0.47 | 0.50 | 0.49–0.58 |
| All requests: RMSE / share inside | 26.6 / 17.7% | 25.8 / 16.6% | 26.5 / 17.0% | 26.9 / 15.5% | 25.5–27 / 15.5–18.5% |
| Copies: all requests / highest request per original | 1.3% / 2.6% | 8.7% / 46.8% | 2.2% / 6.2% | 5.7% / 22.1% | **3.5–7.5% / 12–35%** |

| Other dimensions | v3 · targets | v5 · targets | CLiC-it 2026 0.6B | Prediction for v7 |
|---|---|---|---|---|
| Base: mean achieved at 70–80 / RMSE 60–80 | 66.9 / 19.8 | 64.0 / 20.6 | 71.7 / 18.9 | 65–72 / 18.8–20.5 |
| Base: copies, all requests / highest request per original | 1.0% / 2.6% | 1.2% / 3.7% | 3.9% / 16.0% | 1.5–4% / 6–18% |
| Base at requests 0–10 (v6: 42.2) | 28.4 | 28.2 | 27.9 | 26–31 |
| Lexical: inside / ρ within | 12.8% / 0.05 | 12.5% / 0.08 | 14.4% / 0.13 | 11–15% / ≤ 0.12 |
| Joint: all three inside | 3.6% | 3.5% | – | 3–5% |

- **Falsified if** copies at the highest syntax request fall outside 8–40%. Then the share among examples showing the token does not set copying in this format either.
- **Risks:**
  - Three times the steps on the same pairs could overfit. Watch the validation loss (v3: best 0.7015) and meaning preservation (BERTScore, SBERT).
  - Lexical alone is now 26,733 examples per epoch instead of 3,252, so lexical could change even though its labels do not.
- **Confidence:**
  - about 60% that syntax at 70–80 rises at least 5 points over v3 (to 61.6 or more);
  - about 55% that copies at the highest request land between 11% and 33% (half to one and a half times the CLiC-it model's);
  - about 30% that all four syntax goal conditions are met;
  - about 70% that base at 70–80 is at least 64.

## One epoch: `v7_1ep_targets`

`v7_targets` overfits:
- its validation loss is lowest at step 2,500 of 12,534 (0.708) and 0.815 at the end;
- its training loss ends at 0.38, against v3's 0.58.

Its one missed goal condition, syntax RMSE at 60–80, comes from edits heavier than the CLiC-it model's. `v7_1ep_targets` is the same data and recipe, trained for one epoch:
- **Steps:** 4,178, about as many as v3's 4,155.
- **Exposure:** each pair is seen about three times (once per token, plus the combined example), as often as in v3's three epochs.
- **Checkpoint:** validation loss every 250 steps; the adapter at the lowest one (`runs/v7_1ep_targets/adapter_best`) is the model evaluated.
- **Interpretability:** the same model then goes through the interpretability experiments run on v5 ([`interp/`](../v1/experiments/interp/README.md)).
- **Run:** `bash data/v1/experiments/run_v7_1ep_targets.sh` (about 2.6 hours of training at 2.2 s/step, started 2026-11-13 01:56).

**Result: one epoch is worse than three; the training length is not what makes the edits heavy.**
- **Validation loss** converged: lowest at step 3,000 (0.7054), then flat at about 0.707 to the end.
- **The outputs that are not copies are not lighter.** At syntax request 70 they land at 58.9 (three epochs: 59.8; CLiC-it: 63.0). The test above therefore points to the format, not overfitting, as the cause of the heavy edits.
- **It copies without regard to the request.** Copies reach 3.1% at syntax requests 0–50 (three epochs: 1.1%) and 4.6% over all base requests (1.5%). Base ρ within falls to 0.67 (0.73), base RMSE rises to 23.1 (20.2), and syntax ρ within to 0.47 (0.52).
- **Syntax at the top is about as before:** 64.1 at 70–80 (three epochs: 65.7), RMSE at 60–80 25.0 (24.4). One of the four syntax goal conditions is met (share inside 15.9%).
- **Lexical goes back to v3's level** (share inside 12.5%, ρ within 0.04), so the lexical gain of the three-epoch run came with the longer training.
- The extra epochs taught the model when to copy, which the validation loss does not measure. The three-epoch run stays the v7 model.

### Predicted before the run (2026-11-13, while training)

| Main evaluation | v7 · 3 epochs | CLiC-it 2026 0.6B | Prediction for v7 · 1 epoch |
|---|---|---|---|
| Best validation loss / its step | 0.708 / 2,500 | – | **0.700–0.712 / after step 2,000** |
| Syntax: mean achieved at requests 60–70 / 70–80 / 80–90 | 53.0 / 65.7 / 73.5 | 56.2 / 68.1 / 81.1 | 53–58 / **60–68** / 65–80 |
| Syntax 60–80: RMSE / inside | 24.4 / 11.3% | 23.2 / 12.2% | **23–25** / 10–14% |
| Syntax: ρ within original | 0.52 | 0.50 | 0.48–0.55 |
| Syntax copies: all requests / highest request per original | 4.2% / 21.0% | 5.7% / 22.1% | 2.5–5.5% / **12–25%** |
| Syntax request 70: mean of the outputs that are not copies | 59.8 | 63.0 | **59–63** |
| Base: mean achieved at 70–80 / RMSE 60–80 | 68.3 / 19.1 | 71.7 / 18.9 | 64–70 / 19–20.5 |
| Lexical: inside / ρ within | 14.9% / 0.10 | 14.4% / 0.13 | 12.5–15% / ≤ 0.12 |
| Joint: all three inside | 3.8% | – | 3–5% |
| Syntax BERTScore | 0.844 | 0.853 | 0.844–0.852 |

- **Test:** if v7's heavier edits come from overfitting, the outputs that are not copies at syntax request 70 land at 61 or above. If they land at 60 or below, the format, not the training length, makes the edits heavier.
- **Confidence:**
  - about 55% that the outputs that are not copies at request 70 land at 61 or above;
  - about 60% that copies at the highest request land between 12% and 25%;
  - about 35% that RMSE at 60–80 is at most 23.2, and with it all four syntax goal conditions.
