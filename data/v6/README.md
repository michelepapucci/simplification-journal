# Dataset v6: v3 plus identity pairs, dosed for syntax-only requests

Dataset v6 is [dataset v3](../v3/README.md) plus 240 identity pairs: training originals given as their own simplification, with a syntax target at the original's level. It is [dataset v5](../v5/README.md) with the number of identity pairs corrected. Everything in v3 (pairs, splits, labels, test sets) is unchanged.

| | |
|---|---|
| Source | dataset v3 (`data/v3/training_set_v3.zip`); shares from the CLiC-it 2026 data (`data/clicit2026/`) |
| Training pairs | **89,700**: v3's 89,460 + 240 identity pairs, all in `train` (88,839 `train` + 861 `validation`) |
| Identity pairs | as in v5: original → original; target syntax = the original's level, base and lexical `any`; one per original, drawn at random (seed 0) |
| Test sets | the v1 test sets, byte for byte (`*_test_set_v6.csv`) |
| Built by | `python scripts/build_dataset_v6.py` (about 20 seconds, after `build_dataset_v3.py`; it reuses `build_dataset_v5.py`) |
| Explore | [data explorer](https://michelepapucci.github.io/simplification-journal/#data=v6), dataset "v6" (the identity pairs are in the "All pairs" and "Syntax" slices) |

## Why

v5 added 989 identity pairs, as many as give v3's syntax targets at levels 60–90 the CLiC-it data's share of identical pairs. Syntax at the top moved past the CLiC-it 2026 0.6B, but its copies doubled the CLiC-it model's ([v5 results](../v1/experiments/README.md#dataset-v5-identity-pairs)):
- syntax requests at 70–80 landed at 73.7 (v3: 56.6; CLiC-it: 68.1);
- 46.8% of outputs at the highest syntax request were copies (CLiC-it: 22.1%);
- copies over all syntax requests were 8.7% (CLiC-it: 5.7%).

The count was right for all syntax targets, but wrong for the format the syntax evaluation uses. That evaluation asks for the syntax token alone. The CLiC-it data shows every pair under one token, so its shares hold in that format. v3's pairs carry up to three targets, and show the syntax token alone only when target dropout removes the other two. The identity pairs have only a syntax target and always show it alone. Among the syntax-only examples per epoch, v5's identity share was far above the CLiC-it data's (the run showed that this was the wrong share to match; see the result below):

| Syntax token | 60 | 70 | 80 | 90 |
|---|---|---|---|---|
| CLiC-it 2026 data | 5.0% | 18.6% | 33.5% | 50.0% |
| v5, all examples with a syntax token | 6.4% | 23.4% | 40.6% | 57.6% |
| v5, syntax-only examples | 14.6% | 48.4% | 70.4% | 82.7% |
| **v6, syntax-only examples** | **4.9%** | **18.7%** | **33.5%** | **50.2%** |

## The rule

v6 adds, for each syntax level L from 60 to 90, as many identity pairs as give the CLiC-it share among the syntax-only examples at L. A v3 pair with k set targets, one of them syntax, shows syntax alone with probability 0.7 · 0.3^(k−1) + 0.3^k / k. That is the training script's dropout: each set target is dropped with p 0.3, and one is kept at random if all are dropped.

| Syntax level L | 60 | 70 | 80 | 90 |
|---|---|---|---|---|
| v3 syntax-only examples per epoch at L | 1,156 | 392 | 133 | 23 |
| Identity pairs added (v5) | **60** (197) | **90** (367) | **67** (316) | **23** (109) |

Per epoch, syntax-only examples rise from 22.7% to 22.9% (v5: 23.5%).

## Files

| File | Content |
|---|---|
| `training_set_v6.zip` | v3's columns and rows, plus the identity pairs (as in v5) |
| `{base,lexical,syntax}_test_set_v6.csv` | identical to the v1 to v5 test sets |
| `construction_v6.json` | the CLiC-it shares, the syntax-only exposure, the counts per level and the per-epoch composition |

## Experiment

`v6_targets` is Qwen3-0.6B-Base trained on v6 with exactly the recipe and evaluation of `v3_targets` and `v5_targets` (4,167 steps). It differs from v5 only in the number of identity pairs.

- Run: `bash data/v1/experiments/run_v6_targets.sh`.
- Results: [`data/v1/experiments/README.md`](../v1/experiments/README.md#dataset-v6-identity-pairs-dosed-for-syntax-only-requests).

**Result: at this dose the model barely copies, and syntax at the top did not move.**
- **Syntax at the top:** requests at 70–80 land at 54.8 (v3: 56.6; v5: 73.7; CLiC-it: 68.1).
- **Copies:** 6.2% at the highest request per original (CLiC-it: 22.1%), 2.2% over all syntax requests. This is outside the 8–35% band, so the prediction below is falsified.
- **Goal:** two of the four syntax conditions are met: the share inside (17.0%) and the copies. RMSE at 60–80 is 27.5 and ρ within 0.47.
- **Why:** the model does not treat a syntax token shown alone as a separate format. Among all examples that show the syntax token, v6's identity share is about a third of the CLiC-it data's (12.6% against 33.5% at 80), and copies follow that share. Matching the CLiC-it data on that measure takes 738 identity pairs.
- **Base:** at base request 0, 20% of the outputs are near-copies, which raises base RMSE from 20.8 to 24.3. Without that request, base is v3's. Base 0 is the rarest base target, and the identity pairs never show it, so this is most likely training variance; it was not tested. Requests at 70–80 land at 63.7 (v3: 66.9).

### Goal

As for v5. On syntax requests:
- RMSE at 60–80 is at most 23.2 (CLiC-it 2026 0.6B);
- within-original ρ is at least 0.50;
- the share inside is at least 15.5% overall;
- copies stay below CLiC-it's 5.7%.

On base, requests at 70–80 stay at 64 or above.

### Predicted before the run (2026-11-12, while training)

Points of reference:
- **v3** (no identity pairs): RMSE at 60–80 is 25.3, with 1.3% copies.
- **v5** (2–3 times CLiC-it's dose): RMSE 22.4, with 8.7% copies.
- **CLiC-it:** RMSE 23.2, with 5.7% copies.

A straight line between v3 and v5 reaches CLiC-it's copy rate at an RMSE of about 23.6.

| Main evaluation, syntax requests | v3 · targets | v5 · targets | CLiC-it 2026 0.6B | Prediction for v6 |
|---|---|---|---|---|
| Mean achieved at requests 60–70 / 70–80 / 80–90 (46 requests) | 54.1 / 56.6 / 60.0 | 58.8 / 73.7 / 84.1 | 56.2 / 68.1 / 81.1 | **55–59 / 62–70 / 65–82** |
| RMSE / share inside at 60–80 | 25.3 / 13.9% | 22.4 / 9.5% | 23.2 / 12.2% | **22.8–24.5 / 10.5–14%** |
| ρ within original | 0.48 | 0.59 | 0.50 | 0.51–0.57 |
| All requests: RMSE / share inside | 26.6 / 17.7% | 25.8 / 16.6% | 26.9 / 15.5% | 25.7–26.7 / 16.5–18.5% |
| Copies: all requests / highest request per original | 1.3% / 2.6% | 8.7% / 46.8% | 5.7% / 22.1% | **3–6% / 12–30%** |

| Other dimensions | v3 · targets | v5 · targets | Prediction for v6 |
|---|---|---|---|
| Base: mean achieved at requests 70–80 / RMSE at 60–80 | 66.9 / 19.8 | 64.0 / 20.6 | 64–68 / 19.5–20.7 |
| Lexical: inside / ρ within | 12.8% / 0.05 | 12.5% / 0.08 | 11–14% / ≤ 0.10 |
| Joint: all three inside | 3.6% | 3.5% | 3–4.5% |

- **Falsified if** copies at the highest syntax request land outside 8–35%. That would mean matching the share in syntax-only examples does not reproduce the CLiC-it model's copying. Short of that, if RMSE at 60–80 misses 23.2 with copies at or below 5.7%, our model trades copies for top-of-scale accuracy no better than the CLiC-it model does.
- **Confidence:**
  - about 70% that copies over all syntax requests stay below 5.7%;
  - about 45% that RMSE at 60–80 is at most 23.2;
  - about 35% that all four syntax goal conditions are met.
