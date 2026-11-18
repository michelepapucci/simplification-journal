# g1 experiments: one control token on READ-IT global

Qwen3-0.6B-Base fine-tuned on [dataset g1](../data/g1/README.md) to rewrite a sentence at a requested READ-IT global
level. This is the LREC 2026 setup: one control token, `<|readability_L|>`, where L is the level of the
simplification's READ-IT global. The training data follows the journal's selection rules.

The earlier work, with three tokens (base, lexical, syntax) and datasets v0 to v7, is archived on the
`multidim-archive` branch. Its explorer is at
https://michelepapucci.github.io/simplification-journal/archive/multidim/. It concluded that the three tokens act
as one "simplify" dial, which is why the work came back to one token.

## Models

All three use the same recipe. It is the CLiC-it 2026 recipe as reproduced in the earlier journal models:
- LoRA r 32, alpha 64, dropout 0.05 on every projection, with the embeddings and the output layer trained in full;
- learning rate 2.5e-5 with linear decay, effective batch 64, 3 epochs, bf16;
- prompt `f"{token}\n{original}\n"`, loss on the simplification only.

The **final** adapter after 3 epochs is the model.

| Model | Training examples per epoch | What changes |
|---|---|---|
| **g1** | 80,251 (every pair once) | nothing: the plain model |
| **g1 · balanced** | 97,932 | levels 0 to 70 are each shown as often as the most frequent level (12,039 times per epoch); the extra copies of the rarer levels are drawn anew every epoch. Level 80, which no test original can request, is left alone |
| **g1 · guided** | 80,251 | 10% of the examples (redrawn every time an example is read) get `<|readability_any|>` ("no level") instead of their level. At generation time this allows classifier-free guidance |

**Classifier-free guidance.** At every step the model reads the original twice, once with the requested level and
once with "no level". The next token is sampled from:

  log p(·|no level) + w · (log p(·|level) − log p(·|no level))

w = 1 is plain generation, and w > 1 strengthens whatever the level token changes. The weight is chosen on the
dev requests before the test run, by a fixed rule (`choose_guidance.py`): the lowest RMSE among w ∈ {1, 1.5, 2, 3}
whose BERTScore against the original is at most 0.01 below w = 1's and whose share of degenerate or echoing
outputs is at most 1 point above it.

**Best-of-n (inference only, g1).** Each test request also gets 3 more samples. Best-of-k keeps, among the plain
output and the first k − 1 extra samples, the output whose READ-IT global is closest to the middle of the
requested level. It measures how much control the model already has but loses in decoding.

## Evaluation

- **Requests:** the 1,000 test originals of g1 (418 at level 70, 582 at level 80), each asked for every level
  below its own: 7,582 requests. None of these originals is in training.
- **Decoding:** sampling at temperature 0.7, top-p 1.0, at most 128 new tokens, seed 42. This is the protocol of
  the journal's earlier evaluation.
- **Control:**
  - RMSE between the requested level and the achieved READ-IT global, ×100. This is the CLiC-it convention:
    achieved minus the level's lower bound, so an output in the middle of its level scores 5.
  - Bias, share inside the level, share within one level, and where the outputs land.
  - Spearman between requested and achieved levels, and the within-original slope (achieved points per requested
    point, averaged over originals).
- **READ-IT movement:** how base, lexical and syntax move.
- **Meaning and quality:**
  - BERTScore and SBERT against the original;
  - copies, empty outputs, and outputs that break g1's cleaning rules (degenerate, echo, too long);
  - the LREC reference metrics: SARI, BLEU and BERTScore against the IMPaCTS simplifications of the same original
    at the requested level. These exist for 4,216 of the 7,582 requests.
- **Profiles:** changes on key Profiling-UD features.
- **Comparisons with g1:** 95% paired bootstrap intervals over test originals.

READ-IT is computed exactly as the IMPaCTS labels, with the copy in `tools/readit`.

| File | What it does |
|---|---|
| `train.py` | training (`--balance`, `--null-share`) |
| `generate.py` | test or dev requests; `--samples k` for best-of-n; `--guidance w` |
| `choose_guidance.py` | the dev-set choice of w |
| `score_texts.py` | READ-IT and Profiling-UD of outputs and originals |
| `meaning.py` | meaning, quality and reference metrics |
| `analyze.py` | `results/summary.json` and `results/summary.md` |
| `run_g1.sh` | the whole run: the three models one after the other on GPU0 (CPUs 9–25), scoring in the background |

## Predicted before the run (2026-11-13, while g1 was training)

Reference points from the archived three-token models, for their base token (the closest to a global dial):
- within-original slope about 0.6;
- within-original ρ 0.6–0.7;
- RMSE about 19–23.

READ-IT global is as noisy as READ-IT syntax: near-identical rewrites differ by 3.8 points at the median and 9.2
at the 75th percentile. Even a perfect model would therefore often miss a 10-point level.

| Measure | g1 | g1 · balanced (vs g1) | g1 · guided, w = 1 (vs g1) | g1 · guided, chosen w (vs g1) |
|---|---|---|---|---|
| RMSE, all requests | **15–21** | within ±0.7 | within ±0.7 | −2 to +0.5 |
| Inside the level / within one level | **18–28% / 45–65%** | +0–3 points | ±2 points | +2 to +6 points |
| ρ within original / slope within original | **0.60–0.80 / 0.55–0.80** | slope +0.02 to +0.08 | ±0.03 | ρ +0.02 to +0.06, **slope +0.10 to +0.25** |
| Mean achieved at request 0 / 60 (level-70 originals) / 70 (level-80 originals) | 15–28 / 50–62 / 55–68 | request 0: 1–4 lower; request 70: 1–4 higher | about the same | spread wider at both ends |
| Copies, all requests / top request | < 2% / 2–10% | about the same | about the same | top request: up to +3 points |
| BERTScore F1 with the original | 0.80–0.86 | about the same | about the same | −0.003 to −0.01 |
| Degenerate / echo outputs | < 1% / < 0.3% | about the same | about the same | up to +1 point |
| Reference metrics: SARI / BLEU / BERTScore | 40–46 / 30–45 / 0.80–0.86 | about the same | about the same | SARI ±1 |

- **Chosen w:** 1.5 or 2 (confidence about 60%). If no w beats w = 1 on dev within the meaning limits, the rule
  keeps w = 1.
- **Best-of-k for g1:**
  - best of 2: inside 28–40%, RMSE 12–16;
  - best of 4: inside 40–60%, RMSE 8–13, ρ within 0.85–0.93.

  Gains will be largest at middle requests and smallest at request 0, where the samples rarely get low enough.
- **Confidence:**
  - g1's ranges: moderate. This is the first global-only model on this data.
  - The balanced model's effect being small: fairly confident, since without the margin only levels 0 and 70 are
    short.
  - Guidance stretching the slope: moderate. Whether it also lowers RMSE depends on whether the plain model's
    errors are mostly compression (helped) or noise (not helped).

## Results (2026-11-14)

All numbers come from `results/summary.json`, produced by `analyze.py`, on the 7,582 test requests. Each comparison
with g1 comes with a 95% paired bootstrap interval over the test originals. A difference is called significant when
its interval excludes 0.

| Condition | RMSE | Inside | Within one level | Bias | ρ within | Slope within | BERTScore | Copies | Degenerate |
|---|---|---|---|---|---|---|---|---|---|
| **g1** | 24.1 | 18.9% | 51.7% | +7.4 | 0.55 | 0.59 | 0.851 | 1.5% | 0.9% |
| g1, best of 2 | 16.6 | 32.4% | 72.3% | +6.2 | 0.81 | 0.81 | 0.852 | 1.1% | 0.9% |
| g1, best of 3 | 13.3 | 42.9% | 82.9% | +5.8 | 0.89 | 0.90 | 0.851 | 0.9% | 0.8% |
| **g1, best of 4** | **11.3** | **51.6%** | **88.7%** | +5.7 | **0.93** | **0.93** | 0.852 | 0.7% | 0.8% |
| g1 · balanced | 24.4 | 20.0% | 52.0% | +7.6 | 0.55 | 0.58 | 0.853 | 1.5% | 1.0% |
| g1 · guided, w = 1 | 24.6 | 18.8% | 51.2% | +7.2 | 0.53 | 0.57 | 0.852 | 1.7% | 0.6% |
| **g1 · guided, w = 3** | **20.1** | 22.7% | 59.7% | +7.4 | **0.80** | **1.05** | 0.846 | 3.5% | 0.8% |

Difference from g1, with 95% intervals:

| | RMSE | Inside (points) | ρ within | Slope within | BERTScore |
|---|---|---|---|---|---|
| g1 · balanced | +0.28 [−0.24, +0.78] | +1.1 [−0.0, +2.3] | −0.005 [−0.031, +0.020] | −0.010 [−0.038, +0.017] | +0.002 [0.000, +0.004] |
| g1 · guided, w = 1 | +0.48 [+0.01, +0.98] | −0.1 [−1.3, +1.1] | −0.023 [−0.049, +0.003] | −0.020 [−0.048, +0.006] | +0.001 [−0.001, +0.003] |
| g1 · guided, w = 3 | **−4.03** [−4.52, −3.53] | **+3.8** [+2.6, +5.0] | **+0.246** [+0.225, +0.267] | **+0.466** [+0.440, +0.490] | −0.005 [−0.007, −0.003] |
| g1, best of 4 | **−12.76** [−13.20, −12.34] | **+32.7** [+31.6, +33.8] | **+0.379** [+0.360, +0.398] | **+0.345** [+0.325, +0.365] | +0.001 [−0.001, +0.003] |

Mean achieved READ-IT global by requested level, with the share inside the level in brackets:

| Requested | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|---|
| g1 | 22.8 (29%) | 26.8 (26%) | 31.0 (20%) | 37.9 (14%) | 43.9 (14%) | 48.7 (13%) | 57.8 (16%) | 65.6 (20%) |
| g1 · balanced | 22.0 (32%) | 27.6 (25%) | 33.2 (21%) | 37.8 (19%) | 43.4 (15%) | 50.0 (12%) | 56.7 (16%) | 65.5 (19%) |
| g1 · guided, w = 1 | 21.5 | 27.7 | 32.1 | 38.7 | 41.9 | 49.8 | 56.7 | 63.4 |
| g1 · guided, w = 3 | 10.4 (67%) | 16.3 (24%) | 22.4 (19%) | 31.4 (15%) | 46.1 (14%) | 61.8 (11%) | 71.9 (13%) | 79.5 (19%) |
| g1, best of 4 | 9.9 (66%) | 16.7 | 25.5 | 35.0 | 44.4 | 53.8 | 64.8 | 75.2 |

Request 70 exists only for originals at level 80. Requests 60 and 70 are the mildest ones.

**Guidance weight on the dev requests** (300 originals, 2,275 requests):

| w | RMSE | Inside | BERTScore | Degenerate or echo | Copies |
|---|---|---|---|---|---|
| 1 | 24.6 | 17.9% | 0.851 | 0.7% | 1.2% |
| 1.5 | 22.3 | 21.2% | 0.851 | 1.1% | 2.4% |
| 2 | 20.9 | 21.6% | 0.847 | 1.1% | 2.7% |
| **3 (chosen)** | **20.4** | 23.6% | 0.848 | 0.6% | 3.4% |

Every weight was within the meaning limits, so the rule took the lowest RMSE, which is the largest weight in the grid.

**Reference metrics (LREC 2026)** on the 4,216 requests that have references, as SARI / BLEU / BERTScore:

| Model | SARI | BLEU | BERTScore |
|---|---|---|---|
| g1 | 43.1 | 33.5 | 0.822 |
| g1 · balanced | 42.9 | 33.4 | 0.823 |
| g1 · guided, w = 1 | 42.9 | 33.6 | 0.822 |
| g1 · guided, w = 3 | 42.3 | 35.2 | 0.819 |

**Training:**

| Model | Steps | Hours | Final validation loss |
|---|---|---|---|
| g1 | 3,762 | 2.2 | 0.610 |
| g1 · balanced | 4,593 | 3.1 | 0.620 |
| g1 · guided | 3,762 | 2.3 | 0.610 |

### Predictions vs observed

| Prediction | Observed | |
|---|---|---|
| g1 RMSE 15–21 | 24.1 | **wrong**: worse |
| g1 inside 18–28%, within one level 45–65% | 18.9%, 51.7% | right, at the low end |
| g1 ρ within 0.60–0.80, slope within 0.55–0.80 | 0.55, 0.59 | ρ **wrong** (below); slope right, at the low end |
| g1 mean at request 0 / 60 (originals at 70) / 70 | 15–28 / 50–62 / 55–68 | 22.8 / 54.9 / 65.6: right |
| g1 copies < 2%, top request 2–10% | 1.5%, 5.4% | right |
| g1 BERTScore 0.80–0.86; degenerate < 1%, echo < 0.3% | 0.851; 0.9%, 0.1% | right |
| SARI / BLEU / BERTScore against references: 40–46 / 30–45 / 0.80–0.86 | 43.1 / 33.5 / 0.822 | right |
| Balanced: RMSE within ±0.7; inside +0 to +3 points | +0.28; +1.1 | right |
| Balanced: slope +0.02 to +0.08; request 0 1–4 lower; request 70 1–4 higher | −0.010; −0.8; −0.05 | **wrong**: no effect at either end |
| Guided, w = 1: same as g1 | RMSE +0.48 (just significant), the rest not | right |
| Chosen w 1.5 or 2 | 3 | **wrong** |
| Guided, chosen w: RMSE −2 to +0.5; inside +2 to +6 points | −4.0; +3.8 | RMSE better than predicted; inside right |
| Guided, chosen w: ρ +0.02 to +0.06; slope +0.10 to +0.25 | +0.25; +0.47 | **wrong**: much larger |
| Guided, chosen w: BERTScore −0.003 to −0.01; degenerate up to +1 point | −0.005; −0.1 points | right |
| Best of 2: inside 28–40%, RMSE 12–16 | 32.4%, 16.6 | inside right; RMSE just outside |
| Best of 4: inside 40–60%, RMSE 8–13, ρ within 0.85–0.93 | 51.6%, 11.3, 0.93 | right |

### What this suggests

1. **One sample hides most of the model's control.** g1's mean follows the request, but compressed: about 0.6 points
   per requested point, from 22.8 at request 0 to 65.6 at request 70. Each single output is also very variable:
   four samples of the same request differ in READ-IT global by a median standard deviation of 15.4 points, more
   than a level. The closest of four samples reaches ρ within 0.93 and puts 52% of outputs inside the level, with
   unchanged meaning. Most of g1's RMSE is sampling spread, not missing control.
2. **Balancing the levels does nothing.** Levels 0 and 70 had about half as many pairs as the others, and showing
   them as often moved neither end: request 0 lands at 22.0 instead of 22.8, and request 70 at 65.5 instead of 65.6.
   The compressed curve is not a frequency problem.
3. **Guidance stretches the curve, and at w = 3 too far.** The slope within original goes from 0.59 to 1.05, and ρ
   within from 0.55 to 0.80, as good as best of 2. It costs a second forward pass instead of READ-IT scoring.
   - **Strong requests** are now reached: request 0 lands at 10.4, with 67% inside (g1: 22.8, 29%).
   - **Mild requests** are now under-simplified. At requests 60 and 70, 57% and 67% of outputs stay at or above the
     original's level (g1: 27% and 34%), and 13.8% of request-70 outputs are copies.
   - **Meaning** drops where the edits get heavier: BERTScore 0.730 at request 0, against 0.789 for g1.
   - The dev rule minimises overall RMSE, which the strong requests dominate, so it took the largest weight in the
     grid.
4. **Training with the "no level" token costs nothing.** At w = 1 the guided model is g1's equal: RMSE +0.48, just
   significant, and nothing else significant.

Not tested, possible next steps:
- a guidance weight that depends on the requested level, or a lower one for mild requests;
- guidance combined with best-of-n;
- a lower sampling temperature, against the 15-point spread.
