# How the model uses the control tokens: questions 1 and 2

The first two experiments of [Section 5.2 of the controllability report](../../../../reports/improving_controllability.md#52-experiments-from-cheapest-to-most-involved), run first on `v5_targets` (Qwen3-0.6B-Base trained on [dataset v5](../../../v5/README.md)), then on the two v7 models ([below](#v7--targets-1-epoch)):

1. **Does each token change *what* is edited, beyond the overall score?**
2. **Are the three dimensions controlled separately?**

## Design

**The same original under every request.**
- **Originals:** 1,000 test originals, drawn at random (seed 0) among the 2,369 whose three levels are all at least 30. Each dimension can then be lowered by 1, 2 and 3 levels on every one of them.
- **Requests:** for each original, every level of each dimension below the original's own level (as in the main evaluation), in two modes, plus one anchor. 34,702 requests in all.
- **Decoding:** greedy, so two outputs of the same original differ only because the control tokens differ. The main evaluation samples at temperature 0.7.

| Mode | Moved dimension at level L | The other two | Requests |
|---|---|---|---|
| **free** | base, lexical or syntax | `any` | 16,851 |
| **keep** | base | lexical `any`, syntax at the original's level | 6,029 |
| | lexical | base and syntax at the original's levels | 3,521 |
| | syntax | base at the original's level, lexical `any` | 7,301 |
| **anchor** | none | base and syntax at the original's levels, lexical `any` | 1,000 |

Keep mode holds the other dimensions where the training data allows it. v5 has same-level targets for base (11.9% of training pairs) and syntax (5.9%), but never for lexical (0 pairs), so lexical stays `any` unless it is the one moved.

**What is recorded:**
- READ-IT of every output (computed as the IMPaCTS labels);
- Profiling-UD;
- five edit operations against the original: length ratio, and the shares of tokens kept, deleted, replaced and inserted (difflib on word tokens);
- the residual stream at the last prompt token, at every layer;
- the trained embeddings of the 33 target tokens.

### Question 1: what is edited

- **Effect of a token.** For each token and mode, outputs three levels below the original are compared with outputs one level below, on the same original. The comparison uses paired Wilcoxon tests and rank-biserial r per feature (positive = larger at the stronger request), with Benjamini–Hochberg q.
  - This is the counterfactual contrast the report asks for: the effect of the token itself, not of original versus simplified.
  - The page also shows the full dose-response from one to nine levels below the original.
- **Same edits or different edits?**
  - **Profile similarity:** the Spearman correlation, across features, between two tokens' r. If the three tokens were one "simplify more" dial, their profiles would correlate near 1 and differ only in strength.
  - **Identical outputs:** how often two requests give the same text, for adjacent levels of one token and for different tokens at the same distance from the original.
  - **Which token was moved?** A logistic regression (5-fold cross-validation, grouped by original) on the outputs one to three levels below the original. It is fed either the change in the overall READ-IT score alone, or the change in the linguistic profile and the edit operations.
  - **Matched outputs:** for each original and pair of tokens, the two outputs (any levels, different texts) whose overall READ-IT scores are closest, kept if within 2 points. The features that still differ show what each token edits at equal overall simplicity.

### Question 2: separate control

- **Specificity matrix.** For each moved token, the within-original slope of each achieved READ-IT score (base, lexical, syntax, global) on the requested level, in achieved points per requested point.
  - Separate control means 1 on the diagonal and 0 off it.
  - There are 95% bootstrap intervals over originals, and the training data's co-movement for reference: the slope of each score's change on the moved dimension's change, across v5's training pairs.
- **Level directions.** For each token, the direction along which its level moves the representation:
  - **Input embeddings:** the regression of the ten level embeddings on the level, with a permutation null.
  - **Residual stream at the last prompt token, at every layer:** the within-original regression of the state on the requested level.
  - **Reported:** cosines between the three directions, and the split-half reliability of each.
  - **Readout:** each token's level read along each direction, with the direction estimated on half the originals and the level read on the other half. A dimension is represented separately if its own direction reads it and the others' do not.

## Predicted before the run (2026-11-12)

From v5's sampled main evaluation (free mode), the within-original slopes are already known, in points of achieved score per requested point:

| Moved token | base | lexical | syntax | global |
|---|---|---|---|---|
| base | 0.58 | 0.03 | 0.55 | 0.57 |
| lexical | 0.22 | 0.06 | 0.19 | 0.21 |
| syntax | 0.37 | 0.03 | 0.65 | 0.65 |

Base and syntax tokens move base and syntax alike. The lexical token moves base and syntax more than lexical, and no token moves READ-IT lexical, which is mostly word-length dispersion, not vocabulary.

| | Prediction |
|---|---|
| Free mode, greedy | the matrix above, within ±0.1 per cell |
| Keep mode, base token (syntax held) | base 0.35–0.6; syntax falls to 0.15–0.35 |
| Keep mode, syntax token (base held) | syntax 0.45–0.7; base falls to 0.1–0.25 |
| Keep mode, lexical token (base and syntax held) | every cell below 0.1 |
| Identical outputs, adjacent levels of one token (free) | 30–55% |
| Profile similarity, base vs syntax (free / keep) | 0.6–0.85 / 0.3–0.7 |
| Profile similarity, base vs lexical (free) | ≥ 0.8: the lexical token is a weak base token |
| Which token: accuracy from the overall READ-IT change alone / from the profile (keep, 3 classes) | 40–50% / 55–70% |
| Profile of the syntax token beyond base | more sentences, fewer tokens per sentence, less subordination, shallower trees at matched overall score |
| Lexical token and basic vocabulary (in_dict) | \|r\| < 0.15 in both modes |
| Embedding directions: cosine base–syntax / base–lexical | 0.2–0.5 / 0.1–0.4 |
| Residual stream, middle layers: cosine base–syntax | 0.4–0.7 (a shared "simplify more" component) |

**Confidence:**
- about 70% that keep mode cuts the base token's leakage into syntax by at least a third;
- about 60% that the profile beats the overall score at telling the tokens apart by at least 10 points;
- about 50% that the lexical token's profile correlates at least 0.8 with the base token's.

## Files

| File | Content |
|---|---|
| [`build_requests.py`](build_requests.py) | `requests.csv` |
| [`generate.py`](generate.py) | greedy outputs, states and embeddings → `outputs/<model>/interp/` |
| [`score.py`](score.py) | READ-IT and Profiling-UD of every distinct text (outputs and originals), with the parse check |
| [`analyze.py`](analyze.py) | `results/interp_<model>.{json,md}` |

```
CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python data/v1/experiments/interp/generate.py \
    --adapter data/v1/experiments/runs/v5_targets/adapter --out data/v1/experiments/outputs/v5_targets/interp
CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python data/v1/experiments/interp/score.py --out data/v1/experiments/outputs/v5_targets/interp
python data/v1/experiments/interp/analyze.py --model v5_targets
```

`generate.py` caps its GPU memory (`--memory-fraction 0.25`) and computes the hidden states with the transformer body only, so it can run next to a training job on GPU0.

## Results (2026-11-12)

34,702 greedy generations, 7,872 distinct outputs. READ-IT could not score 18 distinct texts (80 requests; all very short outputs).

**Two versions of the held mode.** In the first run, lexical stayed `any` when base or syntax was lowered, because v5 never saw a lexical target at the original's own level. The run was then redone symmetrically: whichever token is lowered, the other two are held, lexical included (`build_requests.py`; the first version is `--lexical-any`, kept in `outputs/v5_targets/interp_lexical_any/` and `results/interp_v5_targets_lexical_any.*`). Only the 14,330 changed requests were generated again (`generate.py --reuse`, `score.py --reuse`). The numbers below are the symmetric version, with the first version where it differs. Full numbers are in `results/interp_v5_targets.md`; the page is `interp.html` of the explorer (gh-pages), where experiments 1–4 answer question 1 and experiments 5–7 question 2.

### Question 2: separate control? No.

Specificity matrix (experiment 5): achieved points per requested point, within original, with 95% bootstrap intervals.

| Moved token | base | lexical | syntax | global |
|---|---|---|---|---|
| base · others `any` | **0.62** [0.60, 0.65] | 0.04 | **0.63** [0.60, 0.66] | 0.65 |
| lexical · others `any` | 0.21 [0.18, 0.24] | **0.04** [0.02, 0.05] | 0.20 [0.15, 0.25] | 0.21 |
| syntax · others `any` | 0.48 [0.46, 0.50] | 0.05 | **0.71** [0.69, 0.74] | 0.73 |
| base · lexical and syntax held | **−0.04** [−0.07, −0.01] | 0.00 | −0.05 [−0.07, −0.02] | −0.05 |
| lexical · base and syntax held | −0.01 | **0.00** | −0.03 | −0.02 |
| syntax · base and lexical held | 0.28 [0.26, 0.30] | 0.03 | **0.46** [0.44, 0.49] | 0.48 |
| *first version:* base · syntax held, lexical `any` | −0.04 | 0.00 | −0.04 | −0.03 |
| *first version:* syntax · base held, lexical `any` | 0.30 | 0.03 | 0.48 | 0.50 |
| training data (natural v1 pairs): base | 1 | 0.05 | 0.34 | 0.38 |
| training data: lexical | 0.47 | 1 | −0.12 | 0.26 |
| training data: syntax | 0.25 | −0.01 | 1 | 0.97 |

- **Base and syntax move together, more tightly than in the data.** Per point gained on the moved dimension, the other moves 1.01 (base token) and 0.67 (syntax token), against 0.34 and 0.25 in the training pairs.
- **Nothing moves READ-IT lexical** (at most 0.05). The lexical token moves base and syntax (0.21, 0.20) instead.
- **Held mode does not decouple, in either version.**
  - **Base lowered, lexical and syntax held:** the base token is ignored. 50% of its outputs are copies (first version 55%), and adjacent base levels give the same text 82% of the time. In v5, a syntax target at the original's own level is what the 989 identity pairs carry.
  - **Syntax lowered, base and lexical held:** the syntax token still moves base by 0.28 per point for 0.46 on syntax (ratio 0.61; 0.67 with nothing held). The held tokens weaken the syntax token without making it specific.
  - **Lexical lowered, base and syntax held** (the same requests in both versions): no effect on any score.
  - **Holding lexical as well rarely matters.** It changes 14% of the base-lowered outputs and 26% of the syntax-lowered ones, and lowers copies a little (base lowered 55% → 50%, syntax lowered 13% → 11%, anchor 49% → 44%).
- **Representations (experiments 6 and 7).**
  - Input embeddings: the three level directions are unrelated (cosines base–lexical 0.09, base–syntax −0.03, lexical–syntax 0.28; 95% of shuffled-level cosines within ±0.34). The first principal component orders the levels for syntax (Spearman 0.99) and base (0.82), not for lexical (0.07).
  - Residual stream at the last prompt token, free mode: from layer 8 on, the three tokens move the state along largely one direction. Cosines are base–lexical 0.90–0.94, base–syntax 0.75–0.82 and lexical–syntax 0.62–0.75 at layers 8–16, with split-half reliability ≥ 0.99. The base level is read along the syntax direction (0.95) as well as along its own (0.96), and syntax along base at 0.92 against 0.98 along its own. Base and syntax separate somewhat in the last layers (cosine 0.55 at layer 24).
  - Held mode (symmetric): the base and syntax directions are close to orthogonal at layers 8–16 (0.07–0.30) and opposed at 20–24 (−0.15 to −0.23). Base and lexical are aligned (0.83–0.92 from layer 8 on). Behaviourally, neither the base token nor the lexical token does anything in this mode.

### Question 1: does each token change what is edited? Mostly not.

- **Same edits, different strengths** (free mode, experiments 2 and 3). All three tokens shorten the sentence, drop verbal heads and subordinate clauses, and flatten the tree (r −0.60 to −0.88, 3 levels below vs 1). Their profiles correlate 0.82 (base–lexical), 0.73 (base–syntax) and 0.58 (lexical–syntax).
- **Which token was moved** (3 classes, chance 33%):

  | | free | free, no copies | held | held, no copies |
  |---|---|---|---|---|
  | overall READ-IT change alone | 48.8% | 47.1% | 40.5% | 43.0% |
  | three READ-IT changes | 49.2% | 46.8% | 41.0% | 42.9% |
  | profile change + edit operations | 51.2% | 49.0% | 40.5% | 41.6% |
  | all of the above | 51.8% | 49.5% | 40.8% | 41.9% |

  In held mode the base-lowered and lexical-lowered outputs are confused with each other (77% of them are the same text at the same distance).

  The linguistic profile adds 2 points over the overall score: the tokens differ mainly in how much they simplify.
- **How they get there** (experiment 1, free).
  - **Syntax token:** as the request strengthens, it rewrites: replaced words r 0.83, inserted 0.68, extra sentences 0.72, kept −0.93.
  - **Base token:** mostly deletes (deleted 0.73, replaced 0.19, inserted −0.07).
  - **Lexical token:** a weaker base (deleted 0.56, replaced −0.05).
- **Same sentence, same distance** (experiment 4, free; 1–3 levels below, averaged per sentence). The same distance is not the same strength: lexical 1–3 levels below asks for lexical 0–30, syntax for 40–70. The differences are mostly about how much is simplified.
  - Lowering syntax instead of base gives the same text 31% of the time. Otherwise the output is milder: READ-IT global 9.4 points higher, copies 18 points more often, 12.5% more of the original's words kept (r 0.51–0.85).
  - Lowering lexical gives much stronger simplification than base or syntax at the same distance: READ-IT global 17–27 points lower, 12–14 fewer tokens.
- **At equal overall READ-IT** (experiment 4, free, no copies):
  - Base vs lexical outputs (336 originals): no feature differs at q < 0.01.
  - Base vs syntax (471): base-token outputs delete more (r 0.30) and have fewer subordinate clauses (−3.5 points, r −0.30), shorter maximum dependency links (−0.33) and fewer indicative auxiliaries (−0.47).
  - Lexical vs syntax (382): lexical-token outputs are about one token shorter (r −0.25).
- **No token simplifies vocabulary.** The basic-vocabulary share (in_dict) changes by |r| ≤ 0.10 in both modes.
- **Identical outputs.**
  - Free mode: adjacent levels of a token give the same text 48–50% of the time. Two tokens at the same distance give the same text 23–34% (base with lexical or syntax) and 12–14% (lexical with syntax).
  - Copies: 47% at the mildest syntax request, 11% base, 3% lexical. Held mode: about half for base and lexical at every distance, 29% for syntax one level below.

### Predictions vs observed

The predictions were written for the first version of the held mode, and the held rows below are that version. The symmetric version gives:
- held base −0.04 (base) and −0.05 (syntax);
- held syntax 0.46 (syntax) and 0.28 (base);
- held profile similarity base–syntax −0.03;
- the held classifier 41% / 41%.

The verdicts are the same.

| | Predicted | Observed | |
|---|---|---|---|
| Free mode, greedy, within ±0.1 of the sampled matrix | ±0.1 per cell | largest gap 0.11 (syntax token → base, 0.48 vs 0.37) | mostly yes |
| Held, base token: base / syntax | 0.35–0.6 / 0.15–0.35 | −0.04 / −0.04 (the model copies) | no |
| Held, syntax token: syntax / base | 0.45–0.7 / 0.1–0.25 | 0.48 / 0.30 | half |
| Held, lexical token: every cell below 0.1 | yes | yes (all ≤ 0.03) | yes |
| Identical outputs, adjacent levels (free) | 30–55% | 48–50% | yes |
| Profile similarity base–syntax, free / held | 0.6–0.85 / 0.3–0.7 | 0.73 / 0.29 | yes / just below |
| Profile similarity base–lexical (free) | ≥ 0.8 | 0.82 | yes |
| Classifier: overall score / profile (held) | 40–50% / 55–70% | 40% / 41% (free: 49% / 51%) | yes / no |
| Syntax token at matched score: more sentences, less subordination, shallower | yes | no: at equal overall score, base-token outputs have less subordination | no |
| Lexical token and basic vocabulary | \|r\| < 0.15 | 0.07 (free), 0.03 (held) | yes |
| Embedding cosines base–syntax / base–lexical | 0.2–0.5 / 0.1–0.4 | −0.03 / 0.09, within the shuffled null | no |
| Residual stream, middle layers, base–syntax | 0.4–0.7 | 0.75–0.82 | no (higher) |

Confidence statements:
- **Held mode cuts the base token's leakage into syntax by a third (70%).** No, in substance: leakage fell to zero only because the base token stopped working.
- **The profile beats the overall score by at least 10 points (60%).** No: 2 points.
- **The lexical profile correlates at least 0.8 with base (50%).** Yes: 0.82.

### What this suggests

- **Most of the control is one dial.** v5's three tokens reach the generator as largely one "how much to simplify" signal. The tokens differ in strength, and somewhat in how the edit is made (the syntax token rewrites, the base token deletes), but at equal overall readability they make nearly the same edits. A per-dimension request cannot be met by this architecture as trained: the inputs are separate, but the network merges them by layer 8.
- **Held requests are confounded in v5.** Its identity pairs make "syntax at the original's level" mean "copy". The same experiment on v3_targets (same-level targets, no identity pairs) would test whether holding works when that confound is absent.
- **Lexical.** The lexical token has no ordinal embedding and no effect on READ-IT lexical, consistent with READ-IT lexical measuring word-length dispersion rather than vocabulary. A lexical target defined on vocabulary would be needed before lexical control can be studied mechanistically.

## v7 · targets, 1 epoch

The same experiments, on the same 34,702 requests, run on `v7_1ep_targets`: dataset v7 trained for one epoch, the adapter at the lowest validation loss ([data/v7/README.md](../../../v7/README.md#one-epoch-v7_1ep_targets)). It differs from v5 in its training format:
- **v5:** each pair is one example with all its targets, each dropped with probability 0.3.
- **v7:** one example per target with that token alone, plus a combined example per pair. Its identity pairs carry base as well as syntax targets.
- **Exposure:** both models see each pair about three times.

Run by `run_v7_1ep_targets.sh` after the main evaluation; `score.py --reuse outputs/v5_targets/interp` copies the scores of texts that v5 also produced.

### Predicted before the run (2026-11-13, while training)

What the training data says about the held prompts (counts of training examples; v5 before target dropout):

| Prompt | v5: count · share of identity pairs | v7: count · share of identity pairs |
|---|---|---|
| syntax at the original's level, base lowered | 3,679 · 0% | 3,679 · 0% |
| syntax at the original's level, base `any` | 1,194 · 83% | 5,499 · 19% |
| syntax and base at the original's level | 379 · 0% | 1,287 · 71% |
| base at the original's level, syntax lowered | 9,675 | 9,675 |

In v5, "syntax at the original's level" alone means "copy". In v7 it does not, and copying is tied to base and syntax both at the original's level. A held lexical token is outside both models' data.

| | v5 (observed) | Prediction for v7 · 1 epoch |
|---|---|---|
| Others any, base token: base / syntax | 0.62 / 0.63 | 0.5–0.75 / 0.45–0.75 |
| Others any, syntax token: base / syntax | 0.48 / 0.71 | 0.3–0.55 / 0.6–0.8 |
| Others any, lexical token: base / lexical / syntax | 0.21 / 0.04 / 0.20 | 0.2–0.4 / ≤ 0.08 / 0.2–0.4 |
| Others any: what the other dimension moves per point gained, base token / syntax token (data: 0.34 / 0.25) | 1.01 / 0.67 | **0.6–1.1 / 0.4–0.8**: still well above the data |
| Held, base token: base / syntax / copies | −0.04 / −0.05 / 50% | **0.15–0.5 / ≤ 0.2 / 10–35%** |
| Held, syntax token: syntax / base | 0.46 / 0.28 | 0.4–0.65 / 0.15–0.35 |
| Held, lexical token: every cell | ≤ 0.03 | below 0.1 |
| Identical outputs at adjacent levels (others any) | 48–50% | 35–55% |
| Profile similarity base–syntax (others any) | 0.73 | 0.6–0.85 |
| Classifier from the overall score / the profile (others any) | 49% / 51% | 45–55% / at most 5 points more |
| Embeddings: lexical levels ordered (PC1 Spearman) | 0.07 | below 0.5 (lexical 60–90 are almost never trained) |
| Residual stream, layers 8–16, cosine base–syntax (others any) | 0.75–0.82 | 0.6–0.85 |

- **Main test:** held base. If the base token works when syntax is held (base slope at least 0.15, copies at most 35%), v5's held failure came from its syntax-only identity pairs, not from the architecture.
- **Confidence:**
  - about 60% that held base reaches a slope of at least 0.15 with at most 35% copies;
  - about 75% that, with the other tokens `any`, base and syntax still move together at least twice as much as in the data;
  - about 70% that the residual stream still carries base and syntax on largely one direction (cosine at least 0.6 at layers 8–16).

### Results (2026-11-13)

The one-epoch model did worse than the three-epoch one on the main evaluation ([One epoch](../README.md#one-epoch-is-the-heavy-editing-overfitting)), so the three-epoch `v7_targets` was run as well (`outputs/v7_targets/interp/`, scores reused from the one-epoch run). Both are reported. Distinct outputs: v5 7,872; v7 · 3 epochs 10,247 (125 requests unscored); v7 · 1 epoch 5,795 (56). Full numbers: `results/interp_v7_targets.md` and `results/interp_v7_1ep_targets.md`.

**Question 2: separate control? Still no, in either v7 run.** Points of achieved score per requested point, within original:

| Moved token, mode | v5 | v7 · 3 epochs | v7 · 1 epoch |
|---|---|---|---|
| base, others `any`: base / syntax | 0.62 / 0.63 | 0.61 / 0.62 | 0.69 / 0.76 |
| lexical, others `any`: base / lexical / syntax | 0.21 / 0.04 / 0.20 | 0.21 / 0.05 / 0.24 | 0.12 / 0.04 / 0.17 |
| syntax, others `any`: base / syntax | 0.48 / 0.71 | 0.33 / 0.55 | 0.41 / 0.60 |
| Other dimension per point gained, base token / syntax token (training data: 0.34 / 0.25) | 1.01 / 0.67 | 1.01 / 0.60 | 1.10 / 0.68 |
| base, others held: base / syntax · copies | −0.04 / −0.05 · 50% | **0.51 / 0.52** · 18% | **0.45 / 0.46** · 32% |
| lexical, others held: base / syntax · copies | −0.01 / −0.03 · 46% | 0.06 / 0.12 · 42% | 0.00 / 0.02 · 55% |
| syntax, others held: base / syntax · copies | 0.28 / 0.46 · 11% | 0.34 / 0.48 · 14% | 0.31 / 0.45 · 31% |
| Anchor (all three held): copies | 44% | 47% | 60% |

- **With the other tokens `any`, nothing changed.** The base token moves syntax as much as base in both runs, and leakage per point gained is still 2.4 to 3 times the training data's.
- **Holding now lets the base token work, but does not hold.**
  - v5's held failure came from its identity pairs. In v7, base lowered with syntax held moves base 0.51 per point, with 18% copies instead of 50%.
  - Syntax moves just as much (0.52), as if it were not held. Lowering syntax with base held moves base 0.34 per point, a larger share than with nothing held (0.71 of the syntax change against 0.60).
  - So the held tokens stop the copying, but the model does not read them as constraints on the other dimensions.
- **The lexical token** is about v5's in the three-epoch run, and weaker in the one-epoch run. No token moves READ-IT lexical.

**Question 1: does each token change what is edited? Even less than in v5.**

| | v5 | v7 · 3 epochs | v7 · 1 epoch |
|---|---|---|---|
| Profile similarity, others `any`: base–syntax / base–lexical | 0.73 / 0.82 | 0.84 / 0.74 | 0.93 / 0.88 |
| Which token was moved, others `any`: from the overall READ-IT change / from everything (chance 33%) | 48.8% / 51.8% | 44.5% / 48.8% | 45.6% / 47.3% |
| Replaced words as the request strengthens (r): base / syntax token | 0.19 / 0.83 | 0.40 / 0.58 | 0.54 / 0.79 |
| Deleted words (r): base / syntax token | 0.73 / 0.81 | 0.59 / 0.49 | 0.81 / 0.72 |
| Copies, others `any`: base / syntax token, all requests (one level below the original) | 4% (11%) / 12% (47%) | 4% (12%) / 6% (22%) | 10% (28%) / 12% (34%) |
| Base vs lexical token, matched on overall READ-IT | no feature differs | no feature differs | no feature differs |

- **The tokens' edits are more alike than v5's.** The syntax token still rewrites more than the base token, but the gap is smaller (replaced words: v5 0.19 against 0.83; v7 · 3 epochs 0.40 against 0.58).
- **Telling which token was moved is harder:** 44.5% from the overall READ-IT change alone, against v5's 48.8%.

**Representations.**

| | v5 | v7 · 3 epochs | v7 · 1 epoch |
|---|---|---|---|
| Embedding cosines base–lexical / base–syntax / lexical–syntax (95% of shuffled levels within) | 0.09 / −0.03 / 0.28 (±0.34) | 0.29 / −0.10 / 0.02 (±0.26) | 0.31 / 0.14 / 0.22 (±0.27) |
| Embedding levels ordered (PC1 Spearman): base / lexical / syntax | 0.82 / 0.07 / 0.99 | 0.92 / 0.42 / 1.00 | 0.89 / 0.44 / 1.00 |
| Residual stream, others `any`, layers 8–16: cosine base–syntax / base–lexical | 0.75–0.82 / 0.90–0.94 | 0.85–0.93 / 0.88–0.92 | 0.97–0.99 / 0.91–0.97 |
| Residual stream, others `any`, layers 20–28: base–syntax | 0.55–0.67 | 0.67–0.80 | 0.99 |
| Residual stream, held, layers 8–16: base–syntax | 0.07–0.30 | 0.73–0.82 | 0.94–0.98 |
| Readout of the base level along the syntax direction / its own, layers 8–16 | 0.83–0.95 / 0.95–0.96 | 0.97–0.99 / 0.97 | 0.98–0.99 / 0.95 |

- **The embeddings are still separate inputs.** Their level directions are unrelated, apart from base–lexical at the edge of the null. Lexical embeddings are now partly ordered (0.42; v5: 0.07); lexical-alone examples are eight times as frequent in v7.
- **The residual stream merges base and syntax even more than in v5.** The cosine is 0.85–0.93 at layers 8–16 in the three-epoch run and 0.97–0.99 in the one-epoch run, where they never separate.
- **Held mode in v5 was misleading.** Its near-orthogonal held directions came from a base token that did nothing. In v7, where the held base token works, it moves the state along the syntax direction (0.73–0.82).

### Predictions vs observed

The predictions were written for the one-epoch model.

| | Predicted | v7 · 1 epoch | v7 · 3 epochs | |
|---|---|---|---|---|
| Others any, base token: base / syntax | 0.5–0.75 / 0.45–0.75 | 0.69 / 0.76 | 0.61 / 0.62 | yes / just above |
| Others any, syntax token: base / syntax | 0.3–0.55 / 0.6–0.8 | 0.41 / 0.60 | 0.33 / 0.55 | yes |
| Others any, lexical token: base / lexical / syntax | 0.2–0.4 / ≤ 0.08 / 0.2–0.4 | 0.12 / 0.04 / 0.17 | 0.21 / 0.05 / 0.24 | no / yes / no (3 epochs: yes) |
| Other dimension per point gained, base / syntax token | 0.6–1.1 / 0.4–0.8 | 1.10 / 0.68 | 1.01 / 0.60 | yes |
| Held base: base / syntax / copies | 0.15–0.5 / ≤ 0.2 / 10–35% | 0.45 / 0.46 / 32% | 0.51 / 0.52 / 18% | yes / no / yes |
| Held syntax: syntax / base | 0.4–0.65 / 0.15–0.35 | 0.45 / 0.31 | 0.48 / 0.34 | yes |
| Held lexical: every cell below 0.1 | yes | yes (≤ 0.02) | no (syntax 0.12) | yes |
| Identical outputs at adjacent levels (others any) | 35–55% | 47–58% | 38–47% | mostly |
| Profile similarity base–syntax | 0.6–0.85 | 0.93 | 0.84 | no (higher) |
| Classifier: overall score / everything | 45–55% / ≤ +5 points | 45.6% / 47.3% | 44.5% / 48.8% | yes |
| Lexical embeddings ordered (PC1) | below 0.5 | 0.44 | 0.42 | yes |
| Residual stream, layers 8–16, base–syntax | 0.6–0.85 | 0.97–0.99 | 0.85–0.93 | no (higher) |

- **Main test: passed.** Held base reaches 0.45 (three epochs: 0.51) with 32% copies (18%). v5's held failure came from its syntax-only identity pairs.
- **What the prediction missed:** that the held syntax token would keep syntax in place (≤ 0.2). It does not: holding stops the copying but constrains nothing.
- **Confidences:** about 60% that held base works (yes), 75% that base and syntax still move together at least twice as much as in the data (yes: 3.2 times with the base token), 70% that the residual stream still carries them on one direction (yes, more than v5).

### What this suggests

- **The CLiC-it format did not separate the dimensions.** It fixed what the identity pairs broke in v5 (held base now works), but the three tokens are still one "simplify" dial, read along one direction from layer 8. Giving each token its own examples makes them more alike, not less.
- **The training signal is the likely cause.** In a single-token example the other dimensions are `any`, and the simplification carries whatever the pair did to them. The model learns from these examples how much a request usually changes everything, not which dimension to change. Separate control needs examples where one dimension moves and the others demonstrably do not, and natural IMPaCTS pairs rarely have them.
- **The one-epoch run is a cleaner picture of the same mechanism.** Base and syntax are on one direction at every layer from 8 to 28. The two extra epochs pull them apart a little (0.67 at layer 28), which may be why the three-epoch model copies only when the request is mild.
