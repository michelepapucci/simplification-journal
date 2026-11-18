# How the g1 models use their control token

This study looks at g1 · guided (main) and g1 (check). At w = 1 the two control the output equally well.

The guided model also has the "no level" token `<|readability_any|>`. It gives a reference point for what each
level adds, and it is what guidance works with. g1 · balanced changed nothing and is left out.

## Questions and design

**A. The level tokens, layer by layer** (`states.py`). Are the 10 levels organised in order, hierarchically, in the
model's representation, and from which layer? The 11 tokens are the levels 0–90 plus "no level".
- **Alone:** each token as the only input, followed through the embeddings and all 28 blocks. This is the token's
  own representation before it sees any text.
- **In context:** every one of the 1,000 test originals under each token. The state is taken at the last prompt
  token, the newline where generation starts, at every layer.
- **Measures per layer,** on levels 0–80 (level 90 is never a training target and stays untrained, so it serves as
  a reference):
  - **ordering:** Spearman ρ between level and the first principal component of the 9 level means;
  - **betweenness:** the share of triples a < b < c where a and c are further apart than either is from b;
  - **hierarchy:** in average-linkage clustering of the level means, the share of merges that join neighbouring
    levels, and the correlation between cophenetic distance and the gap between levels;
  - **spacing:** the distance between consecutive levels;
  - **readout:** cross-validated R² of a ridge regression from the in-context state to the requested level, with
    folds by original.

**B. Readout of the outcome.** On the 7,582 test requests, does the in-context state predict the READ-IT global the
output reaches better than the request alone? Better than the request plus the original's READ-IT scores and
length?

**C. "No level"** (guided model only). Where does its default simplification land (1,000 outputs)? Where does the
"no level" state sit on the level axis?

**D. The logits and De Mauro's basic vocabulary** (`lexicon.py`, `logits.py`). At every decoding step, how much of
the probability goes to words of the Nuovo vocabolario di base (NVdB 2016: fondamentale FO, alto uso AU, alta
disponibilità AD)?
- **Lexicon:** the word forms of the g1 texts and the test outputs, lemmatised with the profiler's stanza 1.11.0.
  A form counts as basic if its lemma, or the form itself, is in NVdB, as in READ-IT. Proper nouns, numbers and
  foreign words are left out.
- **Vocabulary items:** an item counts as a word if it is a space followed by letters (`" casa"`, `" Il"`).
- **Basic share:** at every step, of the probability given to classified words, the share on NVdB words.
- **Generated:** every test output is read back through the model that wrote it, which gives the distribution it
  was drawn from (temperature 1). This covers g1, guided at w = 1, guided at w = 3 (the guided distribution) and
  "no level".
- **Fixed text:** one text per original (the model's own output at request 40), read under all 11 tokens. Only the
  token changes, so the differences are the token's own effect.
- **Which items each level pushes up or down** compared with "no level": the mean log-probability difference per
  item.

## Predicted before the run (2026-11-16)

- **A:**
  - The input embeddings of levels 0–80 are ordered: PC1 ρ ≥ 0.9 (confidence about 70%; in v5 the base and syntax
    embeddings were ordered, the lexical ones were not).
  - In context, the levels are ordered from about layer 4 onwards (ρ ≥ 0.95). Readout R² of the requested level
    peaks at 0.85–0.98, somewhere between layers 8 and 20.
  - "Hierarchical" takes the form of an ordered line: neighbours merge first (≥ 80% of the merges in the middle
    layers), and distance grows with the gap between levels (ρ ≥ 0.9). I don't expect separate clusters.
  - Alone, the order survives the early layers but may fade in the middle layers, where the first position becomes
    an attention sink (uncertain).
  - Spacing: the low end (0–10–20) is closer together than the middle, matching the outputs, whose mean moves 4
    points per level at the low end against 6–9 in the middle (confidence about 40%).
  - Level 90, never trained, sits off the line.
- **B:** the request alone explains R² ≈ 0.25–0.30 of the achieved score. The state adds 0.03–0.10, mostly
  through what it knows about the original: it does about as well as request + original features (within 0.05). The
  15-point spread between samples caps everything near 0.5.
- **C:** "no level" outputs land at a mean READ-IT global of 33–45, about the average training target, and the
  "no level" state projects to an equivalent level of 30–50. The compressed curve is partly a pull toward this
  default.
- **D:**
  - **Generated:** the basic share rises for lower requested levels, but only slightly: request 0 vs request 70
    +1 to +4 points. READ-IT global follows syntax and length far more than vocabulary, and no earlier token moved
    the profiler's vocabulary share (`in_dict`). The realised NVdB share of the output words behaves the same.
  - **Guidance (w = 3):** the basic share at request 0 rises 1–3 points over w = 1.
  - **Fixed text:** the token alone moves the basic share by less than 2 points between levels 0 and 70.
  - **Pushed items:** for low levels, the items pushed up most are sentence breaks (`.`) and sentence-opening
    function words (` Il`, ` La`, ` è`). For high levels, they are commas and subordinators (` che`) that keep the
    original's structure. As a class, NVdB words gain slightly against words outside NVdB.

## Results (2026-11-16)

All numbers come from `results/interp_g1.json`, produced by `analyze.py`. The two models agree everywhere: unless
noted, the g1 · guided value is given, with g1 in brackets where it differs.

**A. The level tokens, layer by layer.**

| | Embeddings | Layers 1–2 (in context) | Layers 3–28 (in context) | Alone, every layer |
|---|---|---|---|---|
| Order along PC1 (ρ, levels 0–80) | 0.98 | 0.98 | **1.00** | ≥ 0.98 |
| In-between share of triples | 0.98 | 0.96 | **1.00** | ≥ 0.96 |
| Clustering merges that join neighbours | 100% | 100% | **100%** | 100% |
| Distance vs gap between levels (ρ) | – | 0.85 | 0.93–0.99 | – |
| Requested level read back (CV R²) | – | 0.01–0.14 | 0.91 at layer 3, **≥ 0.99 at layers 7–17**, 0.74–0.87 at the last layers | – |

- **One graded scale.** The levels form a single ordered line, not separate groups: every level lies between its
  neighbours, and clustering always merges adjacent levels first. The first principal component holds 83–90% of the
  spread between the level means.
- **The spacing is widest at the bottom.** At layer 12, the step 0 → 10 is 1.9 times the mean step (g1: 1.8), and
  the step 70 → 80 is 0.6 times. In the outputs, 0 → 10 is an ordinary step: 4.0 points for g1 and 6.2 for
  g1 · guided, against mean steps of 6.1 and 6.0, so 0.7 and 1.0 times. Request 0 still lands at 21–23. The extra
  separation at the bottom does not reach the outputs.
- **The output layer never predicts the control tokens,** so its rows keep their initial values.

**B. Readout of the outcome** (7,552 scored requests, cross-validated R²):

| Predictor | g1 · guided | g1 |
|---|---|---|
| Request alone | 0.259 | 0.280 |
| Request + the original's READ-IT and length | 0.269 | 0.290 |
| Original alone | 0.018 | 0.018 |
| Hidden state, best layer | 0.244 (layer 15) | 0.269 (layer 14) |

The state where generation starts predicts the outcome no better than the request does. Beyond the request, where an
output lands is sampling noise, not a target the model forms before writing.

**C. "No level".**
- **Outputs:** 996 scored, mean READ-IT global 42.6 (quartiles 22.7 / 40.1 / 61.5). That is the level of request
  40, whose outputs average 41.9.
- **State:** it sits where level 40 would be on the main direction (equivalent level 38–44 across layers), the same
  place as the never-trained level 90. Both sit at the centre of the scale.

**D. De Mauro's basic vocabulary.**
- **Generated outputs, next-token distribution:** 88.5–89% of the probability on classified words goes to basic
  vocabulary at every requested level.
  - Request 0 minus request 60: −0.45 points [−1.0, +0.1] for g1, and −0.5 [−1.0, −0.0] for g1 · guided.
  - Guided at w = 3 has 87.7% at request 0 against 88.5% at w = 1.
- **Words in the outputs:** 91–92% basic at every requested level, against 90.2% in the originals.
- **IMPaCTS references do move:** 94.4% at level 0, 91.1% at level 70.
- **Fixed text, the token alone:** the share changes by less than 0.2 points between level 0 and level 80.
- **What the token pushes,** level 0 against "no level" (g1 · guided; g1 is the same), as mean probability ratios:
  - **up:** end-of-text ×3.1; sentence openings with an article (Il ×1.46, Le ×1.33, La ×1.31); full stop ×1.41;
    simple verbs (ha ×1.19, è ×1.16);
  - **down:** semicolon ×0.28; openings with a preposition or adverb (Per ×0.29, Dur(ante) ×0.43, Con ×0.43,
    Non ×0.45, Si ×0.48, In ×0.58);
  - **by class:** fondamentale ×0.99, alto uso ×1.00, alta disponibilità ×1.03, outside NVdB ×0.99;
  - **level 70 is the mirror:** semicolon ×2.4, Per ×1.85, colon ×1.34, comma ×1.23 up; the paragraph break ×0.56
    and the full stop ×0.74 down.

### Predictions vs observed

| Prediction | Observed | |
|---|---|---|
| Input embeddings ordered, ρ ≥ 0.9 | 0.98 | right |
| In context ordered from about layer 4 (ρ ≥ 0.95) | from layer 3, ρ = 1.00 | right |
| Readout R² peaks at 0.85–0.98 between layers 8 and 20 | ≥ 0.99 at layers 7–17 | right place, higher |
| Neighbours merge first (≥ 80%); distance grows with the gap (ρ ≥ 0.9) | 100%; 0.93–0.99 | right |
| Alone, the order may fade in the middle layers | it doesn't (ρ ≥ 0.98) | wrong |
| Low end closer together, like the outputs | the low end is the **widest**; in the outputs, 0 → 10 is an ordinary step (0.7–1.0 times the mean) | **wrong** |
| Level 90 sits off the line | at its centre (equivalent level about 40) | wrong |
| Request alone R² 0.25–0.30; state adds 0.03–0.10 | 0.26–0.28; the state adds nothing (−0.01) | half right |
| "No level" outputs at 33–45; its state at an equivalent level of 30–50 | 42.6; 38–44 | right |
| Basic share rises for lower requests by 1–4 points | flat (−0.5) | **wrong** |
| Guidance raises it at request 0 by 1–3 points | lowers it 0.8 points | **wrong** |
| The token alone moves it by < 2 points | < 0.2 points | right |
| Low levels push full stops and sentence-opening function words; high levels push commas and subordinators | full stops, articles and end-of-text up at level 0; semicolon, colon, comma and prepositional openings up at level 70 | right (but no " che") |
| NVdB words gain slightly as a class | no gain (×0.99) | wrong |

### What this suggests

1. **The model has a clean, graded representation of the request.** A single ordered scale from layer 3 to the
   last layer, with the requested level recoverable almost perfectly. The weak control (slope 0.59) does not come
   from a muddled representation.
2. **The shortfall happens while writing.** The scale is widest at the bottom, where the outputs fall shortest
   (request 0 lands at 21–23). The model distinguishes those requests internally but cannot turn the difference into
   text. Nor does
   the state predict more about the outcome than the request: the spread is sampling.
3. **"No level" is the middle of the scale** (level 40). This explains why guidance against it stretches the curve
   in both directions, and overshoots at the top, where requests above 40 are pushed away from 40 toward the
   original.
4. **The token controls structure, not vocabulary.** It raises or lowers the probability of ending the text, of
   starting a new sentence with a subject, and of semicolons and prepositional openings. The share of basic
   vocabulary does not move, although the real simplifications at low levels do use more of it. This matches what
   READ-IT global rewards (length and syntax), and it means lexical simplification would need its own signal.

## Second round (requested 2026-11-16)

These changes came from reviewing the first round:
- **"No level" appears only in C.** Since it behaves as level 40, comparisons elsewhere use level 40 as the baseline.
- **Level 90 is dropped from A.** In g1 it shares its untrained starting vector with "no level", and in g1 · guided
  the two sit within a fifth of a level-step of each other near level 40, so one hid the other.
- **A gets more projections.** MDS of the level means, plus t-SNE and UMAP of single originals (300 originals ×
  levels 0–80, every second layer), each both as is and with each original's own mean removed.
- **C compares "no level" with the levels themselves:** with the outputs of requests 30–50 on the same originals,
  through the textual similarity (chrF) of its output to each level's output, and through its state's distance to
  every level's state, layer by layer.
- **D adds the training data** (basic share of the g1 training simplifications by target level) and drops the
  word-by-word tables.
- **E is new.** Each original's level-40 output is read under levels 40, 30, 20, 10 and 0. It measures the
  probability of word classes (the part of speech of each vocabulary word in the stanza lexicon, punctuation
  types, end-of-text, word pieces), the tokens that change most steadily, the same classes in the texts the models
  write against the IMPaCTS references (Profiling-UD), and one original step by step.

### Predicted before the second round

- **A:** t-SNE and UMAP of single originals as is group points by original, not by level: the content of the sentence
  dominates the state. With each original's mean removed, the levels form an ordered band in the middle layers.
  MDS shows the same ordered curve as PCA.
- **C:**
  - The "no level" output is textually closest to the outputs of requests 30–50. Requests 40 ± 10 are the closest
    for 40–55% of originals (sampling spreads the rest).
  - The "no level" state is nearest to level 40 in every layer from 3 on.
  - Its outputs' READ-IT distribution matches request 40's more closely than request 30's or 50's.
- **E** (level 0 against 40, fixed texts):
  - end-of-text ×2.5–3.5, full stop ×1.3–1.6, semicolon ×0.2–0.4, comma ×0.7–0.9;
  - adpositions ×0.85–0.95, subordinating conjunctions ×0.7–0.9;
  - verbs and auxiliaries ×1.05–1.2, determiners ×1.05–1.2, nouns ×0.95–1.05.
  - End-of-text, full stop and semicolon change monotonically from 40 to 0. The other classes change in the same
    direction, more weakly.
  - In the texts, lower requested levels have more verbs and auxiliaries and fewer adpositions, in the same direction
    as the IMPaCTS references at those levels. I don't predict the size.

### Second-round results (2026-11-17)

Results are in `results/interp_g1.json`, with the projections in `outputs/interp/projections.json`. g1 · guided
values are given; g1 agrees within a few hundredths.

**A. Projections.** For each point of the t-SNE and UMAP maps (300 originals × 9 levels), I counted its 10
nearest neighbours: how many come from the same original, and how many from the same level or an adjacent one.

| UMAP map, raw states | Layers 2–6 | Layer 8 | Layers 10–16 | Layer 18 | Layers 20–28 |
|---|---|---|---|---|---|
| Neighbours from the same original | 0.72–0.80 | 0.42–0.45 | **0.10–0.19** | 0.53–0.59 | 0.66–0.80 |
| Neighbours within one level | 0.26–0.30 (chance 0.31) | 0.46–0.51 | **0.73–0.88** | 0.35–0.40 | 0.27–0.31 |

- **The middle layers are about the level.** In layers 10–16 the requested level is the main thing the state
  encodes. Before and after, the sentence's content dominates.
- **Within original, the levels are ordered at every layer.** With each original's own mean removed, 87–100% of
  neighbours are within one level. t-SNE gives the same picture.
- **MDS** places the level means on the same ordered curve as PCA.

**C. "No level"** (g1 · guided).
- **Its state is nearest to level 40's at every layer.** At layer 12 it is 0.27 of a level-step from level 40 and
  0.66 from level 30.
- **Its outputs' READ-IT is closest to request 40's:**

  | Request | Kolmogorov–Smirnov distance to "no level" | Mean READ-IT reached |
  |---|---|---|
  | No level | – | 42.6 |
  | 30 | 0.076 | 38.7 |
  | 40 | **0.047** | 41.9 |
  | 50 | 0.151 | 49.7 |

- **Textual similarity can't tell.** The "no level" output is the most similar (chrF) to the level-40 output for 16%
  of originals, against 12–16% for every other level (chance: 14%). Mean chrF rises toward the mild levels
  (51.5 at level 0, 58.3 at level 60), which keep more of the original's wording. Sampling makes two outputs
  differ more than two neighbouring levels do.

**D. Basic vocabulary in the training data.**
- **The data has only a small gradient.** The g1 training simplifications are 92.5% basic at target level 0,
  91.2% at 40 and 90.3% at 70; their originals are 89.8%.
- **The model keeps the average, not the gradient.** It reproduces the average rise over the originals (+1.5
  points), not the 2-point gradient across levels.

**E. Word classes on fixed level-40 texts.**

- **Words by part of speech:** probability relative to level 40, geometric mean over the text's words, 95%
  interval over originals for level 0.

  | Class | Words | Level 30 | Level 20 | Level 10 | Level 0 [95%] |
  |---|---|---|---|---|---|
  | Commas | 1,554 | ×0.92 | ×0.82 | ×0.68 | **×0.49** [0.46, 0.52] |
  | Subordinating conjunctions | 264 | ×0.97 | ×0.93 | ×0.87 | **×0.74** [0.68, 0.81] |
  | Adverbs | 964 | ×0.97 | ×0.92 | ×0.85 | **×0.74** [0.71, 0.78] |
  | Pronouns | 901 | ×0.98 | ×0.95 | ×0.90 | **×0.82** [0.79, 0.85] |
  | Verbs | 2,784 | ×0.98 | ×0.96 | ×0.92 | **×0.87** [0.86, 0.89] |
  | Adpositions | 5,231 | ×0.99 | ×0.97 | ×0.94 | **×0.89** [0.88, 0.89] |
  | Coordinating conjunctions | 897 | ×0.99 | ×0.98 | ×0.95 | ×0.91 [0.89, 0.93] |
  | Adjectives | 2,024 | ×0.99 | ×0.98 | ×0.97 | ×0.95 [0.93, 0.96] |
  | Determiners | 3,050 | ×1.01 | ×1.01 | ×1.00 | ×0.97 [0.96, 0.98] |
  | Nouns, numerals, auxiliaries, proper nouns | 10,740 | ×1.00–1.01 | ×1.00–1.02 | ×0.99–1.02 | ×0.99–1.00 |
  | Full stops | 1,279 | ×1.07 | ×1.13 | ×1.18 | **×1.19** [1.17, 1.22] |

- **At every step, from 40 to 30, 20, 10 and 0:**
  - end-of-text 0.16%, 0.18%, 0.21%, 0.29%, 0.51% (×3.2);
  - full stop 2.6%, 2.8%, 3.1%, 3.4%, 3.7% (×1.43);
  - comma 3.0%, 2.8%, 2.6%, 2.4%, 2.1% (×0.70);
  - semicolon or colon 0.06%, 0.05%, 0.04%, 0.04%, 0.03% (×0.49).

  The probability of ending exactly where the text ends rises from 88.7% at level 40 to 93.9% at level 0.
- **Single items:** of 135 items with at least 0.1% mean probability, 73 change in the same direction at every
  step.
  - **Steadily up most:** end-of-text ×3.2, "Il" ×1.49, "." ×1.44, "La" ×1.32, " ha" ×1.20, " è" ×1.17,
    " hanno" ×1.15, " sono" ×1.14.
  - **Steadily down most:** "," ×0.70, " anche" ×0.79, " si" ×0.90, " come" ×0.91, " che" ×0.92, " in" ×0.93.
- **In the texts** (Profiling-UD share of words, request 70 → request 0, against the references at level 70 →
  level 0):
  - auxiliaries 4.0% → 7.4% (references 3.5% → 7.6%);
  - pronouns 3.8% → 1.8% (4.2% → 1.5%);
  - adverbs 3.8% → 2.1% (4.0% → 2.0%);
  - adpositions 16.3% → 14.2% (16.1% → 12.8%);
  - subordinating conjunctions 0.95% → 0.59% (1.12% → 0.76%);
  - proper nouns 7.4% → 9.7% (6.8% → 12.7%).

  The outputs move like the real simplifications, by similar amounts.

### Second-round predictions vs observed

| Prediction | Observed | |
|---|---|---|
| t-SNE and UMAP as is group by original, not by level | true in layers 2–6 and 20–28; **in layers 10–16 they group by level** | partly wrong |
| Within original, an ordered band in the middle layers | ordered at every layer | right, and stronger |
| MDS shows the same curve as PCA | yes | right |
| The no-level output is textually closest to levels 30–50 for 40–55% of originals | 45% (chance 43%) | right, but it doesn't discriminate |
| The no-level state is nearest to level 40 from layer 3 on | at every layer | right |
| The no-level outputs' READ-IT matches request 40 best | KS 0.047 against 0.076 and 0.151 | right |
| End-of-text ×2.5–3.5, full stop ×1.3–1.6 | ×3.2, ×1.43 | right |
| Semicolon ×0.2–0.4, comma ×0.7–0.9 | ×0.49, ×0.70 | semicolon just outside, comma right |
| Adpositions ×0.85–0.95, subordinating conjunctions ×0.7–0.9 | ×0.89, ×0.74 | right |
| Verbs and auxiliaries ×1.05–1.2, determiners ×1.05–1.2, nouns ×0.95–1.05 | verbs ×0.87, auxiliaries ×0.99, determiners ×0.97, nouns ×0.99 | wrong except nouns (but the words measured are the fixed text's own; at the distribution level the auxiliaries and copulas " è", " ha", " sono" do go up ×1.14–1.20) |
| End-of-text, full stop and semicolon monotonic; other classes weaker, same direction | all monotonic | right |
| In the texts, lower levels have more auxiliaries and fewer adpositions, like the references | yes, by similar amounts | right |

*Note.* The class predictions were written for the share of the next-token distribution that falls on each class.
The measure was changed before any result was looked at, because most Italian content words take several
vocabulary pieces: 207 single-piece verbs against 2,784 verbs in the texts. It became the probability of the
text's own words of each class.

### What the second round adds

1. **The level dominates the generation point only in the middle of the network.** In layers 10–16 it is the
   main axis of the state, where readout is near perfect. Afterwards the representation returns to the sentence's
   content, and the level becomes a small direction within it, still cleanly ordered.
2. **"No level" is the middle of the scale,** by its state and by where its outputs land. Textual comparison is
   too noisy to confirm it.
3. **The token shortens and splits, and leaves the vocabulary alone.**
   - At lower levels it makes ending and full stops more likely, and commas, semicolons and clause-building words
     (subordinators, adverbs, pronouns, verbs, adpositions) less likely. Nouns, names and numbers stay as they are.
   - The texts it writes match the real simplifications' part-of-speech shifts.
   - The only missing piece is vocabulary, which the data itself barely varies (2 points).

## Attention and syntax (requested 2026-11-17)

Can the attention patterns give a meaningful index of syntactic complexity, and does that complexity diminish at
lower requests? The models are g1 · guided (main) and g1 (check on step 1).

**Step 1. Syntax heads, validation, and a structural probe.**
- **Parsing.** The texts are parsed with the profiler's stanza pipeline (dependencies included). The units are the
  surface tokens, so "della" (di + la) is one unit. A unit's syntactic neighbours on its left are the units of its
  words' heads and dependents that come before it.
- **Head accuracy.** All 448 heads (28 layers × 16) are scored on whether the earlier unit a non-punctuation unit
  attends to most is one of its syntactic neighbours. Attention from pieces is averaged over query pieces and
  summed over key pieces, and only the text's own span is used, so the first-token sink drops out.
- **Baseline:** the best fixed offset (previous unit, two back, three back).
- **Selection without circularity.** Heads are selected on the texts of 500 test originals (the originals and their
  IMPaCTS references) and reported on the other 500.
- **Criterion, fixed now:** a syntax head beats the best positional baseline by at least 10 points on the held-out
  texts. Syntax heads count as found if there are at least 3. Step 2 runs only then.
- **Indices,** from the syntax heads, per text:
  - expected attention distance, in units;
  - share of attention more than 5 units back;
  - normalised entropy.
- **Validation** on texts the model did not write (test originals and references): Spearman with READ-IT syntax
  and global, and with the parser's mean and maximum dependency length, tree depth and subordinate-clause share,
  also partial on length.
- **Structural probe (cross-check, Hewitt & Manning 2019):** a rank-64 linear map from unit states (mean of the
  pieces) whose squared distances fit tree distances. It is trained on 2,000 g1 training simplifications, at layers
  4–28 (every 4th), and evaluated on the test texts: undirected attachment score of the minimum spanning tree
  (UUAS), distance Spearman, and the predicted dependency length per text, against the parser and READ-IT.

**Step 2.1. Outputs by requested level, against READ-IT global.** The indices on g1 · guided's 7,582 test
outputs: their trend over the requested level, their correlation with the READ-IT global the output reaches, and
the correlation within original.

**Step 2.2. Same input, different request.** Only the token changes, levels 40, 30, 20, 10 and 0:
- **(a) reading the original:** does the request change how the syntax heads, and the probe, process it?
- **(b) reading the fixed level-40 output:** the same question.

Step 2.3 (causal patching) waits for the user.

### Predicted before the run (2026-11-17)

- **Step 1:**
  - Syntax heads exist: 5–20 heads beat the positional baseline by ≥ 10 points on held-out texts. The best one
    reaches 45–65% against a baseline of 30–40%. Most sit in the middle layers (6–18).
  - The indices correlate with the parser's mean dependency length at ρ 0.4–0.7, and with READ-IT syntax at
    ρ 0.3–0.6. Partial on length, both drop by about a third.
  - The probe reaches UUAS 0.55–0.75 at its best layer (middle layers) and correlates with the parser's dependency
    length at ρ ≥ 0.6, better than the attention indices.
- **Step 2.1:** on the outputs, the attention-distance index and the probe's dependency length fall with the
  requested level. Spearman with the achieved READ-IT global is 0.3–0.6, positive (more complex text, longer
  attention), and positive within original too.
- **Step 2.2:**
  - **(a)** Reading the original, the token changes the syntax heads' attention only a little: their expected
    distance moves by less than 5% between levels 40 and 0, and the probe's parse of the original keeps the same
    UUAS within 0.02. The request changes what the model writes more than how it reads.
  - **(b)** Reading the fixed level-40 output, the attention distance shrinks slightly at level 0 (2–8%), in line
    with E's shift away from clause-building words.

*Design change after a smoke test, before any full result.* On 40 texts the task as planned turned out to be
decided by adjacency: the previous unit is a syntactic neighbour 92% of the time. On 300 references, a
non-adjacent version (the previous unit is not a neighbour) was still positional: the unit two back is a
neighbour 47% of the time, and the best head reached 46%. The criterion is therefore applied with the standard
per-relation evaluation of Clark et al. (2019). A **syntax head** is a head that, for some dependency relation with
at least 100 held-out arcs, attends from the dependent to its syntactic head more often than the relation's best
fixed offset (1, 2 or 3 units back) does, by at least 10 points, on the held-out texts. Syntax heads count as found
if there are at least 3. The complexity indices are computed from these heads only.

### Attention and syntax: results (2026-11-17)

Code: `parse.py`, `attention.py`, `probe.py` and `analyze_attention.py`. Results are in
`results/attention_g1.json`.

**Step 1. Syntax heads: found** (the gate to step 2 is passed).
- **108 heads in g1 · guided, 109 in g1, 106 shared.** Of 247 head–relation candidates selected on half the
  texts, 189 were confirmed on the other half. The heads come from the base model.
- **Where they are:** mostly layers 3–6 (32 heads) and 9–20 (58 heads).
- **Best head per relation, held-out:**

| Relation (word → its head) | Best head | Accuracy | Best fixed offset | Arcs |
|---|---|---|---|---|
| obj (object) | L5H7 | **95.6%** | 55.9% | 4,339 |
| nsubj (subject) | L5H7 | 80.3% | 57.0% | 605 |
| xcomp (open complement) | L5H15 | 76.6% | 50.1% | 1,181 |
| conj (conjunct) | L13H0 | **61.4%** | 24.8% | 4,963 |
| acl (clause modifying a noun) | L6H0 | 50.4% | 19.9% | 2,836 |
| advcl (adverbial clause) | L5H10 | 49.2% | 17.9% | 1,328 |
| ccomp (clausal complement) | L17H7 | 37.0% | 14.5% | 732 |
| parataxis | L8H1 | 32.1% | 2.9% | 137 |
| obl (oblique) | L24H0 | 54.3% | 42.0% | 7,574 |

**Validation** on 4,481 held-out originals and references, as Spearman ρ, with the partial correlation
controlling for length in brackets:

| Index | READ-IT syntax | READ-IT global | Dependency length (parser) | Tree depth (parser) | Subordinate clauses | Length |
|---|---|---|---|---|---|---|
| **Entropy of the syntax heads** | 0.50 (**0.41**) | 0.49 (**0.40**) | 0.37 (0.23) | 0.44 (0.31) | 0.39 (0.30) | 0.33 |
| Expected attention distance | 0.50 (0.16) | 0.49 (0.16) | 0.58 (0.31) | 0.60 (−0.05) | 0.37 (−0.03) | **0.99** |
| Long-range share | 0.50 (0.15) | 0.49 (0.15) | 0.59 (0.27) | 0.59 (−0.02) | 0.37 (−0.02) | 0.98 |
| Probe: predicted dependency length (layer 12) | 0.45 (0.24) | 0.44 (0.23) | **0.62 (0.45)** | 0.51 (0.24) | 0.33 (0.14) | 0.58 |

- **The distance indices measure length.** Their correlation with length is 0.98–0.99.
- **The entropy of the syntax heads is the meaningful index.** It has the most correlation left once length is
  controlled for (0.40–0.41 with READ-IT) and keeps a share with depth and subordination.
- **The probe tracks the parser's trees best.** It predicts dependency length at 0.62 (0.45 controlling for
  length). Its undirected attachment score at the best layer (12) is 0.63 on the dev sentences and on the
  references, and 0.50 on the originals read in the prompt.
- **The two cross-check each other.** The probe's dependency length correlates with the attention distance at 0.59
  and with the entropy at 0.33.

**Step 2.1. Outputs by requested level, against READ-IT global** (g1 · guided, 7,582 outputs).

| Requested level | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|---|
| Entropy of the syntax heads | 0.664 | 0.667 | 0.672 | 0.677 | 0.681 | 0.684 | 0.686 | 0.689 |
| Attention distance (units) | 3.65 | 4.34 | 4.82 | 5.23 | 5.48 | 5.68 | 5.95 | 6.42 |
| Probe dependency length | 2.00 | 2.10 | 2.12 | 2.20 | 2.24 | 2.29 | 2.35 | 2.43 |
| READ-IT global reached | 21.5 | 27.7 | 32.1 | 38.7 | 41.9 | 49.7 | 56.7 | 63.4 |
| Words (units) | 16.3 | 20.4 | 23.2 | 25.4 | 26.9 | 27.7 | 29.3 | 31.9 |

| Index | ρ with READ-IT global reached | Controlling for length | Within original, vs request | Within original, vs READ-IT reached |
|---|---|---|---|---|
| Entropy | 0.41 | **0.32** | 0.35 | 0.42 |
| Attention distance | 0.44 | 0.24 | 0.45 | 0.48 |
| Probe dependency length | 0.30 | 0.10 | 0.29 | 0.32 |

All three fall steadily with the requested level and follow the READ-IT global the output reaches. Only the entropy
keeps a clear part of that beyond the outputs being shorter.

**Step 2.2. Same input, different request** (levels 40 → 0; change relative to level 40, 95% intervals over
texts):

| | Entropy | Attention distance | Long-range share | Probe UUAS | Probe dependency length |
|---|---|---|---|---|---|
| (a) Reading the original, level 0 | −0.08% [−0.09, −0.07] | +0.32% [+0.30, +0.33] | +0.30% | 0.502 → 0.499 | +3.0% |
| (b) Reading the fixed level-40 output, level 0 | +0.25% [+0.23, +0.26] | +0.20% [+0.17, +0.23] | +0.20% | 0.624 → 0.620 | +1.1% |

The best heads' accuracies per relation do not change either. For example, conj is 52.5% → 52.3% on the
originals, and advcl 50.0% → 51.0% on the fixed texts.

### Predictions vs observed (attention and syntax)

| Prediction | Observed | |
|---|---|---|
| 5–20 syntax heads, best 45–65% against a 30–40% baseline, mostly layers 6–18 | 108 heads, mostly layers 3–6 and 9–20. Best 32–96% against 3–56% depending on the relation | found, but far more heads than predicted |
| Indices correlate with parser dependency length 0.4–0.7, READ-IT syntax 0.3–0.6; partial on length about a third lower | 0.37–0.59 and 0.50. Partial: entropy loses about a fifth, the distances about two thirds | half right: the distances are mostly length |
| Probe UUAS 0.55–0.75 at a middle layer; vs dependency length ρ ≥ 0.6, better than attention | 0.63 (dev, references), 0.50 (originals) at layer 12; ρ 0.62, the best beyond length | right |
| 2.1: indices fall with the request; ρ 0.3–0.6 with READ-IT global; positive within original | 0.30–0.44; within original 0.29–0.48 | right |
| 2.2a: reading the original, less than 5% change; probe UUAS within 0.02 | ≤ 0.3%; 0.003 | right |
| 2.2b: reading the fixed output, attention distance shrinks 2–8% at level 0 | +0.2% | **wrong** |

### What this suggests

1. **A meaningful attention index exists, and it is the entropy of the syntax heads, not their reach.**
   - How far attention reaches mostly counts words.
   - How spread the syntax heads' attention is tracks READ-IT syntax and global beyond length (partial 0.40–0.41).
     It also tracks the depth and subordination the parser finds.
   - The structural probe is the better tracker of dependency length itself.
2. **Complex behaviour does diminish at lower requests, in what the model writes.** In its own outputs, the syntax
   heads' attention becomes more focused, and the probe's trees get shorter arcs, as the request goes down, in step
   with the READ-IT global reached.
3. **The request does not change how the model parses.** The same text read under level 0 or 40 gets the same
   syntactic attention and the same probed tree, within 0.3% and 0.004 UUAS.
   - Together with E (the request changes the probability of ending a sentence, of commas and of subordinators)
     and A (the level is the main axis of the middle-layer state), the picture is that the token acts on the
     choice of the next word, not on the analysis of the text read so far.
   - Step 2.3, patching the heads, would test this causally. It waits for the user.

## G. Value zeroing and patching (requested 2026-11-17)

**Why.** From layer 3 on, the original's words give 55–88% of their attention to the control token at position 0,
yet its value vectors are only 5–24% as large as the other words' (measured on 50 originals). The request sits on
the attention sink, so attention weights cannot say how much of it is actually read. Part F also left out that
position. Value zeroing (Mohebbi et al., 2023) measures what a position contributes. Position j's value vector is
zeroed in one layer, the rest of the layer is recomputed (output projection, residual, normalisation,
feed-forward), and the change of every other position's output (cosine distance) is that position's mixing score,
normalised over j. Because the attention output is linear in the values, one batched pass per layer covers every
pair. A norm-based score, the size of each position's projected weighted value (Kobayashi et al., 2020), comes from
the same pass.

The model is g1 · guided. All runs use fp32 with TF32 matrix products.

- **G1. Exactness (gate).** With nothing zeroed, the recomputed layer output must match the model's own (mean
  cosine ≥ 0.999 at every layer). Otherwise nothing else runs.
- **G2. Where the request enters.** The control token's value-zeroing share at every position and layer, against
  the attention it receives. It is measured while the model reads the original and while it reads the fixed
  level-40 text, under levels 40, 20 and 0. It is reported by position: the text's words, the generation point
  (the prompt's final newline), and sentence-final positions.
- **G3. Part F with value zeroing.** Per layer and relation, does a word's representation depend most on its
  syntactic head's unit? The comparison is against the best fixed offset, with layers selected on half the
  originals and confirmed on the other half.
  - **Gate:** at least 3 layer–relation pairs beat it by at least 10 points on held-out texts.
  - **If passed:** value-zeroing entropy and distance indices, validated against READ-IT and the parser (held-out
    originals and 2,000 references). Then 2.1 on 2,000 outputs (250 per requested level) and 2.2 (originals and
    fixed texts under 40, 20 and 0).
- **2.3. Patching** (greedy decoding, 500 test originals, the same for every condition):
  - **A. The request pathway.** In the level-0 run, the control token's keys and values are replaced by the
    level-40 run's in layers 0–9, 10–18, 19–27, or all (0–27). Every later position reads the request only through
    them, so "all" must reproduce the level-40 output exactly; this checks the patch.
  - **B. The syntax heads.** Over the prompt positions, the outputs of part F's 108 syntax heads (before the output
    projection) are replaced by the level-40 run's; generation then continues unpatched. The control is 108 random
    other heads (seed 42).
  - **Outcome:** READ-IT global of the outputs (also syntax, words, Profiling-UD dependency length and
    subordination), as the share of the gap between the unpatched level-0 and level-40 outputs that each patch
    recovers, with 95% bootstrap intervals over originals.

### Predicted before the run (2026-11-17)

- **G1:** exact (cosine > 0.9999).
- **G2:**
  - The control token's actual contribution is small: a 1–5% value-zeroing share in the middle layers, against
    55–88% of the attention.
  - It is largest at the generation point and at sentence-final positions.
  - It differs little between levels (within ±20% relative): the request changes what is carried, not how much.
- **G3:**
  - Value-zeroing mixing tracks syntax: some relations (obj, conj, advcl) beat the fixed offsets by ≥ 10 points in
    middle layers, with fewer pairs than for the attention heads.
  - The value-zeroing entropy correlates with READ-IT syntax at a partial ρ of 0.3–0.5 once length is controlled
    for, about as well as the attention entropy.
  - 2.1: the value-zeroing indices fall with the request.
  - 2.2: reading the same text under level 0 changes the mixing among its words by more than attention showed
    (0.3%), but still little (< 3%). The control token's share changes more than the mixing among words.
- **2.3:**
  - **A:** patching the middle layers (10–18) recovers the largest share of the gap (40–70%), early layers 10–30%,
    late layers ≤ 20%, and all layers 100% (exactly the level-40 outputs).
  - **B:** the syntax heads over the prompt recover < 15%, random heads 0–5%. The request acts during writing,
    through position 0, not by changing how the original is read.

### G. Results (2026-11-17)

Results are in `results/vz_g1.json`, from `vz.py`, `patching.py` and `analyze_vz.py`. Patched outputs are in
`outputs/g1_guided_patch/`.

**G1. Exactness: passed.** With nothing zeroed, the recomputed layer outputs match the model's own at every layer
(lowest mean cosine 0.99999), on all 10,000 texts.

**G2. Where the request enters** (the control token's share at the original's words, level 40):

| Layer | 0 | 3 | 5 | 8 | 10 | 12 | 15 | 18 | 21 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Attention it receives | 2% | 81% | 68% | 66% | 47% | 55% | 59% | 64% | 83% | 88% | 87% |
| Value zeroing | 0.4% | 35% | 14% | 16% | 15% | 21% | 11% | 28% | 51% | 51% | 36% |
| Norm | 3% | 23% | 14% | 17% | 14% | 16% | 14% | 15% | 23% | 36% | 30% |

- **It is far from negligible.** Small values times very large attention still make it 11–51% of what is mixed into
  each word from layer 3 on, less than attention suggests but much more than I predicted.
- **The generation point** draws on it as much, or less (5–37%, and already 23% at layer 2).
- **Level 0 against level 40:** the share is about the same from layer 3 on (×0.97–1.07 at the words) and higher
  only in layers 0–2 (×1.05–1.17 on the original, ×1.2–1.3 on the fixed text). How much the request is read
  hardly depends on the level. What changes is what it carries.

**G3. Syntax by value zeroing: gate passed**, with 4 layer–relation pairs on held-out texts (the gate needed 3):
- obj at layer 5: 67% against an offset of 55%;
- acl at layer 6: 46% against 20%;
- conj at layer 12: 42% against 23% (it also passes at a second layer).

The norm-based score passes with 7 pairs, adding advcl at layer 6 (28% against 18%). Both are far below the single
attention heads (obj 96%), because value zeroing measures a whole layer's mixing, all heads together.

**Validation** on 1,468 held-out originals and references, with the indices from the confirmed layers (5, 6, 12),
as Spearman ρ with the partial correlation controlling for length in brackets:

| Index | READ-IT syntax | READ-IT global | Dependency length | Tree depth | Subordination | Length |
|---|---|---|---|---|---|---|
| **Norm-based entropy** | **0.58 (0.47)** | 0.58 (0.47) | 0.48 (0.32) | 0.46 (0.27) | 0.40 (0.27) | 0.41 |
| Value-zeroing entropy | 0.07 (0.39) | 0.07 (0.39) | −0.05 (0.29) | −0.12 (0.24) | −0.01 (0.24) | −0.45 |
| Value-zeroing distance | 0.55 (0.24) | 0.56 (0.25) | 0.65 (0.36) | 0.60 (0.01) | 0.40 (−0.03) | 0.93 |
| Norm-based distance | 0.56 (0.34) | 0.56 (0.35) | 0.63 (0.43) | 0.63 (0.02) | 0.44 (0.02) | 0.98 |
| *Attention entropy, part F, for comparison* | 0.50 (0.41) | 0.49 (0.40) | 0.37 (0.23) | 0.44 (0.31) | 0.39 (0.30) | 0.33 |

- **The norm-based entropy is the best complexity index found.** It reaches 0.47 controlling for length, against
  0.41 for the attention entropy.
- **The value-zeroing entropy falls with length** (−0.45), which hides its relation to READ-IT (raw 0.07). Once
  length is controlled for it reaches 0.39.

**2.1 (outputs, against READ-IT global; 2,000 outputs, 250 per requested level):**
- **The value-zeroing indices don't add anything.** The entropy is flat across requests (0.461–0.467; ρ with READ-IT
  global reached −0.03, partial on length 0.18). The distance falls (2.54 at request 0, 3.63 at 70; ρ 0.38) but is
  pure length (partial 0.00).
- **The norm-based entropy follows the request** (exploratory, added after the validation): 0.716 at request 0,
  0.754 at request 70, steadily; ρ 0.34 with READ-IT global reached, 0.21 controlling for length.

**2.2 (the same text, level 0 against 40):** mixing among the text's words hardly changes.
- **Value-zeroing entropy:** −0.29% reading the original, −0.36% reading the fixed text.
- **Value-zeroing distance:** +0.46% and +0.06%.
- **Norm-based entropy:** −0.16% and −0.17%.

Attention showed the same (part F).

**2.3. Patching** (500 test originals, greedy decoding):

| Condition | READ-IT global | READ-IT syntax | Words | Dependency length | Subordination | Gap recovered (READ-IT global) [95%] | Identical to level 40 |
|---|---|---|---|---|---|---|---|
| Level 40, no patch | 46.2 | 45.4 | 25.3 | 2.44 | 40.5 | – | 100% |
| Level 0, no patch | 20.6 | 20.0 | 15.2 | 2.09 | 17.9 | – | 10% |
| A: the request from level 40 in layers 0–9 | 45.8 | 45.0 | 25.3 | 2.44 | 40.5 | **98%** [95, 101] | 84% |
| A: … in layers 10–18 | 20.6 | 20.0 | 15.3 | 2.09 | 18.4 | 0% [−2, +2] | 10% |
| A: … in layers 19–27 | 20.6 | 20.0 | 15.0 | 2.08 | 18.2 | 0% [−2, +2] | 9% |
| A: … in every layer (check) | 46.2 | 45.4 | 25.3 | 2.44 | 40.5 | 100% | **100%** |
| B: the 108 syntax heads from level 40 (prompt) | 30.2 | 29.5 | 18.0 | 2.18 | 27.0 | 37% [31, 43] | 25% |
| B: 108 random heads from level 40 (prompt) | 37.4 | 36.8 | 21.1 | 2.33 | 32.5 | 66% [60, 71] | 44% |

- **The request is read from the control token entirely in layers 0–9.** Giving the level-0 run the level-40
  token's keys and values in those layers alone gives back 98% of the level-40 outputs' READ-IT, and the same
  output in 84% of cases. After layer 9, what the token offers changes nothing: the request is already in the other
  positions' states.
- **The syntax heads are not the request's main route.** Over the prompt they carry 37% of it, against 66% for as
  many random heads. The request spreads through many heads.
- **The patches move every measure together:** READ-IT global and syntax, length, dependency length and
  subordination.

### G. Predictions vs observed

| Prediction | Observed | |
|---|---|---|
| G1 exact | cosine 0.99999 | right |
| The control token contributes 1–5% in the middle layers | 11–51% (value zeroing), 14–36% (norm) | **wrong** |
| Largest at the generation point and at sentence ends | at the words as much as at the generation point; sentence ends similar | wrong |
| Its share differs little between levels (±20%) | ×0.97–1.07 from layer 3 | right |
| Some relations beat the offsets by ≥ 10 points in middle layers, fewer than the attention heads | 4 pairs (layers 5, 6, 12) | right |
| Value-zeroing entropy partial ρ with READ-IT 0.3–0.5 | 0.39 (but the norm-based entropy is better: 0.47) | right |
| 2.1: the value-zeroing indices fall with the request | entropy flat, distance only length | **wrong** (the norm-based entropy does fall) |
| 2.2: mixing changes more than attention showed, but < 3% | ≤ 0.5%, like attention | half right |
| 2.3 A: middle layers recover most (40–70%), early 10–30%, late ≤ 20%, all 100% | early 98%, middle 0%, late 0%, all 100% | **wrong** except the check |
| 2.3 B: syntax heads < 15%, random 0–5% | 37% and 66% | **wrong** |

### G. What this suggests

1. **The request is taken in early and then lives in the text's states.**
   - In layers 0–9 the other positions read the control token; after that, the token itself no longer matters (2.3 A).
   - The middle layers where the level dominates the state (part A) are reading their own copy, not the token.
   - Steering, or the next interventions, should target layers 0–9.
2. **Value matters, but the conclusions hold.** Counting what is actually mixed rather than where attention points
   gives a better complexity index, the norm-based entropy (partial ρ 0.47 with READ-IT syntax). It confirms the
   two attention results:
   - complexity of processing falls with the request in the outputs;
   - reading the same text under another request leaves the mixing among its words unchanged.
3. **Many heads carry the request, not specifically the syntax heads.** Patching 108 random heads over the prompt
   transfers more of it than the 108 syntax heads do. The level spreads broadly through the early-layer residual
   stream (part A), and the syntax heads are not its gate.

## Not in git

Three groups of intermediates are not in the repository. Everything else the analyses read is.

| Files | Size | Regenerate with | Time |
|---|---|---|---|
| `outputs/interp/<model>/states.npy` | 653 MB each | `states.py` | about 2 min per model |
| `outputs/interp/<model>/attention-*.npz` | 289 MB together | `attention.py`, for the full conditions; then the `--split selection` and `--split heldout` runs of `originals@40` and `references` | about 5 min |
| `outputs/interp/<model>/fixed_prob.npz` and `classes_prob.npz` | 39 MB together | `logits.py --fixed-only` and `classes.py` | a few minutes |

All runs use GPU0 and CPUs 9–25.
