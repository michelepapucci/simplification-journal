# Dataset v2: v1 plus lexical support

Dataset v2 is [dataset v1](../v1/README.md) with two changes:
- **A lexical supplement.** 11,752 pairs from the rest of the Wikipedia release whose READ-IT lexical score dropped by two levels or more, filtered so that the drop comes from rewording rather than from translation, deletion or formatting.
- **A language check.** Every pair whose original or simplification is not Italian is dropped. This removes 754 v1 pairs.

Everything else is v1's: the three target tokens, the target rule, the noise margins, and the test sets, which are identical.

| | |
|---|---|
| Source | Wikipedia part of IMPaCTS on Hugging Face, [`mpapucci/impacts`](https://huggingface.co/datasets/mpapucci/impacts), revision `489c78e` (pinned in [`scripts/impacts.py`](../../scripts/impacts.py)) |
| Training pairs | **89,460** from 17,062 originals: 88,599 `train` + 861 `validation` (column `split`). Of these, 77,708 are v1 pairs and 11,752 are supplement pairs (column `subset`) |
| Test sets | the v1 test sets, byte for byte (`*_test_set_v2.csv`) |
| Built by | `python scripts/build_dataset_v2.py` (about 2 minutes; needs `langdetect`, installed in `envs/pydeps`) |
| Explore | [data explorer](https://michelepapucci.github.io/simplification-journal/#data=v2), dataset "v2" (the "Lexical supplement" slice shows the added pairs on their own) |

## Why

v1's lexical token had almost nothing to learn from:
- 15,226 of its 77,696 training pairs set a lexical target, and 94% of those are exactly one level below the original;
- level 10 appears 255 times and level 0 essentially never.

Both v1 models learned to "lower lexical by one level" whatever level was requested. At every requested level READ-IT lexical moved by about −4.5, and the correlation between requested and achieved lexical level within an original was 0.02–0.05 ([v1 results](../v1/experiments/README.md#results)).

The lexical drops v1 lacks are in the rest of the release. v1 keeps only originals with READ-IT global > 70 that pass the double-filter proxy, and 97% of the release's two-level lexical drops (27,076 of 27,978, test originals held out) come from other originals.

## Files

| File | Content |
|---|---|
| `training_set_v2.zip` | CSV, one row per pair: v1's columns plus `subset` (`v1` or `lexical_supplement`), after `split` |
| `{base,lexical,syntax}_test_set_v2.csv` | identical to `data/v1/*_test_set_v1.csv` |
| `construction_v2.json` | the rules, v1's margins, the count after every step, the test sets' language check and the lexical-target counts |

## Construction

| Step | Pairs | Originals |
|---|---|---|
| Release, Wikipedia | 764,061 | 106,680 |
| 1. Test originals held out, the release's langdetect list removed, exact duplicates removed (as in v1) | 737,108 | 104,063 |
| 2. Targets with v1's rule and margins (base 5.7, lexical 3.8, syntax 17.6, read from `construction_v1.json`) | | |
| 3. v1 part: global > 70, passes the double-filter proxy, at least one target. Checked to be exactly `training_set_v1.zip` | 78,462 | 10,800 |
| 4. v1 part: original and simplification detected as Italian | **77,708** | 10,748 |
| 5a. Outside v1's selection, lexical target two or more levels below the original | 27,076 | 13,141 |
| 5b. … original and simplification detected as Italian | 24,827 | 12,106 |
| 5c. … not a near-copy (difflib character similarity < 0.9) | 24,511 | 12,011 |
| 5d. … more than names changed (the words without a capital letter differ) | 24,473 | 12,004 |
| 5e. … no line break in the simplification | 22,836 | 11,451 |
| 5f. … the simplification keeps at least half the original's tokens | **11,752** | 6,314 |
| 6. Split: v1 pairs keep their v1 split; 1% of the supplement's originals (seed 42) go to validation | 88,599 train + 861 validation | 17,062 |

The supplement's base and syntax targets follow the same rule as every other pair. They are often set: base in 90% of supplement pairs, syntax in 55%.

### Why each supplement filter

Each filter targets a kind of pair whose READ-IT lexical drop is not a lexical simplification. These came up while sampling the candidates:
- **Language (5b).** READ-IT scores English as lexically easy, so a translation looks like a large lexical drop. 6.6% of the candidates have a non-Italian simplification (4.0% English), and 16.7% of all lexical targets at levels 0–10 outside v1 are not Italian. The release's langdetect list never flagged them: 99% of its ids fall inside the CLiC-it-style selection (original global > 70, no READ-IT score higher), so pairs outside it were never checked. Within that selection the list matches langdetect (0.1% of unlisted pairs detect as non-Italian).
- **Near-copies and name-only changes (5c, 5d).** Spelling a name differently can move the score a lot. For example, "Saint-Lambert-du-Lattay" → "St. Lambert du Lattay" moves lexical from 58 to 36.
- **Line breaks (5e).** These are simplifications broken into fragments, such as a list of noun phrases.
- **Length (5f).** 46.5% of the candidates keep less than half of the original's tokens. Their lexical drop comes from cutting content, not from replacing words. This filter removes the most pairs.

### The language check on v1's own pairs (step 4)

754 v1 pairs (1.0%) have a non-Italian original or simplification. They are mostly v1's trade-off pairs, which lie outside the CLiC-it-style selection and so outside the release's langdetect list. They are over-represented among v1's lowest lexical targets: 6% of v1's lexical targets at levels 0–10 were not Italian.

In the test sets, every original is Italian. 15 of the 3,000 reference simplifications are not (base 3, lexical 3, syntax 9). They are kept, so that v1 and v2 models are evaluated on the same pairs. Generation only uses the originals.

## What changes for lexical

Lexical targets in the training split:

| | v1 part | v2 |
|---|---|---|
| Pairs with a lexical target | 15,090 | **26,733** |
| … two or more levels below the original | 837 (6%) | **12,480 (47%)** |
| … at level 10 | 238 | 1,420 |
| … at level 0 | 10 | 104 |
| … lexical the only target | 803 | 1,493 |

By the original's lexical level, where most test originals sit:

| Original lexical → target | 0 | 10 | 20 | 30 |
|---|---|---|---|---|
| 30 | 35 (v1: 6) | 1,098 (v1: 97) | 6,255 | – |
| 40 | 23 (v1: 1) | 172 (v1: 16) | 6,384 (v1: 535) | 7,340 |
| 50 | 4 (v1: 0) | 22 (v1: 0) | 454 (v1: 18) | 3,430 (v1: 144) |

The supplement pairs are more lexical than v1's lexical pairs. Medians, supplement against v1 pairs with a lexical target:

| | Supplement | v1 lexical pairs |
|---|---|---|
| READ-IT lexical change | −16.4 | −7.4 |
| READ-IT base change | −18.4 | −23.3 |
| READ-IT syntax change | −20.2 | −43.6 |
| Share of tokens in the basic Italian vocabulary (`in_dict`) | +5 points | +1 point |
| Characters per token | −0.045 | 0.000 |
| Simplification length, share of original tokens | 0.69 | 0.58 |

## What v2 does not fix

- **Level 0 stays thin.** 104 pairs: almost no IMPaCTS simplification reaches lexical 0–10 from an Italian original.
- **Rewording-only edits stay rare.** Lexical is the only target in 1,493 pairs. Most supplement pairs also lower base, so they teach how far lexical should drop more than how to change only the words.
- **v2 relaxes two v1 rules for the supplement.**
  - 82% of the supplement comes from originals with READ-IT global ≤ 70, and 56% from originals flagged by the double-filter proxy.
  - Their originals sit at lexical 40–50 (84%), the range of the test originals.
  - Overall, 89% of v2's training pairs still come from originals at global 70 or above.
- **IMPaCTS simplifications can still distort meaning**, as in v1. No filter checks meaning.

## Experiment

`v2_targets`: Qwen3-0.6B-Base trained on v2 with exactly the recipe and evaluation of the v1 "targets" model: three target tokens, target dropout 0.3, LoRA r 32, effective batch 64 (4 × 16), 3 epochs (4,155 steps). Only the target-token variant is trained, since source tokens made no difference in v1.

- Run: `bash data/v1/experiments/run_v2_targets.sh` (code, requests and test sets shared with the v1 experiments).
- Results: [`data/v1/experiments/README.md`](../v1/experiments/README.md#dataset-v2-lexical-support).

**Result: the supplement did not make lexical controllable.**
- Lexical requests from 0 to 20 land at 37.7–38.8, against 39.0–39.9 for v1.
- Within an original, requested and achieved lexical are still unrelated (Spearman 0.06; v1 0.05).
- The share inside the requested level goes from 10.5% to 11.6%.
- Base, syntax and the joint evaluation are unchanged.

Two diagnostics on held-out validation pairs point away from the data amount:
- the model fails the same way on originals from the supplement's own pool;
- next to the base token, the lexical token carries no information about the reference.

IMPaCTS lexical drops come with base drops, so the model learns the lexical token as a general "simplify more" signal rather than as a word-level instruction.
