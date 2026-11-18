# Dataset g1: one control token on READ-IT global

Dataset g1 trains a simplifier that is told one thing, the READ-IT global level to reach. This is the setup of the
LREC 2026 paper, built with the selection rules the journal developed for its three-token datasets (archived on
the `multidim-archive` branch):
- complex originals only (READ-IT global > 70);
- the same filters;
- training and test originals kept apart;
- the CLiC-it levels;
- every pair at least one level below its original.

Three cleaning rules are new: degenerate simplifications, simplifications that echo the generator's instructions,
and simplifications more than twice the original's length.

| | |
|---|---|
| Source | Wikipedia part of IMPaCTS on Hugging Face, [`mpapucci/impacts`](https://huggingface.co/datasets/mpapucci/impacts), revision `489c78e` (pinned in [`scripts/impacts.py`](../../scripts/impacts.py)) |
| Training pairs | **81,058** from 10,762 originals: 80,251 `train` + 807 `validation` (column `split`, 1% of the originals) |
| Test | 1,000 originals × every level below their own = **7,582 requests**; references for 4,216 of them |
| Dev | 300 more originals, 2,275 requests (for choosing settings such as the guidance weight, never the test) |
| Built by | `PYTHONPATH=envs/pydeps python scripts/build_dataset_g1.py` (about 3 minutes; needs `langdetect`) |

## Files

| File | Content |
|---|---|
| `training_set_g1.zip` | one row per pair. Columns: `idx`, `original_sentence_idx`, `split`, both texts, the eight READ-IT scores (`original_*`, `simplification_*`: all, base, lexical, syntax), `src_level`, `tgt_level`, `ctrl_token` (`<|readability_{tgt_level}|>`), then the IMPaCTS Profiling-UD features (`*_original`, `*_simplification`) |
| `test_set_g1.csv`, `dev_set_g1.csv` | one row per original: text, READ-IT scores, `src_level`, Profiling-UD features |
| `test_requests_g1.csv`, `dev_requests_g1.csv` | one row per request: `request`, `original_sentence_idx`, `level` (the requested level), `src_level`, `original_text` |
| `test_references_g1.csv` | the test originals' IMPaCTS simplifications that pass the same cleaning rules and lie below the original's level: both texts, the READ-IT scores, `src_level`, `level`, Profiling-UD features |
| `construction_g1.json` | the rules, the count after every step, READ-IT global's noise, the level tables |

## Construction

| Step | Pairs | Originals |
|---|---|---|
| IMPaCTS release, Wikipedia (already without pairs whose READ-IT global went up, and without copies) | 764,061 | 106,680 |
| 1. Hold out the test originals: every original of the 2,575 CLiC-it 2026 test pairs | 740,563 | 104,105 |
| 2. Complex originals: READ-IT global > 70 | 213,703 | 24,355 |
| 3. Remove the pairs on the GFFF langdetect list | 210,720 | 24,313 |
| 4. Remove originals matching the double-filter proxy (a quote, guillemet or opening parenthesis, or over 400 characters) | 94,560 | 10,866 |
| 5. Remove exact duplicates | 94,505 | 10,866 |
| 6. Original and simplification detected as Italian (langdetect, seed 0) | 93,648 | 10,816 |
| 7. **Degenerate simplification** removed | 92,467 | 10,815 |
| 8. **Instruction echo** removed | 90,351 | 10,814 |
| 9. **Simplification at most twice the original's words** | 88,389 | 10,806 |
| 10. **Target:** the simplification's level is below the original's | **81,058** | **10,762** |
| 11. Split by original: 1% to `validation` (seed 42) | 80,251 + 807 | 10,654 + 108 |

Steps 1–6 are the rules of the journal's dataset v2. Levels follow the CLiC-it rule: level = 10 · ⌊10 · score⌋,
capped at 90. There is **no noise margin**. The token is the simplification's level, whatever the size of the
drop, so a pair that goes from 71 to 68 is a level-60 example.

### The three cleaning rules

Each rule removes simplifications that would teach the model something other than simplifying. Counts are on the
pool after step 6; some pairs break more than one rule.

| Rule | Removes | Pairs (before the other two rules) |
|---|---|---|
| **Degenerate** | a word 4-gram that occurs 3 or more times (and not in the original), or a type-token ratio below 0.4: loops such as *"…eccitando i muscoli durante la contrazione, eccitando i muscoli durante la contrazione, …"* | 1,181 (1.3%) |
| **Instruction echo** | the generator's instructions, or a chat assistant's opening, in place of or before the simplification: *"La semplificazione della frase mantenendo il più possibile intatto il significato è la seguente: …"*, *"Sulla base di questo paragrafo, …"*, *"Ecco …"* (18 patterns, listed in `construction_g1.json`; the original must not contain them) | 2,231 (2.4%) |
| **Too long** | more than twice the original's words. These are mostly the generator explaining or continuing past the sentence, with content the original does not have | 2,703 (2.9%) |

Examples (random among originals under 170 characters):

| Rule | Original → simplification |
|---|---|
| degenerate | *Esse non sono presenti sempre, ma solo quando parte della colonia si accinge a sciamare, …* → *Le api non hanno re e regine. Le api operaie costruiscono un alveare e raccogliere il cibo per nutrire le api operaie e le api droni. Le api operaie costruiscono un alveare e raccogliere il cibo …* (global 80.8 → 8.7) |
| echo | *Secondo Luigi Gambero, i sermoni tendevano a seguire una forma standard:* → *Questa frase è stata semplificata mantenendo il più possibile intatto il significato.* (75.8 → 28.6) |
| echo | *Una femmina, che secondo quanto riferito misurava 7,9 m di lunghezza, …* → *Sulla base di questo paragrafo, un caimano di 1,5 metri è stato trovato nel stomaco di un alligatore femminile …* (75.5 → 17.2) |
| too long | *Alwyn predilesse la dissonanza, mettendo a punto la sua alternativa al serialismo dodecafonico.* → *Dopo la seconda guerra mondiale, Alwyn divenne direttore musicale della Sadler's Wells Ballet. Ha composto le sue due sinfonie, …* (74.5 → 34.0) |

Junk is more common at the strong levels, because READ-IT scores repetition and chatty, explanatory text as simple.
In the pool, pairs with a looping 4-gram are 2.3% of those reaching level 0 but 0.4% of those reaching level 70.
Over-long ones are 4.8–5.2% at levels 20–30 but 0.7% at level 70.

## What the training pairs look like

| Target level | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|---|---|
| Training pairs | 4,582 | 9,874 | 11,987 | 11,951 | 12,039 | 11,242 | 11,545 | 5,411 | 1,620 |
| … from originals at level 70 | 3,294 | 6,384 | 7,257 | 7,199 | 7,007 | 6,597 | 6,881 | – | – |
| … at level 80 | 853 | 2,704 | 3,678 | 3,684 | 3,948 | 3,603 | 3,680 | 4,362 | – |
| … at level 90 | 435 | 786 | 1,052 | 1,068 | 1,084 | 1,042 | 984 | 1,049 | 1,620 |

| Levels down from the original | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Training pairs | 12,863 | 11,326 | 11,594 | 12,189 | 12,025 | 11,130 | 7,050 | 1,639 | 435 |

- **Levels 10–60 are about equally frequent; levels 0 and 70 have about half as many pairs.** The `g1 · balanced`
  model (see [the experiments](../../experiments/README.md)) evens them out.
- **The mildest request is well covered.** A pair one level down is as common as one five levels down. With the
  noise margin of the three-token datasets (P90 of READ-IT noise, 17.3 points for global) only 253 one-level pairs
  would have been left.
- Originals: 44,619 pairs start at level 70, 26,512 at 80 and 9,120 at 90.

### READ-IT global's own noise

Near-identical simplifications of the same original (character similarity ≥ 0.9, 36,734 such pairs) differ in
READ-IT global by:

| Median | 75th percentile | 90th percentile | 95th percentile |
|---|---|---|---|
| 3.8 | 9.2 | 17.3 | 23.6 |

READ-IT global is as noisy as READ-IT syntax was (17.6 at P90). A rewrite that lands in the right 10-point level
partly does so by chance, and a perfect model would still often miss. RMSE and the within-original correlation are
the more reliable measures of control.

## Test and dev originals

| | Originals |
|---|---|
| Held-out originals (every original of the CLiC-it 2026 test pairs) | 2,575 |
| … READ-IT global > 70 | 2,575 |
| … pass the double-filter proxy | 2,306 |
| … detected as Italian | 2,306 |
| **Test** (stratified by level, seed 42): 418 at level 70, 582 at level 80 | **1,000** |
| **Dev** (from the rest): 125 at 70, 175 at 80 | **300** |
| Unused, still held out | 1,006 |

- **Requests:** each test or dev original is asked for every level below its own: levels 0–60 for originals at
  70, and 0–70 for originals at 80.
- **References:** the test originals' 9,131 IMPaCTS simplifications pass the same cleaning as training (langdetect
  list, duplicates, Italian, degenerate, echo, length). 7,968 lie below the original's level, and 4,216 of the
  7,582 requests have at least one reference at the requested level.
