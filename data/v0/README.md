# Dataset v0: the CLiC-it 2026 selection, fixed

The training data of the CLiC-it 2026 paper, rebuilt from the IMPaCTS release on Hugging Face (Wikipedia, revision `489c78e`) by [`scripts/build_dataset_v0.py`](../../scripts/build_dataset_v0.py). It keeps the paper's format: one control token per strategy, every pair repeated under each strategy's token. Use it as the fixed baseline (E0 in `reports/improving_controllability.md`).

| File | Content |
|---|---|
| `training_set_v0.zip` | 55,877 pairs × 3 strategies (167,631 rows), with READ-IT scores, Profiling-UD features, `ctrl_token` and `ctrl_int` |
| `{base,lexical,syntax}_test_set_v0.csv` | 1,000 test pairs per strategy, same pairs as clicit2026, with correct tokens |

**Selection.** Originals with READ-IT global > 0.7, keeping pairs with no READ-IT score (base, lexical, syntax, global) above the original's. Then the GFFF langdetect and double-filter lists are removed, and the test pairs are set aside.

**Differences from clicit2026:**
- each strategy's token comes from that strategy's own score (the lexical bug is fixed);
- no "no overlap" step;
- no `n_sentences`;
- no identity pairs, which the release drops (55,877 instead of 57,720).

The test sets match clicit2026's pairs exactly.

**Caveat.** Only test *pairs* are removed: 93–98% of test originals still appear in training with other simplifications. For a grouped split, drop every pair of a test original, which leaves 42,907 pairs.

The public data explorer (branch `gh-pages`) shows this version.
