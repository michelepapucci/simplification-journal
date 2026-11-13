# Version clicit2026 (GFFF)

Unmodified copies of the files in `GFFF/` of the old `text-simplification-experiments` repository: the data the CLiC-it 2026 models (Qwen3 0.6B / 1.7B / 4B) were trained and tested on. Keep them only for reference and for reproducing the paper. Do not train on them.

| File | Content |
|---|---|
| `training_dataset_final_bin_10_no_overlap.zip` | 45,209 pairs, each repeated under the base, syntax and lexical token (135,627 rows) |
| `{base,lexical,syntax}_test_set.csv` | 1,000 test pairs per strategy |

Known problems, all fixed from v0 on:
- **Lexical tokens alias base.** The lexical tokens were built from the base score, so every lexical token equals the base token.
- **Some test-set tokens are wrong.** The base and lexical tokens in the test sets were taken from the rows of the syntax test set. This is harmless for the paper's results, since evaluation generates from a fixed token list.
- **An undocumented "no overlap" step** dropped 12,511 of the 57,720 training pairs.
- **`n_sentences` is constant.** It is always 1, because Profiling-UD averages its features over sub-sentences.
- **Identity pairs.** 1,598 training pairs (3.5%) are the original sentence unchanged.
