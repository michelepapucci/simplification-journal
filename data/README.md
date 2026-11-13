# Data

Every training and test set lives in a folder named after its version. The test **pairs** are the same in all versions: the same 1,000 pairs per strategy, by `idx`. Only their control columns differ.

| Version | Folder | Training set | Test sets | Built by | What it is |
|---|---|---|---|---|---|
| **clicit2026** (GFFF) | [`clicit2026/`](clicit2026/) | `training_dataset_final_bin_10_no_overlap.zip`: 45,209 pairs × 3 strategies | `{base,lexical,syntax}_test_set.csv`, 1,000 each | GFFF notebooks (unmodified copies) | Exactly what the CLiC-it 2026 models were trained and tested on, bugs included |
| **v0** | [`v0/`](v0/) | `training_set_v0.zip`: 55,877 pairs × 3 strategies | `{base,lexical,syntax}_test_set_v0.csv` | [`scripts/build_dataset_v0.py`](../scripts/build_dataset_v0.py) | The CLiC-it selection rebuilt from the Hugging Face release, with the bugs fixed. One token per strategy. |
| **v1** | [`v1/`](v1/) | `training_set_v1.zip`: 78,462 pairs, each once (77,696 train + 766 validation) | `{base,lexical,syntax}_test_set_v1.csv` | [`scripts/build_dataset_v1.py`](../scripts/build_dataset_v1.py) | Three target tokens per pair (base, lexical, syntax), each a level or `any`. Complex originals (READ-IT global > 70); a target needs one control token down and a drop beyond READ-IT's noise. See [`v1/README.md`](v1/README.md). |

The data explorer on the `gh-pages` branch (https://michelepapucci.github.io/simplification-journal/) shows v0, v1 and the CLiC-it 2025 sample side by side.

All built versions come from the Wikipedia part of IMPaCTS on Hugging Face, [`mpapucci/impacts`](https://huggingface.co/datasets/mpapucci/impacts), pinned at revision `489c78e` in [`scripts/impacts.py`](../scripts/impacts.py).

[`source/`](source/) holds the inputs copied from the old `GFFF/` folder:
- the test-pair lists `idx_to_remove/list_idx_{base,lexical,syntax}.txt`;
- the noise lists `idx_to_remove/list_dropped_idx_{langdetect,double_filter}.txt`;
- the GFFF pre-filtered file `dataset_profiled_filtered_with_4_control_tokens.zip`. No build reads it any more; the evidence script still uses it for the provenance check.
