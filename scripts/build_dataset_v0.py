"""Build dataset v0: the CLiC-it 2026 training selection, fixed, and the three test sets.

The pairs come from the Wikipedia part of the public IMPaCTS release on Hugging Face
(mpapucci/impacts, config wikipedia_profiling, pinned in scripts/impacts.py). The selection
is the one behind the CLiC-it 2026 paper (GFFF/training_dataset_generation.ipynb and
GFFF/test_set_generation.ipynb; copies of their outputs are in data/clicit2026/), written out
as explicit rules instead of being read from a pre-filtered file:

    original READ-IT global > 0.7, and no READ-IT score (base, lexical, syntax, global)
    of the simplification above the original's; then the langdetect and double-filter
    lists are removed, and the test pairs are set aside.

The GFFF pre-filtered file (data/source/dataset_profiled_filtered_with_4_control_tokens.zip)
is exactly these rules applied to the LREC 2026 corpus. The Hugging Face release has also
dropped every simplification that is an exact copy of its original, so this build has 1,843
fewer training pairs per strategy than a build from the GFFF file (55,877 instead of 57,720).
The test pairs are unaffected.

Changes from the CLiC-it 2026 notebooks:

1. Every strategy's control token is binned from that strategy's own READ-IT score of the
   simplification. The notebooks built the lexical token from the base score, and the test-set
   notebook built the base and lexical tokens from the rows of the syntax test set.
2. The "no overlap" step is dropped: pairs whose three control levels coincide are kept.
3. The n_sentences profiling feature is dropped. Profiling-UD features are averages over the
   sentences of each text, so this count is always 1, even when a simplification has several
   sentences.

Inputs, in data/source/idx_to_remove/:
    list_dropped_idx_*.txt              noisy pairs (langdetect, double filter); they cover only
                                        the pairs selected above
    list_idx_{base,lexical,syntax}.txt  test pairs, one list per strategy

Outputs, in data/v0/:
    training_set_v0.zip                    every clean, non-test pair once per strategy
    {base,lexical,syntax}_test_set_v0.csv  one test set per strategy

Run from anywhere: python scripts/build_dataset_v0.py [--hf-dir SNAPSHOT]
(needs pandas, numpy, pyarrow and huggingface_hub).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from impacts import DIMENSIONS, load_wikipedia

ROOT = Path(__file__).resolve().parent.parent
LISTS = ROOT / "data" / "source" / "idx_to_remove"
OUT = ROOT / "data" / "v0"

# Same order as the CLiC-it training file, which concatenated base, syntax, lexical.
STRATEGIES = ("base", "syntax", "lexical")
# The fine-tuning script registers tokens for levels 0, 10, ..., 90 only.
VALID_LEVELS = set(range(0, 100, 10))
MIN_ORIGINAL_GLOBAL = 0.7
BROKEN_COLUMNS = ["n_sentences_original", "n_sentences_simplification"]  # always 1, see change 3 above


def read_ids(path: Path) -> set[int]:
    return {int(line) for line in path.read_text().split()}


def select_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Hard originals whose simplification is no harder than the original on any READ-IT score."""
    no_harder = np.logical_and.reduce([pairs[f"simplification_{d}"] <= pairs[f"original_{d}"] for d in DIMENSIONS])
    return pairs[(pairs["original_all"] > MIN_ORIGINAL_GLOBAL) & no_harder]


def with_control_token(pairs: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Add the strategy's control token: a READ-IT score of 0.43 becomes <|readability_<strategy>_40|>."""
    level = np.floor(pairs[f"simplification_{strategy}"] * 10).astype(int) * 10
    unknown = set(level) - VALID_LEVELS
    if unknown:
        raise ValueError(f"{strategy}: levels {sorted(unknown)} have no registered control token")
    return pairs.assign(ctrl_token=[f"<|readability_{strategy}_{lvl}|>" for lvl in level], ctrl_int=level)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/v0/training_set_v0.zip and the three v0 test sets.")
    parser.add_argument("--hf-dir", type=Path, help="local snapshot of mpapucci/impacts (default: download)")
    args = parser.parse_args()

    corpus = load_wikipedia(profiling=True, local_dir=args.hf_dir).drop(columns=BROKEN_COLUMNS)
    pairs = select_pairs(corpus).sort_values("idx").reset_index(drop=True)

    noisy = read_ids(LISTS / "list_dropped_idx_langdetect.txt") | read_ids(LISTS / "list_dropped_idx_double_filter.txt")
    test_ids = {s: read_ids(LISTS / f"list_idx_{s}.txt") for s in STRATEGIES}
    all_test_ids = set().union(*test_ids.values())

    clean = pairs[~pairs["idx"].isin(noisy)]
    train_pairs = clean[~clean["idx"].isin(all_test_ids)]
    print(f"wikipedia pairs: {len(corpus)} | selected: {len(pairs)} | after noise filters: {len(clean)} | "
          f"test pairs (unique across strategies): {len(all_test_ids)} | training pairs: {len(train_pairs)}")

    training_set = pd.concat([with_control_token(train_pairs, s) for s in STRATEGIES], ignore_index=True)
    training_set.to_csv(OUT / "training_set_v0.zip", index=False, compression="zip")
    print(f"training_set_v0.zip: {len(training_set)} rows ({len(train_pairs)} pairs x {len(STRATEGIES)} strategies)")

    for strategy in STRATEGIES:
        test_pairs = clean[clean["idx"].isin(test_ids[strategy])]
        if len(test_pairs) != len(test_ids[strategy]):
            raise ValueError(f"{strategy}: {len(test_ids[strategy]) - len(test_pairs)} test ids are missing "
                             "from the selected pairs or were removed as noise")
        test_set = with_control_token(test_pairs, strategy)
        test_set.to_csv(OUT / f"{strategy}_test_set_v0.csv", index=False)
        print(f"{strategy}_test_set_v0.csv: {len(test_set)} pairs")


if __name__ == "__main__":
    main()
