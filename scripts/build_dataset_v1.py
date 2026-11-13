"""Build dataset v1: three target tokens per pair (base, lexical, syntax), each a level or "any".

Source: the Wikipedia part of the IMPaCTS release on Hugging Face (mpapucci/impacts, pinned in
scripts/impacts.py). The release has already dropped pairs whose READ-IT global score went up and
simplifications identical to their original.

Construction, in order (data/v1/README.md explains each step and reports the counts):

1. Hold out the test originals: every pair of an original sentence that appears in one of the
   three test sets (data/source/idx_to_remove/list_idx_*.txt) is removed from training.
2. Keep complex originals only: READ-IT global of the original > 0.7, as in the test sets.
3. Drop the pairs on the GFFF langdetect list (it only covers the CLiC-it selection).
4. Drop originals matching the reverse-engineered double filter: the original contains a double
   quote, a guillemet, a curly quote or an opening parenthesis, or is longer than 400 characters.
5. Drop exact duplicates (same original, same simplification text), keeping the first.
6. Measure READ-IT's noise on these pairs: for every two simplifications of the same original that
   are near-identical (difflib character similarity >= 0.9, not identical), the absolute difference
   of each READ-IT score. The margin of a dimension is the 90th percentile of those differences.
7. Bin every READ-IT score with the CLiC-it rule, level = 10 * floor(10 * score), capped at 90.
   For each dimension, the target is the simplification's level if that level is below the
   original's AND the score dropped by at least the dimension's margin; otherwise "any". Rows
   whose three targets are all "any" are dropped.
8. Mark 1% of the remaining originals (seed 42) as validation, so training and validation never
   share an original.

The control string is <|readability_base_T|><|readability_lexical_T|><|readability_syntax_T|>,
with T a level (0-90) or "any" (column ctrl_tokens). Column ctrl_tokens_with_source prefixes it with
the source levels: <|original_readability_base_S|><|original_readability_lexical_S|>
<|original_readability_syntax_S|>, S = 0-90 (also stored as src_*).

The test sets are the v0 test pairs (same 1,000 pairs per strategy, all columns including the
Profiling-UD features), with the v1 columns computed from the reference simplification with the
same margins.

Outputs, in data/v1/:
    training_set_v1.zip                    training and validation pairs (column split), with Profiling-UD features
    {base,lexical,syntax}_test_set_v1.csv  one test set per strategy
    construction_v1.json                   the counts of every step

Run from anywhere: python scripts/build_dataset_v1.py [--hf-dir SNAPSHOT]
(needs pandas, numpy, pyarrow and huggingface_hub).
"""

import argparse
import json
from difflib import SequenceMatcher
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from impacts import REPO, REVISION, SCORE_COLUMNS, load_wikipedia

ROOT = Path(__file__).resolve().parent.parent
LISTS = ROOT / "data" / "source" / "idx_to_remove"
OUT = ROOT / "data" / "v1"

TOKEN_DIMENSIONS = ("base", "lexical", "syntax")
TEST_STRATEGIES = ("base", "lexical", "syntax")
MIN_ORIGINAL_GLOBAL = 0.7
DOUBLE_FILTER_CHARS = r'["«»“”(]'
DOUBLE_FILTER_MAX_CHARS = 400
NEAR_DUPLICATE_SIMILARITY = 0.9
NOISE_PERCENTILE = 90
VALIDATION_SHARE = 0.01
SEED = 42
BROKEN_COLUMNS = ["n_sentences_original", "n_sentences_simplification"]  # always 1 (see build_dataset_v0.py)


def read_ids(path: Path) -> set[int]:
    return {int(line) for line in path.read_text().split()}


def level(score: pd.Series) -> pd.Series:
    """READ-IT score (0-1) to control level: 0.43 -> 40; 1.0 -> 90."""
    return (np.minimum(np.floor(score * 10), 9) * 10).astype(int)


def double_filter_proxy(text: pd.Series) -> pd.Series:
    return text.str.contains(DOUBLE_FILTER_CHARS, regex=True) | (text.str.len() > DOUBLE_FILTER_MAX_CHARS)


def _near_duplicate_differences(group: np.ndarray) -> list:
    out = []
    for a, b in combinations(group, 2):
        if a[0] == b[0]:
            continue
        sm = SequenceMatcher(None, a[0], b[0])
        if (sm.real_quick_ratio() >= NEAR_DUPLICATE_SIMILARITY and sm.quick_ratio() >= NEAR_DUPLICATE_SIMILARITY
                and sm.ratio() >= NEAR_DUPLICATE_SIMILARITY):
            out.append([100 * abs(a[i] - b[i]) for i in range(1, len(a))])
    return out


def readit_noise(pairs: pd.DataFrame) -> dict:
    """Differences in READ-IT (0-100) between near-identical simplifications of the same original."""
    cols = ["simplification"] + [f"simplification_{d}" for d in TOKEN_DIMENSIONS]
    groups = [g.to_numpy() for _, g in pairs.groupby("original_sentence_idx")[cols] if len(g) > 1]
    with Pool(16) as pool:
        diffs = np.array([x for r in pool.map(_near_duplicate_differences, groups, chunksize=200) for x in r])
    return {"near_duplicate_pairs": int(len(diffs)),
            **{d: {f"p{q}": round(float(np.percentile(diffs[:, i], q)), 1) for q in (50, 75, 90, 95)}
               for i, d in enumerate(TOKEN_DIMENSIONS)}}


def with_targets(pairs: pd.DataFrame, margins: dict) -> pd.DataFrame:
    """Add src_* (source levels), tgt_* (target level, or "any" unless the bin went down by at least
    the dimension's margin) and ctrl_tokens."""
    pairs = pairs.copy()
    for d in TOKEN_DIMENSIONS:
        src, out = level(pairs[f"original_{d}"]), level(pairs[f"simplification_{d}"])
        dropped = 100 * (pairs[f"original_{d}"] - pairs[f"simplification_{d}"])
        pairs[f"src_{d}"] = src
        pairs[f"tgt_{d}"] = np.where((out < src) & (dropped >= margins[d]), out.astype(str), "any")
    pairs["ctrl_tokens"] = ["".join(f"<|readability_{d}_{t}|>" for d, t in zip(TOKEN_DIMENSIONS, row))
                            for row in pairs[[f"tgt_{d}" for d in TOKEN_DIMENSIONS]].itertuples(index=False)]
    source = ["".join(f"<|original_readability_{d}_{lv}|>" for d, lv in zip(TOKEN_DIMENSIONS, row))
              for row in pairs[[f"src_{d}" for d in TOKEN_DIMENSIONS]].itertuples(index=False)]
    pairs["ctrl_tokens_with_source"] = [a + b for a, b in zip(source, pairs["ctrl_tokens"])]
    return pairs


def has_target(pairs: pd.DataFrame) -> pd.Series:
    return (pairs[[f"tgt_{d}" for d in TOKEN_DIMENSIONS]] != "any").any(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/v1/: training_set_v1.zip and the three v1 test sets.")
    parser.add_argument("--hf-dir", type=Path, help="local snapshot of mpapucci/impacts (default: download)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    corpus = load_wikipedia(profiling=True, local_dir=args.hf_dir).drop(columns=BROKEN_COLUMNS)
    corpus = corpus.sort_values("idx").reset_index(drop=True)
    steps = []

    def step(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        steps.append({"step": name, "pairs": int(len(frame)), "originals": int(frame.original_sentence_idx.nunique())})
        print(f"{name:60s} {len(frame):8d} pairs {frame.original_sentence_idx.nunique():7d} originals")
        return frame

    core = ["idx", "original_sentence_idx", "original_text", "simplification"] + SCORE_COLUMNS
    pairs = step("release, Wikipedia", corpus[core])

    test_ids = {s: read_ids(LISTS / f"list_idx_{s}.txt") for s in TEST_STRATEGIES}
    test_originals = set(corpus.loc[corpus.idx.isin(set().union(*test_ids.values())), "original_sentence_idx"])
    pairs = step("1. test originals held out", pairs[~pairs.original_sentence_idx.isin(test_originals)])
    pairs = step("2. original READ-IT global > 70", pairs[pairs.original_all > MIN_ORIGINAL_GLOBAL])
    pairs = step("3. langdetect list removed", pairs[~pairs.idx.isin(read_ids(LISTS / "list_dropped_idx_langdetect.txt"))])
    pairs = step("4. double-filter proxy on the original", pairs[~double_filter_proxy(pairs.original_text)])
    pairs = step("5. exact duplicates removed", pairs.drop_duplicates(["original_sentence_idx", "simplification"]))
    noise = readit_noise(pairs)
    margins = {d: noise[d][f"p{NOISE_PERCENTILE}"] for d in TOKEN_DIMENSIONS}
    print(f"6. READ-IT noise on {noise['near_duplicate_pairs']} near-duplicate pairs; margins (p{NOISE_PERCENTILE}): {margins}")
    pairs = with_targets(pairs, margins)
    pairs = step("7. at least one target (one bin lower and drop >= margin)", pairs[has_target(pairs)])

    originals = np.sort(pairs.original_sentence_idx.unique())
    rng = np.random.default_rng(SEED)
    validation = set(rng.choice(originals, size=round(VALIDATION_SHARE * len(originals)), replace=False))
    pairs.insert(2, "split", np.where(pairs.original_sentence_idx.isin(validation), "validation", "train"))
    for split, frame in pairs.groupby("split"):
        step(f"8. split: {split}", frame)

    profiling = [c for c in corpus.columns if c not in pairs.columns]
    pairs = pairs.merge(corpus[["idx"] + profiling], on="idx", how="left", validate="one_to_one")
    pairs.to_csv(OUT / "training_set_v1.zip", index=False, compression="zip")
    print(f"training_set_v1.zip: {len(pairs)} pairs")

    tests = {}
    for s in TEST_STRATEGIES:
        test = with_targets(corpus[corpus.idx.isin(test_ids[s])], margins)
        if len(test) != len(test_ids[s]):
            raise ValueError(f"{s}: {len(test_ids[s]) - len(test)} test ids are missing from the release")
        test.to_csv(OUT / f"{s}_test_set_v1.csv", index=False)
        tests[s] = {"pairs": int(len(test)), "reference_with_no_target": int((~has_target(test)).sum())}
        print(f"{s}_test_set_v1.csv: {len(test)} pairs")

    (OUT / "construction_v1.json").write_text(json.dumps({
        "source": {"repo": REPO, "revision": REVISION, "config": "wikipedia_profiling"},
        "rules": {"min_original_global": MIN_ORIGINAL_GLOBAL, "level": "10 * min(floor(10 * score), 9)",
                  "target": "level of the simplification if below the original's and the score dropped by >= margin, else any",
                  "margins": margins, "margin_rule": f"p{NOISE_PERCENTILE} of |READ-IT difference| between near-identical "
                  f"(similarity >= {NEAR_DUPLICATE_SIMILARITY}) simplifications of the same original",
                  "readit_noise": noise,
                  "double_filter_proxy": {"original_matches": DOUBLE_FILTER_CHARS, "or_longer_than_chars": DOUBLE_FILTER_MAX_CHARS},
                  "validation_share_of_originals": VALIDATION_SHARE, "seed": SEED},
        "steps": steps, "held_out_test_originals": len(test_originals), "test_sets": tests,
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
