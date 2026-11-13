"""Build dataset v2: dataset v1 plus a lexical supplement, with a language check on every pair.

v1's lexical targets are almost all one level below the original (94%), because v1 only keeps
originals with READ-IT global > 0.7 that pass the double-filter proxy. v2 adds, from the rest of the
Wikipedia release, the pairs whose lexical score dropped by two levels or more, after removing the
kinds of pair whose lexical drop is not a lexical simplification. Everything else is v1's rule.

Construction, in order (data/v2/README.md reports the counts):

1. As in v1: hold out the test originals, drop the release's langdetect list, drop exact duplicates.
2. Targets with v1's rule and v1's margins (read from data/v1/construction_v1.json, not re-measured):
   a dimension's target is the simplification's level if it is below the original's and the score
   dropped by at least the margin, otherwise "any".
3. v1 part: the pairs v1 keeps (original global > 0.7, passes the double-filter proxy, at least one
   target). The script checks that they are exactly the pairs of data/v1/training_set_v1.zip.
4. Language check (new): langdetect (seed 0) on the simplification and on the original; pairs where
   either is not detected as Italian are dropped. The release's langdetect list only covers the
   141,984 pairs of the CLiC-it selection, so v1's other pairs were never checked.
5. Lexical supplement (new): pairs outside v1's selection (original global <= 0.7, or flagged by
   the double-filter proxy) with a lexical target at least two levels below the original's lexical
   level, keeping only pairs where
   - the simplification and the original are Italian (step 4);
   - the simplification is not a near-copy (difflib character similarity < 0.9);
   - more than names changed: the words that do not start with a capital letter differ;
   - the simplification has no line break (fragmented output);
   - the simplification has at least half the original's tokens (Profiling-UD n_tokens), so the drop
     does not come from deleting most of the sentence.
   Their base and syntax targets follow the same rule as every other pair.
6. Split: v1 pairs keep their v1 split; 1% of the supplement's originals (seed 42) are validation.
   Supplement originals are never v1 originals, so no original is in both splits.

Column `subset` says where a pair comes from: "v1" or "lexical_supplement". The test sets are the v1
test sets, unchanged (copied as *_test_set_v2.csv), so v1 and v2 models are evaluated on the same
pairs.

Outputs, in data/v2/:
    training_set_v2.zip                    training and validation pairs, with Profiling-UD features
    {base,lexical,syntax}_test_set_v2.csv  identical to the v1 test sets
    construction_v2.json                   rules and the counts of every step

Run from anywhere: python scripts/build_dataset_v2.py [--hf-dir SNAPSHOT]
(needs pandas, numpy, pyarrow, huggingface_hub and langdetect).
"""

import argparse
import json
import re
import shutil
import sys
from difflib import SequenceMatcher
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset_v1 as v1  # noqa: E402  same lists, levels, proxy and target rule
from impacts import REPO, REVISION, load_wikipedia  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V1_DIR = ROOT / "data" / "v1"
OUT = ROOT / "data" / "v2"
DIMS = v1.TOKEN_DIMENSIONS
LANGUAGE = "it"
MIN_LEXICAL_DROP_LEVELS = 2
MAX_SIMILARITY = 0.9
MIN_TOKEN_RATIO = 0.5
VALIDATION_SHARE = v1.VALIDATION_SHARE
SEED = v1.SEED


def _init_langdetect():
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0


def _detect(text: str) -> str:
    from langdetect import detect
    try:
        return detect(text)
    except Exception:  # no letters at all
        return "unknown"


def languages(texts: list[str]) -> list[str]:
    with Pool(16, initializer=_init_langdetect) as pool:
        return pool.map(_detect, texts, chunksize=500)


def _similarity(pair: tuple[str, str]) -> float:
    return SequenceMatcher(None, *pair).ratio()


def similarities(a: pd.Series, b: pd.Series) -> np.ndarray:
    with Pool(16) as pool:
        return np.array(pool.map(_similarity, list(zip(a, b)), chunksize=500))


def lowercase_words(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text) if not w[:1].isupper()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/v2/: training_set_v2.zip and the three v2 test sets.")
    parser.add_argument("--hf-dir", type=Path, help="local snapshot of mpapucci/impacts (default: download)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    corpus = load_wikipedia(profiling=True, local_dir=args.hf_dir).drop(columns=v1.BROKEN_COLUMNS)
    corpus = corpus.sort_values("idx").reset_index(drop=True)
    margins = json.loads((V1_DIR / "construction_v1.json").read_text())["rules"]["margins"]
    steps = []

    def step(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        steps.append({"step": name, "pairs": int(len(frame)), "originals": int(frame.original_sentence_idx.nunique())})
        print(f"{name:72s} {len(frame):8d} pairs {frame.original_sentence_idx.nunique():7d} originals", flush=True)
        return frame

    core = ["idx", "original_sentence_idx", "original_text", "simplification"] + v1.SCORE_COLUMNS + ["n_tokens_original", "n_tokens_simplification"]
    pairs = step("release, Wikipedia", corpus[core])
    test_ids = {s: v1.read_ids(v1.LISTS / f"list_idx_{s}.txt") for s in v1.TEST_STRATEGIES}
    test_originals = set(corpus.loc[corpus.idx.isin(set().union(*test_ids.values())), "original_sentence_idx"])
    pairs = step("1a. test originals held out", pairs[~pairs.original_sentence_idx.isin(test_originals)])
    pairs = step("1b. langdetect list removed", pairs[~pairs.idx.isin(v1.read_ids(v1.LISTS / "list_dropped_idx_langdetect.txt"))])
    pairs = step("1c. exact duplicates removed", pairs.drop_duplicates(["original_sentence_idx", "simplification"]))
    pairs = v1.with_targets(pairs, margins)
    in_v1_selection = (pairs.original_all > v1.MIN_ORIGINAL_GLOBAL) & ~v1.double_filter_proxy(pairs.original_text)

    # 3. v1 part
    part_v1 = pairs[in_v1_selection & v1.has_target(pairs)]
    v1_split = pd.read_csv(V1_DIR / "training_set_v1.zip", usecols=["idx", "split"]).set_index("idx").split
    if set(part_v1.idx) != set(v1_split.index):
        raise ValueError(f"v1 part differs from training_set_v1.zip: {len(set(part_v1.idx) ^ set(v1_split.index))} pairs")
    step("3. v1 part (= training_set_v1.zip)", part_v1)

    # 5 (candidates first, so that the language check only runs where it is needed)
    drop_levels = (pairs.src_lexical - v1.level(pairs.simplification_lexical)) // 10
    candidates = step(f"5a. outside v1's selection, lexical target >= {MIN_LEXICAL_DROP_LEVELS} levels below the original",
                      pairs[~in_v1_selection & (pairs.tgt_lexical != "any") & (drop_levels >= MIN_LEXICAL_DROP_LEVELS)])

    # 4. language of both sides
    checked = pd.concat([part_v1, candidates])
    lang_simplification = dict(zip(checked.idx, languages(checked.simplification.tolist())))
    originals = checked.drop_duplicates("original_sentence_idx")
    lang_original = dict(zip(originals.original_sentence_idx, languages(originals.original_text.tolist())))
    italian = lambda f: (f.idx.map(lang_simplification) == LANGUAGE) & (f.original_sentence_idx.map(lang_original) == LANGUAGE)
    part_v1_it = step("4. v1 part, original and simplification detected as Italian", part_v1[italian(part_v1)])

    sup = step("5b. supplement: original and simplification detected as Italian", candidates[italian(candidates)])
    sup = sup.assign(similarity=similarities(sup.original_text, sup.simplification))
    sup = step(f"5c. supplement: not a near-copy (similarity < {MAX_SIMILARITY})", sup[sup.similarity < MAX_SIMILARITY])
    sup = step("5d. supplement: more than names changed",
               sup[[lowercase_words(a) != lowercase_words(b) for a, b in zip(sup.original_text, sup.simplification)]])
    sup = step("5e. supplement: no line break in the simplification", sup[~sup.simplification.str.contains("\n")])
    sup = step(f"5f. supplement: simplification has >= {MIN_TOKEN_RATIO}x the original's tokens",
               sup[sup.n_tokens_simplification >= MIN_TOKEN_RATIO * sup.n_tokens_original]).drop(columns="similarity")

    # 6. split
    sup_originals = np.sort(sup.original_sentence_idx.unique())
    validation = set(np.random.default_rng(SEED).choice(sup_originals, size=round(VALIDATION_SHARE * len(sup_originals)), replace=False))
    part_v1_it = part_v1_it.assign(subset="v1", split=part_v1_it.idx.map(v1_split))
    sup = sup.assign(subset="lexical_supplement", split=np.where(sup.original_sentence_idx.isin(validation), "validation", "train"))
    out = pd.concat([part_v1_it, sup]).sort_values("idx").drop(columns=["n_tokens_original", "n_tokens_simplification"])
    for (subset, split), frame in out.groupby(["subset", "split"]):
        step(f"6. {subset}, {split}", frame)
    pairs_out = step("v2", out)

    order = list(pd.read_csv(V1_DIR / "training_set_v1.zip", nrows=0).columns)
    profiling = [c for c in corpus.columns if c not in pairs_out.columns]
    pairs_out = pairs_out.merge(corpus[["idx"] + profiling], on="idx", how="left", validate="one_to_one")
    pairs_out = pairs_out[order[:3] + ["subset"] + order[3:]]
    pairs_out.to_csv(OUT / "training_set_v2.zip", index=False, compression=dict(method="zip", archive_name="training_set_v2"))
    print(f"training_set_v2.zip: {len(pairs_out)} pairs", flush=True)

    tests = {}
    for s in v1.TEST_STRATEGIES:
        shutil.copyfile(V1_DIR / f"{s}_test_set_v1.csv", OUT / f"{s}_test_set_v2.csv")
        t = pd.read_csv(OUT / f"{s}_test_set_v2.csv", usecols=["original_text", "simplification"])
        tests[s] = {"pairs": int(len(t)), "identical_to": f"data/v1/{s}_test_set_v1.csv",
                    "original_not_italian": int(sum(x != LANGUAGE for x in languages(t.original_text.tolist()))),
                    "reference_not_italian": int(sum(x != LANGUAGE for x in languages(t.simplification.tolist())))}

    def lexical_targets(f: pd.DataFrame) -> dict:
        t = f[(f.split == "train") & (f.tgt_lexical != "any")]
        below = (t.src_lexical - t.tgt_lexical.astype(int)) // 10
        return {"pairs": int(len(t)), "two_or_more_levels_below": int((below >= 2).sum()),
                "by_level": {int(k): int(v) for k, v in t.tgt_lexical.astype(int).value_counts().sort_index().items()},
                "lexical_only": int(((t.tgt_base == "any") & (t.tgt_syntax == "any")).sum())}

    (OUT / "construction_v2.json").write_text(json.dumps({
        "source": {"repo": REPO, "revision": REVISION, "config": "wikipedia_profiling"},
        "rules": {"base": "dataset v1 (scripts/build_dataset_v1.py), same margins", "margins": margins,
                  "language": f"langdetect 1.0.9, seed 0, on original and simplification; both must be '{LANGUAGE}'",
                  "lexical_supplement": {"outside": "v1's selection (original global > 0.7 and passes the double-filter proxy)",
                                         "min_lexical_drop_levels": MIN_LEXICAL_DROP_LEVELS, "max_similarity": MAX_SIMILARITY,
                                         "more_than_names_changed": True, "no_line_break": True, "min_token_ratio": MIN_TOKEN_RATIO},
                  "validation_share_of_supplement_originals": VALIDATION_SHARE, "seed": SEED},
        "steps": steps, "held_out_test_originals": len(test_originals), "test_sets": tests,
        "lexical_targets_train": {"v1_part": lexical_targets(pairs_out[pairs_out.subset == "v1"]), "v2": lexical_targets(pairs_out)},
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
