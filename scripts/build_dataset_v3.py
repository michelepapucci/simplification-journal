"""Build dataset v3: dataset v2 with same-level targets for base and syntax.

In v1 and v2 a dimension gets a target only if the simplification falls to a lower READ-IT level and
the score drops by at least the noise margin; otherwise "any". So "any" means both "free" and "stayed
where it was", and the models never see a token asking for a light edit. The CLiC-it 2026 data does
have such tokens (18% of its base tokens are at the original's own level), and the CLiC-it 0.6B is
the only model that follows mild base requests (levels 60-80), also on test originals it never saw
(data/v1/experiments/README.md).

v3 changes one thing. For base and syntax, a pair whose simplification stays in the original's level
gets that level as target, unless the score rose by at least the margin (then it stays "any"):

    simplification in a lower level, drop >= margin   -> that level          (as v1/v2)
    simplification in the original's level, no rise   -> the original's level (new)
    anything else (crossing a level within noise, a rise) -> "any"           (as v1/v2)

Lexical is left as in v2: the same rule would relabel 53% of all pairs as "keep lexical", a separate
experiment. Pairs, splits, columns and test sets are v2's; only tgt_base, tgt_syntax, ctrl_tokens and
ctrl_tokens_with_source change. The test sets are copied unchanged (their target columns follow the
v1 rule; the evaluation requests do not use them).

Outputs, in data/v3/:
    training_set_v3.zip                    v2's pairs with the v3 targets
    {base,lexical,syntax}_test_set_v3.csv  identical to the v2 (and v1) test sets
    construction_v3.json                   rule, margins and counts

Run after scripts/build_dataset_v2.py: python scripts/build_dataset_v3.py
"""

import itertools
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset_v1 as v1  # noqa: E402  levels and the token format

ROOT = Path(__file__).resolve().parent.parent
V2_DIR, OUT = ROOT / "data" / "v2", ROOT / "data" / "v3"
DIMS = v1.TOKEN_DIMENSIONS
SAME_LEVEL_DIMS = ("base", "syntax")
TARGET_DROPOUT = 0.3   # data/v1/experiments/train.py, for the per-epoch composition only


def epoch_composition(targets: pd.DataFrame) -> dict:
    """Expected share of training examples per set of targets after target dropout (train.py's rule:
    each set target is dropped with p 0.3, at least one is kept)."""
    seen = Counter()
    for key, n in targets.apply(lambda r: tuple(d for d in DIMS if r[f"tgt_{d}"] != "any"), axis=1).value_counts().items():
        for k in range(1, len(key) + 1):
            for kept in itertools.combinations(key, k):
                seen[kept] += n * (1 - TARGET_DROPOUT) ** len(kept) * TARGET_DROPOUT ** (len(key) - len(kept))
        for d in key:
            seen[(d,)] += n * TARGET_DROPOUT ** len(key) / len(key)
    total = sum(seen.values())
    return {"+".join(k): round(v / total, 4) for k, v in sorted(seen.items(), key=lambda t: -t[1])}


def by_level(col: pd.Series) -> dict:
    return {int(k): int(v) for k, v in col[col != "any"].astype(int).value_counts().sort_index().items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    construction_v2 = json.loads((V2_DIR / "construction_v2.json").read_text())
    margins = construction_v2["rules"]["margins"]
    pairs = pd.read_csv(V2_DIR / "training_set_v2.zip", dtype={f"tgt_{d}": str for d in DIMS}, low_memory=False)
    before = pairs[[f"tgt_{d}" for d in DIMS] + ["split"]].copy()

    counts = {}
    for d in DIMS:
        src, out = v1.level(pairs[f"original_{d}"]), v1.level(pairs[f"simplification_{d}"])
        assert (src == pairs[f"src_{d}"]).all(), f"src_{d} differs from the original's level"
        dropped = 100 * (pairs[f"original_{d}"] - pairs[f"simplification_{d}"])
        v1_rule = np.where((out < src) & (dropped >= margins[d]), out.astype(str), "any")
        assert (v1_rule == pairs[f"tgt_{d}"]).all(), f"tgt_{d} does not follow the v1 rule"
        if d in SAME_LEVEL_DIMS:
            same = (pairs[f"tgt_{d}"] == "any") & (out == src) & (dropped > -margins[d])
            pairs.loc[same, f"tgt_{d}"] = src[same].astype(str)
            counts[d] = {"same_level_targets": int(same.sum()),
                         "same_level_train": int((same & (pairs.split == "train")).sum()),
                         "kept_any_rise_within_level": int(((before[f"tgt_{d}"] == "any") & (out == src) & ~(dropped > -margins[d])).sum()),
                         "median_change_same_level": round(float(-dropped[same].median()), 2),
                         "train_targets_by_level": {"v2": by_level(before.loc[before.split == "train", f"tgt_{d}"]),
                                                    "v3": by_level(pairs.loc[pairs.split == "train", f"tgt_{d}"])}}
            print(f"{d}: {same.sum():,} same-level targets (median change {-dropped[same].median():+.1f} points)", flush=True)
    pairs["ctrl_tokens"] = ["".join(f"<|readability_{d}_{t}|>" for d, t in zip(DIMS, row))
                            for row in pairs[[f"tgt_{d}" for d in DIMS]].itertuples(index=False)]
    source = ["".join(f"<|original_readability_{d}_{lv}|>" for d, lv in zip(DIMS, row))
              for row in pairs[[f"src_{d}" for d in DIMS]].itertuples(index=False)]
    pairs["ctrl_tokens_with_source"] = [a + b for a, b in zip(source, pairs["ctrl_tokens"])]

    pairs.to_csv(OUT / "training_set_v3.zip", index=False, compression=dict(method="zip", archive_name="training_set_v3"))
    print(f"training_set_v3.zip: {len(pairs):,} pairs", flush=True)
    for s in v1.TEST_STRATEGIES:
        shutil.copyfile(V2_DIR / f"{s}_test_set_v2.csv", OUT / f"{s}_test_set_v3.csv")

    train = pairs.split == "train"
    (OUT / "construction_v3.json").write_text(json.dumps({
        "based_on": "data/v2/training_set_v2.zip (same pairs, splits and columns)",
        "rules": {"same_level_targets": list(SAME_LEVEL_DIMS),
                  "same_level": "target = the original's level when the simplification stays in it and the score did not rise by the margin or more",
                  "otherwise": "as v2 (dataset v1 rule)", "margins": margins},
        "pairs": int(len(pairs)), "train": int(train.sum()), "validation": int((~train).sum()),
        "changed": counts,
        "pairs_with_all_three_targets": {"v2": int((before.loc[train, [f"tgt_{d}" for d in DIMS]] != "any").all(axis=1).sum()),
                                         "v3": int((pairs.loc[train, [f"tgt_{d}" for d in DIMS]] != "any").all(axis=1).sum())},
        "epoch_composition_with_dropout": {"v2": epoch_composition(before[train]), "v3": epoch_composition(pairs[train])},
        "test_sets": {s: f"identical to data/v2/{s}_test_set_v2.csv" for s in v1.TEST_STRATEGIES},
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
