"""Build dataset v4: dataset v3 with syntax targets on every pair whose syntax did not rise.

v3 gave base and syntax same-level targets. Mild base requests improved; syntax did not
(data/v1/experiments/README.md, "Dataset v3"). The reason is the noise margin: a syntax drop gets a
target only if it is at least 17.6 points, which is wider than a level. So at the top of the scale
almost no one-level syntax drop is labelled (originals at 80: 77 of 1,483 drops to 70; originals at
90: none of 417), and the model never sees "syntax one level below" there.

v4 changes one thing. For syntax, a pair whose simplification crosses into a lower level within the
margin gets the simplification's level as target, instead of "any". Together with v3's rules, syntax
then follows the CLiC-it 2026 rule without its exclusions:

    simplification's syntax level <= the original's   -> that level  (v4: also within the margin)
    simplification's syntax level  > the original's    -> "any"       (as before)

Base and lexical are left as in v3. Pairs, splits, columns and test sets are v3's (and v2's); only
tgt_syntax, ctrl_tokens and ctrl_tokens_with_source change. The test sets are copied unchanged.

Outputs, in data/v4/:
    training_set_v4.zip                    v3's pairs with the v4 syntax targets
    {base,lexical,syntax}_test_set_v4.csv  identical to the v3 (and v1) test sets
    construction_v4.json                   rule, margins and counts

Run after scripts/build_dataset_v3.py: python scripts/build_dataset_v4.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset_v1 as v1  # noqa: E402  levels and the token format
from build_dataset_v3 import by_level, epoch_composition  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V2_DIR, V3_DIR, OUT = ROOT / "data" / "v2", ROOT / "data" / "v3", ROOT / "data" / "v4"
DIMS = v1.TOKEN_DIMENSIONS
DIM = "syntax"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    margins = json.loads((V2_DIR / "construction_v2.json").read_text())["rules"]["margins"]
    pairs = pd.read_csv(V3_DIR / "training_set_v3.zip", dtype={f"tgt_{d}": str for d in DIMS}, low_memory=False)
    before = pairs[[f"tgt_{d}" for d in DIMS] + ["split"]].copy()

    src, out = v1.level(pairs[f"original_{DIM}"]), v1.level(pairs[f"simplification_{DIM}"])
    dropped = 100 * (pairs[f"original_{DIM}"] - pairs[f"simplification_{DIM}"])
    v3_rule = np.where((out < src) & (dropped >= margins[DIM]), out.astype(str),
                       np.where((out == src) & (dropped > -margins[DIM]), src.astype(str), "any"))
    assert (v3_rule == pairs[f"tgt_{DIM}"]).all(), f"tgt_{DIM} does not follow the v3 rule"
    cross = (pairs[f"tgt_{DIM}"] == "any") & (out < src)
    pairs.loc[cross, f"tgt_{DIM}"] = out[cross].astype(str)
    assert ((pairs[f"tgt_{DIM}"] == "any") == (out > src)).all(), "every syntax target is set unless syntax rose"
    print(f"{DIM}: {cross.sum():,} crossings within the margin labelled (median change {-dropped[cross].median():+.1f} points); "
          f"{(out > src).sum():,} rises stay any", flush=True)

    pairs["ctrl_tokens"] = ["".join(f"<|readability_{d}_{t}|>" for d, t in zip(DIMS, row))
                            for row in pairs[[f"tgt_{d}" for d in DIMS]].itertuples(index=False)]
    source = ["".join(f"<|original_readability_{d}_{lv}|>" for d, lv in zip(DIMS, row))
              for row in pairs[[f"src_{d}" for d in DIMS]].itertuples(index=False)]
    pairs["ctrl_tokens_with_source"] = [a + b for a, b in zip(source, pairs["ctrl_tokens"])]

    pairs.to_csv(OUT / "training_set_v4.zip", index=False, compression=dict(method="zip", archive_name="training_set_v4"))
    print(f"training_set_v4.zip: {len(pairs):,} pairs", flush=True)
    for s in v1.TEST_STRATEGIES:
        shutil.copyfile(V3_DIR / f"{s}_test_set_v3.csv", OUT / f"{s}_test_set_v4.csv")

    train = pairs.split == "train"
    one_below = {}
    for lv in (60, 70, 80, 90):
        sel = train & (src == lv)
        one_below[str(lv)] = {"v3": int((sel & (before[f"tgt_{DIM}"] == str(lv - 10))).sum()),
                              "v4": int((sel & (pairs[f"tgt_{DIM}"] == str(lv - 10))).sum())}
    (OUT / "construction_v4.json").write_text(json.dumps({
        "based_on": "data/v3/training_set_v3.zip (same pairs, splits and columns)",
        "rules": {"changed_dimension": DIM,
                  "syntax": "target = the simplification's syntax level unless it is higher than the original's (then any)",
                  "base": "as v3", "lexical": "as v3 (= v2)", "margins": margins},
        "pairs": int(len(pairs)), "train": int(train.sum()), "validation": int((~train).sum()),
        "changed": {"crossings_labelled": int(cross.sum()), "crossings_train": int((cross & train).sum()),
                    "crossings_by_levels_dropped": {str(int(k)): int(v) for k, v in (src[cross] - out[cross]).value_counts().sort_index().items()},
                    "median_change_crossings": round(float(-dropped[cross].median()), 2),
                    "rises_kept_any": int((out > src).sum()),
                    "train_targets_by_level": {"v3": by_level(before.loc[train, f"tgt_{DIM}"]), "v4": by_level(pairs.loc[train, f"tgt_{DIM}"])},
                    "train_one_level_below_by_original_level": one_below},
        "pairs_with_all_three_targets": {"v3": int((before.loc[train, [f"tgt_{d}" for d in DIMS]] != "any").all(axis=1).sum()),
                                         "v4": int((pairs.loc[train, [f"tgt_{d}" for d in DIMS]] != "any").all(axis=1).sum())},
        "epoch_composition_with_dropout": {"v3": epoch_composition(before[train]), "v4": epoch_composition(pairs[train])},
        "test_sets": {s: f"identical to data/v3/{s}_test_set_v3.csv" for s in v1.TEST_STRATEGIES},
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
