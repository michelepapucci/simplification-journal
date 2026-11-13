"""Build the evaluation requests (written to eval_requests/, generated once, shared by every model).

main-<dim>.csv  One-dimension requests, the main evaluation. For each original of the <dim> test
                set, every level from 0 up to one level below the original's own level in <dim>
                (from its IMPaCTS label); the other two dimensions are "any". Base 8,054 requests,
                lexical 3,862, syntax 7,905.
joint.csv       Three-dimension requests: every target set, to measure whether all three are met
                at once. Each of the 2,575 test originals gets at most two:
                - "achievable": the targets of one of its own simplifications in IMPaCTS that lowers
                  all three dimensions beyond READ-IT's noise (v1's labelling rule), i.e. a
                  combination the generator reached for this very sentence (77% of originals);
                - "typical": the targets of a training pair with all three targets set whose
                  original starts at the same three levels (nearest levels if none), restricted to
                  targets below this original's own levels; a realistic combination not tied to
                  this sentence.
                Test originals were held out of training, so no request reuses a training pair of
                the same original.

    python data/v1/experiments/build_requests.py [--hf-dir SNAPSHOT]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import DIMS, HERE, LEVELS, ROOT, SEED, V1

sys.path.insert(0, str(ROOT / "scripts"))
from build_dataset_v1 import with_targets  # noqa: E402
from impacts import SCORE_COLUMNS, load_wikipedia  # noqa: E402

REQUESTS = HERE / "eval_requests"


def originals() -> pd.DataFrame:
    """One row per test original, with its IMPaCTS labels and source levels."""
    cols = ["original_sentence_idx", "original_text"] + [f"original_{d}" for d in (*DIMS, "all")] + [f"src_{d}" for d in DIMS]
    return pd.concat([pd.read_csv(V1 / f"{d}_test_set_v1.csv", usecols=cols) for d in DIMS]).drop_duplicates("original_sentence_idx")


def main_requests() -> None:
    for dim in DIMS:
        test = pd.read_csv(V1 / f"{dim}_test_set_v1.csv", usecols=["original_sentence_idx"] + [f"src_{d}" for d in DIMS])
        rows = [{"idx": r.original_sentence_idx, "dimension": dim, "level": lv,
                 **{f"tgt_{d}": (str(lv) if d == dim else "any") for d in DIMS},
                 **{f"src_{d}": getattr(r, f"src_{d}") for d in DIMS}}
                for r in test.itertuples(index=False) for lv in LEVELS if lv < getattr(r, f"src_{dim}")]
        pd.DataFrame(rows).to_csv(REQUESTS / f"main-{dim}.csv", index=False)
        print(f"main-{dim}.csv: {len(rows)} requests")


def joint_requests(hf_dir: Path | None) -> None:
    rng = np.random.default_rng(SEED)
    margins = json.loads((V1 / "construction_v1.json").read_text())["rules"]["margins"]
    orig = originals()

    rel = load_wikipedia(columns=["idx", "original_sentence_idx", "original_text", "simplification"] + SCORE_COLUMNS, local_dir=hf_dir)
    siblings = with_targets(rel[rel.original_sentence_idx.isin(orig.original_sentence_idx)], margins)
    siblings = siblings[(siblings[[f"tgt_{d}" for d in DIMS]] != "any").all(axis=1)]

    train = pd.read_csv(V1 / "training_set_v1.zip", usecols=["split"] + [f"src_{d}" for d in DIMS] + [f"tgt_{d}" for d in DIMS],
                        dtype={f"tgt_{d}": str for d in DIMS})
    pool = train[(train.split == "train") & (train[[f"tgt_{d}" for d in DIMS]] != "any").all(axis=1)]
    src_cols = [f"src_{d}" for d in DIMS]
    pool_src = pool[src_cols].to_numpy()
    pool_tgt = pool[[f"tgt_{d}" for d in DIMS]].astype(int).to_numpy()

    rows = []
    for r in orig.itertuples(index=False):
        src = {d: getattr(r, f"src_{d}") for d in DIMS}
        mine = siblings[siblings.original_sentence_idx == r.original_sentence_idx]
        if len(mine):
            pick = mine.iloc[rng.integers(len(mine))]
            rows.append({"idx": r.original_sentence_idx, "type": "achievable", "sibling_idx": int(pick.idx),
                         **{f"tgt_{d}": pick[f"tgt_{d}"] for d in DIMS}, **{f"src_{d}": src[d] for d in DIMS}})
        below = (pool_tgt < np.array([src[d] for d in DIMS])).all(axis=1)   # every target below this original's level
        dist = np.where(below, np.abs(pool_src - np.array([src[d] for d in DIMS])).sum(axis=1), np.iinfo(np.int64).max)
        candidates = pool[dist == dist.min()]
        pick = candidates.iloc[rng.integers(len(candidates))]
        rows.append({"idx": r.original_sentence_idx, "type": "typical", "source_distance": int(dist.min()),
                     **{f"tgt_{d}": pick[f"tgt_{d}"] for d in DIMS}, **{f"src_{d}": src[d] for d in DIMS}})
    joint = pd.DataFrame(rows)
    joint.to_csv(REQUESTS / "joint.csv", index=False)
    typical = joint[joint.type == "typical"]
    print(f"joint.csv: {len(joint)} requests ({joint.type.value_counts().to_dict()}); typical requests whose training "
          f"source levels match exactly: {(typical.source_distance == 0).mean():.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hf-dir", type=Path, help="local snapshot of mpapucci/impacts (default: download)")
    args = parser.parse_args()
    REQUESTS.mkdir(exist_ok=True)
    main_requests()
    joint_requests(args.hf_dir)


if __name__ == "__main__":
    main()
