"""Build the requests of the interpretability experiments (questions 1 and 2 of
reports/improving_controllability.md, Section 5.2): the same originals under every request of one
dimension, in two modes.

- free ("others any"): the lowered dimension at level L, the other two "any" (as in the main evaluation);
- keep ("others held"): the lowered dimension at level L, the other two at the original's own levels:
    base lowered:    base L,   lexical src, syntax src
    lexical lowered: base src, lexical L,   syntax src
    syntax lowered:  base src, lexical src, syntax L
  v5 has same-level targets for base and syntax (11.9% and 5.9% of training pairs) but never for lexical
  (0 pairs), so a held lexical token is outside its training data. With --lexical-any, lexical stays "any"
  when it is not the lowered dimension (the first, asymmetric version of the run).
- anchor (one per original): every dimension at the original's own level (lexical "any" with --lexical-any).

L runs over every level below the original's own level in the lowered dimension (0 ... src - 10), as in
the main evaluation. Originals: 1,000 test originals, drawn at random (seed 0) among the 2,369 whose three
levels are all at least 30, so that every dimension can be lowered by 1, 2 and 3 levels.

    python data/v1/experiments/interp/build_requests.py [--lexical-any] [--out interp/requests.csv]
Writes idx, mode, dimension, level, drop (levels below the original), tgt_*, src_*.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DIMS  # noqa: E402

N_ORIGINALS, SEED, MIN_LEVEL = 1000, 0, 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lexical-any", action="store_true", help="held mode leaves lexical 'any' (first version)")
    parser.add_argument("--out", type=Path, default=HERE / "requests.csv")
    args = parser.parse_args()
    held_lexical = (lambda s: "any") if args.lexical_any else (lambda s: s["lexical"])
    req = pd.concat([pd.read_csv(HERE.parent / "eval_requests" / f"main-{d}.csv") for d in DIMS]
                    + [pd.read_csv(HERE.parent / "eval_requests" / "joint.csv")])
    src = req.drop_duplicates("idx")[["idx", *[f"src_{d}" for d in DIMS]]].sort_values("idx")
    eligible = src[(src[[f"src_{d}" for d in DIMS]] >= MIN_LEVEL).all(axis=1)]
    chosen = eligible.sample(N_ORIGINALS, random_state=SEED).sort_values("idx")
    rows = []
    for o in chosen.itertuples(index=False):
        s = {d: int(getattr(o, f"src_{d}")) for d in DIMS}
        for mode in ("free", "keep"):
            for d in DIMS:
                for level in range(0, s[d], 10):
                    if mode == "free":
                        tgt = {e: "any" for e in DIMS}
                    else:
                        tgt = {"base": s["base"], "lexical": held_lexical(s), "syntax": s["syntax"]}
                    tgt[d] = level
                    rows.append({"idx": o.idx, "mode": mode, "dimension": d, "level": level, "drop": (s[d] - level) // 10,
                                 **{f"tgt_{e}": str(tgt[e]) for e in DIMS}, **{f"src_{e}": s[e] for e in DIMS}})
        rows.append({"idx": o.idx, "mode": "anchor", "dimension": "none", "level": -1, "drop": 0,
                     "tgt_base": str(s["base"]), "tgt_lexical": str(held_lexical(s)), "tgt_syntax": str(s["syntax"]),
                     **{f"src_{e}": s[e] for e in DIMS}})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"{len(chosen)} originals of {len(eligible)} eligible; {len(out)} requests:")
    print(out.groupby(["mode", "dimension"]).size().to_string())


if __name__ == "__main__":
    main()
