"""Build dataset v6: dataset v3 plus identity pairs, dosed for the format the syntax evaluation uses.

v5 added 989 identity pairs to v3, as many as give v3's syntax targets at levels 60-90 the CLiC-it 2026
data's share of identical pairs (5%, 18.6%, 33.5%, 50%). Syntax at the top rose past the CLiC-it 0.6B,
but copies doubled theirs (data/v1/experiments/README.md, "Dataset v5"). The count was wrong for the
format: the CLiC-it data shows every pair under one token, and its evaluation asks for one token. Our
syntax evaluation also asks for the syntax token alone, but v3's pairs carry up to three targets and
show the syntax token alone only when target dropout removes the others. The identity pairs, which have
only a syntax target, always show it alone. Among the syntax-only examples, v5's identity share was
14.6% / 48.4% / 70.4% / 82.7% at 60 / 70 / 80 / 90, two to three times the CLiC-it data's.

v6 is v3 plus identity pairs whose number gives the CLiC-it share among the syntax-only examples per
epoch. A v3 pair with k set targets, one of them syntax, shows syntax alone with probability
0.7 * 0.3^(k-1) + 0.3^k / k (train.py: each set target dropped with p 0.3, one kept at random if all
are dropped). Everything else is as in v5: identity pairs from distinct training originals at the
level, drawn at random (seed 0); syntax target = the original's level; base and lexical "any".

Outputs, in data/v6/:
    training_set_v6.zip                    v3's pairs plus the identity pairs
    {base,lexical,syntax}_test_set_v6.csv  identical to the v3 (and v1) test sets
    construction_v6.json                   shares, exposures, counts and the per-epoch composition

Run after scripts/build_dataset_v3.py: python scripts/build_dataset_v6.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset_v5 as v5  # noqa: E402  identity pairs, CLiC-it shares
from build_dataset_v3 import by_level, epoch_composition  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "v6"
DIMS, LEVELS, TARGET_DROPOUT = v5.DIMS, v5.LEVELS, 0.3


def syntax_only_exposure(pairs) -> dict:
    """Expected syntax-only examples per epoch, by syntax target level, for the training pairs."""
    t = pairs[pairs.split == "train"]
    k = (t[[f"tgt_{d}" for d in DIMS]] != "any").sum(axis=1)
    p = (1 - TARGET_DROPOUT) * TARGET_DROPOUT ** (k - 1) + TARGET_DROPOUT ** k / k
    return {L: float(p[t.tgt_syntax == str(L)].sum()) for L in LEVELS}


def main() -> None:
    pairs = v5.load_v3()
    share = v5.clicit_identity_share()
    exposure = syntax_only_exposure(pairs)
    counts = {L: int(round(share[L]["share"] / (1 - share[L]["share"]) * exposure[L])) for L in LEVELS}
    ident, candidates = v5.pick_originals(pairs, counts)
    plan = {}
    for L in LEVELS:
        f = share[L]["share"]
        plan[str(L)] = {"clicit_share": round(f, 4), "v3_syntax_only_examples_per_epoch": round(exposure[L], 1),
                        "identity_pairs": counts[L], "identity_share_of_syntax_only_examples": round(counts[L] / (counts[L] + exposure[L]), 4),
                        "candidate_originals": candidates[L]}
        print(f"syntax {L}: CLiC-it share {100 * f:.1f}%; v3 shows {exposure[L]:,.0f} syntax-only examples per epoch "
              f"-> {counts[L]:,} identity pairs ({100 * counts[L] / (counts[L] + exposure[L]):.1f}% of syntax-only examples)", flush=True)
    ident = v5.identity_rows(pairs, ident)
    out = v5.write(pairs, ident, OUT, "v6")

    train = pairs.split == "train"
    t3, t6 = pairs[train], out[out.split == "train"]
    (OUT / "construction_v6.json").write_text(json.dumps({
        "based_on": "data/v3/training_set_v3.zip (same pairs, splits, labels and columns)",
        "rules": {"identity_pairs": "for syntax levels 60-90: original -> original, from distinct training originals at that syntax level, "
                                    "as many as give the CLiC-it data's share of identical pairs among the syntax-only examples per epoch",
                  "syntax_only_probability": "0.7 * 0.3^(k-1) + 0.3^k / k for a pair with k set targets including syntax",
                  "identity_targets": "syntax = the original's level; base = any; lexical = any",
                  "seed": v5.SEED, "levels": list(LEVELS)},
        "pairs": int(len(out)), "train": int((out.split == "train").sum()), "validation": int((out.split == "validation").sum()),
        "identity_pairs": int(len(ident)), "identity_idx_from": int(ident.idx.min()),
        "by_level": plan,
        "train_syntax_targets_by_level": {"v3": by_level(t3.tgt_syntax), "v6": by_level(t6.tgt_syntax)},
        "epoch_composition_with_dropout": {"v3": epoch_composition(t3), "v6": epoch_composition(t6)},
        "test_sets": {s: f"identical to data/v3/{s}_test_set_v3.csv" for s in v5.v1.TEST_STRATEGIES},
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
