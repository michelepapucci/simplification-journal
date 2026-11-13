"""Build dataset v5: dataset v3 plus identity pairs at the top of the syntax scale.

v4 labelled every syntax crossing and syntax at the top did not move (data/v1/experiments/README.md,
"Dataset v4"). In this data, simplifications of originals at syntax 80 are equally heavy edits whatever
level their syntax score lands in, so no label can teach "change little" for a high target. The CLiC-it
2026 data teaches it with identity pairs: the original sentence as its own simplification, under the
original's level. At its highest syntax request the CLiC-it 0.6B copies 22% of the time.

v5 tests whether those identity pairs explain the CLiC-it model's syntax at the top. It is dataset v3
(not v4: the v4 relabelling did not help syntax and cost base) plus identity pairs:
- for each syntax level L in 60-90, as many identity pairs as give the share of identical pairs among
  the syntax targets at L that the CLiC-it data has (5% at 60, 19% at 70, 33% at 80, 50% at 90);
- each from a different training original at syntax level L, drawn at random (seed 0);
- target: syntax = L; base and lexical = "any". The CLiC-it data shows each identity pair under one
  token at a time; here only its syntax row is reproduced, so the base and lexical tokens keep v3's
  meaning.

Pairs, splits, labels and test sets of v3 are unchanged; the identity pairs are appended to `train`.

Outputs, in data/v5/:
    training_set_v5.zip                    v3's pairs plus the identity pairs
    {base,lexical,syntax}_test_set_v5.csv  identical to the v3 (and v1) test sets
    construction_v5.json                   shares, counts and the per-epoch composition

Run after scripts/build_dataset_v3.py: python scripts/build_dataset_v5.py
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
V3_DIR, OUT = ROOT / "data" / "v3", ROOT / "data" / "v5"
CLICIT = ROOT / "data" / "clicit2026" / "training_dataset_final_bin_10_no_overlap.zip"
DIMS = v1.TOKEN_DIMENSIONS
READIT = ("base", "lexical", "syntax", "all")
LEVELS = (60, 70, 80, 90)
SEED = 0


def same_text(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.astype(str).str.strip() == b.astype(str).str.strip()


def clicit_identity_share() -> dict:
    """Share of identical pairs among the CLiC-it training rows whose syntax token is L.
    Its rows repeat each pair once per token, and its syntax token is the simplification's level."""
    c = pd.read_csv(CLICIT, usecols=["original_text", "simplification", "simplification_syntax"], low_memory=False)
    same, out = same_text(c.original_text, c.simplification), v1.level(c.simplification_syntax)
    return {L: {"identical": int((same & (out == L)).sum() // 3), "tokens": int((out == L).sum() // 3),
                "share": float((same & (out == L)).sum() / (out == L).sum())} for L in LEVELS}


def load_v3() -> pd.DataFrame:
    pairs = pd.read_csv(V3_DIR / "training_set_v3.zip", dtype={f"tgt_{d}": str for d in DIMS}, low_memory=False)
    assert not same_text(pairs.original_text, pairs.simplification).any(), "v3 has no identity pairs"
    assert (v1.level(pairs.original_syntax) == pairs.src_syntax.astype(int)).all()
    return pairs


def pick_originals(pairs: pd.DataFrame, counts: dict) -> tuple[pd.DataFrame, dict]:
    """counts[L] training originals at syntax level L, drawn at random (one row each, as the template)."""
    train = pairs.split == "train"
    originals = pairs[train].drop_duplicates("original_sentence_idx")
    picked, candidates_by_level = [], {}
    for L in LEVELS:
        candidates = originals[v1.level(originals.original_syntax) == L]
        assert counts[L] <= len(candidates), (L, counts[L], len(candidates))
        picked.append(candidates.sample(n=counts[L], random_state=SEED + L))
        candidates_by_level[L] = int(len(candidates))
    return pd.concat(picked).copy(), candidates_by_level


def identity_rows(pairs: pd.DataFrame, ident: pd.DataFrame) -> pd.DataFrame:
    """Turn template rows into identity pairs: simplification = original, syntax target = its level, rest any."""
    ident["simplification"] = ident["original_text"]
    for d in READIT:
        ident[f"simplification_{d}"] = ident[f"original_{d}"]
    for c in [c for c in ident.columns if c.endswith("_simplification")]:
        o = c[: -len("_simplification")] + "_original"
        ident[c] = ident[o] if o in ident.columns else np.nan
    ident["tgt_base"], ident["tgt_lexical"] = "any", "any"
    ident["tgt_syntax"] = v1.level(ident.original_syntax).astype(str)
    ident["split"] = "train"
    ident["idx"] = np.arange(len(ident)) + int(pairs.idx.max()) + 1
    ident["ctrl_tokens"] = ["".join(f"<|readability_{d}_{t}|>" for d, t in zip(DIMS, row))
                            for row in ident[[f"tgt_{d}" for d in DIMS]].itertuples(index=False)]
    source = ["".join(f"<|original_readability_{d}_{lv}|>" for d, lv in zip(DIMS, row))
              for row in ident[[f"src_{d}" for d in DIMS]].itertuples(index=False)]
    ident["ctrl_tokens_with_source"] = [a + b for a, b in zip(source, ident["ctrl_tokens"])]
    return ident


def write(pairs: pd.DataFrame, ident: pd.DataFrame, out_dir: Path, name: str) -> pd.DataFrame:
    out = pd.concat([pairs, ident], ignore_index=True)
    assert same_text(out.original_text, out.simplification).sum() == len(ident)
    assert out.idx.is_unique
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"training_set_{name}.zip", index=False, compression=dict(method="zip", archive_name=f"training_set_{name}"))
    print(f"training_set_{name}.zip: {len(out):,} pairs ({len(ident):,} identity pairs added to train)", flush=True)
    for s in v1.TEST_STRATEGIES:
        shutil.copyfile(V3_DIR / f"{s}_test_set_v3.csv", out_dir / f"{s}_test_set_{name}.csv")
    return out


def main() -> None:
    pairs = load_v3()
    train = pairs.split == "train"
    share = clicit_identity_share()
    targets = {L: int((pairs.loc[train, "tgt_syntax"] == str(L)).sum()) for L in LEVELS}
    counts = {L: int(round(share[L]["share"] / (1 - share[L]["share"]) * targets[L])) for L in LEVELS}
    ident, candidates = pick_originals(pairs, counts)
    plan = {}
    for L in LEVELS:
        f = share[L]["share"]
        plan[str(L)] = {"clicit_identical": share[L]["identical"], "clicit_syntax_tokens": share[L]["tokens"],
                        "clicit_share": round(f, 4), "v3_syntax_targets": targets[L], "identity_pairs": counts[L],
                        "candidate_originals": candidates[L]}
        print(f"syntax {L}: CLiC-it {share[L]['identical']:,} of {share[L]['tokens']:,} tokens identical ({100 * f:.1f}%); "
              f"v3 has {targets[L]:,} targets -> {counts[L]:,} identity pairs from {candidates[L]:,} originals", flush=True)
    ident = identity_rows(pairs, ident)
    out = write(pairs, ident, OUT, "v5")

    t3, t5 = pairs[train], out[out.split == "train"]
    after = {str(L): round(float(len(ident[ident.tgt_syntax == str(L)]) / (t5.tgt_syntax == str(L)).sum()), 4) for L in LEVELS}
    (OUT / "construction_v5.json").write_text(json.dumps({
        "based_on": "data/v3/training_set_v3.zip (same pairs, splits, labels and columns)",
        "rules": {"identity_pairs": "for syntax levels 60-90: original -> original, from distinct training originals at that "
                                    "syntax level, as many as give the CLiC-it data's share of identical pairs among syntax tokens",
                  "identity_targets": "syntax = the original's level; base = any; lexical = any",
                  "seed": SEED, "levels": list(LEVELS)},
        "pairs": int(len(out)), "train": int((out.split == "train").sum()), "validation": int((out.split == "validation").sum()),
        "identity_pairs": int(len(ident)), "identity_idx_from": int(ident.idx.min()),
        "by_level": plan, "identity_share_of_syntax_targets_after": after,
        "train_syntax_targets_by_level": {"v3": by_level(t3.tgt_syntax), "v5": by_level(t5.tgt_syntax)},
        "epoch_composition_with_dropout": {"v3": epoch_composition(t3), "v5": epoch_composition(t5)},
        "test_sets": {s: f"identical to data/v3/{s}_test_set_v3.csv" for s in v1.TEST_STRATEGIES},
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
