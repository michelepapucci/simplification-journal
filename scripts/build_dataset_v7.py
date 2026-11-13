"""Build dataset v7: dataset v3 in the CLiC-it 2026 format (one token per example), plus one combined
example per pair, plus identity pairs under all three tokens.

v5 and v6 added identity pairs with a syntax target only, and the model's copying did not follow the
doses we computed (data/v1/experiments/README.md, "Dataset v6"). The CLiC-it data differs in format:
it shows every pair once per token (base, lexical, syntax), one token at a time, and its identity pairs
under all three. Our pairs carry up to three targets at once, thinned by target dropout. Per epoch
the CLiC-it model sees a syntax token at 60-90 in 11,069 examples, ours in 4,636.

v7 trains on v3's pairs in the CLiC-it format, keeping our multi-token control:
- each training pair gives one example per set target, with that token alone (the others "any");
- a pair with two or three targets also gives one combined example with all of them;
- target dropout is off (train.py --target-dropout 0): every example is shown as written;
- identity pairs (original -> original) get syntax and base targets at the original's own levels and
  lexical "any", and are expanded the same way (base alone, syntax alone, combined). The CLiC-it
  data's lexical token is a copy of its base token, so it has no lexical share to match, and lexical
  is left as in v3;
- per syntax level L in 60-90, the identity pairs give the CLiC-it data's share of identical pairs
  among all training examples that show the syntax token L (5.0% / 18.6% / 33.5% / 50%), each from a
  different training original at that level, drawn at random (seed 0);
- per base level B, at most as many identity pairs keep their base target as give the CLiC-it share
  among the examples showing base token B (identity originals drawn by syntax level sit high on base,
  so this thins base 80 and 90). A pair without its base target shows syntax once, not twice, and
  further identity pairs (syntax target only) restore the syntax share.
The validation split is v3's, one example per pair with all its targets, so validation losses stay
comparable with v3 to v6.

Outputs, in data/v7/:
    training_set_v7.zip                    one row per training example; `idx` is the pair, `example`
                                           the token shown alone or "combined", `pair_row` marks one
                                           row per pair with all its targets (the pair-level view)
    {base,lexical,syntax}_test_set_v7.csv  identical to the v3 (and v1) test sets
    construction_v7.json                   shares, counts and the composition of an epoch

Run after scripts/build_dataset_v3.py: python scripts/build_dataset_v7.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset_v5 as v5  # noqa: E402  identity pairs, CLiC-it shares

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "v7"
DIMS, LEVELS, TGT = v5.DIMS, v5.LEVELS, [f"tgt_{d}" for d in v5.DIMS]


def n_set(frame: pd.DataFrame) -> pd.Series:
    return (frame[TGT] != "any").sum(axis=1)


def set_control_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame["ctrl_tokens"] = ["".join(f"<|readability_{d}_{t}|>" for d, t in zip(DIMS, row))
                            for row in frame[TGT].itertuples(index=False)]
    source = ["".join(f"<|original_readability_{d}_{lv}|>" for d, lv in zip(DIMS, row))
              for row in frame[[f"src_{d}" for d in DIMS]].itertuples(index=False)]
    frame["ctrl_tokens_with_source"] = [a + b for a, b in zip(source, frame["ctrl_tokens"])]
    return frame


def expand(pairs: pd.DataFrame) -> pd.DataFrame:
    """Training pairs -> one example per set target (that token alone) + one combined example if two or
    three targets are set. Validation pairs stay one example with all their targets."""
    train, valid = pairs[pairs.split == "train"], pairs[pairs.split == "validation"].copy()
    k = n_set(train)
    parts = []
    for d in DIMS:
        s = train[train[f"tgt_{d}"] != "any"].copy()
        for o in DIMS:
            if o != d:
                s[f"tgt_{o}"] = "any"
        s["example"], s["pair_row"] = d, (k[s.index] == 1).values
        parts.append(s)
    comb = train[k >= 2].copy()
    comb["example"], comb["pair_row"] = "combined", True
    valid["example"], valid["pair_row"] = "pair", True
    out = pd.concat(parts + [comb, valid], ignore_index=True)
    order = {"base": 0, "lexical": 1, "syntax": 2, "combined": 3, "pair": 4}
    out = out.assign(_o=out.example.map(order)).sort_values(["idx", "_o"]).drop(columns="_o").reset_index(drop=True)
    out.insert(0, "row", range(len(out)))
    return set_control_columns(out)


def shows(examples: pd.DataFrame, d: str, level: int) -> pd.Series:
    t = examples[examples.split == "train"]
    return t[f"tgt_{d}"] == str(level)


def clicit_base_share() -> dict:
    """Share of identical pairs among the CLiC-it rows whose base token is L (its base token is the
    simplification's base level)."""
    c = pd.read_csv(v5.CLICIT, usecols=["original_text", "simplification", "simplification_base"], low_memory=False)
    same, lv = v5.same_text(c.original_text, c.simplification), v5.v1.level(c.simplification_base)
    return {L: round(float((same & (lv == L)).sum() / max(1, (lv == L).sum())), 4) for L in range(0, 100, 10)}


def main() -> None:
    pairs = v5.load_v3()
    share, base_share = v5.clicit_identity_share(), clicit_base_share()
    base_ex = expand(pairs)
    # identity examples needed, per level, among the training examples that show that token
    syn_exp = {L: int(shows(base_ex, "syntax", L).sum()) for L in LEVELS}
    syn_need = {L: int(round(share[L]["share"] / (1 - share[L]["share"]) * syn_exp[L])) for L in LEVELS}
    base_exp = {B: int(shows(base_ex, "base", B).sum()) for B in range(0, 100, 10)}
    base_need = {B: int(round(base_share[B] / (1 - base_share[B]) * base_exp[B])) for B in range(0, 100, 10)}

    # a pool of originals per syntax level, in sampling order: the first ones are used first
    pool, candidates = v5.pick_originals(pairs, {L: min(2 * syn_need[L], n) for L, n in
                                                 v5.pick_originals(pairs, {L: 0 for L in LEVELS})[1].items()})
    pool = v5.identity_rows(pairs, pool).copy()
    pool["tgt_base"] = pool.src_base.astype(int).astype(str)
    pool["tgt_lexical"] = "any"
    pool["_s"] = pool.src_syntax.astype(int)
    first = pd.concat([g.head(int(round(syn_need[L] / 2))) for L, g in pool.groupby("_s", sort=True)])
    # base: keep the base target on at most base_need[B] / 2 pairs per base level (each shows base twice)
    rng = np.random.RandomState(v5.SEED)
    for B, g in first.groupby(first.src_base.astype(int)):
        keep = min(len(g), base_need[B] // 2)
        drop = g.index[rng.permutation(len(g))[keep:]]
        first.loc[drop, "tgt_base"] = "any"
    # syntax: a pair with base shows syntax twice (alone and combined), one without it once
    extra = []
    for L in LEVELS:
        f = first[first._s == L]
        shown = 2 * int((f.tgt_base != "any").sum()) + int((f.tgt_base == "any").sum())
        more = max(0, syn_need[L] - shown)
        rest = pool[(pool._s == L) & ~pool.idx.isin(first.idx)].head(more).copy()
        assert len(rest) == more, (L, more, len(rest))
        rest["tgt_base"] = "any"
        extra.append(rest)
    ident = pd.concat([first] + extra).drop(columns="_s")
    ident["idx"] = np.arange(len(ident)) + int(pairs.idx.max()) + 1
    assert (ident.tgt_syntax.astype(int) == v5.v1.level(ident.original_syntax)).all()
    assert ident.original_sentence_idx.is_unique

    all_pairs = pd.concat([pairs, ident], ignore_index=True)
    assert all_pairs.idx.is_unique
    out = expand(all_pairs)

    # checks: examples per pair, identity examples, validation unchanged
    tr_pairs = all_pairs[all_pairs.split == "train"]
    k = n_set(tr_pairs)
    train = out[out.split == "train"]
    assert len(train) == int(k.sum() + (k >= 2).sum())
    assert (n_set(train[train.example != "combined"]) == 1).all()
    assert (n_set(out) >= 1).all()
    assert out.groupby("idx").pair_row.sum().eq(1).all()
    assert (out[out.split == "validation"][TGT].values == pairs[pairs.split == "validation"][TGT].values).all()
    same = v5.same_text(out.original_text, out.simplification)
    with_base = int((ident.tgt_base != "any").sum())
    assert same.sum() == 3 * with_base + (len(ident) - with_base)

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "training_set_v7.zip", index=False, compression=dict(method="zip", archive_name="training_set_v7"))
    for s in v5.v1.TEST_STRATEGIES:
        shutil.copyfile(v5.V3_DIR / f"{s}_test_set_v3.csv", OUT / f"{s}_test_set_v7.csv")

    same_tr = same[out.split == "train"]
    def share_of(d, L):
        m = shows(out, d, L)
        return int(m.sum()), round(float((m & same_tr).sum() / m.sum()), 4) if m.sum() else None
    by_syntax, by_base, by_lexical = {}, {}, {}
    for L in LEVELS:
        n_show, sh = share_of("syntax", L)
        g = ident[ident.tgt_syntax == str(L)]
        by_syntax[str(L)] = {"clicit_share": round(share[L]["share"], 4), "v3_examples_showing_token": syn_exp[L],
                             "identity_pairs": int(len(g)), "of_which_with_base_target": int((g.tgt_base != "any").sum()),
                             "candidate_originals": candidates[L], "examples_showing_token": n_show, "identity_share": sh}
        print(f"syntax {L}: CLiC-it {100 * share[L]['share']:.1f}%; {len(g)} identity pairs ({int((g.tgt_base != 'any').sum())} with base) "
              f"-> {100 * sh:.1f}% of {n_show:,} examples", flush=True)
    for B in range(0, 100, 10):
        n_show, sh = share_of("base", B)
        if n_show:
            by_base[str(B)] = {"clicit_share": base_share[B], "examples_showing_token": n_show, "identity_share": sh}
    for X in range(0, 100, 10):
        n_show, sh = share_of("lexical", X)
        if n_show:
            by_lexical[str(X)] = {"examples_showing_token": n_show, "identity_share": sh}
    print("base identity share, CLiC-it vs v7:", {B: (v["clicit_share"], v["identity_share"]) for B, v in by_base.items()}, flush=True)
    comp = train.example.value_counts().to_dict()
    print(f"training_set_v7.zip: {len(out):,} rows ({len(train):,} training examples from {len(tr_pairs):,} pairs, "
          f"{len(ident)} identity pairs, {with_base} with a base target; {int((out.split == 'validation').sum())} validation pairs); {comp}", flush=True)

    (OUT / "construction_v7.json").write_text(json.dumps({
        "based_on": "data/v3/training_set_v3.zip (same pairs, splits and labels), in the CLiC-it format plus a combined example",
        "rules": {"examples": "per training pair, one example per set target with that token alone, plus one combined example "
                              "with all targets if two or three are set; validation pairs unchanged (one example each)",
                  "target_dropout": 0.0,
                  "identity_pairs": "original -> original from distinct training originals at syntax 60-90; syntax and base targets "
                                    "at the original's levels, lexical any; as many as give the CLiC-it share of identical pairs among "
                                    "training examples showing syntax token L; base target kept on at most as many as give the CLiC-it "
                                    "share among examples showing base token B, the others syntax only",
                  "seed": v5.SEED, "levels": list(LEVELS)},
        "rows": int(len(out)), "training_examples": int(len(train)), "training_pairs": int(len(tr_pairs)),
        "validation_pairs": int((out.split == "validation").sum()), "identity_pairs": int(len(ident)),
        "identity_pairs_with_base_target": with_base, "identity_idx_from": int(ident.idx.min()),
        "examples_by_kind": {k_: int(v_) for k_, v_ in comp.items()},
        "steps_for_3_epochs_at_batch_64": int(-(-3 * len(train) // 64)),
        "by_syntax_level": by_syntax, "by_base_level": by_base, "by_lexical_level": by_lexical,
        "test_sets": {s: f"identical to data/v3/{s}_test_set_v3.csv" for s in v5.v1.TEST_STRATEGIES},
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
