"""Compute the statistics behind the data explorer (the gh-pages branch), one payload per dataset.

Re-runs the analyses of Papucci, Venturi & Dell'Orletta (CLiC-it 2025), "Generating and Evaluating
Multi-Level Text Simplification: A Case Study on Italian":

- READ-IT distributions (Gaussian KDE) of original and simplified sentences, including the most,
  randomly selected and least simplified version of each original (Figure 1 of the 2025 paper);
- a two-group MANOVA, original vs simplified, reported as Pillai's trace (Table 4);
- per feature: mean (sd) on both sides, Wilcoxon signed-rank test, and matched-pairs rank-biserial
  r computed on original minus simplified, so r > 0 means the feature is higher in the original
  (Tables 5-8);
- per feature: Spearman correlation between its original-minus-simplified difference and the
  READ-IT difference (Figure 2).

Datasets:
- clicit2025: the Wikipedia sample analysed in the 2025 paper (2,000 originals), read from the old
  repository (data/sampled_final_dataset/simdataset_wikipedia_aligned_2000.csv);
- v0: data/v0/ (the CLiC-it 2026 / GFFF selection, fixed), by split, strategy and control token;
- v1: data/v1/ (three target tokens), by split, targeted dimension and target level; a pair that
  targets several dimensions appears in each of their slices;
- v2: data/v2/ (v1 plus a lexical supplement, language-checked), sliced like v1, plus the lexical
  supplement on its own, by lexical target level;
- v3: data/v3/ (v2 with same-level base and syntax targets), sliced like v1, plus the relabelled
  pairs on their own (base and syntax, at the original's level);
- v4: data/v4/ (v3 with syntax targets on every pair whose syntax did not rise), sliced like v1, plus
  the relabelled pairs on their own (syntax crossings into a lower level within the noise margin);
- v5: data/v5/ (v3 plus identity pairs at syntax 60-90, at the CLiC-it 2026 data's share), sliced like v1;
- v6: data/v6/ (as v5, with the identity pairs dosed for syntax-only examples), sliced like v1.
- v7: data/v7/ (v3 in the CLiC-it format plus combined examples and identity pairs), one row per pair, sliced like v1.
In every slice an original contributes one simplification, picked at random with a fixed seed, as
in the 2025 analysis, so the pairs entering each test are independent.

Usage: python scripts/build_webview_data.py --out <gh-pages checkout> [--clicit2025 PATH] [--datasets clicit2025 v0 v1 v2 v3 v4 v5 v6 v7]
Writes <out>/data_<dataset>.js for each dataset; index.html loads them. Needs numpy, pandas and scipy.
"""

import argparse
import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 42
STRATEGIES = ("base", "lexical", "syntax")
READIT_DIMS = ("all", "base", "lexical", "syntax")  # "all" is READ-IT global
LEVELS = tuple(range(0, 100, 10))
KDE_GRID = np.arange(-10, 111, 1.0)  # READ-IT x 100; the KDE spills past 0 and 100, as in the 2025 figure
MIN_PAIRS = 10  # below this a stratum gets counts and examples only
N_EXAMPLES = 6
# Dropped before the MANOVA in the 2025 analysis, as near-duplicates of n_tokens and max_links_len.
# (The 2025 list also had n_sentences, which build_dataset_v0.py removes from the data.)
MANOVA_EXCLUDED = ("tokens_per_sent", "avg_max_links_len")

CATEGORIES = [  # (category, test on the feature name), first match wins; grouping of the 2025 paper, Table 3
    ("Raw text properties", lambda f: f in {"n_tokens", "tokens_per_sent", "char_per_tok"}),
    ("Lexical variety", lambda f: f.startswith("in_")),
    ("Morphosyntactic information", lambda f: f.startswith("upos_dist_") or f == "lexical_density"),
    ("Inflectional morphology", lambda f: f.startswith(("verbs_", "aux_"))),
    ("Verbal predicate structure", lambda f: f.startswith(("verbal_", "verb_edges_")) or f == "avg_verb_edges"),
    ("Dependency relations", lambda f: f.startswith("dep_dist_")),
    ("Order of elements", lambda f: f in {"subj_pre", "subj_post", "obj_pre", "obj_post"}),
    ("Use of subordination", lambda f: "subordinate" in f or f == "principal_proposition_dist"),
    ("Global and local parse-tree structures", lambda f: True),
]


def category(feature: str) -> str:
    return next(name for name, test in CATEGORIES if test(feature))


def sig(x, digits=4):
    """Round to significant digits for a compact JSON; NaN and inf become null."""
    if x is None or not np.isfinite(x):
        return None
    x = float(x)
    if x == 0:
        return 0.0
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


def sig_list(values, digits=4):
    return [sig(v, digits) for v in values]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def one_per_original(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs.sample(frac=1, random_state=SEED).drop_duplicates("original_sentence_idx")


def kde(values: np.ndarray):
    values = values[np.isfinite(values)]
    if len(values) < MIN_PAIRS or np.ptp(values) == 0:
        return None
    return sig_list(stats.gaussian_kde(values)(KDE_GRID), 3)


def readit_summary(values: np.ndarray) -> dict:
    return {"n": int(len(values)), "mean": sig(values.mean(), 3), "median": sig(np.median(values), 3),
            "sd": sig(values.std(ddof=1), 3) if len(values) > 1 else None}


def pillai_trace(orig: np.ndarray, simp: np.ndarray):
    """Two-group MANOVA with original and simplified sentences as the groups, as in the 2025 analysis.

    With two groups E^-1 H has one non-zero eigenvalue, so Pillai's trace and its F test are exact.
    Collinear features are removed by a rank-revealing QR on the standardized data. Under the null
    hypothesis Pillai's trace has expectation p / (N - 1); it is reported as `chance` because it
    depends on the stratum size.
    """
    X = np.vstack([orig, simp]).astype(float)
    X = X - X.mean(axis=0)
    sd = X.std(axis=0)
    X = X[:, sd > 1e-12] / sd[sd > 1e-12]
    _, R, piv = linalg.qr(X, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    X = X[:, np.sort(piv[: int((diag > 1e-6 * diag[0]).sum())])]
    n_o, n_s, (N, p) = len(orig), len(simp), X.shape
    if N - p - 1 < p:  # too few pairs for the number of features
        return None
    a, b = X[:n_o], X[n_o:]
    d = a.mean(axis=0) - b.mean(axis=0)
    E = (a - a.mean(axis=0)).T @ (a - a.mean(axis=0)) + (b - b.mean(axis=0)).T @ (b - b.mean(axis=0))
    try:
        lam = (n_o * n_s / N) * d @ np.linalg.solve(E, d)
    except np.linalg.LinAlgError:  # a feature constant within both groups but different between them
        return None
    F = lam * (N - p - 1) / p
    return {"V": sig(lam / (1 + lam), 3), "F": sig(F, 4), "df1": int(p), "df2": int(N - p - 1),
            "p": sig(stats.f.sf(F, p, N - p - 1), 2), "chance": sig(p / (N - 1), 3)}


def feature_stats(pairs: pd.DataFrame, features: list[str]) -> dict:
    O = pairs[[f"{f}_original" for f in features]].to_numpy(float)
    S = pairs[[f"{f}_simplification" for f in features]].to_numpy(float)
    D = O - S
    r_rb, p_w, n_diff = [], [], []
    for j in range(len(features)):
        d = D[:, j][D[:, j] != 0]
        n_diff.append(int(len(d)))  # r is +-1 whenever all differing pairs go one way, however few they are
        if len(d) == 0:
            r_rb.append(None)
            p_w.append(None)
            continue
        ranks = stats.rankdata(np.abs(d))
        w_plus, w_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
        r_rb.append(sig((w_plus - w_minus) / (w_plus + w_minus), 3))
        p_w.append(sig(stats.wilcoxon(O[:, j], S[:, j]).pvalue, 2))

    # Spearman = Pearson on ranks; p from the t approximation, as scipy.stats.spearmanr does.
    n = len(pairs)
    ranked = stats.rankdata(D, axis=0)
    ranked = (ranked - ranked.mean(axis=0)) / ranked.std(axis=0)
    rho, rho_p = {}, {}
    for dim in READIT_DIMS:
        target = stats.rankdata(pairs[f"original_{dim}"] - pairs[f"simplification_{dim}"])
        target = (target - target.mean()) / target.std()
        r = np.clip((ranked * target[:, None]).mean(axis=0), -1, 1)
        t = r * np.sqrt((n - 2) / np.maximum(1 - r**2, 1e-300))
        rho[dim] = sig_list(r, 3)
        rho_p[dim] = sig_list(2 * stats.t.sf(np.abs(t), n - 2), 2)

    keep = [i for i, f in enumerate(features) if f not in MANOVA_EXCLUDED]
    return {
        "mean_o": sig_list(O.mean(axis=0)), "sd_o": sig_list(O.std(axis=0, ddof=1)),
        "mean_s": sig_list(S.mean(axis=0)), "sd_s": sig_list(S.std(axis=0, ddof=1)),
        "r": r_rb, "p": p_w, "n_diff": n_diff, "rho": rho, "rho_p": rho_p,
        "manova": pillai_trace(O[:, keep], S[:, keep]),
    }


def examples(pairs: pd.DataFrame) -> list[dict]:
    sample = pairs.sample(n=min(N_EXAMPLES, len(pairs)), random_state=SEED)
    return [{"original": row.original_text, "simplification": row.simplification,
             "readit_o": {d: sig(100 * getattr(row, f"original_{d}"), 3) for d in READIT_DIMS},
             "readit_s": {d: sig(100 * getattr(row, f"simplification_{d}"), 3) for d in READIT_DIMS}}
            for row in sample.itertuples()]


def analyse_stratum(rows: pd.DataFrame, features: list[str], dedupe: bool) -> dict:
    pairs = one_per_original(rows) if dedupe else rows
    out = {"n_rows": int(len(rows)), "n_pairs": int(len(pairs)), "examples": examples(pairs)}
    if len(pairs) < MIN_PAIRS:
        return out
    out["kde"] = {d: {"original": kde(100 * pairs[f"original_{d}"].to_numpy()),
                      "simplified": kde(100 * pairs[f"simplification_{d}"].to_numpy())} for d in READIT_DIMS}
    out["summary"] = {d: {"original": readit_summary(100 * pairs[f"original_{d}"].to_numpy()),
                          "simplified": readit_summary(100 * pairs[f"simplification_{d}"].to_numpy())}
                      for d in READIT_DIMS}
    out["stats"] = feature_stats(pairs, features)
    return out


def figure1(pairs: pd.DataFrame, features: list[str]) -> dict:
    """Most, randomly selected and least simplified version of each original (2025 paper, Figure 1 and Table 4).

    As in the paper, versions are ranked by READ-IT global: most / least = lowest / highest score among the
    original's simplifications. The random one is drawn from the remaining simplifications, or from all of
    them when the original has fewer than three.
    """
    originals = pairs.drop_duplicates("original_sentence_idx")
    most = pairs.loc[pairs.groupby("original_sentence_idx")["simplification_all"].idxmin()]
    least = pairs.loc[pairs.groupby("original_sentence_idx")["simplification_all"].idxmax()]
    shuffled = pairs.sample(frac=1, random_state=SEED)
    random = (shuffled.assign(_extreme=shuffled["idx"].isin(set(most["idx"]) | set(least["idx"])))
              .sort_values("_extreme", kind="stable").drop_duplicates("original_sentence_idx"))
    versions = {"most": most, "random": random, "least": least}
    keep = [f for f in features if f not in MANOVA_EXCLUDED]
    per_original = pairs.groupby("original_sentence_idx").size().value_counts().sort_index()
    return {
        "n_originals": int(len(originals)), "n_pairs": int(len(pairs)),
        "per_original": {str(k): int(v) for k, v in per_original.items()},
        "kde": {"original": kde(100 * originals["original_all"].to_numpy()),
                **{k: kde(100 * v["simplification_all"].to_numpy()) for k, v in versions.items()}},
        "summary": {"original": readit_summary(100 * originals["original_all"].to_numpy()),
                    **{k: readit_summary(100 * v["simplification_all"].to_numpy()) for k, v in versions.items()}},
        "manova": {k: pillai_trace(v[[f"{f}_original" for f in keep]].to_numpy(float),
                                   v[[f"{f}_simplification" for f in keep]].to_numpy(float))
                   for k, v in versions.items()},
    }


def non_constant_features(frames: list[pd.DataFrame]) -> tuple[list[str], list[str]]:
    both = pd.concat(frames)
    all_features = [c[: -len("_simplification")] for c in both.columns if c.endswith("_simplification")
                    and not c.startswith("simplification_") and f"{c[: -len('_simplification')]}_original" in both.columns]
    constant = [f for f in all_features if both[f"{f}_original"].nunique() <= 1 and both[f"{f}_simplification"].nunique() <= 1]
    return [f for f in all_features if f not in constant], constant


def payload(meta: dict, splits: dict, figure_pairs: pd.DataFrame, sources: list[Path]) -> dict:
    """splits: {split: (dedupe, {strategy: DataFrame with a `level` column, NaN where there is none})}."""
    features, constant = non_constant_features([df for _, frames in splits.values() for df in frames.values()])
    strata, levels = {}, set()
    for split, (dedupe, frames) in splits.items():
        for strategy, rows in frames.items():
            lv = sorted(int(x) for x in rows["level"].dropna().unique())
            levels.update(lv)
            for level in ["all"] + lv:
                subset = rows if level == "all" else rows[rows["level"] == level]
                strata[f"{split}|{strategy}|{level}"] = analyse_stratum(subset, features, dedupe)
                print(f"{meta['id']:10s} {split:6s} {strategy:7s} {str(level):>3s}: {len(subset):6d} rows -> "
                      f"{strata[f'{split}|{strategy}|{level}']['n_pairs']:6d} pairs", flush=True)
    return {
        "meta": {**meta, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "seed": SEED,
                 "sources": {p.name: sha256(p)[:12] for p in sources}, "dropped_constant_features": constant,
                 "manova_excluded": list(MANOVA_EXCLUDED), "min_pairs": MIN_PAIRS},
        "features": [{"name": f, "category": category(f)} for f in features],
        "categories": [name for name, _ in CATEGORIES],
        "levels": sorted(levels),
        "readit_dims": list(READIT_DIMS),
        "kde_grid": KDE_GRID.tolist(),
        "strata": strata,
        "figure1": figure1(figure_pairs, features),
    }


def dataset_v0() -> dict:
    train_path = DATA / "v0" / "training_set_v0.zip"
    test_paths = {s: DATA / "v0" / f"{s}_test_set_v0.csv" for s in STRATEGIES}
    train = pd.read_csv(train_path, compression="zip")
    train["strategy"] = train["ctrl_token"].str.extract(r"<\|readability_(\w+?)_\d+\|>", expand=False)
    train["level"] = train["ctrl_int"]
    tests = {s: pd.read_csv(p).assign(level=lambda d: d["ctrl_int"]) for s, p in test_paths.items()}
    meta = {
        "id": "v0", "label": "v0", "title": "v0 · CLiC-it 2026 (GFFF) selection, fixed",
        "description": "The training data of the CLiC-it 2026 models, rebuilt from the IMPaCTS release with the lexical-token bug fixed: "
                       "55,877 pairs from originals with READ-IT global > 70, each repeated under one token per strategy.",
        "splits": {"train": "Training", "test": "Test"}, "strategies": {s: s.capitalize() for s in STRATEGIES},
        "strategy_label": "Strategy", "level_label": "CT", "overall_key": "train|base|all",
        "counts_note": "Each training pair appears once per strategy, with that strategy’s control token, so the three “all” training "
                       "slices hold the same 55,877 pairs.",
        "data_note": "dataset v0 (data/v0/), built by scripts/build_dataset_v0.py from the Wikipedia part of IMPaCTS (mpapucci/impacts): "
                     "originals with READ-IT global > 70, no READ-IT score higher after simplification, GFFF noise filters.",
        "fig1_note": "training set",
    }
    return payload(meta, {"train": (True, {s: train[train.strategy == s] for s in STRATEGIES}), "test": (False, tests)},
                   train[train.strategy == STRATEGIES[0]], [train_path, *test_paths.values()])


def dataset_v1() -> dict:
    train_path = DATA / "v1" / "training_set_v1.zip"
    test_paths = {s: DATA / "v1" / f"{s}_test_set_v1.csv" for s in STRATEGIES}
    tgt = {f"tgt_{s}": str for s in STRATEGIES}
    train = pd.read_csv(train_path, dtype=tgt)
    as_level = lambda col: pd.to_numeric(col.where(col != "any"), errors="coerce")
    frames = {"all": train.assign(level=np.nan)}
    frames.update({s: train[train[f"tgt_{s}"] != "any"].assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s in STRATEGIES})
    tests = {s: pd.read_csv(p, dtype=tgt).assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s, p in test_paths.items()}
    meta = {
        "id": "v1", "label": "v1", "title": "v1 · three target tokens",
        "description": f"{len(train):,} pairs from originals with READ-IT global > 70. Each pair is used once with three target tokens "
                       "(base, lexical, syntax); a dimension gets a target only if it drops a bin and by more than READ-IT’s noise, otherwise “any”.",
        "splits": {"train": "Training", "test": "Test"},
        "strategies": {"all": "All pairs", **{s: s.capitalize() for s in STRATEGIES}},
        "strategy_label": "Targeted dimension", "level_label": "target", "overall_key": "train|all|all",
        "counts_note": "Training: each pair appears in the slice of every dimension it targets, at that dimension’s target level; "
                       "“All pairs” holds every pair once. Test: the reference simplification’s target in the test set’s own dimension.",
        "data_note": "dataset v1 (data/v1/), built by scripts/build_dataset_v1.py from the Wikipedia part of IMPaCTS (mpapucci/impacts). "
                     "See data/v1/README.md for the construction.",
        "fig1_note": "training set",
    }
    return payload(meta, {"train": (True, frames), "test": (False, tests)}, train, [train_path, *test_paths.values()])


def dataset_v2() -> dict:
    train_path = DATA / "v2" / "training_set_v2.zip"
    test_paths = {s: DATA / "v2" / f"{s}_test_set_v2.csv" for s in STRATEGIES}
    tgt = {f"tgt_{s}": str for s in STRATEGIES}
    train = pd.read_csv(train_path, dtype=tgt)
    as_level = lambda col: pd.to_numeric(col.where(col != "any"), errors="coerce")
    frames = {"all": train.assign(level=np.nan)}
    frames.update({s: train[train[f"tgt_{s}"] != "any"].assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s in STRATEGIES})
    supplement = train[train.subset == "lexical_supplement"]
    frames["supplement"] = supplement.assign(level=as_level(supplement.tgt_lexical))
    tests = {s: pd.read_csv(p, dtype=tgt).assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s, p in test_paths.items()}
    meta = {
        "id": "v2", "label": "v2", "title": "v2 · v1 plus lexical support",
        "description": f"{len(train):,} pairs: v1’s pairs with a non-Italian side removed, plus {len(supplement):,} pairs from the rest of the release "
                       "whose lexical score dropped two levels or more (Italian, not near-copies, at least half the original’s tokens). "
                       "Same three target tokens and rule as v1; same test sets.",
        "splits": {"train": "Training", "test": "Test"},
        "strategies": {"all": "All pairs", **{s: s.capitalize() for s in STRATEGIES}, "supplement": "Lexical supplement"},
        "strategy_label": "Targeted dimension", "level_label": "target", "overall_key": "train|all|all",
        "counts_note": "Training: each pair appears in the slice of every dimension it targets, at that dimension’s target level; "
                       "“All pairs” holds every pair once; “Lexical supplement” holds only the added pairs, by lexical target. "
                       "Test: identical to v1.",
        "data_note": "dataset v2 (data/v2/), built by scripts/build_dataset_v2.py from the Wikipedia part of IMPaCTS (mpapucci/impacts). "
                     "See data/v2/README.md for the construction.",
        "fig1_note": "training set",
    }
    return payload(meta, {"train": (True, frames), "test": (False, tests)}, train, [train_path, *test_paths.values()])


def dataset_v3() -> dict:
    train_path = DATA / "v3" / "training_set_v3.zip"
    test_paths = {s: DATA / "v3" / f"{s}_test_set_v3.csv" for s in STRATEGIES}
    tgt = {f"tgt_{s}": str for s in STRATEGIES}
    train = pd.read_csv(train_path, dtype=tgt)
    as_level = lambda col: pd.to_numeric(col.where(col != "any"), errors="coerce")
    frames = {"all": train.assign(level=np.nan)}
    frames.update({s: train[train[f"tgt_{s}"] != "any"].assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s in STRATEGIES})
    same = {}
    for s in ("base", "syntax"):   # a target equal to the original's level exists only under the v3 rule
        rows = train[train[f"tgt_{s}"] == train[f"src_{s}"].astype(str)]
        same[s] = len(rows)
        frames[f"{s}_same"] = rows.assign(level=as_level(rows[f"tgt_{s}"]))
    tests = {s: pd.read_csv(p, dtype=tgt).assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s, p in test_paths.items()}
    meta = {
        "id": "v3", "label": "v3", "title": "v3 · v2 plus same-level targets",
        "description": f"v2’s {len(train):,} pairs with one rule change: when base or syntax stays in the original’s level, the target is that level "
                       f"instead of “any” ({same['base']:,} base and {same['syntax']:,} syntax targets), so the model sees light edits. "
                       "Lexical targets and test sets as in v2.",
        "splits": {"train": "Training", "test": "Test"},
        "strategies": {"all": "All pairs", **{s: s.capitalize() for s in STRATEGIES},
                       "base_same": "Base, same level", "syntax_same": "Syntax, same level"},
        "strategy_label": "Targeted dimension", "level_label": "target", "overall_key": "train|all|all",
        "counts_note": "Training: each pair appears in the slice of every dimension it targets, at that dimension’s target level; "
                       "“All pairs” holds every pair once; the two “same level” slices hold only the pairs relabelled by v3, at the original’s level. "
                       "Test: identical to v1 and v2.",
        "data_note": "dataset v3 (data/v3/), built by scripts/build_dataset_v3.py from dataset v2. See data/v3/README.md for the rule.",
        "fig1_note": "training set",
    }
    return payload(meta, {"train": (True, frames), "test": (False, tests)}, train, [train_path, *test_paths.values()])


def dataset_v4() -> dict:
    train_path = DATA / "v4" / "training_set_v4.zip"
    test_paths = {s: DATA / "v4" / f"{s}_test_set_v4.csv" for s in STRATEGIES}
    tgt = {f"tgt_{s}": str for s in STRATEGIES}
    train = pd.read_csv(train_path, dtype=tgt)
    margin = json.loads((DATA / "v4" / "construction_v4.json").read_text())["rules"]["margins"]["syntax"]
    as_level = lambda col: pd.to_numeric(col.where(col != "any"), errors="coerce")
    frames = {"all": train.assign(level=np.nan)}
    frames.update({s: train[train[f"tgt_{s}"] != "any"].assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s in STRATEGIES})
    # the pairs v4 relabels: syntax crossed into a lower level by less than the noise margin
    dropped = 100 * (train["original_syntax"] - train["simplification_syntax"])
    lower = as_level(train["tgt_syntax"]) < train["src_syntax"]
    cross = train[lower & (dropped < margin)]
    frames["syntax_cross"] = cross.assign(level=as_level(cross["tgt_syntax"]))
    tests = {s: pd.read_csv(p, dtype=tgt).assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s, p in test_paths.items()}
    meta = {
        "id": "v4", "label": "v4", "title": "v4 · v3 plus syntax crossings",
        "description": f"v3’s {len(train):,} pairs with one rule change: a simplification whose syntax falls into a lower level by less "
                       f"than the noise margin ({margin} points) gets that level as syntax target instead of “any” ({len(cross):,} pairs). "
                       "Syntax is then targeted in every pair except those where it rose. Base, lexical and test sets as in v3.",
        "splits": {"train": "Training", "test": "Test"},
        "strategies": {"all": "All pairs", **{s: s.capitalize() for s in STRATEGIES}, "syntax_cross": "Syntax, crossed within the margin"},
        "strategy_label": "Targeted dimension", "level_label": "target", "overall_key": "train|all|all",
        "counts_note": "Training: each pair appears in the slice of every dimension it targets, at that dimension’s target level; "
                       "“All pairs” holds every pair once; “Syntax, crossed within the margin” holds only the pairs relabelled by v4. "
                       "Test: identical to v1–v3.",
        "data_note": "dataset v4 (data/v4/), built by scripts/build_dataset_v4.py from dataset v3. See data/v4/README.md for the rule.",
        "fig1_note": "training set",
    }
    return payload(meta, {"train": (True, frames), "test": (False, tests)}, train, [train_path, *test_paths.values()])


def dataset_v5() -> dict:
    train_path = DATA / "v5" / "training_set_v5.zip"
    test_paths = {s: DATA / "v5" / f"{s}_test_set_v5.csv" for s in STRATEGIES}
    tgt = {f"tgt_{s}": str for s in STRATEGIES}
    train = pd.read_csv(train_path, dtype=tgt)
    n_identity = json.loads((DATA / "v5" / "construction_v5.json").read_text())["identity_pairs"]
    as_level = lambda col: pd.to_numeric(col.where(col != "any"), errors="coerce")
    frames = {"all": train.assign(level=np.nan)}
    frames.update({s: train[train[f"tgt_{s}"] != "any"].assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s in STRATEGIES})
    # no slice of the identity pairs alone: every feature difference in it is zero
    tests = {s: pd.read_csv(p, dtype=tgt).assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s, p in test_paths.items()}
    meta = {
        "id": "v5", "label": "v5", "title": "v5 · v3 plus identity pairs",
        "description": f"v3’s pairs plus {n_identity:,} identity pairs: training originals at syntax levels 60–90 given as their own "
                       "simplification, with a syntax target at the original’s level (base and lexical “any”). Their number gives each "
                       "of these syntax levels the CLiC-it 2026 data’s share of identical pairs (5% at 60, 19% at 70, 33% at 80, 50% at 90). "
                       "Everything else as in v3.",
        "splits": {"train": "Training", "test": "Test"},
        "strategies": {"all": "All pairs", **{s: s.capitalize() for s in STRATEGIES}},
        "strategy_label": "Targeted dimension", "level_label": "target", "overall_key": "train|all|all",
        "counts_note": "Training: each pair appears in the slice of every dimension it targets, at that dimension’s target level; "
                       "“All pairs” holds every pair once. The identity pairs are in “All pairs” and “Syntax” (levels 60–90), where an "
                       "original that has one may contribute it as its simplification. Test: identical to v1–v4.",
        "data_note": "dataset v5 (data/v5/), built by scripts/build_dataset_v5.py from dataset v3. See data/v5/README.md for the rule.",
        "fig1_note": "training set",
    }
    return payload(meta, {"train": (True, frames), "test": (False, tests)}, train, [train_path, *test_paths.values()])


def dataset_v6() -> dict:
    train_path = DATA / "v6" / "training_set_v6.zip"
    test_paths = {s: DATA / "v6" / f"{s}_test_set_v6.csv" for s in STRATEGIES}
    tgt = {f"tgt_{s}": str for s in STRATEGIES}
    train = pd.read_csv(train_path, dtype=tgt)
    n_identity = json.loads((DATA / "v6" / "construction_v6.json").read_text())["identity_pairs"]
    as_level = lambda col: pd.to_numeric(col.where(col != "any"), errors="coerce")
    frames = {"all": train.assign(level=np.nan)}
    frames.update({s: train[train[f"tgt_{s}"] != "any"].assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s in STRATEGIES})
    tests = {s: pd.read_csv(p, dtype=tgt).assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s, p in test_paths.items()}
    meta = {
        "id": "v6", "label": "v6", "title": "v6 · v3 plus identity pairs, dosed for syntax-only requests",
        "description": f"v3’s pairs plus {n_identity:,} identity pairs at syntax levels 60–90, as in v5 but fewer: their number gives the "
                       "CLiC-it 2026 data’s share of identical pairs among the training examples that show the syntax token alone, the format "
                       "of the syntax evaluation (5% at 60, 19% at 70, 33% at 80, 50% at 90). Everything else as in v3.",
        "splits": {"train": "Training", "test": "Test"},
        "strategies": {"all": "All pairs", **{s: s.capitalize() for s in STRATEGIES}},
        "strategy_label": "Targeted dimension", "level_label": "target", "overall_key": "train|all|all",
        "counts_note": "Training: each pair appears in the slice of every dimension it targets, at that dimension’s target level; "
                       "“All pairs” holds every pair once. The identity pairs are in “All pairs” and “Syntax” (levels 60–90). "
                       "Test: identical to v1–v5.",
        "data_note": "dataset v6 (data/v6/), built by scripts/build_dataset_v6.py from dataset v3. See data/v6/README.md for the rule.",
        "fig1_note": "training set",
    }
    return payload(meta, {"train": (True, frames), "test": (False, tests)}, train, [train_path, *test_paths.values()])


def dataset_v7() -> dict:
    train_path = DATA / "v7" / "training_set_v7.zip"
    test_paths = {s: DATA / "v7" / f"{s}_test_set_v7.csv" for s in STRATEGIES}
    tgt = {f"tgt_{s}": str for s in STRATEGIES}
    rows = pd.read_csv(train_path, dtype=tgt)
    train = rows[rows.pair_row].drop(columns=["row", "example", "pair_row"]).reset_index(drop=True)  # one row per pair
    c = json.loads((DATA / "v7" / "construction_v7.json").read_text())
    as_level = lambda col: pd.to_numeric(col.where(col != "any"), errors="coerce")
    frames = {"all": train.assign(level=np.nan)}
    frames.update({s: train[train[f"tgt_{s}"] != "any"].assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s in STRATEGIES})
    tests = {s: pd.read_csv(p, dtype=tgt).assign(level=lambda d, s=s: as_level(d[f"tgt_{s}"])) for s, p in test_paths.items()}
    meta = {
        "id": "v7", "label": "v7", "title": "v7 · v3 in the CLiC-it format, plus combined examples and identity pairs",
        "description": f"v3’s pairs, trained the way the CLiC-it 2026 data is: one example per target with that token alone, plus one "
                       f"combined example per pair with all its targets ({c['training_examples']:,} training examples, no target dropout). "
                       f"{c['identity_pairs']:,} identity pairs with syntax and base targets at the original’s levels give the CLiC-it data’s "
                       "share of identical pairs among the examples that show each syntax token at 60–90 (5%, 19%, 33%, 50%).",
        "splits": {"train": "Training", "test": "Test"},
        "strategies": {"all": "All pairs", **{s: s.capitalize() for s in STRATEGIES}},
        "strategy_label": "Targeted dimension", "level_label": "target", "overall_key": "train|all|all",
        "counts_note": "Training: one row per pair, as in v1–v6 (the training file repeats each pair once per target and once combined). "
                       "Each pair appears in the slice of every dimension it targets, at that dimension’s target level; “All pairs” holds "
                       "every pair once. The identity pairs are in “All pairs”, “Syntax” (levels 60–90) and, most of them, “Base”. "
                       "Test: identical to v1–v6.",
        "data_note": "dataset v7 (data/v7/), built by scripts/build_dataset_v7.py from dataset v3. See data/v7/README.md for the rule.",
        "fig1_note": "training set",
    }
    return payload(meta, {"train": (True, frames), "test": (False, tests)}, train, [train_path, *test_paths.values()])


CLICIT2025_SAMPLE = ROOT.parent / "data" / "sampled_final_dataset" / "simdataset_wikipedia_aligned_2000.csv"


def load_clicit2025_sample(path: Path = CLICIT2025_SAMPLE) -> pd.DataFrame:
    """The 2025 Wikipedia sample, with this repository's column names (original_* / simplification_*)."""
    df = pd.read_csv(path)
    rename = {f"og_{d}": f"original_{d}" for d in READIT_DIMS} | {d: f"simplification_{d}" for d in READIT_DIMS}
    rename |= {c: c[: -len("_og")] + "_original" for c in df.columns if c.endswith("_og")}
    rename |= {c: c[: -len("_simp")] + "_simplification" for c in df.columns if c.endswith("_simp")}
    df = df.rename(columns=rename).drop(columns=["n_sentences_original", "n_sentences_simplification"], errors="ignore")
    return df.loc[:, ~df.columns.str.endswith("_diff")]


def dataset_clicit2025(path: Path) -> dict:
    df = load_clicit2025_sample(path).assign(level=np.nan)
    meta = {
        "id": "clicit2025", "label": "CLiC-it 2025", "title": "CLiC-it 2025 · Wikipedia sample",
        "description": f"The Wikipedia sample analysed in the CLiC-it 2025 paper: {df.original_sentence_idx.nunique():,} originals with "
                       f"{len(df):,} simplifications, before any READ-IT selection and without control tokens.",
        "splits": {"sample": "2025 sample"}, "strategies": {"all": "All pairs"},
        "strategy_label": "Subset", "level_label": "CT", "overall_key": "sample|all|all",
        "counts_note": "No control tokens: the 2025 sample is analysed as one set.",
        "data_note": f"{path.name}, the sample of the 2025 paper, from the old repository (data/sampled_final_dataset/).",
        "fig1_note": "2025 sample",
    }
    return payload(meta, {"sample": (True, {"all": df})}, df, [path])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="directory that receives the data_*.js files")
    parser.add_argument("--clicit2025", type=Path, default=CLICIT2025_SAMPLE,
                        help="the 2025 Wikipedia sample (default: in the parent text-simplification-experiments repository)")
    parser.add_argument("--datasets", nargs="+", default=["clicit2025", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"], choices=["clicit2025", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"])
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    builders = {"clicit2025": lambda: dataset_clicit2025(args.clicit2025), "v0": dataset_v0, "v1": dataset_v1, "v2": dataset_v2, "v3": dataset_v3, "v4": dataset_v4, "v5": dataset_v5, "v6": dataset_v6, "v7": dataset_v7}
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        data = builders[name]()
        body = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        path = args.out / f"data_{name}.js"
        path.write_text(f"(window.WEBVIEW_DATASETS = window.WEBVIEW_DATASETS || {{}})[{json.dumps(name)}] = {body};\n", encoding="utf-8")
        print(f"wrote {path} ({len(body) / 1e6:.1f} MB, {len(data['strata'])} strata, {len(data['features'])} features)")


if __name__ == "__main__":
    main()
