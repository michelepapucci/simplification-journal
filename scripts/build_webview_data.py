"""Compute the statistics behind part 1 of the explorer (the gh-pages branch): dataset g1.

Re-runs the analyses of Papucci, Venturi & Dell'Orletta (CLiC-it 2025), "Generating and Evaluating
Multi-Level Text Simplification: A Case Study on Italian", on dataset g1 (data/g1/):

- READ-IT distributions (Gaussian KDE) of original and simplified sentences, including the most,
  randomly selected and least simplified version of each original (Figure 1 of the 2025 paper);
- a two-group MANOVA, original vs simplified, reported as Pillai's trace (Table 4);
- per feature: mean (sd) on both sides, Wilcoxon signed-rank test, and matched-pairs rank-biserial
  r computed on original minus simplified, so r > 0 means the feature is higher in the original
  (Tables 5-8);
- per feature: Spearman correlation between its original-minus-simplified difference and the
  READ-IT difference (Figure 2).

Slices: training pairs and test references, by the original's level and by target level. In every slice an
original contributes one simplification, picked at random with a fixed seed, as in the 2025 analysis, so the
pairs entering each test are independent.

Usage: python scripts/build_webview_data.py --out <gh-pages checkout>
Writes <out>/data_g1.js; index.html loads it. Needs numpy, pandas and scipy. The same script for datasets v0 to v7
is on the multidim-archive branch.
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


def dataset_g1() -> dict:
    train_path, refs_path = DATA / "g1" / "training_set_g1.zip", DATA / "g1" / "test_references_g1.csv"
    train = pd.read_csv(train_path).assign(level=lambda d: d.tgt_level)
    refs = pd.read_csv(refs_path)
    by_source = lambda f: {"all": f, **{f"from{lv}": f[f.src_level == lv] for lv in (70, 80, 90) if (f.src_level == lv).any()}}
    meta = {
        "id": "g1", "label": "g1", "title": "g1 · one token on READ-IT global",
        "description": f"{len(train):,} training pairs from {train.original_sentence_idx.nunique():,} originals with READ-IT global > 70, "
                       "each labelled with the READ-IT global level of its simplification, one or more levels below the original’s. "
                       f"Test references: the {len(refs):,} IMPaCTS simplifications of the 1,000 test originals that pass the same rules.",
        "splits": {"train": "Training", "test": "Test references"},
        "strategies": {"all": "All pairs", "from70": "Originals at 70", "from80": "Originals at 80", "from90": "Originals at 90"},
        "strategy_label": "Original’s level", "level_label": "target", "overall_key": "train|all|all",
        "counts_note": "Each pair appears once, at the level of its simplification’s READ-IT global (its control token). "
                       "Training includes the 1% validation split. Test references: simplifications of the test originals, which never appear in training.",
        "data_note": "dataset g1 (data/g1/), built by scripts/build_dataset_g1.py from the Wikipedia part of IMPaCTS (mpapucci/impacts). "
                     "See data/g1/README.md for the construction.",
        "fig1_note": "training set",
    }
    return payload(meta, {"train": (True, by_source(train)), "test": (True, by_source(refs))}, train, [train_path, refs_path])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="directory that receives data_g1.js")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args.out.mkdir(parents=True, exist_ok=True)
    data = dataset_g1()
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path = args.out / "data_g1.js"
    path.write_text(f"(window.WEBVIEW_DATASETS = window.WEBVIEW_DATASETS || {{}})[\"g1\"] = {body};\n", encoding="utf-8")
    print(f"wrote {path} ({len(body) / 1e6:.1f} MB, {len(data['strata'])} strata, {len(data['features'])} features)")


if __name__ == "__main__":
    main()
