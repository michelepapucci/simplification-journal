"""Analyses of dataset v1 (data/v1/), in the style of the data explorer (scripts/build_webview_data.py).

- size, sources and target-token coverage, compared with dataset v0;
- which dimensions each pair targets, and what "any" stands for;
- how the CLiC-it 2026 test requests (CT 10-70 per test original) sit against v1's training rule;
- READ-IT distributions (Gaussian KDE) of original and simplified sentences, overall and for the
  pairs that lower a single dimension;
- per stratum, the 2025 analyses on Profiling-UD features: Wilcoxon signed-rank and matched-pairs
  rank-biserial r on original minus simplified (r > 0: higher in the original), two-group MANOVA
  (Pillai's trace), Spearman of feature differences with READ-IT differences.
As in the explorer, each stratum keeps one simplification per original (seed 42).

Usage: python scripts/analyze_dataset_v1.py
Writes data/v1/analysis_v1.json and data/v1/figures/*.png. Needs numpy, pandas, scipy and matplotlib.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from build_webview_data import (KDE_GRID, MANOVA_EXCLUDED, READIT_DIMS, category, feature_stats, figure1,
                                load_clicit2025_sample, one_per_original, pillai_trace, sig)

ROOT = Path(__file__).resolve().parent.parent
V0, V1 = ROOT / "data" / "v0", ROOT / "data" / "v1"
DIMS = ("base", "lexical", "syntax")
SHORT = {"base": "B", "lexical": "L", "syntax": "S"}
LEVELS = tuple(range(0, 100, 10))
EVAL_LEVELS = tuple(range(10, 80, 10))
SEED = 42
KEY_FEATURES = ["n_tokens", "char_per_tok", "in_dict", "in_FO", "lexical_density", "avg_max_depth", "avg_links_len",
                "max_links_len", "avg_token_per_clause", "verbal_head_per_sent", "subordinate_proposition_dist",
                "n_prepositional_chains", "avg_prepositional_chain_len"]

THEMES = {  # dataviz reference palette: categorical slots 1-2, chart chrome
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "original": "#2a78d6", "simplified": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "original": "#3987e5", "simplified": "#d95926"},
}


def targeted(df: pd.DataFrame, d: str) -> pd.Series:
    return df[f"tgt_{d}"] != "any"


def pattern(df: pd.DataFrame) -> pd.Series:
    return pd.Series(["+".join(SHORT[d] for d, t in zip(DIMS, row) if t) for row in
                      zip(*(targeted(df, d) for d in DIMS))], index=df.index)


def overview(v1: pd.DataFrame, v0: pd.DataFrame) -> dict:
    first = v1.drop_duplicates("original_sentence_idx")
    first0 = v0.drop_duplicates("original_sentence_idx")
    return {
        "pairs": int(len(v1)), "originals": int(len(first)),
        "per_split": {s: {"pairs": int(len(g)), "originals": int(g.original_sentence_idx.nunique())} for s, g in v1.groupby("split")},
        "pairs_per_original_mean": round(len(v1) / len(first), 2),
        "source_global_quartiles": {"v1": [round(float(q), 1) for q in (100 * first.original_all).quantile([.25, .5, .75])],
                                    "v0": [round(float(q), 1) for q in (100 * first0.original_all).quantile([.25, .5, .75])]},
        "original_chars_median": {"v1": int(first.original_text.str.len().median()), "v0": int(first0.original_text.str.len().median())},
        "v0_pairs": int(len(v0)), "v0_originals": int(len(first0)),
    }


def patterns(v1: pd.DataFrame) -> dict:
    p = pattern(v1).value_counts()
    states = {}
    for d in DIMS:
        lower = targeted(v1, d)
        out = (np.minimum(np.floor(v1[f"simplification_{d}"] * 10), 9) * 10).astype(int)
        states[d] = {"lower": round(float(lower.mean()), 3), "same_bin": round(float((out == v1[f"src_{d}"]).mean()), 3),
                     "higher_bin": round(float((out > v1[f"src_{d}"]).mean()), 3)}
    return {"patterns": {k: {"pairs": int(v), "share": round(v / len(v1), 3)} for k, v in p.items()},
            "per_dimension_state": states}


def per_dimension(v1: pd.DataFrame, v0: pd.DataFrame) -> dict:
    v0 = v0.assign(strategy=v0.ctrl_token.str.extract(r"<\|readability_(\w+?)_\d+\|>", expand=False))
    out = {}
    for d in DIMS:
        t = v1[targeted(v1, d)]
        tgt = t[f"tgt_{d}"].astype(int)
        drop = (t[f"src_{d}"] - tgt) // 10
        per_orig = tgt.groupby(t.original_sentence_idx).nunique()
        v0d = v0[v0.strategy == d]
        out[d] = {
            "targeted_pairs": int(len(t)), "targeted_share": round(len(t) / len(v1), 3),
            "target_level_counts": {lv: int((tgt == lv).sum()) for lv in LEVELS},
            "v0_level_counts": {lv: int((v0d.ctrl_int == lv).sum()) for lv in LEVELS},
            "source_level_counts": {lv: int((v1[f"src_{d}"] == lv).sum()) for lv in LEVELS},
            "bins_lowered": {"1": int((drop == 1).sum()), "2": int((drop == 2).sum()), "3": int((drop == 3).sum()),
                             "4+": int((drop >= 4).sum())},
            "target_levels_per_original_mean": round(float(per_orig.mean()), 2),
            "originals_with_one_target_level": round(float((per_orig == 1).mean()), 3),
            "v0_levels_per_original_mean": round(float(v0d.groupby("original_sentence_idx").ctrl_int.nunique().mean()), 2),
        }
    return out


def co_targeting(v1: pd.DataFrame) -> dict:
    t = {d: targeted(v1, d) for d in DIMS}
    alone = {d: t[d] & ~np.logical_or.reduce([t[e] for e in DIMS if e != d]) for d in DIMS}
    return {
        "p_row_targets_column_given_row": {d: {e: round(float(t[e][t[d]].mean()), 3) for e in DIMS} for d in DIMS},
        "single_target_share_of_targeted": {d: round(float(alone[d].sum() / t[d].sum()), 3) for d in DIMS},
    }


def test_requests(v1: pd.DataFrame) -> dict:
    """Each test original x CT 10-70 request, against v1's rule: a target is below the source level and at
    least the dimension's margin below the source score. Also how the test originals compare with v1's."""
    from build_dataset_v1 import double_filter_proxy
    margins = json.loads((V1 / "construction_v1.json").read_text())["rules"]["margins"]
    out = {"margins": margins}
    for s in DIMS:
        test = pd.read_csv(V1 / f"{s}_test_set_v1.csv", usecols=[f"src_{s}", f"original_{s}", "original_text"])
        src = np.repeat(test[f"src_{s}"].to_numpy(), len(EVAL_LEVELS))
        score = np.repeat(100 * test[f"original_{s}"].to_numpy(), len(EVAL_LEVELS))
        req = np.tile(EVAL_LEVELS, len(test))
        out[s] = {"in_training_rule": round(float(((req < src) & (score - margins[s] >= req)).mean()), 3),
                  "below_source_level": round(float((req < src).mean()), 3),
                  "at_source_level": round(float((req == src).mean()), 3),
                  "above_source_level": round(float((req > src).mean()), 3),
                  "originals_the_proxy_would_remove": round(float(double_filter_proxy(test.original_text).mean()), 3),
                  "originals_longer_than_400_chars": round(float((test.original_text.str.len() > 400).mean()), 3),
                  "original_chars_median": int(test.original_text.str.len().median()),
                  "source_level_counts": {int(k): int(v) for k, v in test[f"src_{s}"].value_counts().sort_index().items()}}
    return out


def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def small_edits(v1: pd.DataFrame) -> dict:
    """How many targets come from tiny score drops (a bin boundary crossed by a few points) or near-copies."""
    from multiprocessing import Pool
    with Pool(16) as pool:
        sim = np.array(pool.starmap(_similarity, zip(v1.original_text, v1.simplification), chunksize=2000))
    drops = pd.DataFrame({d: np.where(targeted(v1, d), 100 * (v1[f"original_{d}"] - v1[f"simplification_{d}"]), np.nan)
                          for d in DIMS})
    largest = drops.max(axis=1)
    return {
        "character_similarity_quartiles": [round(float(q), 3) for q in np.quantile(sim, [.25, .5, .75])],
        "near_copies": {f">={t}": round(float((sim >= t).mean()), 3) for t in (0.9, 0.95, 0.98)},
        "target_score_drop": {d: {"quartiles": [round(float(q), 1) for q in drops[d].dropna().quantile([.25, .5, .75])],
                                  "below_2_points": round(float((drops[d].dropna() < 2).mean()), 3),
                                  "below_5_points": round(float((drops[d].dropna() < 5).mean()), 3)} for d in DIMS},
        "rows_whose_largest_target_drop_is_below": {"2_points": round(float((largest < 2).mean()), 3),
                                                    "5_points": round(float((largest < 5).mean()), 3)},
        "near_copies_by_pattern": {k: round(float((sim[(pattern(v1) == k).to_numpy()] >= 0.95).mean()), 3)
                                   for k in ("B", "L", "S", "B+S", "B+L+S")},
    }


def strata(v1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = pattern(v1)
    out = {"all": v1}
    out.update({f"{d} targeted": v1[targeted(v1, d)] for d in DIMS})
    out.update({f"only {d}": v1[p == SHORT[d]] for d in DIMS})
    out["all three"] = v1[p == "B+L+S"]
    return {k: one_per_original(v) for k, v in out.items()}


def profiling(sets: dict[str, pd.DataFrame], features: list[str]) -> dict:
    out = {}
    for name, pairs in sets.items():
        st = feature_stats(pairs, features)
        r = pd.Series(st["r"], index=features, dtype=float)
        n_diff = pd.Series(st["n_diff"], index=features)
        # Rank only features that change in at least 10% of pairs: sparse ones reach r = +-1 on a handful of pairs.
        top = r[n_diff >= 0.1 * len(pairs)].abs().sort_values(ascending=False).index[:10]
        cats = pd.Series({f: category(f) for f in features})
        rho_top = {d: [(features[i], st["rho"][d][i]) for i in
                       np.argsort([-abs(x) if x is not None else 0 for x in st["rho"][d]])[:5]] for d in READIT_DIMS}
        out[name] = {
            "n_pairs": int(len(pairs)),
            "readit_change_mean": {d: sig(100 * (pairs[f"original_{d}"] - pairs[f"simplification_{d}"]).mean(), 3) for d in READIT_DIMS},
            "manova": st["manova"],
            "top_features": [{"feature": f, "category": category(f), "r": sig(r[f], 3),
                              "mean_o": st["mean_o"][features.index(f)], "mean_s": st["mean_s"][features.index(f)]} for f in top],
            "key_features": {f: {"r": sig(r[f], 3), "mean_o": st["mean_o"][features.index(f)], "mean_s": st["mean_s"][features.index(f)],
                                 "share_changed": sig(n_diff[f] / len(pairs), 3)}
                             for f in KEY_FEATURES if f in features},
            "category_mean_abs_r": {c: sig(r[cats == c].abs().mean(), 3) for c in cats.unique()},
            "top_spearman_with_readit_change": rho_top,
        }
        print(f"profiling: {name:18s} {len(pairs):7d} pairs  Pillai {st['manova']['V'] if st['manova'] else None}", flush=True)
    return out


def examples(sets: dict[str, pd.DataFrame]) -> dict:
    """Two short examples per single-dimension stratum, for the README."""
    out = {}
    for d in DIMS:
        pool = sets[f"only {d}"]
        pool = pool[pool.original_text.str.len().between(80, 170)].sample(n=2, random_state=SEED)
        out[f"only {d}"] = [{"original": r.original_text, "simplification": r.simplification, "ctrl_tokens": r.ctrl_tokens,
                             "readit_o": {k: round(100 * getattr(r, f"original_{k}"), 1) for k in DIMS},
                             "readit_s": {k: round(100 * getattr(r, f"simplification_{k}"), 1) for k in DIMS}}
                            for r in pool.itertuples()]
    return out


def comparison(v1_full: pd.DataFrame, features: list[str]) -> dict:
    """The CLiC-it 2025 analyses with the same code on: the 2025 Wikipedia sample, the GFFF (CLiC-it 2026)
    training file, dataset v0 and dataset v1. One simplification per original; common features only."""
    base_rows = lambda df: df[df.ctrl_token.str.contains("_base_")]
    sets = {
        "clicit2025_sample": load_clicit2025_sample(),
        "clicit2026_gfff_training": base_rows(pd.read_csv(ROOT / "data" / "clicit2026" / "training_dataset_final_bin_10_no_overlap.zip")),
        "v0_training": base_rows(pd.read_csv(V0 / "training_set_v0.zip")),
        "v1_training": v1_full,
        # v1 without its trade-off pairs: isolates what keeping them changes
        "v1_training_no_score_higher": v1_full[np.logical_and.reduce([v1_full[f"simplification_{d}"] <= v1_full[f"original_{d}"]
                                                                      for d in READIT_DIMS])],
    }
    common = [f for f in features if all(f"{f}_original" in d.columns and f"{f}_simplification" in d.columns for d in sets.values())]
    out = {"features_compared": len(common), "datasets": {}}
    for name, df in sets.items():
        pairs = one_per_original(df)
        st = feature_stats(pairs, common)
        fig = figure1(df, common)
        keep = [f for f in common if f not in MANOVA_EXCLUDED]
        out["datasets"][name] = {
            "pairs": int(len(df)), "originals": int(df.original_sentence_idx.nunique()),
            "readit_original_mean": {d: sig(100 * pairs[f"original_{d}"].mean(), 3) for d in READIT_DIMS},
            "readit_drop_mean": {d: sig(100 * (pairs[f"original_{d}"] - pairs[f"simplification_{d}"]).mean(), 3) for d in READIT_DIMS},
            "share_harder_in": {d: sig(float((pairs[f"simplification_{d}"] > pairs[f"original_{d}"]).mean()), 3) for d in DIMS},
            "original_chars_median": int(pairs.original_text.str.len().median()),
            "manova_one_per_original": pillai_trace(pairs[[f"{f}_original" for f in keep]].to_numpy(float),
                                                    pairs[[f"{f}_simplification" for f in keep]].to_numpy(float)),
            "manova_most_random_least": fig["manova"],
            "r": dict(zip(common, st["r"])), "p": dict(zip(common, st["p"])),
            "share_changed": {f: sig(n / len(pairs), 3) for f, n in zip(common, st["n_diff"])},
            "rho_global": dict(zip(common, st["rho"]["all"])),
        }
        print(f"comparison: {name:26s} {len(pairs):6d} pairs", flush=True)
    # Features whose effect moves by >= 0.15 between two datasets, among features that change in >= 10% of pairs
    # in both (sparse features reach r = +-1 on a handful of pairs).
    def moved(a: str, b: str) -> list:
        A, B = out["datasets"][a], out["datasets"][b]
        rows = [(f, A["r"][f], B["r"][f]) for f in common if A["r"][f] is not None and B["r"][f] is not None
                and A["share_changed"][f] >= 0.1 and B["share_changed"][f] >= 0.1 and abs(B["r"][f] - A["r"][f]) >= 0.15]
        return [{"feature": f, "category": category(f), "r_from": x, "r_to": y} for f, x, y in sorted(rows, key=lambda r: -abs(r[2] - r[1]))]
    out["r_changes"] = {f"{a} -> {b}": moved(a, b) for a, b in (("clicit2025_sample", "clicit2026_gfff_training"),
                                                                 ("clicit2025_sample", "v1_training"),
                                                                 ("clicit2026_gfff_training", "v1_training"),
                                                                 ("v1_training_no_score_higher", "v1_training"))}
    return out


def kde(values: np.ndarray) -> np.ndarray:
    from scipy import stats
    return stats.gaussian_kde(values)(KDE_GRID)


def draw(sets: dict[str, pd.DataFrame], rows: list[str], cols: list[str], path: Path, theme: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = THEMES[theme]
    names = {"all": "READ-IT global", "base": "READ-IT base", "lexical": "READ-IT lexical", "syntax": "READ-IT syntax"}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": c["ink"], "axes.labelcolor": c["ink2"],
                         "xtick.color": c["muted"], "ytick.color": c["muted"]})
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.3 * len(cols) + 0.6, 1.75 * len(rows) + 0.7),
                             squeeze=False, sharex=True, facecolor=c["surface"])
    for i, row in enumerate(rows):
        pairs = sets[row]
        for j, dim in enumerate(cols):
            ax = axes[i][j]
            ax.set_facecolor(c["surface"])
            for side, col in (("original", "original"), ("simplification", "simplified")):
                ax.plot(KDE_GRID, kde(100 * pairs[f"{side}_{dim}"].to_numpy()), color=c[col], lw=2, solid_capstyle="round")
            ax.grid(axis="y", color=c["grid"], lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(c["axis"])
            ax.tick_params(length=0, labelsize=8)
            ax.set_yticklabels([])
            ax.set_xlim(-10, 110)
            ax.set_xticks([0, 25, 50, 75, 100])
            if i == 0:
                ax.set_title(names[dim], fontsize=9, color=c["ink"], loc="left")
            if j == 0 and len(rows) > 1:
                ax.set_ylabel(f"{row}\n({len(pairs):,} pairs)", fontsize=8.5, color=c["ink2"])
        axes[-1][len(cols) // 2].set_xlabel("score (0 = most readable)", fontsize=8.5)
    handles = [plt.Line2D([], [], color=c["original"], lw=2), plt.Line2D([], [], color=c["simplified"], lw=2)]
    fig.legend(handles, ["original", "simplified"], loc="upper right", ncol=2, frameon=False, fontsize=8.5,
               labelcolor=c["ink2"], bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    v1 = pd.read_csv(V1 / "training_set_v1.zip", dtype={f"tgt_{d}": str for d in DIMS})
    v0 = pd.read_csv(V0 / "training_set_v0.zip", usecols=["idx", "original_sentence_idx", "original_text", "original_all",
                                                         "ctrl_token", "ctrl_int"])
    v0 = v0[v0.ctrl_token.str.contains("_base_")].drop_duplicates("idx")
    feats = [c[: -len("_original")] for c in v1.columns if c.endswith("_original") and c[: -len("_original")] + "_simplification" in v1.columns]
    feats = [f for f in feats if v1[f"{f}_original"].nunique() > 1 or v1[f"{f}_simplification"].nunique() > 1]
    full = v1

    sets = strata(full)
    result = {
        "overview": overview(v1, v0),
        "patterns": patterns(v1),
        "per_dimension": per_dimension(v1, pd.read_csv(V0 / "training_set_v0.zip", usecols=["original_sentence_idx", "ctrl_token", "ctrl_int"])),
        "co_targeting": co_targeting(v1),
        "test_requests": test_requests(v1),
        "comparison": comparison(full, feats),
        "small_edits": small_edits(v1),
        "profiling": profiling(sets, feats),
        "examples": examples(sets),
        "n_features": len(feats), "manova_excluded": list(MANOVA_EXCLUDED), "seed": SEED,
    }
    (V1 / "analysis_v1.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    figs = V1 / "figures"
    figs.mkdir(exist_ok=True)
    for theme in THEMES:
        draw(sets, ["all"], list(READIT_DIMS), figs / f"readit_kde_{theme}.png", theme)
        draw(sets, [f"only {d}" for d in DIMS], list(DIMS), figs / f"single_target_kde_{theme}.png", theme)
    print(json.dumps({k: v for k, v in result.items() if k not in ("profiling", "examples")}, indent=1, ensure_ascii=False))
    print(f"wrote {V1 / 'analysis_v1.json'} and {figs}")


if __name__ == "__main__":
    main()
