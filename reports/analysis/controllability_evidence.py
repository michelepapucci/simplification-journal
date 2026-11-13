"""Evidence behind reports/improving_controllability.md.

Reads, without modifying them:
- the CLiC-it 2026 artefacts in the GFFF folder of the old text-simplification-experiments
  repository (model predictions scored with READ-IT, and the three Qwen3 checkpoints);
- the Wikipedia part of the IMPaCTS release on Hugging Face (mpapucci/impacts, pinned in
  scripts/impacts.py), which is the dataset of the journal version;
- this repository's data/ folder.
Prints a summary and writes reports/analysis/controllability_evidence.json.

Usage: python reports/analysis/controllability_evidence.py --gfff /path/to/text-simplification-experiments/GFFF \
           [--hf-dir SNAPSHOT] [--lrec /path/to/text-simplification-experiments/LREC/data/dataset_profiled.zip]
--hf-dir reads a local snapshot of the release instead of downloading it. --lrec adds a check of
how the release was derived from the LREC 2026 corpus.
Needs numpy, pandas, scipy, pyarrow, huggingface_hub, torch and safetensors.
"""

import argparse
import json
import sys
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
V0_TRAIN = DATA / "v0" / "training_set_v0.zip"
sys.path.insert(0, str(ROOT / "scripts"))
from impacts import DIMENSIONS, SCORE_COLUMNS, load_wikipedia  # noqa: E402
STRATEGIES = ("base", "lexical", "syntax")
EVAL_LEVELS = tuple(range(10, 80, 10))  # control tokens evaluated in the CLiC-it 2026 paper
MODELS = {"0.6B": "06", "1.7B": "17", "4B": "40"}


def level_of(token: pd.Series) -> pd.Series:
    return token.str.extract(r"_(\d+)\|>$", expand=False).astype(int)


def strategy_of(token: pd.Series) -> pd.Series:
    return token.str.extract(r"<\|readability_(\w+?)_\d+\|>", expand=False)


def training_data() -> dict:
    """How the training pairs cover the control space, for the CLiC-it file and the rebuilt one."""
    cols = ["idx", "original_sentence_idx", "ctrl_token", "ctrl_int"] + \
           [f"{p}_{s}" for p in ("original", "simplification") for s in STRATEGIES]
    out = {}
    for name, path in (("clicit2026", DATA / "clicit2026" / "training_dataset_final_bin_10_no_overlap.zip"),
                       ("v0", DATA / "v0" / "training_set_v0.zip")):
        df = pd.read_csv(path, usecols=cols)
        df["strategy"] = strategy_of(df["ctrl_token"])
        res = {}
        for s in STRATEGIES:
            d = df[df.strategy == s]
            per_orig = d.groupby("original_sentence_idx")["ctrl_int"]
            res[s] = {
                "pairs": int(len(d)),
                "originals": int(d.original_sentence_idx.nunique()),
                "target_harder_than_source": round(float((d[f"simplification_{s}"] > d[f"original_{s}"]).mean()), 3),
                "levels_per_original_mean": round(float(per_orig.nunique().mean()), 2),
                "originals_with_one_level": round(float((per_orig.nunique() == 1).mean()), 3),
                "level_span_per_original_mean": round(float((per_orig.max() - per_orig.min()).mean()), 1),
                "level_counts": {int(k): int(v) for k, v in d.ctrl_int.value_counts().sort_index().items()},
                "original_score_quartiles": [round(float(q), 1) for q in
                                             (100 * d.drop_duplicates("original_sentence_idx")[f"original_{s}"]).quantile([.25, .5, .75])],
            }
        out[name] = res
    return out


def region(source: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Where the source sentence sits relative to the requested bin [level, level + 10)."""
    return np.where(source >= level + 10, "above", np.where(source >= level, "inside", "below"))


def test_feasibility() -> dict:
    """Share of the paper's requests (every test original x CT 10-70) by source position."""
    out = {}
    for s in STRATEGIES:
        src = 100 * pd.read_csv(DATA / "v0" / f"{s}_test_set_v0.csv", usecols=[f"original_{s}"])[f"original_{s}"].to_numpy()
        per_level = {}
        for lv in EVAL_LEVELS:
            reg = region(src, np.full_like(src, lv))
            per_level[lv] = {r: round(float((reg == r).mean()), 3) for r in ("above", "inside", "below")}
        allreg = np.concatenate([region(src, np.full_like(src, lv)) for lv in EVAL_LEVELS])
        out[s] = {"per_level": per_level, "overall": {r: round(float((allreg == r).mean()), 3) for r in ("above", "inside", "below")},
                  "source_quartiles": [round(float(q), 1) for q in np.quantile(src, [.25, .5, .75])]}
    return out


def prediction_files(gfff: Path, tag: str, s: str) -> Path:
    if tag == "40":
        return gfff / "profiling" / f"prediction_test_set_{s}_profiled_readit_40.csv"
    return gfff / "metrics" / tag / f"prediction_test_set_{s}_profiled_readit_{tag}.csv"


def load_predictions(gfff: Path, tag: str, s: str) -> pd.DataFrame:
    cols = ["original_sentence_idx", "prediction", "control_token", "original_text", f"original_{s}", s]
    df = pd.read_csv(prediction_files(gfff, tag, s), usecols=cols)
    df["level"] = level_of(df["control_token"])
    df = df[df.level.isin(EVAL_LEVELS)].copy()
    df["source"] = 100 * df[f"original_{s}"]
    df["achieved"] = 100 * df[s]
    df.attrs["unscored"] = int(df.achieved.isna().sum())  # outputs READ-IT could not score (e.g. empty)
    df = df.dropna(subset=["achieved"])
    df["region"] = region(df.source.to_numpy(), df.level.to_numpy())
    df["copy"] = df.prediction.astype(str).str.strip() == df.original_text.astype(str).str.strip()
    return df


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def prediction_eval(gfff: Path) -> dict:
    """The paper's RMSE and Spearman, reproduced, then split by where the source sits."""
    out = {}
    for model, tag in MODELS.items():
        out[model] = {}
        for s in STRATEGIES:
            df = load_predictions(gfff, tag, s)
            per_orig = []
            for _, g in df.groupby("original_sentence_idx"):
                if g.level.nunique() == len(EVAL_LEVELS) and g.achieved.nunique() > 1:
                    per_orig.append(stats.spearmanr(g.level, g.achieved).statistic)
            by_region = {}
            for r, g in df.groupby("region"):
                by_region[r] = {"share": round(len(g) / len(df), 3), "rmse": round(rmse(g.level, g.achieved), 1),
                                "copy_rate": round(float(g["copy"].mean()), 3),
                                "achieved_above_source": round(float((g.achieved > g.source).mean()), 3)}
            out[model][s] = {
                "rmse_per_level": {lv: round(rmse(g.level, g.achieved), 2) for lv, g in df.groupby("level")},
                "spearman_pooled": round(float(stats.spearmanr(df.level, df.achieved).statistic), 3),
                "spearman_within_original_mean": round(float(np.mean(per_orig)), 3),
                "by_region": by_region,
                "copy_rate": round(float(df["copy"].mean()), 3),
                "unscored_outputs": df.attrs["unscored"],
            }
    return out


def memorization(gfff: Path) -> dict:
    """How often a test output reproduces a training simplification of the same original sentence."""
    train = pd.read_csv(DATA / "clicit2026" / "training_dataset_final_bin_10_no_overlap.zip",
                        usecols=["original_sentence_idx", "simplification", "ctrl_token"])
    train = train[strategy_of(train.ctrl_token) == "base"]  # each pair appears once per strategy
    targets = train.groupby("original_sentence_idx")["simplification"].apply(lambda x: [t.strip() for t in x]).to_dict()
    out = {"juno_ii_in_training": [t for t in train.simplification if "Juno II era un satellite" in t][:2]}
    for model, tag in MODELS.items():
        out[model] = {}
        for s in STRATEGIES:
            df = load_predictions(gfff, tag, s)
            exact = near = with_targets = 0
            for oid, pred in zip(df.original_sentence_idx, df.prediction.astype(str).str.strip()):
                cands = targets.get(oid)
                if not cands:
                    continue
                with_targets += 1
                if pred in cands:
                    exact += 1
                    near += 1
                elif any(SequenceMatcher(None, pred, c).ratio() >= 0.9 for c in cands
                         if SequenceMatcher(None, pred, c).real_quick_ratio() >= 0.9):
                    near += 1
            out[model][s] = {"outputs_with_training_targets": round(with_targets / len(df), 3),
                             "exact_training_target": round(exact / len(df), 3),
                             "near_duplicate_training_target": round(near / len(df), 3)}
    return out


def embeddings(gfff: Path) -> dict:
    """Geometry of the learned control-token embeddings (drift from their shared initialisation)."""
    import torch
    from safetensors import safe_open

    train = pd.read_csv(DATA / "clicit2026" / "training_dataset_final_bin_10_no_overlap.zip", usecols=["ctrl_token"])
    counts = train.ctrl_token.value_counts().to_dict()
    out = {}
    for model in MODELS:
        run = gfff / f"text-simplification-Qwen3-{model}-Base-wikipedia-42-bin-10"
        vocab = json.loads((run / "checkpoint-6351" / "added_tokens.json").read_text())
        tokens = [f"<|readability_{s}_{lv}|>" for s in STRATEGIES for lv in range(0, 100, 10)]
        ids = [vocab[t] for t in tokens]
        lo, hi = min(ids), max(ids) + 1
        with safe_open(str(run / "checkpoint-6351" / "adapter_model.safetensors"), "pt") as f:
            prefix = "base_model.model.model.embed_tokens."
            trained = f.get_slice(prefix + "modules_to_save.weight")[lo:hi].to(torch.float32).numpy()
            init = f.get_slice(prefix + "original_module.weight")[lo:hi].to(torch.float32).numpy()
        rows = {t: i - lo for t, i in zip(tokens, ids)}
        drift = {t: trained[rows[t]] - init[rows[t]] for t in tokens}
        norm = {t: float(np.linalg.norm(v)) for t, v in drift.items()}
        cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        pair = lambda s1, s2: round(float(np.mean([cos(drift[f"<|readability_{s1}_{lv}|>"], drift[f"<|readability_{s2}_{lv}|>"])
                                                   for lv in range(0, 100, 10)])), 3)
        adjacent = lambda s: round(float(np.mean([cos(drift[f"<|readability_{s}_{lv}|>"], drift[f"<|readability_{s}_{lv + 10}|>"])
                                                  for lv in range(0, 90, 10)])), 3)
        ordinality = {}
        for s in STRATEGIES:
            M = np.stack([drift[f"<|readability_{s}_{lv}|>"] for lv in range(0, 100, 10)])
            M = M - M.mean(axis=0)
            u, sv, _ = np.linalg.svd(M, full_matrices=False)
            pc1 = u[:, 0] * sv[0]
            ordinality[s] = {"pc1_vs_level_spearman": round(abs(float(stats.spearmanr(pc1, range(10)).statistic)), 3),
                             "pc1_variance_share": round(float(sv[0] ** 2 / (sv ** 2).sum()), 3)}
        log_counts = [np.log1p(counts.get(t, 0)) for t in tokens]
        out[model] = {
            # init_control_tokens_embeddings() assumes the new tokens are the last rows, but
            # resize_token_embeddings(pad_to_multiple_of=8) appends padding rows after them.
            "tokens_with_mean_initialisation": [t for t in tokens if np.allclose(init[rows[t]], init[rows[tokens[-1]]], atol=1e-6)],
            "same_level_cosine": {"base~lexical": pair("base", "lexical"), "base~syntax": pair("base", "syntax"),
                                  "lexical~syntax": pair("lexical", "syntax")},
            "adjacent_level_cosine": {s: adjacent(s) for s in STRATEGIES},
            "ordinality": ordinality,
            "drift_vs_log_training_count_spearman": round(float(stats.spearmanr([norm[t] for t in tokens], log_counts).statistic), 3),
            "drift_norm": {t: round(v, 4) for t, v in norm.items()},
        }
    return out


def paper_levels(gfff: Path) -> dict:
    """The paper's RMSE tables and Spearman values, recomputed for every control token 0-90."""
    out = {}
    for model, tag in MODELS.items():
        out[model] = {}
        for s in STRATEGIES:
            df = pd.read_csv(prediction_files(gfff, tag, s), usecols=["control_token", s]).dropna()
            lv, ach = level_of(df.control_token), 100 * df[s]
            out[model][s] = {
                "rmse_per_token": {int(l): round(rmse(np.full(int((lv == l).sum()), l), ach[lv == l]), 2) for l in sorted(lv.unique())},
                "spearman_tokens_0_60": round(float(stats.spearmanr(lv[lv <= 60], ach[lv <= 60]).statistic), 3),
                "spearman_tokens_10_70": round(float(stats.spearmanr(lv[lv.between(10, 70)], ach[lv.between(10, 70)]).statistic), 3),
            }
    return out


def near_duplicate_noise(pairs: pd.DataFrame) -> dict:
    """READ-IT differences between near-identical (character similarity >= 0.9, not identical)
    simplifications of the same original."""
    cols = ["simplification"] + [f"simplification_{s}" for s in STRATEGIES]
    diffs = []
    for _, g in pairs.groupby("original_sentence_idx"):
        for a, b in combinations(g[cols].to_numpy(), 2):
            if a[0] == b[0]:
                continue
            sm = SequenceMatcher(None, a[0], b[0])
            if sm.real_quick_ratio() >= 0.9 and sm.quick_ratio() >= 0.9 and sm.ratio() >= 0.9:
                diffs.append([100 * abs(a[i] - b[i]) for i in (1, 2, 3)])
    d = np.array(diffs)
    return {"near_duplicate_pairs": int(len(d)),
            "median_abs_difference": {s: round(float(np.median(d[:, i])), 1) for i, s in enumerate(STRATEGIES)},
            "share_at_least_20_points": {s: round(float((d[:, i] >= 20).mean()), 3) for i, s in enumerate(STRATEGIES)}}


def label_noise() -> dict:
    t = pd.read_csv(V0_TRAIN, usecols=["original_sentence_idx", "ctrl_token", "simplification"] +
                    [f"simplification_{s}" for s in STRATEGIES])
    return near_duplicate_noise(t[strategy_of(t.ctrl_token) == "base"])


# ---------------------------------------------------------------------------------------------
# IMPaCTS Wikipedia, as released on Hugging Face (mpapucci/impacts, pinned in scripts/impacts.py)
# ---------------------------------------------------------------------------------------------

def load_release(hf_dir: Path | None) -> pd.DataFrame:
    cols = ["idx", "original_sentence_idx", "original_text", "simplification"] + SCORE_COLUMNS
    return load_wikipedia(columns=cols, local_dir=hf_dir)


def read_list(name: str) -> set:
    return {int(x) for x in (DATA / "source" / "idx_to_remove" / name).read_text().split()}


def test_originals(release: pd.DataFrame) -> set:
    tests = set().union(*(read_list(f"list_idx_{s}.txt") for s in STRATEGIES))
    return set(release.loc[release.idx.isin(tests), "original_sentence_idx"])


def harder(d: pd.DataFrame, dims=STRATEGIES) -> np.ndarray:
    """Pairs whose simplification scores higher (harder) than the original in at least one of dims."""
    return np.logical_or.reduce([(d[f"simplification_{s}"] > d[f"original_{s}"]).to_numpy() for s in dims])


def clicit_selection(d: pd.DataFrame) -> pd.DataFrame:
    """The CLiC-it 2026 selection (scripts/build_dataset_v0.py): hard originals, no READ-IT score higher."""
    return d[(d.original_all > 0.7) & ~harder(d, DIMENSIONS)]


def provenance(release: pd.DataFrame, lrec: Path | None) -> dict:
    """How the release relates to the GFFF source file and, if given, to the LREC 2026 corpus."""
    gfff = pd.read_csv(DATA / "source" / "dataset_profiled_filtered_with_4_control_tokens.zip",
                       usecols=["idx", "original_text", "simplification"])
    missing = gfff[~gfff.idx.isin(release.idx)]
    copy = lambda d: (d.simplification == d.original_text).to_numpy()
    clicit = pd.read_csv(DATA / "clicit2026" / "training_dataset_final_bin_10_no_overlap.zip",
                         usecols=["idx", "ctrl_token", "original_text", "simplification"])
    clicit = clicit[strategy_of(clicit.ctrl_token) == "base"]
    out = {
        "release_pairs": int(len(release)), "release_originals": int(release.original_sentence_idx.nunique()),
        "release_harder_globally": int(harder(release, ("all",)).sum()), "release_exact_copies": int(copy(release).sum()),
        "gfff_source_pairs": int(len(gfff)), "gfff_source_pairs_missing_from_release": int(len(missing)),
        "missing_pairs_that_are_exact_copies": round(float(copy(missing).mean()), 3),
        "release_selection_equals_gfff_source_without_copies":
            set(clicit_selection(release).idx) == set(gfff.idx[~copy(gfff)]),
        "exact_copies_in_clicit2026_training": int(copy(clicit).sum()),
        "exact_copies_in_clicit2026_training_share": round(float(copy(clicit).mean()), 3),
        "noise_list_pairs_still_in_release": {n: int(len(read_list(f"list_dropped_idx_{n}.txt") & set(release.idx)))
                                              for n in ("langdetect", "double_filter")},
    }
    if lrec is not None:
        cols = ["idx", "domain", "original_text", "simplification"] + SCORE_COLUMNS
        wiki = pd.read_csv(lrec, usecols=cols)
        wiki = wiki[wiki.domain == "wikipedia"]
        dropped_globally, dropped_copy = harder(wiki, ("all",)), copy(wiki)
        rule_keeps = ~(dropped_globally | dropped_copy)
        out["lrec2026_wikipedia"] = {
            "pairs": int(len(wiki)), "removed": int(len(wiki) - wiki.idx.isin(release.idx).sum()),
            "removed_global_score_higher": int(dropped_globally.sum()),
            "removed_exact_copy_only": int((dropped_copy & ~dropped_globally).sum()),
            "rule_matches_release_membership": round(float((rule_keeps == wiki.idx.isin(release.idx).to_numpy()).mean()), 5),
        }
    return out


def double_filter(release: pd.DataFrame) -> dict:
    """What the GFFF double filter removed, at the level of original sentences."""
    sel = clicit_selection(release).copy()
    sel["dropped"] = sel.idx.isin(read_list("list_dropped_idx_double_filter.txt"))
    per_orig = sel.groupby("original_sentence_idx").dropped.mean()
    first = sel.drop_duplicates("original_sentence_idx").set_index("original_sentence_idx").original_text
    text = first.loc[per_orig.index]
    quote_or_paren = text.str.contains(r'["«»“”(]', regex=True)
    long = text.str.len() > 400
    whole = per_orig[(per_orig == 0) | (per_orig == 1)] == 1
    proxy = (quote_or_paren | long).loc[whole.index]
    return {
        "pairs_removed_in_selection": int(sel.dropped.sum()),
        "originals_entirely_removed": round(float((per_orig == 1).mean()), 3),
        "originals_partly_removed": round(float(((per_orig > 0) & (per_orig < 1)).mean()), 3),
        "median_original_chars": {"removed": int(text[per_orig == 1].str.len().median()), "kept": int(text[per_orig == 0].str.len().median())},
        "proxy_quotes_parentheses_or_over_400_chars_agreement": round(float((proxy == whole).mean()), 3),
        "pairs_never_checked": int(len(release) - len(sel)),
    }


def pool_summary(d: pd.DataFrame, held_out: set) -> dict:
    kept = d[~d.original_sentence_idx.isin(held_out)]
    first = d.drop_duplicates("original_sentence_idx")
    return {
        "pairs": int(len(d)), "originals": int(d.original_sentence_idx.nunique()),
        "pairs_per_original_mean": round(len(d) / d.original_sentence_idx.nunique(), 2),
        "harder_in": {s: round(float((d[f"simplification_{s}"] > d[f"original_{s}"]).mean()), 3) for s in STRATEGIES},
        "harder_in_any_dimension": round(float(harder(d).mean()), 3),
        "source_quartiles": {s: [round(float(q), 1) for q in (100 * first[f"original_{s}"]).quantile([.25, .5, .75])]
                             for s in DIMENSIONS},
        "held_out_test_originals": {"pairs": int(len(kept)), "originals": int(kept.original_sentence_idx.nunique()),
                                    "pairs_no_dimension_harder": int((~harder(kept)).sum())},
    }


def pools(release: pd.DataFrame) -> dict:
    """Training pools in the release, with test originals held out (the grouped split)."""
    held_out = test_originals(release)
    hard = release[release.original_all > 0.7]
    sel = clicit_selection(release)
    noisy = read_list("list_dropped_idx_langdetect.txt") | read_list("list_dropped_idx_double_filter.txt")
    e0 = sel[~sel.idx.isin(noisy)]
    tests = set().union(*(read_list(f"list_idx_{s}.txt") for s in STRATEGIES))
    e0_train = e0[~e0.idx.isin(tests)]
    e0_held = e0_train[~e0_train.original_sentence_idx.isin(held_out)]
    return {
        "held_out_originals": len(held_out),
        "all_sources": pool_summary(release, held_out),
        "global_above_70": pool_summary(hard, held_out),
        "clicit_selection": {"pairs": int(len(sel)), "originals": int(sel.original_sentence_idx.nunique()),
                             "after_noise_lists": int(len(e0)),
                             "training_pairs": int(len(e0_train)),
                             "training_pairs_held_out": int(len(e0_held)),
                             "training_originals_held_out": int(e0_held.original_sentence_idx.nunique()),
                             "lost_by_holding_out": int(len(e0_train) - len(e0_held))},
    }


def level_coverage(release: pd.DataFrame) -> dict:
    """Target levels per strategy, and how many distinct levels each original is seen with:
    the CLiC-it selection (dataset v0) vs every held-out pair of the release."""
    held_out = test_originals(release)
    t = pd.read_csv(V0_TRAIN, usecols=["original_sentence_idx", "ctrl_token", "ctrl_int"])
    full = release[~release.original_sentence_idx.isin(held_out)]
    out = {}
    for s in STRATEGIES:
        sel = t[strategy_of(t.ctrl_token) == s]
        lv = np.minimum(np.floor(100 * full[f"simplification_{s}"] / 10) * 10, 90).astype(int)
        not_harder = (full[f"simplification_{s}"] <= full[f"original_{s}"]).to_numpy()
        pools = (("clicit_selection", sel.ctrl_int, sel.original_sentence_idx),
                 ("release_all_sources", lv, full.original_sentence_idx),
                 ("release_all_sources_dimension_not_harder", lv[not_harder], full.original_sentence_idx[not_harder]))
        out[s] = {}
        for name, v, orig in pools:
            v, orig = pd.Series(np.asarray(v)), pd.Series(np.asarray(orig))
            per_orig = v.groupby(orig.to_numpy()).nunique()
            out[s][name] = {"pairs": int(len(v)), "share_20_40": round(float(v.between(20, 40).mean()), 3),
                            "levels_with_1000_pairs": int((v.value_counts() >= 1000).sum()),
                            "levels_per_original_mean": round(float(per_orig.mean()), 2),
                            "originals_with_one_level": round(float((per_orig == 1).mean()), 3),
                            "counts": {int(k): int(c) for k, c in v.value_counts().sort_index().items()}}
    return out


def cluster_pairs(d: pd.DataFrame) -> dict:
    """Ordered pairs s_i -> s_j between simplifications of the same original that would pass the
    release's own rule (global score not higher, text different)."""
    cols = [f"simplification_{s}" for s in DIMENSIONS]
    simpler = dict.fromkeys(STRATEGIES, 0)
    total = passing = any_simpler = same_bins = 0
    for _, g in d.groupby("original_sentence_idx"):
        if len(g) < 2:
            continue
        v = 100 * g[cols].to_numpy()
        text = g.simplification.to_numpy()
        ok = (text[None, :] != text[:, None])  # [source i, target j]
        total += int(ok.sum())
        ok &= v[None, :, 3] <= v[:, None, 3]
        passing += int(ok.sum())
        lower = v[None, :, :3] < v[:, None, :3]
        for k, s in enumerate(STRATEGIES):
            simpler[s] += int((lower[:, :, k] & ok).sum())
        any_simpler += int((lower.any(axis=2) & ok).sum())
        bins = np.floor(v[:, :3] / 10)
        same_bins += int(((bins[None, :, :] == bins[:, None, :]).all(axis=2) & ok).sum())
    return {"original_to_simplification": int(len(d)), "between_simplifications_distinct_text": total,
            "between_simplifications_global_not_higher": passing, "of_which_simpler_in": simpler,
            "of_which_simpler_in_any_dimension": any_simpler, "of_which_same_bin_all_three": same_bins}


def clusters(release: pd.DataFrame) -> dict:
    held_out = test_originals(release)
    kept = release[~release.original_sentence_idx.isin(held_out)]
    t = pd.read_csv(V0_TRAIN, usecols=["idx", "ctrl_token"])
    e0 = kept[kept.idx.isin(t.idx[strategy_of(t.ctrl_token) == "base"])]
    return {"clicit_selection": cluster_pairs(e0),
            "global_above_70": cluster_pairs(kept[kept.original_all > 0.7]),
            "all_sources": cluster_pairs(kept)}


def independence(pairs: pd.DataFrame) -> dict:
    """How independently the three READ-IT dimensions move, and how much of the joint grid has data."""
    out = 100 * pairs[[f"simplification_{s}" for s in STRATEGIES]].to_numpy()
    delta = out - 100 * pairs[[f"original_{s}" for s in STRATEGIES]].to_numpy()
    corr = lambda m: pd.DataFrame(m, columns=STRATEGIES).corr("spearman")
    co, cd = corr(out), corr(delta)
    cells = pd.Series(list(map(tuple, np.minimum(np.floor(out / 10), 9).astype(int)))).value_counts()
    return {
        "pairs": int(len(pairs)),
        "spearman_outputs": {f"{a}-{b}": round(float(co.loc[a, b]), 2) for a, b in combinations(STRATEGIES, 2)},
        "spearman_changes": {f"{a}-{b}": round(float(cd.loc[a, b]), 2) for a, b in combinations(STRATEGIES, 2)},
        "lexical_only_share": round(float(((np.abs(delta[:, 0]) <= 5) & (delta[:, 1] <= -10)).mean()), 3),
        "lexical_only_pairs": int(((np.abs(delta[:, 0]) <= 5) & (delta[:, 1] <= -10)).sum()),
        "syntax_only_share": round(float(((np.abs(delta[:, 0]) <= 5) & (np.abs(delta[:, 1]) <= 5) & (delta[:, 2] <= -10)).mean()), 3),
        "joint_cells_with_20_pairs": int((cells >= 20).sum()),
        "pairs_in_those_cells": round(float(cells[cells >= 20].sum() / len(pairs)), 3),
    }


def independence_pools(release: pd.DataFrame) -> dict:
    hard = release[release.original_all > 0.7]
    return {"clicit_selection": independence(clicit_selection(release)),
            "global_above_70": independence(hard),
            "all_sources_no_dimension_harder": independence(release[~harder(release)]),
            "all_sources": independence(release)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gfff", type=Path, required=True, help="GFFF folder of the old repository")
    parser.add_argument("--hf-dir", type=Path, help="local snapshot of mpapucci/impacts (default: download)")
    parser.add_argument("--lrec", type=Path, help="LREC 2026 corpus (LREC/data/dataset_profiled.zip), for the provenance check")
    parser.add_argument("--skip", nargs="*", default=[], help="sections to leave out, e.g. release_label_noise")
    args = parser.parse_args()
    release = load_release(args.hf_dir)
    sections = {
        # the CLiC-it 2026 models and data
        "training_data": training_data,
        "test_feasibility": test_feasibility,
        "predictions": lambda: prediction_eval(args.gfff),
        "paper_levels": lambda: paper_levels(args.gfff),
        "memorization": lambda: memorization(args.gfff),
        "embeddings": lambda: embeddings(args.gfff),
        "label_noise": label_noise,
        # the journal dataset: IMPaCTS Wikipedia on Hugging Face
        "release_provenance": lambda: provenance(release, args.lrec),
        "release_double_filter": lambda: double_filter(release),
        "release_pools": lambda: pools(release),
        "release_level_coverage": lambda: level_coverage(release),
        "release_clusters": lambda: clusters(release),
        "release_independence": lambda: independence_pools(release),
        "release_label_noise": lambda: near_duplicate_noise(release),
    }
    path = Path(__file__).with_suffix(".json")
    # With --skip, the sections that are not recomputed keep their previous values.
    result = json.loads(path.read_text()) if args.skip and path.exists() else {}
    for name, fn in sections.items():
        if name not in args.skip:
            result[name] = fn()
            print(f"done: {name}", flush=True)
    path.write_text(json.dumps({k: result[k] for k in sections if k in result}, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False)[:20000])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
