"""Results of the two v1 models and of the CLiC-it 2026 Qwen3-0.6B, on the requests of build_requests.py.

READ-IT of every generation is computed exactly as the IMPaCTS labels (tools/readit/score_readit.py),
so generations, originals (their IMPaCTS labels) and the requested levels share one scale.

Main evaluation (one dimension requested, the others free), per test set and level:
- RMSE between the requested level and the achieved score (x100), the CLiC-it convention: an
  output anywhere inside the level scores about 5, so the share inside the level is reported too;
- signed bias, pooled and within-original Spearman;
- movement of the requested and of the two free dimensions: generation minus original label;
- meaning (BERTScore, SBERT, copies) and profile changes (Profiling-UD, generation minus original).
The CLiC-it 0.6B generated levels 0-90 for every original; only the levels requested here (below
the original's own level) are kept.

Joint evaluation (all three dimensions requested), by request type (achievable / typical):
- each dimension: share inside its level, RMSE, bias;
- all three inside their levels at once, and all three within one level (achieved between one
  level below and one level above the requested one).

    python data/v1/experiments/analyze.py      ->  results/summary.json
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

from common import DIMS, OUTPUTS, RESULTS, V1

MODELS = {"clicit2026_0.6B": "CLiC-it 2026 0.6B", "targets": "v1, target tokens", "source_targets": "v1, source + target tokens",
          "v2_targets": "v2, target tokens", "v3_targets": "v3, target tokens", "v4_targets": "v4, target tokens", "v5_targets": "v5, target tokens", "v6_targets": "v6, target tokens", "v7_targets": "v7, one token per example + combined",
          "v7_1ep_targets": "v7, one epoch"}
KEY_FEATURES = ["n_sentences", "n_tokens", "char_per_tok", "in_dict", "lexical_density", "avg_max_depth",
                "verbal_head_per_sent", "subordinate_proposition_dist"]


def level_of(x100) -> np.ndarray:
    return (np.minimum(np.floor(np.asarray(x100, dtype=float) / 10), 9) * 10)


def control_metrics(level: pd.Series, achieved: pd.Series) -> dict:
    ok = achieved.notna()
    level, achieved = level[ok], achieved[ok]
    if not len(level):
        return {"scored": 0}
    err = achieved - level
    q = achieved.quantile([.25, .5, .75])
    return {"scored": int(ok.sum()), "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
            "achieved_mean": round(float(achieved.mean()), 2),
            "achieved_quartiles": [round(float(x), 2) for x in q],
            "in_level": round(float(((achieved >= level) & (achieved < level + 10)).mean()), 3),
            "bias": round(float(err.mean()), 2),
            "spearman": round(float(stats.spearmanr(level, achieved).statistic), 3) if level.nunique() > 1 else None}


def originals() -> pd.DataFrame:
    """One row per test original: IMPaCTS labels (x100) and Profiling-UD features."""
    rows = []
    for dim in DIMS:
        test = pd.read_csv(V1 / f"{dim}_test_set_v1.csv", usecols=["original_sentence_idx"] + [f"original_{d}" for d in (*DIMS, "all")])
        prof = OUTPUTS / "references" / f"profiling-originals-{dim}.csv"
        if prof.exists():
            p = pd.read_csv(prof)
            test = test.assign(**{f"prof_{k}": p[k] for k in KEY_FEATURES if k in p})
        rows.append(test)
    o = pd.concat(rows).drop_duplicates("original_sentence_idx").set_index("original_sentence_idx")
    for d in (*DIMS, "all"):
        o[f"label_{d}"] = 100 * o[f"original_{d}"]
    return o


def scorer_check() -> dict:
    """Share of test originals and references whose re-scored READ-IT is identical to the IMPaCTS label."""
    out = {}
    for dim in DIMS:
        test = pd.read_csv(V1 / f"{dim}_test_set_v1.csv")
        for side, col in (("originals", "original"), ("references", "simplification")):
            path = OUTPUTS / "references" / f"readit-{side}-{dim}.csv"
            if path.exists():
                r = pd.read_csv(path)
                out[f"{side}-{dim}"] = {d: round(float((np.abs(r[f"readit_{d}"] - test[f"{col}_{d}"]) < 1e-9).mean()), 4)
                                         for d in (*DIMS, "all")}
    return out


def load(model: str, name: str, orig: pd.DataFrame) -> pd.DataFrame | None:
    p, r = OUTPUTS / model / f"predictions-{name}.csv", OUTPUTS / model / f"readit-{name}.csv"
    if not (p.exists() and r.exists()):
        return None
    f = pd.read_csv(p, dtype={f"tgt_{d}": str for d in DIMS})
    rd = pd.read_csv(r)
    f = f.assign(**{f"out_{d}": 100 * rd[f"readit_{d}"] for d in DIMS})
    for d in DIMS:
        f[f"src_{d}"] = f.idx.map(orig[f"label_{d}"])
    for extra in ("meaning", "profiling"):
        path = OUTPUTS / model / f"{extra}-{name}.csv"
        if path.exists():
            e = pd.read_csv(path)
            if extra == "meaning":
                f = pd.concat([f, e], axis=1)
            else:
                for k in KEY_FEATURES:
                    if k in e and f"prof_{k}" in orig:
                        f[f"delta_{k}"] = e[k] - f.idx.map(orig[f"prof_{k}"])
    return f


def meaning_block(f: pd.DataFrame) -> dict:
    res = {k: round(float(f[k].mean()), 4) for k in ("bertscore_f1", "sbert_cosine", "copy", "empty") if k in f}
    deltas = {k[6:]: round(float(f[k].mean()), 3) for k in f.columns if k.startswith("delta_")}
    if deltas:
        res["profile_change"] = deltas
    return res


def main_results(model: str, orig: pd.DataFrame) -> dict:
    out = {}
    for dim in DIMS:
        f = load(model, dim, orig)
        if f is None:
            continue
        f = f[f.level < level_of(f[f"src_{dim}"])]          # the requested grid (CLiC-it outputs cover 0-90)
        f["achieved"] = f[f"out_{dim}"]

        def block(sub: pd.DataFrame) -> dict:
            return {**control_metrics(sub.level, sub.achieved),
                    "moved_requested": round(float((sub[f"out_{dim}"] - sub[f"src_{dim}"]).mean()), 2),
                    "moved_free": {d: round(float((sub[f"out_{d}"] - sub[f"src_{d}"]).mean()), 2) for d in DIMS if d != dim},
                    **meaning_block(sub)}

        within = [stats.spearmanr(g.level, g.achieved).statistic for _, g in f.dropna(subset=["achieved"]).groupby("idx")
                  if g.level.nunique() > 2 and g.achieved.nunique() > 1]
        out[dim] = {"requests": int(len(f)), "unscored": int(f.achieved.isna().sum()),
                    "all": {**block(f), "within_original_spearman": round(float(np.mean(within)), 3) if within else None},
                    "by_level": {int(lv): block(s) for lv, s in f.groupby("level")}}
    return out


def joint_results(model: str, orig: pd.DataFrame) -> dict | None:
    f = load(model, "joint", orig)
    if f is None:
        return None

    def block(sub: pd.DataFrame) -> dict:
        scored = sub[[f"out_{d}" for d in DIMS]].notna().all(axis=1)
        sub = sub[scored]
        level = {d: sub[f"tgt_{d}"].astype(int) for d in DIMS}
        inside = {d: (sub[f"out_{d}"] >= level[d]) & (sub[f"out_{d}"] < level[d] + 10) for d in DIMS}
        near = {d: (sub[f"out_{d}"] >= level[d] - 10) & (sub[f"out_{d}"] < level[d] + 20) for d in DIMS}
        return {"requests": int(len(scored)), "scored": int(scored.sum()),
                "per_dimension": {d: control_metrics(level[d], sub[f"out_{d}"]) for d in DIMS},
                "all_three_in_level": round(float(np.logical_and.reduce([inside[d] for d in DIMS]).mean()), 3),
                "all_three_within_one_level": round(float(np.logical_and.reduce([near[d] for d in DIMS]).mean()), 3),
                **meaning_block(sub)}

    return {"all": block(f), **{t: block(g) for t, g in f.groupby("type")}}


def main() -> None:
    orig = originals()
    summary = {"scorer_check": scorer_check(), "models": {}}
    for model, label in MODELS.items():
        res = {"label": label, "main": main_results(model, orig), "joint": joint_results(model, orig)}
        if res["main"] or res["joint"]:
            summary["models"][model] = res
            for dim, r in res["main"].items():
                a = r["all"]
                print(f"{label:28s} {dim:8s} n={r['requests']:5d} RMSE {a.get('rmse')} in-level {a.get('in_level')} bias {a.get('bias')} "
                      f"rho {a.get('spearman')} within {a.get('within_original_spearman')} moved {a.get('moved_requested')} "
                      f"free {a.get('moved_free')} | BERTScore {a.get('bertscore_f1')} SBERT {a.get('sbert_cosine')} copy {a.get('copy')}")
            if res["joint"]:
                j = res["joint"]["all"]
                print(f"{label:28s} joint    n={j['requests']} all three in level {j['all_three_in_level']} "
                      f"within one level {j['all_three_within_one_level']} | "
                      + ", ".join(f"{d} in-level {j['per_dimension'][d].get('in_level')}" for d in DIMS))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["scorer_check"], indent=1))


if __name__ == "__main__":
    main()
