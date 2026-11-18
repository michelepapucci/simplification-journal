"""Results of the g1 models on the test requests (data/g1/test_requests_g1.csv): every READ-IT global level below
each test original's own, 1,000 originals, 7,582 requests.

Conditions: g1 (plain); g1 with best-of-k selection (k = 2, 3, 4: the plain output and 3 more samples, the one
whose READ-IT global is closest to the middle of the requested level); g1_balanced; g1_guided without guidance
(w = 1) and with the weight chosen on the dev requests (results/g1_guided/guidance.json).

For each condition, over all requests, per requested level and per original level (70 / 80):
- control: RMSE between the requested level and the achieved READ-IT global (x100; the CLiC-it convention,
  achieved minus the level's lower bound, so an output in the middle of the level scores 5), bias, share inside the
  requested level and within one level of it, mean and quartiles of the achieved score; Spearman between requested
  and achieved, pooled and within original (mean over originals), and the within-original slope (achieved points
  per requested point, mean over originals);
- where outputs land: the achieved level's distribution for every requested level, and the share at or above the
  original's own level;
- movement of READ-IT base, lexical and syntax (output minus original);
- meaning and quality (meaning.py): BERTScore F1 and SBERT cosine with the original, copies, empty, degenerate,
  echo, too long; LREC reference metrics (SARI, BLEU, BERTScore against the references at the requested level);
- profile changes on key Profiling-UD features (output minus original, both profiled by tools/profiling).
Differences from g1 come with 95% paired bootstrap intervals over test originals (B = 2,000).

    python experiments/analyze.py      ->  results/summary.json, results/summary.md
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

from common import G1, OUTPUTS, RESULTS, RUNS

B, SEED = 2000, 42
KEY_FEATURES = ["n_sentences", "n_tokens", "char_per_tok", "in_dict", "lexical_density", "avg_max_depth",
                "verbal_head_per_sent", "subordinate_proposition_dist"]
MEANING = ["bertscore_f1", "sbert_cosine", "copy", "empty", "degenerate", "echo", "too_long"]
BEST_OF = (2, 3, 4)


def originals() -> pd.DataFrame:
    o = pd.read_csv(G1 / "test_set_g1.csv").set_index("original_sentence_idx")
    for d in ("all", "base", "lexical", "syntax"):
        o[f"label_{d}"] = 100 * o[f"original_{d}"]
    prof = OUTPUTS / "originals" / "profiling-test.csv"
    if prof.exists():
        p = pd.read_csv(prof)
        for k in KEY_FEATURES:
            if k in p:
                o[f"prof_{k}"] = p[k].to_numpy()
    return o


def scorer_check() -> dict | None:
    path = OUTPUTS / "originals" / "readit-test.csv"
    if not path.exists():
        return None
    r, t = pd.read_csv(path), pd.read_csv(G1 / "test_set_g1.csv")
    return {d: round(float((np.abs(r[f"readit_{d}"] - t[f"original_{d}"]) < 1e-9).mean()), 4) for d in ("all", "base", "lexical", "syntax")}


def frame(model: str, name: str, orig: pd.DataFrame) -> pd.DataFrame | None:
    """One row per generation: request, achieved READ-IT (x100), meaning, profile deltas."""
    folder = OUTPUTS / model
    paths = {k: folder / f"{k}-{name}.csv" for k in ("predictions", "readit", "meaning", "profiling")}
    if name.startswith("samples-"):
        paths["predictions"] = folder / f"{name}.csv"
    if not (paths["predictions"].exists() and paths["readit"].exists()):
        return None
    f = pd.read_csv(paths["predictions"])
    rd = pd.read_csv(paths["readit"])
    for d in ("all", "base", "lexical", "syntax"):
        f[f"out_{d}"] = 100 * rd[f"readit_{d}"].to_numpy()
        f[f"delta_{d}"] = f[f"out_{d}"] - f.original_sentence_idx.map(orig[f"label_{d}"]).to_numpy()
    if paths["meaning"].exists():
        m = pd.read_csv(paths["meaning"])
        for k in MEANING:
            f[k] = m[k].to_numpy()
    if paths["profiling"].exists():
        p = pd.read_csv(paths["profiling"])
        for k in KEY_FEATURES:
            if k in p and f"prof_{k}" in orig:
                f[f"pdelta_{k}"] = p[k].to_numpy() - f.original_sentence_idx.map(orig[f"prof_{k}"]).to_numpy()
    return f


def best_of(main: pd.DataFrame, samples: pd.DataFrame, k: int) -> pd.DataFrame:
    """For every request, among the plain output and the first k-1 extra samples, the output whose READ-IT global
    is closest to the middle of the requested level (unscored outputs never chosen)."""
    cand = pd.concat([main.assign(sample=0), samples[samples["sample"] < k]], ignore_index=True)
    cand["dist"] = (cand.out_all - (cand.level + 5)).abs()
    cand = cand.sort_values(["request", "dist", "sample"], na_position="last")
    return cand.drop_duplicates("request").sort_values("request").drop(columns="dist").reset_index(drop=True)


def per_original(f: pd.DataFrame) -> pd.DataFrame:
    """Per-original pieces for pooled metrics and the bootstrap."""
    g = f.assign(err2=(f.out_all - f.level) ** 2, inside=(f.out_all >= f.level) & (f.out_all < f.level + 10),
                 scored=f.out_all.notna())
    rows = []
    for i, x in g.groupby("original_sentence_idx"):
        s = x[x.scored]
        rho = slope = np.nan
        if len(s) >= 3 and s.out_all.nunique() > 1:
            rho = stats.spearmanr(s.level, s.out_all).statistic
            slope = np.polyfit(s.level, s.out_all, 1)[0]
        rows.append({"original": i, "n": len(s), "sse": float(s.err2.sum()), "inside": float(s.inside.sum()), "rho": rho,
                     "slope": slope, "bert_sum": float(x.bertscore_f1.sum()) if "bertscore_f1" in x else np.nan,
                     "bert_n": len(x)})
    return pd.DataFrame(rows).set_index("original")


def pooled(po: pd.DataFrame) -> dict:
    return {"rmse": float(np.sqrt(po.sse.sum() / po.n.sum())), "in_level": float(po.inside.sum() / po.n.sum()),
            "rho_within": float(np.nanmean(po.rho)), "slope_within": float(np.nanmean(po.slope)),
            "bertscore_f1": float(po.bert_sum.sum() / po.bert_n.sum())}


def bootstrap_diff(po: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """95% paired bootstrap intervals of (condition - g1), resampling test originals."""
    ids = ref.index.intersection(po.index)
    draws = np.random.default_rng(SEED).integers(0, len(ids), size=(B, len(ids)))

    def metrics(frame: pd.DataFrame) -> dict:
        x = {k: frame[k].to_numpy(dtype=float)[draws] for k in ("n", "sse", "inside", "rho", "slope", "bert_sum", "bert_n")}
        return {"rmse": np.sqrt(x["sse"].sum(1) / x["n"].sum(1)), "in_level": x["inside"].sum(1) / x["n"].sum(1),
                "rho_within": np.nanmean(x["rho"], axis=1), "slope_within": np.nanmean(x["slope"], axis=1),
                "bertscore_f1": x["bert_sum"].sum(1) / x["bert_n"].sum(1)}

    a, r = po.loc[ids], ref.loc[ids]
    da, dr, pa, pr = metrics(a), metrics(r), pooled(a), pooled(r)
    out = {}
    for key in da:
        lo, hi = np.percentile(da[key] - dr[key], [2.5, 97.5])
        out[key] = {"diff": round(pa[key] - pr[key], 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    return out


def control(f: pd.DataFrame) -> dict:
    s = f[f.out_all.notna()]
    if not len(s):
        return {"scored": 0}
    err = s.out_all - s.level
    q = s.out_all.quantile([.25, .5, .75])
    ach_level = np.minimum(np.floor(s.out_all / 10), 9) * 10
    res = {"requests": int(len(f)), "scored": int(len(s)), "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
           "bias": round(float(err.mean()), 2),
           "in_level": round(float(((s.out_all >= s.level) & (s.out_all < s.level + 10)).mean()), 3),
           "within_one_level": round(float((np.abs(ach_level - s.level) <= 10).mean()), 3),
           "achieved_mean": round(float(s.out_all.mean()), 2), "achieved_quartiles": [round(float(x), 2) for x in q],
           "at_or_above_original_level": round(float((ach_level >= s.src_level).mean()), 3)}
    if s.level.nunique() > 1:
        res["spearman_pooled"] = round(float(stats.spearmanr(s.level, s.out_all).statistic), 3)
    return res


def describe(f: pd.DataFrame) -> dict:
    res = {"control": control(f)}
    res["readit_change"] = {d: round(float(f[f"delta_{d}"].mean()), 2) for d in ("all", "base", "lexical", "syntax")}
    res["meaning"] = {k: round(float(f[k].mean()), 4) for k in MEANING if k in f}
    prof = {k[7:]: round(float(f[k].mean()), 3) for k in f.columns if k.startswith("pdelta_")}
    if prof:
        res["profile_change"] = prof
    return res


def landing(f: pd.DataFrame) -> dict:
    s = f[f.out_all.notna()]
    ach = (np.minimum(np.floor(s.out_all / 10), 9) * 10).astype(int)
    tab = pd.crosstab(s.level, ach, normalize="index").round(4)
    return {int(lv): {int(a): float(v) for a, v in row.items() if v > 0} for lv, row in tab.iterrows()}


def condition(f: pd.DataFrame) -> dict:
    po = per_original(f)
    res = {"all": describe(f), "within_original": {k: round(v, 3) for k, v in pooled(po).items() if k in ("rho_within", "slope_within")}}
    res["by_level"] = {int(lv): describe(x) for lv, x in f.groupby("level")}
    res["by_original_level"] = {int(lv): {**describe(x), "within_original": {k: round(v, 3) for k, v in pooled(per_original(x)).items()
                                                                               if k in ("rho_within", "slope_within")}}
                                for lv, x in f.groupby("src_level")}
    res["by_original_and_level"] = {int(src): {int(lv): control(y) for lv, y in x.groupby("level")} for src, x in f.groupby("src_level")}
    res["landing"] = landing(f)
    return res, po


def main() -> None:
    orig = originals()
    conditions = {}
    g1 = frame("g1", "test", orig)
    if g1 is not None:
        conditions["g1"] = ("g1", g1)
        samples = frame("g1", "samples-test", orig)
        if samples is not None:
            for k in BEST_OF:
                conditions[f"g1_best_of_{k}"] = (f"g1, best of {k}", best_of(g1, samples, k))
    for model, label in (("g1_balanced", "g1 · balanced"), ("g1_guided", "g1 · guided, w = 1")):
        f = frame(model, "test", orig)
        if f is not None:
            conditions[model] = (label, f)
    guidance = RESULTS / "g1_guided" / "guidance.json"
    if guidance.exists():
        w = json.loads(guidance.read_text())["chosen"]
        if w != 1:
            f = frame("g1_guided", f"test-w{w:g}", orig)
            if f is not None:
                conditions[f"g1_guided_w{w:g}"] = (f"g1 · guided, w = {w:g}", f)

    summary = {"scorer_check_originals": scorer_check(), "conditions": {}, "training": {}, "guidance": None}
    for run in ("g1", "g1_balanced", "g1_guided"):
        t = RUNS / run / "training.json"
        if t.exists():
            j = json.loads(t.read_text())
            summary["training"][run] = {k: j[k] for k in ("train_pairs", "examples_per_epoch", "examples_per_level_per_epoch",
                                                          "steps", "hours", "null_share", "balance")}
            summary["training"][run]["final_eval_loss"] = round(j["final_eval"]["eval_loss"], 4)
    if guidance.exists():
        summary["guidance"] = json.loads(guidance.read_text())
    pos = {}
    for key, (label, f) in conditions.items():
        res, po = condition(f)
        pos[key] = po
        refs = RESULTS / key.split("_best_of_")[0].split("_w")[0] / "reference_metrics.json"
        name = f"test-w{key.rsplit('_w', 1)[1]}" if key.startswith("g1_guided_w") else "test"
        if refs.exists() and "best_of" not in key:
            res["reference_metrics"] = json.loads(refs.read_text()).get(name)
        summary["conditions"][key] = {"label": label, **res}
        print(f"{label:24s}", res["all"]["control"], res["within_original"], res["all"]["meaning"], flush=True)
    if "g1" in pos:
        for key, po in pos.items():
            if key != "g1":
                summary["conditions"][key]["vs_g1"] = bootstrap_diff(po, pos["g1"])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    (RESULTS / "summary.md").write_text(markdown(summary))
    print(f"wrote {RESULTS / 'summary.json'}")


def markdown(s: dict) -> str:
    c = s["conditions"]
    lines = ["| Condition | RMSE | Inside | Within one level | Bias | ρ within | Slope within | BERTScore | Copies | Degenerate |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in c.items():
        a, w, m = r["all"]["control"], r["within_original"], r["all"]["meaning"]
        lines.append(f"| {r['label']} | {a['rmse']} | {100 * a['in_level']:.1f}% | {100 * a['within_one_level']:.1f}% | {a['bias']} | "
                     f"{w['rho_within']} | {w['slope_within']} | {m.get('bertscore_f1', '–')} | "
                     f"{100 * m.get('copy', float('nan')):.1f}% | {100 * m.get('degenerate', float('nan')):.1f}% |")
    lines += ["", "Mean achieved READ-IT global by requested level:", "",
              "| Condition | " + " | ".join(str(lv) for lv in range(0, 80, 10)) + " |", "|---|" + "---|" * 8]
    for key, r in c.items():
        lines.append(f"| {r['label']} | " + " | ".join(f"{r['by_level'][lv]['control']['achieved_mean']:.1f}" if lv in r["by_level"] else "–"
                                                       for lv in range(0, 80, 10)) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
