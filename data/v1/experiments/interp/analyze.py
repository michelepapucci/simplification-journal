"""Analyse the interpretability run of one model (questions 1 and 2 of
reports/improving_controllability.md, Section 5.2).

Q1, does each token change *what* is edited?
- per Profiling-UD feature and edit operation: outputs at 3 levels below the original vs 1 level below,
  same original, same mode (paired Wilcoxon, rank-biserial r; positive = larger at the stronger request);
- how similar the three tokens' profiles are (Spearman across features of their r);
- how often two requests give the very same text;
- whether the moved token can be told from the output's change alone (logistic regression, grouped
  5-fold cross-validation by original), with only the change in the overall READ-IT score as baseline;
- two tokens at the same distance from the original (1-3 levels below), same sentence: which features differ;
- outputs of two tokens matched on the overall READ-IT score (within 2 points, same original, neither a
  copy of the original): which features still differ.
Q2, are the three dimensions controlled separately?
- the specificity matrix: within-original slope of each achieved READ-IT score on the moved token's
  requested level (points per point; diagonal 1 and off-diagonal 0 would be separate control), 95%
  bootstrap intervals over originals, and the same slopes in the training data as reference;
- the tokens' level directions: in the trained input embeddings, and in the residual stream at the last
  prompt token, layer by layer (cosines; split-half readout of each token's level along each direction).

    python data/v1/experiments/interp/analyze.py --model v5_targets
Reads outputs/<model>/interp/, writes results/interp_<model>.json and results/interp_<model>.md.
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3] / "scripts"))
from common import DIMS, OUTPUTS, RESULTS, ROOT  # noqa: E402
from generate import originals  # noqa: E402

SCORES = ["readit_base", "readit_lexical", "readit_syntax", "readit_all"]
EDITS = ["len_ratio", "kept", "deleted", "replaced", "inserted"]
KEY = ["n_tokens", "n_sentences", "tokens_per_sent", "char_per_tok", "in_dict", "lexical_density",
       "upos_dist_PUNCT", "subordinate_proposition_dist", "avg_max_depth", "avg_links_len", "verbal_head_per_sent"]
MODES = ("free", "keep")
PAIRS = [("base", "lexical"), ("base", "syntax"), ("lexical", "syntax")]
SEED, B = 0, 2000
TOKEN = re.compile(r"\w+|[^\w\s]")
DATA = {"v5_targets": ROOT / "data" / "v5" / "training_set_v5.zip", "v3_targets": ROOT / "data" / "v3" / "training_set_v3.zip",
        "v7_targets": ROOT / "data" / "v7" / "training_set_v7.zip",
        "v7_1ep_targets": ROOT / "data" / "v7" / "training_set_v7.zip"}


def sig(x, digits=3):
    if x is None or not np.isfinite(x):
        return None
    x = float(x)
    return 0.0 if x == 0 else round(x, digits - 1 - int(np.floor(np.log10(abs(x)))))


def edit_ops(original: str, output: str) -> list[float]:
    a, b = TOKEN.findall(original.lower()), TOKEN.findall(output.lower())
    if not a or not b:
        return [np.nan] * len(EDITS)
    ops = {"equal": 0, "delete": 0, "replace": 0, "insert": 0}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        ops[tag] += (i2 - i1) if tag != "insert" else (j2 - j1)
    return [len(b) / len(a), ops["equal"] / len(a), ops["delete"] / len(a), ops["replace"] / len(a), ops["insert"] / len(b)]


def load(folder: Path) -> tuple[pd.DataFrame, list[str]]:
    pred = pd.read_csv(folder / "predictions.csv", keep_default_na=False, dtype={f"tgt_{d}": str for d in DIMS})
    texts = pd.read_csv(folder / "texts.csv", keep_default_na=False)
    readit = pd.read_csv(folder / "readit-texts.csv")
    prof = pd.read_csv(folder / "profiling-texts.csv")
    assert (prof.row.to_numpy() == np.arange(len(texts))).all() and len(readit) == len(texts)
    readit[SCORES] = readit[SCORES] * 100
    features = [c for c in prof.columns if c != "row" and prof[c].nunique() > 1]
    table = pd.concat([readit[SCORES], prof[features]], axis=1)
    row = dict(zip(texts.text, range(len(texts))))
    text = originals()
    pred["original_text"] = pred.idx.map(text)
    out = table.reindex(pred.prediction.map(row).to_numpy()).reset_index(drop=True)
    orig = table.reindex(pred.original_text.map(row).to_numpy()).reset_index(drop=True)
    edits = pd.DataFrame([edit_ops(o, p) for o, p in zip(pred.original_text, pred.prediction)], columns=EDITS)
    df = pd.concat([pred, out, orig.add_prefix("orig_"), edits], axis=1)
    df["copy"] = (df.prediction == df.original_text).astype(float)
    return df, features


# ---------------------------------------------------------------- Q1

def rank_biserial(d: np.ndarray) -> tuple:
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    if len(nz) < 10:
        return None, None, int(len(nz))
    ranks = stats.rankdata(np.abs(nz))
    r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    return sig(r), sig(stats.wilcoxon(nz).pvalue, 2), int(len(nz))


def bh(p: list) -> list:
    """Benjamini-Hochberg q-values; None stays None."""
    idx = [i for i, v in enumerate(p) if v is not None]
    q = [None] * len(p)
    if not idx:
        return q
    pv = np.array([p[i] for i in idx])
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    for k, j in enumerate(order):
        q[idx[j]] = sig(min(ranked[k], 1.0), 2)
    return q


def contrast(df: pd.DataFrame, mode: str, dim: str, cols: list[str], hi: int = 3, lo: int = 1) -> dict:
    sub = df[(df["mode"] == mode) & (df.dimension == dim) & df["drop"].isin([lo, hi])]
    wide = sub.pivot_table(index="idx", columns="drop", values=cols, aggfunc="first")
    out = {"r": [], "p": [], "n_diff": [], "mean_diff": []}
    for c in cols:
        d = (wide[(c, hi)] - wide[(c, lo)]).to_numpy(float)
        r, p, n = rank_biserial(d)
        out["r"].append(r), out["p"].append(p), out["n_diff"].append(n)
        out["mean_diff"].append(sig(np.nanmean(d)))
    out["q"] = bh(out["p"])
    return out


def curves(df: pd.DataFrame, mode: str, dim: str, cols: list[str], min_n: int = 100) -> dict:
    """Within-original change from the output 1 level below the original, by levels below."""
    sub = df[(df["mode"] == mode) & (df.dimension == dim)]
    base = sub[sub["drop"] == 1].set_index("idx")[cols]
    res = {"drop": [], "n": [], "at1": {c: sig(base[c].mean()) for c in cols}, **{c: [] for c in cols}}
    for k in sorted(sub["drop"].unique()):
        at = sub[sub["drop"] == k].set_index("idx")[cols]
        diff = (at - base.reindex(at.index)).dropna(how="all")
        if len(diff) < min_n:
            continue
        res["drop"].append(int(k)), res["n"].append(int(len(diff)))
        for c in cols:
            res[c].append(sig(diff[c].mean()))
    return res


def identical(df: pd.DataFrame, mode: str) -> dict:
    sub = df[df["mode"] == mode]
    res = {"adjacent": {}, "across": {}, "copies": {}}
    for d in DIMS:
        s = sub[sub.dimension == d].sort_values(["idx", "level"])
        same = (s.prediction.values[1:] == s.prediction.values[:-1]) & (s.idx.values[1:] == s.idx.values[:-1])
        res["adjacent"][d] = sig(same.sum() / (s.idx.values[1:] == s.idx.values[:-1]).sum())
        res["copies"][d] = {"all": sig(s["copy"].mean()), "drop1": sig(s[s["drop"] == 1]["copy"].mean())}
    for a, b in PAIRS:
        res["across"][f"{a}-{b}"] = {}
        for k in (1, 2, 3):
            x = sub[(sub["drop"] == k) & (sub.dimension == a)].set_index("idx").prediction
            y = sub[(sub["drop"] == k) & (sub.dimension == b)].set_index("idx").prediction
            res["across"][f"{a}-{b}"][k] = sig((x == y.reindex(x.index)).mean())
    return res


def classifier(df: pd.DataFrame, mode: str, features: list[str]) -> dict:
    sub = df[(df["mode"] == mode) & df["drop"].isin([1, 2, 3])]
    cs = [*SCORES, *features]
    delta = pd.DataFrame(sub[cs].to_numpy(float) - sub[[f"orig_{c}" for c in cs]].to_numpy(float), columns=[f"d_{c}" for c in cs], index=sub.index)
    sub = pd.concat([sub, delta], axis=1)
    sets = {
        "overall READ-IT change": ["d_readit_all", "drop"],
        "three READ-IT changes": [f"d_{c}" for c in SCORES] + ["drop"],
        "profile change + edits": [f"d_{c}" for c in features] + EDITS + ["drop"],
        "all of the above": [f"d_{c}" for c in [*SCORES, *features]] + EDITS + ["drop"],
    }
    sub = sub.dropna(subset=sets["all of the above"])
    y, groups = sub.dimension.to_numpy(), sub.idx.to_numpy()
    res = {"n": int(len(sub)), "chance": sig(pd.Series(y).value_counts(normalize=True).max()), "accuracy": {}}
    for name, cols in sets.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        guess = cross_val_predict(model, sub[cols].to_numpy(float), y, groups=groups, cv=GroupKFold(5))
        res["accuracy"][name] = sig((guess == y).mean())
        if name == "all of the above":
            res["confusion"] = {t: {g: int(((y == t) & (guess == g)).sum()) for g in DIMS} for t in DIMS}
    return res


def matched(df: pd.DataFrame, mode: str, cols: list[str], tol: float = 2.0) -> dict:
    """For each original and pair of tokens, the two outputs (any levels) closest in overall READ-IT."""
    sub = df[(df["mode"] == mode) & (df["drop"] >= 1)].dropna(subset=["readit_all"])
    res = {}
    for a, b in PAIRS:
        diffs, gaps = [], []
        for _, g in sub[sub.dimension.isin([a, b])].groupby("idx"):
            x, y = g[g.dimension == a], g[g.dimension == b]
            if x.empty or y.empty:
                continue
            gap = np.abs(x.readit_all.to_numpy()[:, None] - y.readit_all.to_numpy()[None, :])
            gap[x.prediction.to_numpy()[:, None] == y.prediction.to_numpy()[None, :]] = np.inf  # the same text
            i, j = np.unravel_index(np.argmin(gap), gap.shape)
            if gap[i, j] <= tol:
                diffs.append(x.iloc[i][cols].to_numpy(float) - y.iloc[j][cols].to_numpy(float))
                gaps.append(x.iloc[i].readit_all - y.iloc[j].readit_all)
        D = np.array(diffs)
        stats_ = [rank_biserial(D[:, k]) for k in range(len(cols))]
        res[f"{a}-{b}"] = {"n": len(D), "mean_gap": sig(np.mean(gaps)), "r": [s[0] for s in stats_], "p": [s[1] for s in stats_],
                           "q": bh([s[1] for s in stats_]), "mean_diff": [sig(np.nanmean(D[:, k])) for k in range(len(cols))]}
    return res


def same_distance(df: pd.DataFrame, mode: str, cols: list[str], drops=(1, 2, 3)) -> dict:
    """Two tokens at the same distance from the original, same sentence: token a's output minus token b's,
    averaged over 1, 2 and 3 levels below the original (one value per sentence), paired Wilcoxon."""
    sub = df[(df["mode"] == mode) & df["drop"].isin(drops)]
    res = {}
    for a, b in PAIRS:
        X = sub[sub.dimension == a].set_index(["idx", "drop"])
        Y = sub[sub.dimension == b].set_index(["idx", "drop"]).reindex(X.index)
        D = (X[cols].astype(float) - Y[cols].astype(float)).groupby(level=0).mean()
        st = [rank_biserial(D[c].to_numpy(float)) for c in cols]
        res[f"{a}-{b}"] = {"n": int(len(D)), "same_text": sig((X.prediction == Y.prediction).mean()),
                           "r": [x[0] for x in st], "p": [x[1] for x in st], "q": bh([x[1] for x in st]),
                           "mean_diff": [sig(D[c].mean()) for c in cols]}
    return res


# ---------------------------------------------------------------- Q2

def slopes(df: pd.DataFrame, mode: str) -> dict:
    """Fixed-effects slope (within original) of each achieved score on the moved token's level."""
    rng = np.random.default_rng(SEED)
    res = {}
    for d in DIMS:
        sub = df[(df["mode"] == mode) & (df.dimension == d) & (df["drop"] >= 1)].dropna(subset=SCORES)
        g = sub.groupby("idx")
        L = (sub["level"] - g["level"].transform("mean")).to_numpy(float)
        ids = sub.idx.to_numpy()
        uniq, code = np.unique(ids, return_inverse=True)
        sxx = np.bincount(code, L * L)
        boot = rng.integers(0, len(uniq), size=(B, len(uniq)))
        res[d] = {}
        for e in SCORES:
            y = (sub[e] - g[e].transform("mean")).to_numpy(float)
            sxy = np.bincount(code, L * y)
            b = sxy[boot].sum(1) / sxx[boot].sum(1)
            res[d][e] = {"slope": sig(sxy.sum() / sxx.sum()), "lo": sig(np.quantile(b, 0.025)), "hi": sig(np.quantile(b, 0.975))}
    return res


def data_slopes(model: str) -> dict | None:
    path = DATA.get(model)
    if path is None or not path.exists():
        return None
    cols = [f"{p}_{d}" for p in ("original", "simplification") for d in [*DIMS, "all"]]
    header = pd.read_csv(path, nrows=0).columns
    extra = ["pair_row"] if "pair_row" in header else []  # dataset v7: one row per example; pair_row marks one per pair
    t = pd.read_csv(path, usecols=["split", "subset", "original_text", "simplification", *cols, *extra], low_memory=False)
    if extra:
        t = t[t.pair_row.astype(str).str.lower().isin(["true", "1"])]
    # the natural IMPaCTS pairs: identity pairs and the lexical supplement (selected for large lexical drops) excluded
    t = t[(t.split == "train") & (t.subset == "v1") & (t.original_text != t.simplification)]
    delta = {d: (t[f"simplification_{d}"] - t[f"original_{d}"]).to_numpy() * 100 for d in [*DIMS, "all"]}
    return {d: {f"readit_{e}": sig(np.cov(delta[d], delta[e])[0, 1] / np.var(delta[d], ddof=1)) for e in [*DIMS, "all"]} for d in DIMS}


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def embedding_directions(folder: Path) -> dict:
    z = np.load(folder / "embeddings.npz")
    tokens, E = list(z["tokens"]), z["input"].astype(float)
    levels = np.arange(0, 100, 10, dtype=float)
    rng = np.random.default_rng(SEED)

    def direction(d, perm=None):
        X = np.stack([E[tokens.index(f"<|readability_{d}_{int(l)}|>")] for l in levels])
        lv = levels if perm is None else levels[perm]
        lc = lv - lv.mean()
        return unit(lc @ (X - X.mean(0)))

    def pc1_rho(d):
        X = np.stack([E[tokens.index(f"<|readability_{d}_{int(l)}|>")] for l in levels])
        X = X - X.mean(0)
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        return sig(abs(stats.spearmanr(u[:, 0], levels).statistic)), sig(s[0] ** 2 / (s ** 2).sum())

    dirs = {d: direction(d) for d in DIMS}
    null = [abs(direction("base", rng.permutation(10)) @ direction("syntax", rng.permutation(10))) for _ in range(2000)]
    return {"cos": {f"{a}-{b}": sig(dirs[a] @ dirs[b]) for a, b in PAIRS}, "null95": sig(np.quantile(null, 0.95)),
            "pc1": {d: dict(zip(["rho", "share"], pc1_rho(d))) for d in DIMS}}


def demean(X: np.ndarray, code: np.ndarray, n: int) -> np.ndarray:
    sums = np.zeros((n, X.shape[1]))
    np.add.at(sums, code, X)
    return X - (sums / np.bincount(code, minlength=n)[:, None])[code]


def activation_directions(df: pd.DataFrame, folder: Path, mode: str) -> dict:
    states = np.load(folder / "states.npy", mmap_mode="r")
    rng = np.random.default_rng(SEED)
    originals_ = np.array(sorted(df.idx.unique()))
    half = set(rng.permutation(originals_)[: len(originals_) // 2])
    rows, L, H = {}, {}, {}
    for d in DIMS:
        sub = df[(df["mode"] == mode) & (df.dimension == d) & (df["drop"] >= 1)]
        rows[d] = sub.index.to_numpy()
        _, code = np.unique(sub.idx.to_numpy(), return_inverse=True)
        lc = sub.level.to_numpy(float)
        L[d] = lc - (np.bincount(code, lc) / np.bincount(code))[code]
        H[d] = (code, code.max() + 1, sub.idx.isin(half).to_numpy())
    res = {"layer": [], "cos": {f"{a}-{b}": [] for a, b in PAIRS}, "reliability": {d: [] for d in DIMS},
           "readout": {f"{a}>{b}": [] for a in DIMS for b in DIMS}, "norm": {d: [] for d in DIMS}}
    for layer in range(states.shape[1]):
        X = {d: demean(states[rows[d], layer, :].astype(np.float64), H[d][0], H[d][1]) for d in DIMS}
        beta = {d: L[d] @ X[d] / (L[d] @ L[d]) for d in DIMS}
        beta_a = {d: L[d][H[d][2]] @ X[d][H[d][2]] / (L[d][H[d][2]] @ L[d][H[d][2]]) for d in DIMS}
        beta_b = {d: L[d][~H[d][2]] @ X[d][~H[d][2]] / (L[d][~H[d][2]] @ L[d][~H[d][2]]) for d in DIMS}
        res["layer"].append(layer)
        if min(np.linalg.norm(beta[d]) for d in DIMS) < 1e-6:  # the embedding of the last prompt token: the same for every request
            for k in ("cos", "reliability", "readout", "norm"):
                for v in res[k].values():
                    v.append(None)
            continue
        for a, b in PAIRS:
            res["cos"][f"{a}-{b}"].append(sig(unit(beta[a]) @ unit(beta[b])))
        for d in DIMS:
            res["reliability"][d].append(sig(unit(beta_a[d]) @ unit(beta_b[d])))
            res["norm"][d].append(sig(np.linalg.norm(beta[d]) * 10))  # per level (10 points)
        for a in DIMS:      # direction of token a, estimated on one half of the originals ...
            for b in DIMS:  # ... read on token b's requests in the other half
                rs = []
                for est, test in ((beta_a, ~H[b][2]), (beta_b, H[b][2])):
                    proj = X[b][test] @ unit(est[a])
                    rs.append(np.corrcoef(proj, L[b][test])[0, 1])
                res["readout"][f"{a}>{b}"].append(sig(np.mean(rs)))
    return res


def examples(df: pd.DataFrame, n: int = 8) -> list[dict]:
    """A few originals with every keep and free output, for the page."""
    rng = np.random.default_rng(SEED)
    picks = rng.choice(sorted(df.idx.unique()), size=n, replace=False)
    out = []
    for i in picks:
        g = df[df.idx == i]
        out.append({"idx": int(i), "original": g.original_text.iloc[0], "src": {d: int(g[f"src_{d}"].iloc[0]) for d in DIMS},
                    "outputs": [{"mode": r.mode, "dim": r.dimension, "level": int(r.level), "text": r.prediction,
                                 "scores": [sig(getattr(r, s)) for s in SCORES]} for r in g.itertuples(index=False)]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="v5_targets")
    parser.add_argument("--folder", type=Path, help="default: outputs/<model>/interp")
    parser.add_argument("--results", type=Path, default=RESULTS, help="default: results/")
    args = parser.parse_args()
    folder = args.folder or OUTPUTS / args.model / "interp"
    df, features = load(folder)
    cols = [*SCORES, *features, *EDITS, "copy"]
    res = {"model": args.model, "n_originals": int(df.idx.nunique()), "n_requests": int(len(df)),
           "n_distinct_outputs": int(df.prediction.nunique()), "unscored": int(df.readit_all.isna().sum()),
           "features": cols, "key": [*SCORES, *KEY, *EDITS, "copy"], "q1": {}, "q2": {}}
    res["q2"]["data"] = data_slopes(args.model)
    res["q2"]["embeddings"] = embedding_directions(folder)
    for mode in MODES:
        res["q1"][mode] = {"contrast": {d: contrast(df, mode, d, cols) for d in DIMS},
                           "curves": {d: curves(df, mode, d, res["key"]) for d in DIMS},
                           "identical": identical(df, mode),
                           "classifier": classifier(df, mode, features),
                           # without copies of the original: a copy is the same text whichever token produced it
                           "classifier_noncopy": classifier(df[df["copy"] == 0], mode, features),
                           "matched": matched(df[df["copy"] == 0], mode, cols),
                           "same_distance": same_distance(df, mode, cols)}
        r = {d: np.array([np.nan if v is None else v for v in res["q1"][mode]["contrast"][d]["r"]]) for d in DIMS}
        res["q1"][mode]["profile_rho"] = {}
        for a, b in PAIRS:
            ok = np.isfinite(r[a]) & np.isfinite(r[b])
            res["q1"][mode]["profile_rho"][f"{a}-{b}"] = sig(stats.spearmanr(r[a][ok], r[b][ok]).statistic)
        res["q2"][mode] = {"slopes": slopes(df, mode), "activations": activation_directions(df, folder, mode)}
        print(f"{mode}: done", flush=True)
    res["anchor"] = {"copies": sig(df[df["mode"] == "anchor"]["copy"].mean()),
                     "scores": {s: sig((df[df["mode"] == "anchor"][s] - df[df["mode"] == "anchor"][f"orig_{s}"]).mean()) for s in SCORES}}
    res["examples"] = examples(df)
    args.results.mkdir(exist_ok=True)
    (args.results / f"interp_{args.model}.json").write_text(json.dumps(res, ensure_ascii=False))
    (args.results / f"interp_{args.model}.md").write_text(report(res))
    print(report(res))


def report(res: dict) -> str:
    lines = [f"# Interpretability, {res['model']}: {res['n_originals']} originals, {res['n_requests']} greedy requests, "
             f"{res['n_distinct_outputs']} distinct outputs, {res['unscored']} unscored", ""]
    short = {"readit_base": "base", "readit_lexical": "lexical", "readit_syntax": "syntax", "readit_all": "global"}
    lines += ["## Q2: specificity matrix (achieved points per requested point, within original)", "",
              "| Mode | Moved token | " + " | ".join(short.values()) + " |", "|---|---|" + "---|" * 4]
    for mode in MODES:
        for d in DIMS:
            s = res["q2"][mode]["slopes"][d]
            lines.append(f"| {mode} | {d} | " + " | ".join(f"{s[e]['slope']} [{s[e]['lo']}, {s[e]['hi']}]" for e in SCORES) + " |")
    if res["q2"]["data"]:
        for d in DIMS:
            lines.append(f"| training data | {d} | " + " | ".join(str(res["q2"]["data"][d][e]) for e in SCORES) + " |")
    e = res["q2"]["embeddings"]
    lines += ["", f"Embedding level directions, cosine: {e['cos']} (95% permutation null |cos| {e['null95']}); PC1: {e['pc1']}", ""]
    for mode in MODES:
        a = res["q2"][mode]["activations"]
        lines.append(f"### Activations, {mode}: layer, cos(b-l, b-s, l-s), reliability(b, l, s), readout diag / base>syntax, syntax>base")
        for k in range(0, len(a["layer"]), 4):
            lines.append(f"- L{a['layer'][k]}: cos {[a['cos'][p][k] for p in a['cos']]}, rel {[a['reliability'][d][k] for d in DIMS]}, "
                         f"diag {[a['readout'][f'{d}>{d}'][k] for d in DIMS]}, b>s {a['readout']['base>syntax'][k]}, s>b {a['readout']['syntax>base'][k]}, "
                         f"b>l {a['readout']['base>lexical'][k]}, l>b {a['readout']['lexical>base'][k]}")
        lines.append("")
    lines += ["## Q1: what is edited", ""]
    for mode in MODES:
        q = res["q1"][mode]
        lines += [f"### {mode}", f"- identical outputs: {q['identical']}", f"- profile rho between tokens: {q['profile_rho']}",
                  f"- classifier: {q['classifier']}", f"- classifier, non-copies: {q['classifier_noncopy']['accuracy']}", ""]
        lines += ["| Feature | " + " | ".join(DIMS) + " |", "|---|---|---|---|"]
        for k, f in enumerate(res["features"]):
            if f in res["key"]:
                lines.append(f"| {f} | " + " | ".join(f"{q['contrast'][d]['r'][k]} ({q['contrast'][d]['mean_diff'][k]})" for d in DIMS) + " |")
        lines.append("")
        for pair, m in q["matched"].items():
            top = sorted([(abs(r), f, r, m["mean_diff"][k]) for k, (f, r) in enumerate(zip(res["features"], m["r"]))
                          if r is not None and m["q"][k] is not None and m["q"][k] < 0.01], reverse=True)[:10]
            lines.append(f"- matched {pair}: n {m['n']}, mean gap {m['mean_gap']}; top: " + "; ".join(f"{f} {r} ({md})" for _, f, r, md in top))
        for pair, m in q["same_distance"].items():
            top = sorted([(abs(r), f, r, m["mean_diff"][k]) for k, (f, r) in enumerate(zip(res["features"], m["r"]))
                          if r is not None and m["q"][k] is not None and m["q"][k] < 0.01], reverse=True)[:10]
            lines.append(f"- same distance {pair}: n {m['n']}, same text {m['same_text']}; top: " + "; ".join(f"{f} {r} ({md})" for _, f, r, md in top))
        lines.append("")
    lines.append(f"Anchor (every dimension at the original's level): {res['anchor']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
