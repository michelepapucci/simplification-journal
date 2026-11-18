"""Analysis of the g1 interpretability runs (see README.md): A level geometry layer by layer, B readout of the outcome,
C the "no level" default, D the probability of De Mauro's basic vocabulary at every decoding step.

    taskset --cpu-list 9-25 python experiments/interp/analyze.py   ->  results/interp_g1.json
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import G1, OUTPUTS, RESULTS, TOKENS  # noqa: E402

INTERP = OUTPUTS / "interp"
MODELS = ("g1_guided", "g1")
LEVELS = list(range(0, 100, 10))
TRAINED = 9                       # levels 0-80; level 90 and (in g1) "any" are untrained
N_CLOUD = 40                      # originals shown per level in the per-layer clouds
CLASSES = ("FO", "AU", "AD", "out")
WORD = re.compile(r"[^\W\d_]+")
SEED = 42


def r(x, d=3):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def geometry(P: np.ndarray) -> dict:
    """P: [11, D] points (levels 0-90, any). Ordering, betweenness, hierarchy and spacing on the 9 trained levels."""
    T = P[:TRAINED]
    c = T.mean(0)
    if np.abs(T - c).max() < 1e-6:          # the token has not reached this position yet (layer 0 in context)
        z = np.zeros((P.shape[0], 2))
        return {"rho_pc1": None, "pc1_share": None, "betweenness": None, "contiguous_merges": None, "cophenetic_rho": None,
                "spacing": [None] * (TRAINED - 1), "any_equivalent_level": None,
                "coords": z[:TRAINED].tolist(), "cosine": [[None] * TRAINED] * TRAINED, "_axes": (c, np.zeros((2, P.shape[1])), 1.0)}
    u, s, vt = np.linalg.svd(T - c, full_matrices=False)
    coords = (P - c) @ vt[:2].T
    pc1 = coords[:TRAINED, 0]
    sign = np.sign(stats.spearmanr(LEVELS[:TRAINED], pc1).statistic or 1)
    coords[:, 0] *= sign
    d = squareform(pdist(T))
    trip = [d[a, cc] >= max(d[a, b], d[b, cc]) - 1e-9 for a in range(TRAINED) for b in range(a + 1, TRAINED) for cc in range(b + 1, TRAINED)]
    Z = linkage(T, "average")
    members = {i: [i] for i in range(TRAINED)}
    contiguous = []
    for k, (a, b, _, _) in enumerate(Z):
        m = sorted(members[int(a)] + members[int(b)])
        members[TRAINED + k] = m
        contiguous.append(m == list(range(m[0], m[-1] + 1)))
    gaps = pdist(np.array(LEVELS[:TRAINED])[:, None])
    coph = stats.spearmanr(cophenet(Z), gaps).statistic
    step = np.array([d[i, i + 1] for i in range(TRAINED - 1)])
    # "no level" (row 10) as a level: its position along PC1, mapped linearly from the trained levels (used in C only)
    slope, icpt = np.polyfit(coords[:TRAINED, 0], LEVELS[:TRAINED], 1)
    Tc = T - c
    cos = (Tc / np.linalg.norm(Tc, axis=1, keepdims=True)) @ (Tc / np.linalg.norm(Tc, axis=1, keepdims=True)).T
    return {"rho_pc1": r(abs(stats.spearmanr(LEVELS[:TRAINED], pc1).statistic)),
            "pc1_share": r(s[0] ** 2 / (s ** 2).sum()), "betweenness": r(np.mean(trip)),
            "contiguous_merges": r(np.mean(contiguous)), "cophenetic_rho": r(coph),
            "spacing": [r(x / step.mean()) for x in step],
            "any_equivalent_level": r(slope * coords[10, 0] + icpt, 1) if len(P) > 10 else None,
            "coords": [[r(a, 4), r(b, 4)] for a, b in coords[:TRAINED]], "cosine": [[r(x, 3) for x in row] for row in cos],
            "_axes": (c, vt[:2], sign)}


def readout_level(S: np.ndarray) -> float:
    """CV R^2 of ridge from the in-context state [orig, 9 trained levels, D] to the level, folds by original."""
    X = S.reshape(-1, S.shape[-1]).astype(np.float32)
    y = np.tile(LEVELS[:TRAINED], S.shape[0]).astype(float)
    g = np.repeat(np.arange(S.shape[0]), TRAINED)
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, g):
        m = X[tr].mean(0)
        pred[te] = Ridge(alpha=100.0).fit(X[tr] - m, y[tr]).predict(X[te] - m)
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def cv_r2(X: np.ndarray, y: np.ndarray, g: np.ndarray, alpha: float) -> float:
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, g):
        m, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        pred[te] = Ridge(alpha=alpha).fit((X[tr] - m) / sd, y[tr]).predict((X[te] - m) / sd)
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def part_a(model: str) -> tuple[dict, np.ndarray]:
    folder = INTERP / model
    S = np.load(folder / "states.npy", mmap_mode="r")                  # [orig, 11, layer, D]
    alone = np.load(folder / "alone.npy")                               # [11, layer, D]
    emb = np.load(folder / "embeddings.npz")
    rng = np.random.default_rng(SEED)
    cloud_idx = rng.choice(S.shape[0], N_CLOUD, replace=False)
    layers = []
    for l in range(S.shape[2]):
        Sl = np.asarray(S[:, :, l], dtype=np.float32)
        g = geometry(Sl.mean(0))
        c, axes, sign = g.pop("_axes")
        cloud = (Sl[cloud_idx, :TRAINED] - c) @ axes.T
        cloud[..., 0] *= sign
        ga = geometry(alone[:, l])
        ga.pop("_axes")
        layers.append({"layer": l, "in_context": {**g, "readout_r2": r(readout_level(Sl[:, :TRAINED])),
                                                 "cloud": [[[r(a, 3), r(b, 3)] for a, b in row] for row in cloud]},
                       "alone": {k: ga[k] for k in ("rho_pc1", "pc1_share", "betweenness", "contiguous_merges", "cophenetic_rho",
                                                    "spacing", "coords", "cosine")}})
        print(f"{model} layer {l:2d}: in context rho {g['rho_pc1']} betw {g['betweenness']} contig {g['contiguous_merges']} "
              f"R2 {layers[-1]['in_context']['readout_r2']} | alone rho {ga['rho_pc1']}", flush=True)
    embeddings = {}
    for k in ("input",):                       # the output rows are never trained (the tokens are never predicted)
        ge = geometry(emb[k])
        ge.pop("_axes")
        embeddings[k] = ge
    return {"layers": layers, "embeddings": embeddings}, S


def part_b(model: str, S: np.ndarray) -> dict:
    test = pd.read_csv(G1 / "test_set_g1.csv")
    pos = {i: k for k, i in enumerate(test.original_sentence_idx)}
    pred = pd.read_csv(OUTPUTS / model / "predictions-test.csv")
    ach = 100 * pd.read_csv(OUTPUTS / model / "readit-test.csv").readit_all
    ok = ach.notna().to_numpy()
    pred, y = pred[ok], ach[ok].to_numpy()
    oi = pred.original_sentence_idx.map(pos).to_numpy()
    li = (pred.level // 10).to_numpy()
    g = pred.original_sentence_idx.to_numpy()
    o = test.set_index("original_sentence_idx").loc[pred.original_sentence_idx]
    feats = np.column_stack([100 * o[f"original_{d}"] for d in ("all", "base", "lexical", "syntax")] + [o.original_text.str.len()])
    base = {"request": r(cv_r2(pred[["level"]].to_numpy(float), y, g, 1.0)),
            "request_and_original": r(cv_r2(np.column_stack([pred.level, feats]).astype(float), y, g, 1.0)),
            "original_only": r(cv_r2(feats.astype(float), y, g, 1.0))}
    by_layer = []
    for l in range(S.shape[2]):
        X = np.asarray(S[oi, li, l], dtype=np.float32)
        by_layer.append(r(cv_r2(X, y, g, 1000.0)))
    best = int(np.nanargmax(by_layer))
    return {"n": int(len(y)), **base, "state_by_layer": by_layer, "best_layer": best, "state_best": by_layer[best]}


def lexicon_class() -> dict:
    lex = pd.read_csv(INTERP / "lexicon.tsv", sep="\t", keep_default_na=False)
    return dict(zip(lex.form, lex["class"]))


def realised_share(texts: pd.Series, form2cls: dict) -> pd.DataFrame:
    rows = []
    for t in texts.fillna(""):
        c = [form2cls.get(w.lower()) for w in WORD.findall(t)]
        n = {k: sum(x == k for x in c) for k in CLASSES}
        tot = sum(n.values())
        rows.append({"words_classified": tot, "basic_share_text": (n["FO"] + n["AU"] + n["AD"]) / tot if tot else np.nan,
                     "fo_share_text": n["FO"] / tot if tot else np.nan})
    return pd.DataFrame(rows)


def shares(f: pd.DataFrame) -> pd.DataFrame:
    basic = f.p_FO + f.p_AU + f.p_AD
    pb = f.pb_FO + f.pb_AU + f.pb_AD
    return f.assign(basic_share=basic / (basic + f.p_out), fo_share=f.p_FO / (basic + f.p_out),
                    basic_share_boundary=pb / (pb + f.pb_out), word_mass=(basic + f.p_out) / f.steps)


def by_group(f: pd.DataFrame, key: str, cols: list[str]) -> dict:
    out = {}
    for k, x in f.groupby(key):
        out[str(k)] = {"n": int(len(x)), **{c: r(x[c].mean(), 4) for c in cols}}
    return out


def boot_diff(a: pd.Series, b: pd.Series, n=2000) -> dict:
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(SEED)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {"diff": r(d.mean(), 4), "ci95": [r(np.percentile(bs, 2.5), 4), r(np.percentile(bs, 97.5), 4)]}


def part_d(form2cls: dict) -> dict:
    """De Mauro's basic vocabulary: in the generated outputs' next-token distributions and words, in the training data
    and the references, and on fixed texts read under every level (baseline: level 40)."""
    res = {"generated": {}, "fixed": {}}
    cols = ["basic_share", "fo_share", "basic_share_boundary", "word_mass", "basic_share_text", "fo_share_text", "entropy"]
    for name in ("g1", "g1_guided", "g1_guided_w3"):
        path = INTERP / f"logits-{name}.csv"
        if not path.exists():
            continue
        f = shares(pd.read_csv(path))
        model, file = {"g1": ("g1", "test"), "g1_guided": ("g1_guided", "test"), "g1_guided_w3": ("g1_guided", "test-w3")}[name]
        texts = pd.read_csv(OUTPUTS / model / f"predictions-{file}.csv").prediction
        f = pd.concat([f, realised_share(texts, form2cls)], axis=1)
        res["generated"][name] = {"all": {c: r(f[c].mean(), 4) for c in cols}, "by_level": by_group(f, "level", cols)}
        lo, hi = f[f.level == 0], f[f.level == 60]
        res["generated"][name]["request0_minus_request60"] = boot_diff(
            lo.set_index("original_sentence_idx").basic_share, hi.set_index("original_sentence_idx").basic_share)
    test = pd.read_csv(G1 / "test_set_g1.csv")
    orig_real = realised_share(test.original_text, form2cls)
    refs = pd.read_csv(G1 / "test_references_g1.csv")
    ref_real = pd.concat([refs[["level"]], realised_share(refs.simplification, form2cls)], axis=1)
    res["originals_text"] = {c: r(orig_real[c].mean(), 4) for c in ("basic_share_text", "fo_share_text")}
    res["references_text_by_level"] = by_group(ref_real, "level", ["basic_share_text", "fo_share_text"])
    train = pd.read_csv(G1 / "training_set_g1.zip", usecols=["original_text", "simplification", "tgt_level"])
    tr_real = pd.concat([train[["tgt_level"]].rename(columns={"tgt_level": "level"}), realised_share(train.simplification, form2cls)], axis=1)
    res["training_text_by_level"] = by_group(tr_real, "level", ["basic_share_text", "fo_share_text"])
    res["training_originals_text"] = {c: r(realised_share(train.original_text.drop_duplicates(), form2cls)[c].mean(), 4)
                                      for c in ("basic_share_text", "fo_share_text")}
    for model in MODELS:
        path = INTERP / model / "fixed.csv"
        if not path.exists():
            continue
        f = shares(pd.read_csv(path, dtype={"token": str}))
        f = f[f.token != "any"]
        piv = f.pivot(index="original_sentence_idx", columns="token", values="basic_share")
        res["fixed"][model] = {"reference": "40", "by_token": by_group(f, "token", ["basic_share", "fo_share", "word_mass", "entropy"]),
                               "vs_reference": {t: boot_diff(piv[t], piv["40"]) for t in piv.columns if t != "40"}}
    return res


E_LEVELS = (40, 30, 20, 10, 0)
STEP_CLASSES = ("end of text", "full stop", "comma", "semicolon or colon")
POS_TEXT = ("NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "DET", "ADP", "PRON", "CCONJ", "SCONJ", "PUNCT")


def part_e(model: str) -> dict | None:
    """E: the text's words by part of speech and the probability of ending or punctuating, on fixed level-40 texts read
    under levels 40-0; single items that change most steadily; parts of speech in the texts against the references."""
    path = INTERP / model / "classes.csv"
    if not path.exists():
        return None
    f = pd.read_csv(path)
    base = f[f.level == 40].set_index("original_sentence_idx")
    word_classes = [c[3:] for c in f.columns if c.startswith("lp_")]
    rng = np.random.default_rng(SEED)
    ids = base.index.to_numpy()
    draws = rng.integers(0, len(ids), (2000, len(ids)))
    words, steps = {}, {}
    for c in word_classes:
        n = base[f"n_{c}"].to_numpy(float)
        if n.sum() < 50:
            continue
        out = {"words": int(n.sum())}
        for lv in E_LEVELS[1:]:
            x = f[f.level == lv].set_index("original_sentence_idx").loc[ids]
            d = (x[f"lp_{c}"] - base.loc[ids, f"lp_{c}"]).to_numpy()
            bs = d[draws].sum(1) / n[draws].sum(1)
            out[str(lv)] = {"ratio": r(np.exp(d.sum() / n.sum()), 3), "ci95": [r(np.exp(np.percentile(bs, 2.5)), 3), r(np.exp(np.percentile(bs, 97.5)), 3)]}
        words[c] = out
    per = lambda x, c: x[f"p_{c}"].sum() / (x.steps - 1).sum()
    for c in STEP_CLASSES:
        steps[c] = {str(lv): r(per(f[f.level == lv], c), 5) for lv in E_LEVELS}
    steps["end of text, at the text's end"] = {str(lv): r(f[f.level == lv].eos_at_end.mean(), 4) for lv in E_LEVELS}
    z = np.load(INTERP / model / "classes_prob.npz")
    lv_order, mp = [int(x) for x in z["levels"]], z["sums"] / z["steps"][:, None]
    from transformers import AutoTokenizer  # noqa: PLC0415 (labels only)
    tok = AutoTokenizer.from_pretrained(OUTPUTS.parent / "runs" / model / "adapter")
    seq = np.stack([mp[lv_order.index(lv)] for lv in E_LEVELS])                    # [5, V], from 40 to 0
    common = np.where(seq.max(0) >= 1e-3)[0]
    diffs = np.diff(seq[:, common], axis=0)
    monotone = common[(np.all(diffs > 0, axis=0)) | (np.all(diffs < 0, axis=0))]
    lr = np.log(seq[-1, monotone] / seq[0, monotone])
    order = monotone[np.argsort(lr)]
    item = lambda i: {"item": tok.decode([int(i)]), "prob": {str(lv): r(seq[k, i], 5) for k, lv in enumerate(E_LEVELS)},
                      "ratio_0_vs_40": r(seq[-1, i] / seq[0, i], 3)}
    tokens = {"up": [item(i) for i in order[::-1][:10]], "down": [item(i) for i in order[:10]],
              "n_items_considered": int(len(common)), "n_monotone": int(len(monotone))}
    texts = {}
    prof = pd.read_csv(OUTPUTS / model / "profiling-test.csv")
    pred = pd.read_csv(OUTPUTS / model / "predictions-test.csv")
    cols = [f"upos_dist_{p}" for p in POS_TEXT]
    texts["outputs"] = {str(k): {p: r(v, 2) for p, v in zip(POS_TEXT, g[cols].mean())} for k, g in pd.concat([pred[["level"]], prof[cols]], axis=1).groupby("level")}
    refp = INTERP / "profiling-references.csv"
    if refp.exists():
        rp = pd.concat([pd.read_csv(G1 / "test_references_g1.csv", usecols=["level"]), pd.read_csv(refp)[cols]], axis=1)
        texts["references"] = {str(k): {p: r(v, 2) for p, v in zip(POS_TEXT, g[cols].mean())} for k, g in rp.groupby("level")}
    op = pd.read_csv(OUTPUTS / "originals" / "profiling-test.csv")
    texts["originals"] = {p: r(v, 2) for p, v in zip(POS_TEXT, op[cols].mean())}
    examples = json.loads((INTERP / model / "classes_examples.json").read_text())["examples"]
    return {"words": words, "steps": steps, "tokens": tokens, "texts": texts, "examples": examples}


def part_c(S: np.ndarray, layers: list) -> dict | None:
    """'No level' (g1 · guided): where its outputs land against requests 30-50, which level's output its output is
    textually closest to (chrF), and which level's state its state is closest to, layer by layer."""
    from sacrebleu.metrics import CHRF
    rd = OUTPUTS / "g1_guided" / "readit-test-null.csv"
    if not rd.exists():
        return None
    null = pd.read_csv(OUTPUTS / "g1_guided" / "predictions-test-null.csv").assign(a=100 * pd.read_csv(rd).readit_all)
    main = pd.read_csv(OUTPUTS / "g1_guided" / "predictions-test.csv")
    main = main.assign(a=100 * pd.read_csv(OUTPUTS / "g1_guided" / "readit-test.csv").readit_all)
    lvl = lambda x: (np.minimum(np.floor(x.dropna() / 10), 9) * 10).astype(int)
    shares = lambda x: {int(k): r(v, 3) for k, v in lvl(x).value_counts(normalize=True).sort_index().items()}
    dist = {"no level": {"mean": r(null.a.mean(), 1), "quartiles": [r(x, 1) for x in null.a.quantile([.25, .5, .75])], "shares": shares(null.a)}}
    for L in (30, 40, 50):
        x = main[main.level == L].a
        dist[str(L)] = {"mean": r(x.mean(), 1), "quartiles": [r(v, 1) for v in x.quantile([.25, .5, .75])], "shares": shares(x),
                        "ks_vs_no_level": r(stats.ks_2samp(x.dropna(), null.a.dropna()).statistic)}
    chrf = CHRF()
    nt = dict(zip(null.original_sentence_idx, null.prediction.fillna("")))
    sims = []
    for row in main.itertuples():
        if row.original_sentence_idx in nt:
            sims.append((row.original_sentence_idx, row.level, chrf.sentence_score(nt[row.original_sentence_idx], [row.prediction if isinstance(row.prediction, str) else ""]).score))
    sim = pd.DataFrame(sims, columns=["o", "level", "chrf"])
    closest = sim.loc[sim.groupby("o").chrf.idxmax()].level.value_counts(normalize=True).sort_index()
    full = sim.groupby("o")["level"].max() >= 60                         # originals with every level 0-60
    sub = sim[sim.o.isin(full[full].index) & (sim.level <= 60)]
    closest_full = sub.loc[sub.groupby("o").chrf.idxmax()].level.value_counts(normalize=True).sort_index()
    state = []
    for l in range(S.shape[2]):
        M = np.asarray(S[:, :, l], dtype=np.float32).mean(0)
        step = np.mean([np.linalg.norm(M[i] - M[i + 1]) for i in range(TRAINED - 1)])
        if step < 1e-6:
            state.append(None)
            continue
        d = [np.linalg.norm(M[10] - M[i]) / step for i in range(TRAINED)]
        state.append({"distance_in_steps": [r(x, 2) for x in d], "nearest": LEVELS[int(np.argmin(d))],
                      "equivalent_level": layers[l]["in_context"]["any_equivalent_level"]})
    return {"outputs": dist, "chrf_by_level": {int(k): r(v, 1) for k, v in sim.groupby("level").chrf.mean().items()},
            "closest_level_share": {int(k): r(v, 3) for k, v in closest.items()},
            "closest_level_share_levels_0_60": {int(k): r(v, 3) for k, v in closest_full.items()},
            "state_by_layer": state}


def main() -> None:
    res = {"models": {}}
    for model in MODELS:
        a, S = part_a(model)
        res["models"][model] = {"geometry": a, "readout": part_b(model, S)}
        print(model, "readout", {k: v for k, v in res["models"][model]["readout"].items() if k != "state_by_layer"}, flush=True)
        if model == "g1_guided":
            res["no_level"] = part_c(S, a["layers"])
            print("no level", {k: v for k, v in res["no_level"].items() if k != "state_by_layer"}, flush=True)
    res["classes"] = {m: part_e(m) for m in MODELS}
    res["vocabulary"] = part_d(lexicon_class())
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "interp_g1.json").write_text(json.dumps(res, ensure_ascii=False))
    print("wrote", RESULTS / "interp_g1.json")


if __name__ == "__main__":
    main()
