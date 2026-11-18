"""Analysis of attention.py and probe.py: syntax heads (step 1), validation of the attention and probe indices, then,
if syntax heads are found, step 2.1 (outputs by requested level, against READ-IT global) and step 2.2 (same input,
different request).

    taskset --cpu-list 9-25 python experiments/interp/analyze_attention.py   ->  results/attention_g1.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import G1, OUTPUTS, RESULTS  # noqa: E402

INTERP = OUTPUTS / "interp"
MIN_ARCS, MARGIN, MIN_HEADS = 100, 0.10, 3
LEVELS = (40, 30, 20, 10, 0)
SEED = 42
INDICES = ("dist", "long", "ent")


def r(x, d=3):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def rel_table(model: str, split: str) -> dict:
    """Per relation: correct [448], total, best offset accuracy, summed over the split's originals and references."""
    out = {}
    for cond in ("originals@40", "references"):
        z = np.load(INTERP / model / f"attention-{cond}_{split}.npz")
        for name, c, t, b in zip(z["rel_names"], z["rel_correct"], z["rel_total"], z["rel_base"]):
            C, T, B = out.get(str(name), (np.zeros(448), 0, np.zeros(3)))
            out[str(name)] = (C + c, T + t, B + b)
    return out


def syntax_heads(model: str) -> dict:
    sel, ho = rel_table(model, "selection"), rel_table(model, "heldout")
    confirmed, candidates = [], 0
    for rel in sorted(sel):
        if rel not in ho or sel[rel][1] < MIN_ARCS or ho[rel][1] < MIN_ARCS:
            continue
        acc_s, base_s = sel[rel][0] / sel[rel][1], sel[rel][2].max() / sel[rel][1]
        acc_h, base_h = ho[rel][0] / ho[rel][1], ho[rel][2].max() / ho[rel][1]
        for h in np.where(acc_s - base_s >= MARGIN)[0]:
            candidates += 1
            if acc_h[h] - base_h >= MARGIN:
                confirmed.append({"relation": rel, "head": int(h), "layer": int(h // 16), "index": int(h % 16),
                                  "acc_heldout": r(acc_h[h]), "baseline_heldout": r(base_h), "arcs_heldout": int(ho[rel][1])})
    heads = sorted({c["head"] for c in confirmed})
    best = {}
    for c in confirmed:
        if c["relation"] not in best or c["acc_heldout"] > best[c["relation"]]["acc_heldout"]:
            best[c["relation"]] = c
    layers = pd.Series([h // 16 for h in heads]).value_counts().sort_index()
    return {"candidates": candidates, "confirmed_pairs": len(confirmed), "heads": heads, "n_heads": len(heads),
            "found": len(heads) >= MIN_HEADS, "best_per_relation": best, "heads_per_layer": {int(k): int(v) for k, v in layers.items()},
            "relations_tested": sorted(r_ for r_ in sel if r_ in ho and sel[r_][1] >= MIN_ARCS and ho[r_][1] >= MIN_ARCS)}


def indices(z, heads: list) -> pd.DataFrame:
    """Per text: the syntax heads' mean expected attention distance, long-range share and entropy."""
    nq = np.where((z["nq"] > 0) & z["ok"], z["nq"], np.nan)
    return pd.DataFrame({k: z[k][:, heads].mean(1) / nq for k in INDICES})


def spearman(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return stats.spearmanr(a[m], b[m]).statistic if m.sum() > 10 else np.nan


def partial(a, b, c) -> float:
    """Spearman of a and b controlling for c (on ranks)."""
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    ra, rb, rc = (stats.rankdata(x[m]) for x in (a, b, c))
    res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)
    return stats.pearsonr(res(ra), res(rb)).statistic


def validation_frame(model: str, heads: list) -> pd.DataFrame:
    """Held-out originals and references: indices, probe, parser statistics, READ-IT, Profiling-UD."""
    test = pd.read_csv(G1 / "test_set_g1.csv")
    refs = pd.read_csv(G1 / "test_references_g1.csv")
    gold = json.loads((INTERP / "parse_gold.json").read_text())
    held = set(test.original_sentence_idx[500:])
    prof_o = pd.read_csv(OUTPUTS / "originals" / "profiling-test.csv")
    prof_r = pd.read_csv(INTERP / "profiling-references.csv")
    feats = ["avg_links_len", "max_links_len", "avg_max_depth", "subordinate_proposition_dist", "n_tokens"]
    frames = []
    for kind, z, src, prof, g, sc in (
            ("original", np.load(INTERP / model / "attention-originals@40.npz"), test, prof_o, gold["originals"], "original"),
            ("reference", np.load(INTERP / model / "attention-references.npz"), refs, prof_r, gold["references"], "simplification")):
        f = indices(z, heads)
        f["kind"] = kind
        f["original_sentence_idx"] = src.original_sentence_idx.to_numpy()
        f["readit_syntax"] = 100 * src[f"{sc}_syntax"].to_numpy()
        f["readit_global"] = 100 * src[f"{sc}_all"].to_numpy()
        for k in feats:
            f[k] = prof[k].to_numpy()
        for k in ("mean_dep_len", "max_dep_len", "mean_max_depth", "units"):
            f[f"parse_{k}"] = [x[k] for x in g]
        pf = INTERP / model / f"probe-{'originals@40' if kind == 'original' else 'references'}.npz"
        if pf.exists():
            p = np.load(pf)
            for li, layer in enumerate(p["layers"]):
                f[f"probe_uuas_L{layer}"] = p["scores"][:, li, 0]
                f[f"probe_len_L{layer}"] = p["scores"][:, li, 2]
        frames.append(f[f.original_sentence_idx.isin(held)])
    return pd.concat(frames, ignore_index=True)


def validate(f: pd.DataFrame, cols: list[str]) -> dict:
    targets = ["readit_syntax", "readit_global", "parse_mean_dep_len", "parse_max_dep_len", "parse_mean_max_depth",
               "avg_links_len", "max_links_len", "avg_max_depth", "subordinate_proposition_dist"]
    L = f.parse_units.to_numpy(float)
    out = {}
    for c in cols:
        a = f[c].to_numpy(float)
        out[c] = {t: {"rho": r(spearman(a, f[t].to_numpy(float))), "partial_length": r(partial(a, f[t].to_numpy(float), L))} for t in targets}
        out[c]["length"] = {"rho": r(spearman(a, L))}
    return out


def within(df: pd.DataFrame, x: str, y: str) -> float:
    vals = [stats.spearmanr(g[x], g[y]).statistic for _, g in df.groupby("original_sentence_idx") if len(g) >= 3 and g[x].nunique() > 1 and g[y].nunique() > 1]
    return float(np.nanmean(vals))


def step21(model: str, heads: list, probe_layer: int | None) -> dict:
    outs = pd.read_csv(OUTPUTS / model / "predictions-test.csv")
    ach = 100 * pd.read_csv(OUTPUTS / model / "readit-test.csv").readit_all
    f = indices(np.load(INTERP / model / "attention-outputs.npz"), heads)
    f["level"], f["original_sentence_idx"], f["achieved"] = outs.level.to_numpy(), outs.original_sentence_idx.to_numpy(), ach.to_numpy()
    f["units"] = [x["units"] for x in json.loads((INTERP / "parse_gold.json").read_text())["outputs"]]
    cols = list(INDICES)
    pf = INTERP / model / "probe-outputs.npz"
    if pf.exists() and probe_layer is not None:
        p = np.load(pf)
        li = list(p["layers"]).index(probe_layer)
        f["probe_len"], f["probe_uuas"] = p["scores"][:, li, 2], p["scores"][:, li, 0]
        cols.append("probe_len")
    res = {"by_level": {int(lv): {c: r(g[c].mean(), 4) for c in cols + ["achieved", "units"]} for lv, g in f.groupby("level")}}
    for c in cols:
        a = f[c].to_numpy(float)
        res[c] = {"rho_achieved_global": r(spearman(a, f.achieved.to_numpy(float))),
                  "partial_length": r(partial(a, f.achieved.to_numpy(float), f.units.to_numpy(float))),
                  "rho_requested_level": r(spearman(a, f.level.to_numpy(float))),
                  "within_original_vs_level": r(within(f.dropna(subset=[c]), c, "level")),
                  "within_original_vs_achieved": r(within(f.dropna(subset=[c, "achieved"]), c, "achieved"))}
    return res


def step22(model: str, heads: list, best_pairs: dict, probe_layer: int | None) -> dict:
    res = {}
    rng = np.random.default_rng(SEED)
    for kind in ("originals", "fixed"):
        base = indices(np.load(INTERP / model / f"attention-{kind}@40.npz"), heads)
        draws = rng.integers(0, len(base), (2000, len(base)))
        out = {}
        for lv in LEVELS:
            z = np.load(INTERP / model / f"attention-{kind}@{lv}.npz")
            f = indices(z, heads)
            row = {}
            for c in INDICES:
                a, b = f[c].to_numpy(), base[c].to_numpy()
                m = np.isfinite(a) & np.isfinite(b)
                rel = a[m].mean() / b[m].mean() - 1
                bs = [a[m][d % m.sum()].mean() / b[m][d % m.sum()].mean() - 1 for d in draws[:500]]
                row[c] = {"mean": r(a[m].mean(), 4), "relative_to_40": r(rel, 4), "ci95": [r(np.percentile(bs, 2.5), 4), r(np.percentile(bs, 97.5), 4)]}
            acc = {}
            for rel_name, pair in best_pairs.items():
                k = list(z["rel_names"]).index(rel_name) if rel_name in list(z["rel_names"]) else None
                if k is not None and z["rel_total"][k] > 0:
                    acc[rel_name] = r(z["rel_correct"][k][pair["head"]] / z["rel_total"][k])
            row["best_head_accuracy"] = acc
            pf = INTERP / model / f"probe-{kind}@{lv}.npz"
            if pf.exists() and probe_layer is not None:
                p = np.load(pf)
                li = list(p["layers"]).index(probe_layer)
                row["probe_uuas"] = r(np.nanmean(p["scores"][:, li, 0]), 4)
                row["probe_len"] = r(np.nanmean(p["scores"][:, li, 2]), 4)
            out[str(lv)] = row
        res[kind] = out
    return res


def probe_summary(model: str) -> tuple[dict | None, int | None]:
    pj = INTERP / model / "probe.json"
    if not pj.exists():
        return None, None
    p = json.loads(pj.read_text())
    best = {int(l): max(c) for l, c in p["dev_uuas"].items()}
    layer = max(best, key=best.get)
    return {"dev_uuas_best_epoch": best, "best_layer": layer, "train_sentences": p["train_sentences"]}, layer


def main() -> None:
    res = {"criterion": {"min_arcs": MIN_ARCS, "margin": MARGIN, "min_heads": MIN_HEADS}, "models": {}}
    for model in ("g1_guided", "g1"):
        res["models"][model] = {"syntax_heads": syntax_heads(model)}
        sh = res["models"][model]["syntax_heads"]
        print(model, "candidates", sh["candidates"], "confirmed pairs", sh["confirmed_pairs"], "heads", sh["n_heads"], "found", sh["found"], flush=True)
    g, c = set(res["models"]["g1_guided"]["syntax_heads"]["heads"]), set(res["models"]["g1"]["syntax_heads"]["heads"])
    res["head_overlap"] = {"both": len(g & c), "only_guided": len(g - c), "only_g1": len(c - g)}
    model = "g1_guided"
    sh = res["models"][model]["syntax_heads"]
    probe, layer = probe_summary(model)
    res["probe"] = probe
    if sh["n_heads"]:
        f = validation_frame(model, sh["heads"])
        cols = list(INDICES) + ([f"probe_len_L{layer}"] if layer is not None and f"probe_len_L{layer}" in f else [])
        res["validation"] = validate(f, cols)
        res["validation_n"] = int(len(f))
        if layer is not None and f"probe_uuas_L{layer}" in f:
            res["probe"]["test_uuas"] = {kind: r(g[f"probe_uuas_L{layer}"].mean()) for kind, g in f.groupby("kind")}
            res["probe"]["attention_vs_probe"] = {c_: r(spearman(f[c_].to_numpy(float), f[f"probe_len_L{layer}"].to_numpy(float))) for c_ in INDICES}
    if sh["found"]:
        res["step_2_1"] = step21(model, sh["heads"], layer)
        res["step_2_2"] = step22(model, sh["heads"], sh["best_per_relation"], layer)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "attention_g1.json").write_text(json.dumps(res, indent=1))
    print("wrote", RESULTS / "attention_g1.json")


if __name__ == "__main__":
    main()
