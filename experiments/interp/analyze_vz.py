"""Analysis of vz.py (G1-G3, with 2.1 and 2.2 by value zeroing) and patching.py (2.3).

    taskset --cpu-list 9-25 python experiments/interp/analyze_vz.py   ->  results/vz_g1.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from analyze_attention import partial, r, spearman, within  # noqa: E402
from common import G1, OUTPUTS, RESULTS  # noqa: E402

INTERP = OUTPUTS / "interp"
MODEL = "g1_guided"
MIN_ARCS, MARGIN, MIN_PAIRS = 100, 0.10, 3
SEED = 42
G2_KEYS = ("vz_text", "vz_gen", "vz_stop", "att_text", "att_gen", "att_stop", "norm_text", "norm_gen", "norm_stop")


def load(name):
    return np.load(INTERP / MODEL / f"vz-{name}.npz")


def g1_g2() -> dict:
    exact = []
    out = {}
    for name in ("originals@40", "originals@20", "originals@0", "fixed@40", "fixed@20", "fixed@0", "references", "outputs"):
        z = load(name)
        exact.append(np.nanmean(z["g2_exact"], 0))
        if name.split("@")[0] in ("originals", "fixed"):
            out[name] = {k: [r(x, 4) for x in np.nanmean(z[f"g2_{k}"], 0)] for k in G2_KEYS}
    ex = np.nanmin(np.stack(exact), 0)
    ratios = {}
    for kind in ("originals", "fixed"):
        a, b = load(f"{kind}@0"), load(f"{kind}@40")
        ratios[kind] = {k: [r(x, 3) for x in np.nanmean(a[f"g2_{k}"], 0) / np.nanmean(b[f"g2_{k}"], 0)] for k in ("vz_text", "vz_gen", "att_text", "norm_text")}
    return {"exactness_min_by_layer": [r(x, 6) for x in ex], "exact": bool(np.all(ex >= 0.999)), "control_token": out, "level0_over_level40": ratios}


def rel_table(kind: str, split: str) -> dict:
    out = {}
    for name in ("originals@40", "references"):
        z = load(name)
        for n_, c, t, b in zip(z[f"{kind}_{split}_rel_names"], z[f"{kind}_{split}_rel_correct"], z[f"{kind}_{split}_rel_total"], z[f"{kind}_{split}_rel_base"]):
            C, T, B = out.get(str(n_), (np.zeros(28), 0, np.zeros(3)))
            out[str(n_)] = (C + c, T + t, B + b)
    return out


def syntax_layers(kind: str) -> dict:
    sel, ho = rel_table(kind, "selection"), rel_table(kind, "heldout")
    confirmed, cand = [], 0
    for rel in sorted(sel):
        if rel not in ho or sel[rel][1] < MIN_ARCS or ho[rel][1] < MIN_ARCS:
            continue
        acc_s, base_s = sel[rel][0] / sel[rel][1], sel[rel][2].max() / sel[rel][1]
        acc_h, base_h = ho[rel][0] / ho[rel][1], ho[rel][2].max() / ho[rel][1]
        for l in np.where(acc_s - base_s >= MARGIN)[0]:
            cand += 1
            if acc_h[l] - base_h >= MARGIN:
                confirmed.append({"relation": rel, "layer": int(l), "acc_heldout": r(acc_h[l]), "baseline_heldout": r(base_h), "arcs_heldout": int(ho[rel][1])})
    best = {}
    for c in confirmed:
        if c["relation"] not in best or c["acc_heldout"] > best[c["relation"]]["acc_heldout"]:
            best[c["relation"]] = c
    by_layer = {}
    for rel in sel:
        if rel in ho and ho[rel][1] >= MIN_ARCS:
            by_layer[rel] = {"acc": [r(x) for x in ho[rel][0] / ho[rel][1]], "baseline": r(ho[rel][2].max() / ho[rel][1]), "arcs": int(ho[rel][1])}
    layers = sorted({c["layer"] for c in confirmed})
    return {"candidates": cand, "confirmed_pairs": len(confirmed), "layers": layers, "found": len(confirmed) >= MIN_PAIRS,
            "best_per_relation": best, "heldout_accuracy_by_layer": by_layer}


def idx(z, kind: str, layers: list) -> pd.DataFrame:
    nq = np.where(z[f"{kind}_nq"] > 0, z[f"{kind}_nq"], np.nan)
    return pd.DataFrame({k: z[f"{kind}_{k}"][:, layers].mean(1) / nq for k in ("ent", "dist", "long")})


def validation(layers: list) -> tuple[dict, int]:
    test = pd.read_csv(G1 / "test_set_g1.csv")
    refs = pd.read_csv(G1 / "test_references_g1.csv")
    gold = json.loads((INTERP / "parse_gold.json").read_text())
    held = set(test.original_sentence_idx[500:])
    prof_o = pd.read_csv(OUTPUTS / "originals" / "profiling-test.csv")
    prof_r = pd.read_csv(INTERP / "profiling-references.csv")
    frames = []
    for name, src, prof, g, sc in (("originals@40", test, prof_o, gold["originals"], "original"), ("references", refs, prof_r, gold["references"], "simplification")):
        z = load(name)
        rows = z["rows"] if name == "references" else np.arange(len(test))
        f = idx(z, "vz", layers).add_prefix("vz_").join(idx(z, "norm", layers).add_prefix("norm_"))
        f["original"] = z["originals"]
        f["readit_syntax"] = 100 * src[f"{sc}_syntax"].to_numpy()[rows]
        f["readit_global"] = 100 * src[f"{sc}_all"].to_numpy()[rows]
        f["subordinate_proposition_dist"] = prof["subordinate_proposition_dist"].to_numpy()[rows]
        for k in ("mean_dep_len", "mean_max_depth", "units"):
            f[f"parse_{k}"] = np.array([x[k] for x in g])[rows]
        frames.append(f[f.original.isin(held)])
    f = pd.concat(frames, ignore_index=True)
    targets = ["readit_syntax", "readit_global", "parse_mean_dep_len", "parse_mean_max_depth", "subordinate_proposition_dist"]
    L = f.parse_units.to_numpy(float)
    out = {}
    for c in [c for c in f.columns if c.startswith(("vz_", "norm_"))]:
        a = f[c].to_numpy(float)
        out[c] = {t: {"rho": r(spearman(a, f[t].to_numpy(float))), "partial_length": r(partial(a, f[t].to_numpy(float), L))} for t in targets}
        out[c]["length"] = {"rho": r(spearman(a, L))}
    return out, int(len(f))


def step21(layers: list, kind: str = "vz") -> dict:
    z = load("outputs")
    outs = pd.read_csv(OUTPUTS / MODEL / "predictions-test.csv")
    ach = (100 * pd.read_csv(OUTPUTS / MODEL / "readit-test.csv").readit_all).to_numpy()
    rows = z["rows"]
    f = idx(z, kind, layers)
    f["level"], f["achieved"], f["original_sentence_idx"] = outs.level.to_numpy()[rows], ach[rows], outs.original_sentence_idx.to_numpy()[rows]
    f["units"] = np.array([x["units"] for x in json.loads((INTERP / "parse_gold.json").read_text())["outputs"]])[rows]
    res = {"by_level": {int(lv): {c: r(g[c].mean(), 4) for c in ("ent", "dist", "achieved", "units")} for lv, g in f.groupby("level")}}
    for c in ("ent", "dist"):
        a = f[c].to_numpy(float)
        res[c] = {"rho_achieved_global": r(spearman(a, f.achieved.to_numpy(float))), "partial_length": r(partial(a, f.achieved.to_numpy(float), f.units.to_numpy(float))),
                  "rho_requested_level": r(spearman(a, f.level.to_numpy(float)))}
    return res


def step22(layers: list, kind: str = "vz") -> dict:
    rng = np.random.default_rng(SEED)
    res = {}
    for text in ("originals", "fixed"):
        base = idx(load(f"{text}@40"), kind, layers)
        draws = rng.integers(0, len(base), (1000, len(base)))
        res[text] = {}
        for lv in (40, 20, 0):
            f = idx(load(f"{text}@{lv}"), kind, layers)
            row = {}
            for c in ("ent", "dist"):
                a, b = f[c].to_numpy(), base[c].to_numpy()
                m = np.isfinite(a) & np.isfinite(b)
                aa, bb = a[m], b[m]
                bs = [aa[d % m.sum()].mean() / bb[d % m.sum()].mean() - 1 for d in draws[:500]]
                row[c] = {"relative_to_40": r(aa.mean() / bb.mean() - 1, 4), "ci95": [r(np.percentile(bs, 2.5), 4), r(np.percentile(bs, 97.5), 4)]}
            res[text][str(lv)] = row
    return res


def patching() -> dict | None:
    folder = OUTPUTS / "g1_guided_patch"
    if not (folder / "patching.json").exists():
        return None
    meta = json.loads((folder / "patching.json").read_text())
    conds = ["base40", "base0", "A_0-9", "A_10-18", "A_19-27", "A_all", "B_syntax", "B_random"]
    data = {}
    for c in conds:
        p = pd.read_csv(folder / f"predictions-{c}.csv")
        rd = pd.read_csv(folder / f"readit-{c}.csv") if (folder / f"readit-{c}.csv").exists() else None
        pf = pd.read_csv(folder / f"profiling-{c}.csv") if (folder / f"profiling-{c}.csv").exists() else None
        f = pd.DataFrame({"o": p.original_sentence_idx})
        if rd is not None:
            for d in ("all", "syntax", "base"):
                f[d] = 100 * rd[f"readit_{d}"].to_numpy()
        if pf is not None:
            for k in ("n_tokens", "avg_links_len", "subordinate_proposition_dist"):
                f[k] = pf[k].to_numpy()
        data[c] = f
    measures = [m for m in ("all", "syntax", "base", "n_tokens", "avg_links_len", "subordinate_proposition_dist") if m in data["base0"]]
    rng = np.random.default_rng(SEED)
    n = len(data["base0"])
    draws = rng.integers(0, n, (2000, n))
    res = {"n": n, "identical_to_base40": meta["identical_to_base40"], "identical_to_base0": meta["identical_to_base0"], "means": {}, "recovered": {}}
    for c in conds:
        res["means"][c] = {m: r(data[c][m].mean(), 3) for m in measures}
    for c in conds:
        rec = {}
        for m in measures:
            a, b0, b40 = data[c][m].to_numpy(float), data["base0"][m].to_numpy(float), data["base40"][m].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b0) & np.isfinite(b40)
            share = lambda ix: (np.nanmean(a[ix]) - np.nanmean(b0[ix])) / (np.nanmean(b40[ix]) - np.nanmean(b0[ix]))
            idx_ok = np.where(ok)[0]
            bs = [share(idx_ok[d % len(idx_ok)]) for d in draws[:1000]]
            rec[m] = {"share": r(share(idx_ok), 3), "ci95": [r(np.percentile(bs, 2.5), 3), r(np.percentile(bs, 97.5), 3)]}
        res["recovered"][c] = rec
    return res


def main() -> None:
    res = {"criterion": {"min_arcs": MIN_ARCS, "margin": MARGIN, "min_pairs": MIN_PAIRS}}
    res.update(g1_g2())
    print("G1 exact:", res["exact"], "min", min(res["exactness_min_by_layer"]), flush=True)
    res["syntax"] = {kind: syntax_layers(kind) for kind in ("vz", "norm")}
    sv = res["syntax"]["vz"]
    print("G3 vz: candidates", sv["candidates"], "confirmed", sv["confirmed_pairs"], "layers", sv["layers"], "found", sv["found"], flush=True)
    if res["exact"] and sv["found"]:
        res["validation"], res["validation_n"] = validation(sv["layers"])
        res["step_2_1"] = step21(sv["layers"])
        res["step_2_2"] = step22(sv["layers"])
        # exploratory, added after the validation showed the norm-based entropy to be the best index
        res["step_2_1_norm"] = step21(sv["layers"], "norm")
        res["step_2_2_norm"] = step22(sv["layers"], "norm")
    res["patching"] = patching()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "vz_g1.json").write_text(json.dumps(res, indent=1))
    print("wrote", RESULTS / "vz_g1.json")


if __name__ == "__main__":
    main()
