"""Payload for the "Model generation analysis" section of the data explorer (gh-pages).

Reads data/v1/experiments/results/summary.json (written by data/v1/experiments/analyze.py) and the
models' outputs, and writes <out>/data_generation.js with:
- per test set and model: achieved READ-IT per requested level (mean and quartiles), RMSE, share
  inside the level, bias, Spearman; movement of the requested and free dimensions; meaning scores;
- the joint evaluation (all three dimensions requested);
- one worked example per test set: an original and every model's output at each requested level.

    python scripts/build_generation_webview.py --out <gh-pages checkout>
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "data" / "v1" / "experiments"
DIMS = ("base", "lexical", "syntax")
SEED = 42
SHORT = {"clicit2026_0.6B": "CLiC-it 2026 0.6B", "targets": "v1 · targets", "source_targets": "v1 · source + targets",
         "v2_targets": "v2 · targets", "v3_targets": "v3 · targets", "v4_targets": "v4 · targets",
         "v5_targets": "v5 · targets", "v6_targets": "v6 · targets", "v7_targets": "v7 · 3 epochs",
         "v7_1ep_targets": "v7 · 1 epoch"}


def pick(d: dict, *keys):
    return {k: d.get(k) for k in keys}


def example(dim: str, models: list[str]) -> dict | None:
    """One original of the <dim> test set (80-200 characters, middle of the source range) with every model's
    output at each requested level."""
    preds = {}
    for m in models:
        p = EXP / "outputs" / m / f"predictions-{dim}.csv"
        r = EXP / "outputs" / m / f"readit-{dim}.csv"
        if p.exists() and r.exists():
            f = pd.read_csv(p)
            preds[m] = f.assign(score=100 * pd.read_csv(r)[f"readit_{dim}"])
    if not preds:
        return None
    test = pd.read_csv(ROOT / "data" / "v1" / f"{dim}_test_set_v1.csv", usecols=["original_sentence_idx", "original_text", f"original_{dim}", f"src_{dim}"])
    test = test[test.original_text.str.len().between(80, 200)]
    test = test[test[f"src_{dim}"] == test[f"src_{dim}"].median()] if len(test) else test
    row = test.sample(1, random_state=SEED).iloc[0]
    levels = [lv for lv in range(0, 100, 10) if lv < row[f"src_{dim}"]]
    rows = []
    for lv in levels:
        out = {"level": lv}
        for m, f in preds.items():
            g = f[(f.idx == row.original_sentence_idx) & (f.level == lv)]
            if len(g):
                out[m] = {"text": str(g.prediction.iloc[0]), "score": None if pd.isna(g.score.iloc[0]) else round(float(g.score.iloc[0]), 1)}
        rows.append(out)
    return {"original": row.original_text, "original_score": round(100 * float(row[f"original_{dim}"]), 1), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((EXP / "results" / "summary.json").read_text())
    models = [m for m in SHORT if m in summary["models"]]

    main_, joint = {}, {}
    for dim in DIMS:
        main_[dim] = {}
        for m in models:
            r = summary["models"][m]["main"].get(dim)
            if not r:
                continue
            a = r["all"]
            main_[dim][m] = {
                "requests": r["requests"], "unscored": r["unscored"],
                "overall": {**pick(a, "rmse", "in_level", "bias", "spearman", "within_original_spearman", "moved_requested",
                                   "moved_free", "bertscore_f1", "sbert_cosine", "copy", "empty"),
                            "profile_change": a.get("profile_change")},
                "levels": {lv: {**pick(b, "scored", "rmse", "in_level", "bias", "achieved_mean", "achieved_quartiles",
                                       "bertscore_f1", "sbert_cosine", "copy")} for lv, b in r["by_level"].items()},
            }
    for m in models:
        j = summary["models"][m].get("joint")
        if j:
            joint[m] = {t: {**pick(b, "requests", "all_three_in_level", "all_three_within_one_level", "bertscore_f1", "sbert_cosine", "copy"),
                            "per_dimension": {d: pick(b["per_dimension"][d], "in_level", "rmse", "bias") for d in DIMS}}
                        for t, b in j.items()}

    data = {
        "meta": {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "models": [{"id": m, "label": SHORT[m]} for m in models], "dims": list(DIMS),
                 "scorer_check": summary.get("scorer_check")},
        "main": main_, "joint": joint,
        "examples": {dim: example(dim, models) for dim in DIMS},
    }
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path = args.out / "data_generation.js"
    path.write_text(f"window.GENERATION_DATA = {body};\n", encoding="utf-8")
    print(f"wrote {path} ({len(body) / 1e3:.0f} kB, models: {', '.join(models)})")


if __name__ == "__main__":
    main()
