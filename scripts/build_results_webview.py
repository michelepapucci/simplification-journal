"""Compute the data behind part 2 of the explorer (the gh-pages branch): the g1 models on the test requests.

Reads experiments/results/summary.json (experiments/analyze.py) and, for the worked examples, the models' outputs.
Writes <out>/data_results.js (window.RESULTS_DATA).

    python scripts/build_results_webview.py --out <gh-pages checkout>
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
import analyze  # noqa: E402

SEED = 42
N_EXAMPLES = 4
DROP = ("by_original_level",)  # not shown


def short(key: str) -> str:
    return {"g1": "g1", "g1_balanced": "balanced", "g1_guided": "guided, w = 1"}.get(key, key.replace("g1_best_of_", "best of ")
                                                                                       .replace("g1_guided_w", "guided, w = "))


def examples(conditions: dict, orig: pd.DataFrame) -> list[dict]:
    """A few test originals (half at each level, 80 to 200 characters), with every condition's output at every level."""
    frames = {}
    for key in conditions:
        if key == "g1" or key.startswith("g1_best_of_"):
            continue
        model, name = ("g1_guided", f"test-w{key.rsplit('_w', 1)[1]}") if key.startswith("g1_guided_w") else (key, "test")
        frames[key] = analyze.frame(model, name, orig)
    g1 = analyze.frame("g1", "test", orig)
    frames["g1"] = g1
    samples = analyze.frame("g1", "samples-test", orig)
    if samples is not None:
        best = max(int(k.rsplit("_", 1)[1]) for k in conditions if k.startswith("g1_best_of_"))
        frames[f"g1_best_of_{best}"] = analyze.best_of(g1, samples, best)
    pool = orig[(orig.original_text.str.len() >= 80) & (orig.original_text.str.len() <= 200)]
    rng = np.random.default_rng(SEED)
    picked = [i for lv in (80, 70) for i in rng.choice(pool[pool.src_level == lv].index, size=N_EXAMPLES // 2, replace=False)]
    out = []
    for i in picked:
        rows = []
        for lv in range(0, int(orig.loc[i, "src_level"]), 10):
            row = {"level": lv}
            for key, f in frames.items():
                if f is None:
                    continue
                x = f[(f.original_sentence_idx == i) & (f.level == lv)]
                if len(x):
                    score = x.out_all.iloc[0]
                    row[key] = {"text": x.prediction.iloc[0] if isinstance(x.prediction.iloc[0], str) else "",
                                "score": None if pd.isna(score) else round(float(score), 1)}
            rows.append(row)
        out.append({"original": orig.loc[i, "original_text"], "original_score": round(float(orig.loc[i, "label_all"]), 1),
                    "src_level": int(orig.loc[i, "src_level"]), "rows": rows})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((analyze.RESULTS / "summary.json").read_text())
    conds = summary["conditions"]
    orig = analyze.originals()
    data = {
        "meta": {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "conditions": [{"id": k, "label": v["label"], "short": short(k)} for k, v in conds.items()],
                 "training": summary["training"], "guidance": summary["guidance"],
                 "scorer_check_originals": summary["scorer_check_originals"],
                 "test_originals": int(len(orig)), "test_requests": int(sum(1 for _ in open(analyze.G1 / "test_requests_g1.csv")) - 1)},
        "conditions": {k: {kk: vv for kk, vv in v.items() if kk not in DROP} for k, v in conds.items()},
        "examples": examples(conds, orig),
    }
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path = args.out / "data_results.js"
    path.write_text(f"window.RESULTS_DATA = {body};\n", encoding="utf-8")
    print(f"wrote {path} ({len(body) / 1e6:.2f} MB, {len(conds)} conditions)")


if __name__ == "__main__":
    main()
