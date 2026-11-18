"""Data for the interpretability page of the explorer (interp.html on gh-pages), from experiments/results/interp_g1.json
(experiments/interp/analyze.py) and experiments/results/summary.json (the achieved READ-IT by requested level) and
experiments/outputs/interp/projections.json (MDS, t-SNE and UMAP).

    python scripts/build_interp_webview.py --out <gh-pages checkout>     ->  <out>/data_interp.js (window.INTERP_G1)
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
LABELS = {"g1_guided": "g1 · guided", "g1": "g1"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    res = json.loads((RESULTS / "interp_g1.json").read_text())
    summary = json.loads((RESULTS / "summary.json").read_text())["conditions"]
    behaviour = {m: {lv: v["control"]["achieved_mean"] for lv, v in summary[m]["by_level"].items()} for m in LABELS if m in summary}
    proj_path = ROOT / "experiments" / "outputs" / "interp" / "projections.json"
    proj = json.loads(proj_path.read_text()) if proj_path.exists() else None
    if proj:   # two decimals are enough for a scatter scaled to [-1, 1]
        for m in proj["models"].values():
            for entry in m["layers"].values():
                for k, v in entry.items():
                    entry[k] = [[round(a, 2), round(b, 2)] for a, b in v]
    data = {"meta": {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                     "models": [{"id": m, "label": LABELS[m]} for m in res["models"]],
                     "levels": [str(lv) for lv in range(0, 90, 10)]},
            "models": res["models"], "no_level": res["no_level"], "vocabulary": res["vocabulary"], "classes": res["classes"],
            "projections": proj, "behaviour": behaviour,
            "attention": json.loads((RESULTS / "attention_g1.json").read_text()) if (RESULTS / "attention_g1.json").exists() else None,
            "vz": json.loads((RESULTS / "vz_g1.json").read_text()) if (RESULTS / "vz_g1.json").exists() else None}
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path = args.out / "data_interp.js"
    path.write_text(f"window.INTERP_G1 = {body};\n", encoding="utf-8")
    print(f"wrote {path} ({len(body) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
