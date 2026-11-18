"""Choose the guidance weight w of the guided model on the dev requests, by a rule fixed before any result:
the w with the lowest RMSE between requested level and achieved READ-IT global, among the candidates whose mean
BERTScore F1 against the original is at most 0.01 below w = 1's and whose share of degenerate or echoing outputs is
at most 1 point above w = 1's. RMSE is the CLiC-it one: achieved minus the requested level's lower bound (x100).

    python experiments/choose_guidance.py --model g1_guided     # prints w, writes results/<model>/guidance.json
"""

import argparse
import json

import numpy as np
import pandas as pd

from common import OUTPUTS, RESULTS

CANDIDATES = (1.0, 1.5, 2.0, 3.0)
MAX_BERTSCORE_DROP, MAX_BAD_RISE = 0.01, 0.01


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    folder, rows = OUTPUTS / args.model, {}
    for w in CANDIDATES:
        name = f"dev-w{w:g}"
        pred, rd, mn = (pd.read_csv(folder / f"{k}-{name}.csv") for k in ("predictions", "readit", "meaning"))
        achieved = 100 * rd.readit_all
        ok = achieved.notna()
        err = (achieved - pred.level)[ok]
        rows[w] = {"rmse": round(float(np.sqrt((err ** 2).mean())), 2), "bias": round(float(err.mean()), 2),
                   "in_level": round(float(((achieved >= pred.level) & (achieved < pred.level + 10))[ok].mean()), 3),
                   "bertscore_f1": round(float(mn.bertscore_f1.mean()), 4),
                   "bad": round(float((mn.degenerate | mn.echo).mean()), 4), "copy": round(float(mn["copy"].mean()), 4),
                   "scored": int(ok.sum())}
    base = rows[1.0]
    allowed = [w for w in CANDIDATES if rows[w]["bertscore_f1"] >= base["bertscore_f1"] - MAX_BERTSCORE_DROP
               and rows[w]["bad"] <= base["bad"] + MAX_BAD_RISE]
    chosen = min(allowed, key=lambda w: rows[w]["rmse"])
    out = {"rule": __doc__.split("\n\n")[0], "candidates": {f"{w:g}": r for w, r in rows.items()},
           "allowed": [f"{w:g}" for w in allowed], "chosen": chosen}
    (RESULTS / args.model).mkdir(parents=True, exist_ok=True)
    (RESULTS / args.model / "guidance.json").write_text(json.dumps(out, indent=1))
    for w, r in rows.items():
        print(f"w={w:g}: {r}", flush=True)
    print(f"{chosen:g}")


if __name__ == "__main__":
    main()
