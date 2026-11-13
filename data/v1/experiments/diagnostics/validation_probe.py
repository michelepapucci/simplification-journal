"""Lexical requests on held-out originals from the training pool: is the lexical failure specific to the test set?

Originals: the validation split of dataset v2, one row per original (never trained on by any v1 or v2 model):
63 from the lexical supplement and 107 from v1's part. Each gets the main evaluation's lexical requests (every
lexical level below its own, base and syntax any), generated exactly as generate.py does and scored with READ-IT
exactly as score_texts.py does. Writes outputs/diagnostics/validation_probe-<model>.csv and readit-<model>.csv.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python data/v1/experiments/diagnostics/validation_probe.py targets v2_targets
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DIMS, OUTPUTS, ROOT, RUNS, SEED, control_string, prompt  # noqa: E402
from generate import generate, load  # noqa: E402
from score_texts import readit  # noqa: E402

OUT = OUTPUTS / "diagnostics"


def requests() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "v2" / "training_set_v2.zip",
                    usecols=["original_sentence_idx", "split", "subset", "original_text"] + [f"src_{x}" for x in DIMS])
    d = d[d.split == "validation"].drop_duplicates("original_sentence_idx")
    rows = [dict(original_sentence_idx=r.original_sentence_idx, subset=r.subset, original_text=r.original_text,
                 **{f"src_{x}": getattr(r, f"src_{x}") for x in DIMS}, level=L)
            for r in d.itertuples() for L in range(0, int(min(np.floor(r.src_lexical / 10), 9) * 10), 10)]
    return pd.DataFrame(rows)


def summary(name: str) -> None:
    p = pd.read_csv(OUT / f"validation_probe-{name}.csv")
    r = pd.read_csv(OUT / f"readit-{name}.csv")
    assert len(p) == len(r)
    f = p.assign(out_lexical=100 * r.readit_lexical.values, out_base=100 * r.readit_base.values).dropna(subset=["out_lexical"])
    for sub, g in f.groupby("subset"):
        def rho(col):
            return np.nanmean([stats.spearmanr(h.level, h[col]).statistic for _, h in g.groupby("original_sentence_idx")
                               if h.level.nunique() > 2 and h[col].nunique() > 1])
        inside = ((g.out_lexical >= g.level) & (g.out_lexical < g.level + 10)).mean()
        by = " ".join(f"{L}: {g[g.level == L].out_lexical.mean():.1f}" for L in (0, 10, 20, 30))
        print(f"{name:11s} {sub:18s} originals {g.original_sentence_idx.nunique():3d} scored {len(g):3d}/{(p.subset == sub).sum():3d}"
              f" | original lexical {g.drop_duplicates('original_sentence_idx').src_lexical.mean():.1f} | achieved at {by}"
              f" | Δ lexical {(g.out_lexical - g.src_lexical).mean():+.1f} Δ base {(g.out_base - g.src_base).mean():+.1f}"
              f" | inside {100 * inside:.1f}% | ρ within: lexical {rho('out_lexical'):+.2f} base {rho('out_base'):+.2f}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    req = requests()
    for name in sys.argv[1:]:
        if not (OUT / f"readit-{name}.csv").exists():
            tok, model = load(RUNS / name / "adapter")
            ctrl = [control_string("targets", {"base": "any", "lexical": str(L), "syntax": "any"}) for L in req.level]
            torch.manual_seed(SEED)
            preds = generate(tok, model, [prompt(c, t) for c, t in zip(ctrl, req.original_text)], 96)
            req.assign(control_token=ctrl, prediction=preds).to_csv(OUT / f"validation_probe-{name}.csv", index=False)
            del model
            torch.cuda.empty_cache()
            readit(OUT / f"validation_probe-{name}.csv", "prediction", OUT / f"readit-{name}.csv")
        summary(name)


if __name__ == "__main__":
    main()
