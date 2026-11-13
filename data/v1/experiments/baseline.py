"""Bring the CLiC-it 2026 Qwen3-0.6B outputs into this folder's layout, read only.

The CLiC-it 2026 0.6B model generated the same grid (each test original x control token 0-90 of
its strategy, temperature 0.7). Its outputs are reused, not regenerated:
GFFF/metrics/06/predictions-<dim>_test_set-<dim>.csv -> outputs/clicit2026_0.6B/predictions-<dim>.csv.
They are then scored like the new models (meaning.py, score_texts.py), so READ-IT comes from the
same reproduced IMPaCTS pipeline for every model.
"""

import pandas as pd

from common import DIMS, OLD_REPO, OUTPUTS

MODEL = "clicit2026_0.6B"


def main() -> None:
    src = OLD_REPO / "GFFF" / "metrics" / "06"
    out = OUTPUTS / MODEL
    out.mkdir(parents=True, exist_ok=True)
    for dim in DIMS:
        pred = pd.read_csv(src / f"predictions-{dim}_test_set-{dim}.csv")
        pred["level"] = pred.control_token.str.extract(r"_(\d+)\|>$", expand=False).astype(int)
        pred[["idx", "level", "control_token", "prediction"]].to_csv(out / f"predictions-{dim}.csv", index=False)
        print(dim, len(pred), "generations")


if __name__ == "__main__":
    main()
