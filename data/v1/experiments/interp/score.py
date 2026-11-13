"""Score the interpretability generations: READ-IT (as the IMPaCTS labels) and Profiling-UD, once per
distinct text. Greedy outputs repeat across requests, and the originals are scored with them.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python data/v1/experiments/interp/score.py --out data/v1/experiments/outputs/v5_targets/interp
Writes <out>/texts.csv (text, is_original), <out>/readit-texts.csv and <out>/profiling-texts.csv,
row-aligned with texts.csv, after the parse check of score_texts.py --repair.
With --reuse <folder>, texts already scored there are copied, and only the others are scored (in
<out>/new/). Profiling-UD writes a column only for features that occur in its input, so a feature missing
from one of the two parts is 0 there.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from generate import originals  # noqa: E402
from score_texts import profiling, readit, repair  # noqa: E402


def score(csv: Path, folder: Path) -> None:
    readit(csv, "text", folder / "readit-texts.csv")
    repair(csv, "text", folder / "readit-texts.csv")
    profiling(csv, "text", folder / "profiling-texts.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, help="a previous output folder whose scored texts are copied")
    args = parser.parse_args()
    pred = pd.read_csv(args.out / "predictions.csv", keep_default_na=False)
    text = originals()
    origs = [text[i] for i in sorted(pred.idx.unique())]
    texts = pd.Series(pd.unique(pd.Series([*origs, *pred.prediction])))
    texts = texts[texts.str.strip() != ""]
    frame = pd.DataFrame({"text": texts.values, "is_original": texts.isin(set(origs)).values})
    csv = args.out / "texts.csv"
    frame.to_csv(csv, index=False)
    print(f"{len(pred)} generations, {len(frame)} distinct texts ({frame.is_original.sum()} originals)", flush=True)
    if not args.reuse:
        score(csv, args.out)
        return

    old_texts = pd.read_csv(args.reuse / "texts.csv", keep_default_na=False).text
    old_row = dict(zip(old_texts, range(len(old_texts))))
    old_readit = pd.read_csv(args.reuse / "readit-texts.csv")
    old_prof = pd.read_csv(args.reuse / "profiling-texts.csv")
    new = frame[~frame.text.isin(old_row)]
    print(f"reusing {len(frame) - len(new)} scored texts from {args.reuse}; scoring {len(new)}", flush=True)
    folder = args.out / "new"
    folder.mkdir(exist_ok=True)
    new[["text"]].to_csv(folder / "texts.csv", index=False)
    score(folder / "texts.csv", folder)
    new_row = dict(zip(new.text, range(len(new))))
    new_readit = pd.read_csv(folder / "readit-texts.csv")
    new_prof = pd.read_csv(folder / "profiling-texts.csv")

    def assemble(old: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in dict.fromkeys([*old.columns, *fresh.columns]) if c != "row"]
        rows = []
        for t in frame.text:
            rows.append(old.iloc[old_row[t]] if t in old_row else fresh.iloc[new_row[t]])
        out = pd.DataFrame(rows).reindex(columns=cols).reset_index(drop=True)
        out.insert(0, "row", range(len(out)))
        return out

    readit_all = assemble(old_readit, new_readit)
    prof_all = assemble(old_prof, new_prof)
    prof_all = prof_all.fillna({c: 0 for c in prof_all.columns if c not in old_prof.columns or c not in new_prof.columns})
    readit_all.to_csv(args.out / "readit-texts.csv", index=False)
    prof_all.to_csv(args.out / "profiling-texts.csv", index=False)
    print(f"wrote readit-texts.csv and profiling-texts.csv for {len(frame)} texts; "
          f"{int(readit_all.readit_all.isna().sum())} unscored", flush=True)


if __name__ == "__main__":
    main()
