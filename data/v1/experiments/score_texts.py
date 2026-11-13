"""Run READ-IT (tools/readit) and Profiling-UD (tools/profiling) on a model's generations, or on the
test originals and reference simplifications.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python data/v1/experiments/score_texts.py --model targets
    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python data/v1/experiments/score_texts.py --references

--model writes outputs/<model>/{readit,profiling}-<set>.csv, row-aligned with predictions-<set>.csv
(<set> = base, lexical, syntax, joint).
--references writes outputs/references/{readit,profiling}-{originals,references}-<dim>.csv,
row-aligned with data/v1/<dim>_test_set_v1.csv.
READ-IT is computed exactly as the IMPaCTS labels (tools/readit/score_readit.py): the t2k virtualenv of
the old repository (read only), the t2k copy in tools/readit, stanza 1.10.1 from envs/pydeps-t2k.

--sets limits the run to some sets. --repair checks existing READ-IT files instead
(tools/readit/check_parses.py: every row's parse must spell the row's text), re-scores only the rows
that fail, patches them into the file and records them in repairs/readit-<set>.json. It exists because
two scoring runs once overlapped while t2k's parser shared fixed temporary files, which swapped a few
parses between the runs (tools/readit/t2k_v3 now uses per-process files).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import DIMS, OLD_REPO, OUTPUTS, ROOT, V1

T2K_PYTHON = OLD_REPO / "LREC" / "evaluation" / ".venv_t2k" / "bin" / "python"
READIT_DIR = ROOT / "tools" / "readit"
T2K_PACKAGE = READIT_DIR / "t2k_v3" / "t2k"
STANZA_OVERLAY = ROOT / "envs" / "pydeps-t2k"
PROFILING = ROOT / "tools" / "profiling" / "profile_texts.py"
sys.path.insert(0, str(READIT_DIR))
from check_parses import documents, mismatches  # noqa: E402


def readit(csv: Path, column: str, out: Path) -> None:
    env = {**os.environ, "PYTHONPATH": f"{STANZA_OVERLAY}:{READIT_DIR / 't2k_v3'}"}
    with open(out.with_suffix(".log"), "w") as log:
        subprocess.run([str(T2K_PYTHON), str(READIT_DIR / "score_readit.py"), str(csv.resolve()), "--column", column,
                        "--out", str(out.resolve())], cwd=T2K_PACKAGE, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(open(out.with_suffix(".log")).read().strip().splitlines()[-1], flush=True)


def profiling(csv: Path, column: str, out: Path) -> None:
    subprocess.run([sys.executable, str(PROFILING), str(csv), "--column", column, "--out", str(out)], check=True)


def repair(csv: Path, column: str, out: Path) -> None:
    """Re-score the rows of <out> whose parse does not spell their text; patch <out> and its texts.conllu."""
    texts = pd.read_csv(csv)[column]
    work = out.with_suffix(".work")
    bad = mismatches(texts, work)
    if not bad:
        print(f"{out.name}: every parse matches its text", flush=True)
        return
    folder = out.parent / "repairs"
    folder.mkdir(exist_ok=True)
    subset = folder / f"{out.stem}.input.csv"
    pd.DataFrame({column: texts.iloc[bad].values}).to_csv(subset, index=False)
    fixed = folder / out.name
    readit(subset, column, fixed)
    if mismatches(pd.read_csv(subset)[column], fixed.with_suffix(".work")):
        raise SystemExit(f"{fixed}: the repair itself has parses that do not match their texts")
    scores, new = pd.read_csv(out), pd.read_csv(fixed)
    for k, row in enumerate(bad):
        for col in new.columns.drop("row"):
            scores.loc[row, col] = new.loc[k, col]
    scores.to_csv(out, index=False)
    docs, new_docs = documents(work / "texts.conllu"), documents(fixed.with_suffix(".work") / "texts.conllu")
    for k, row in enumerate(bad):
        if k in new_docs:
            docs[row] = new_docs[k]
        else:
            docs.pop(row, None)
    with open(work / "texts.conllu", "w", encoding="utf-8") as f:
        for i in sorted(docs):
            f.write(f'<doc id="{i}">' + chr(10) + docs[i] + "</doc>" + chr(10) * 2)
    left = mismatches(texts, work)
    (folder / f"{out.stem}.json").write_text(json.dumps({"rows": bad, "unscored_after": int(scores.loc[bad, "readit_all"].isna().sum()),
                                                         "mismatches_after": left}, indent=1))
    print(f"{out.name}: re-scored {len(bad)} rows whose parse belonged to another text; {len(left)} mismatches left", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="folder name under outputs/")
    group.add_argument("--references", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[], choices=["readit", "profiling"])
    parser.add_argument("--sets", nargs="+", help="only these sets (e.g. lexical joint, or originals-base)")
    parser.add_argument("--repair", action="store_true", help="check READ-IT files and re-score the rows with a wrong parse")
    args = parser.parse_args()
    jobs = []
    if args.model:
        for name in (*DIMS, "joint"):
            src = OUTPUTS / args.model / f"predictions-{name}.csv"
            if src.exists():
                jobs.append((src, "prediction", OUTPUTS / args.model, name))
    else:
        for dim in DIMS:
            src = V1 / f"{dim}_test_set_v1.csv"
            jobs += [(src, "original_text", OUTPUTS / "references", f"originals-{dim}"),
                     (src, "simplification", OUTPUTS / "references", f"references-{dim}")]
    if args.sets:
        jobs = [j for j in jobs if j[3] in args.sets]
    for src, column, folder, name in jobs:
        folder.mkdir(parents=True, exist_ok=True)
        if args.repair:
            repair(src, column, folder / f"readit-{name}.csv")
            continue
        if "readit" not in args.skip:
            readit(src, column, folder / f"readit-{name}.csv")
        if "profiling" not in args.skip:
            profiling(src, column, folder / f"profiling-{name}.csv")


if __name__ == "__main__":
    main()
