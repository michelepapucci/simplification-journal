"""Run READ-IT (tools/readit) and Profiling-UD (tools/profiling) on a model's generations, or on the test and dev
originals.

    taskset --cpu-list 9-25 python experiments/score_texts.py --model g1             # every unscored file
    taskset --cpu-list 9-25 python experiments/score_texts.py --model g1 --sets test
    taskset --cpu-list 9-25 python experiments/score_texts.py --originals

--model scores outputs/<model>/predictions-<set>.csv and samples-<set>.csv into readit-<name>.csv and
profiling-<name>.csv (row-aligned; <name> is the file name without "predictions-", e.g. test, dev-w2, samples-test).
Files that already have a READ-IT file are skipped unless named with --sets. --originals writes
outputs/originals/{readit,profiling}-{test,dev}.csv, row-aligned with data/g1/{test,dev}_set_g1.csv.
READ-IT is computed exactly as the IMPaCTS labels (tools/readit/score_readit.py): the t2k virtualenv of the old
repository (read only), the t2k copy in tools/readit, stanza 1.10.1 from envs/pydeps-t2k.

--repair checks existing READ-IT files instead (tools/readit/check_parses.py: every row's parse must spell the
row's text), re-scores only the rows that fail and patches them in (a safeguard kept from the earlier experiments,
where two overlapping t2k runs once swapped parses through shared temporary files).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import G1, OLD_REPO, OUTPUTS, ROOT

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
    (folder / f"{out.stem}.json").write_text(json.dumps({"rows": bad, "mismatches_after": left}, indent=1))
    print(f"{out.name}: re-scored {len(bad)} rows whose parse belonged to another text; {len(left)} mismatches left", flush=True)


def jobs_for_model(model: str, sets: list[str] | None) -> list[tuple]:
    folder = OUTPUTS / model
    jobs = []
    for src in sorted(folder.glob("predictions-*.csv")) + sorted(folder.glob("samples-*.csv")):
        name = src.stem.removeprefix("predictions-")
        if sets is not None and name not in sets:
            continue
        if sets is None and (folder / f"readit-{name}.csv").exists():
            continue
        jobs.append((src, "prediction", folder, name))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="folder name under outputs/")
    group.add_argument("--originals", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[], choices=["readit", "profiling"])
    parser.add_argument("--sets", nargs="+", help="only these files (names as in readit-<name>.csv)")
    parser.add_argument("--repair", action="store_true", help="check READ-IT files and re-score the rows with a wrong parse")
    args = parser.parse_args()
    if args.model:
        jobs = jobs_for_model(args.model, args.sets if (args.sets or not args.repair) else [p.stem.removeprefix("readit-")
                              for p in (OUTPUTS / args.model).glob("readit-*.csv")])
    else:
        jobs = [(G1 / f"{s}_set_g1.csv", "original_text", OUTPUTS / "originals", s) for s in ("test", "dev")]
        if args.sets:
            jobs = [j for j in jobs if j[3] in args.sets]
    for src, column, folder, name in jobs:
        folder.mkdir(parents=True, exist_ok=True)
        if args.repair:
            repair(src, column, folder / f"readit-{name}.csv")
            continue
        if "readit" not in args.skip:
            readit(src, column, folder / f"readit-{name}.csv")
        if "profiling" not in args.skip and not name.startswith("samples-"):
            profiling(src, column, folder / f"profiling-{name}.csv")


if __name__ == "__main__":
    main()
