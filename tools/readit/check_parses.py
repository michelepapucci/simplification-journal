"""Check that every text of a READ-IT run was scored with its own parse.

score_readit.py writes the parse of row i to <out>.work/texts.conllu as <doc id="i">. A parse belongs
to its text when its surface tokens spell the text, whitespace aside (CoNLL-U multiword tokens
count once, e.g. "della" rather than "di la"). On the 36,000 texts of the reference and CLiC-it
baseline runs every parse passes, so a mismatch means the text was scored with someone else's parse.

Why this exists: t2k's dependency parser used fixed temporary files (/tmp/pre_processed_tags.tmp,
/tmp/post_processed_tags.tmp), so two t2k processes running at once could swap parses silently.
tools/readit/t2k_v3 now uses one temporary file per process, and score_readit.py checks each parse
as it goes; this script checks runs after the fact.

    python tools/readit/check_parses.py IN.csv --column prediction --work OUT.work
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def spelled(conllu: str) -> str:
    """The text a CoNLL-U parse spells: its surface tokens, whitespace removed."""
    out, skip_to = [], 0
    for line in conllu.splitlines():
        if not line.strip():
            skip_to = 0
            continue
        f = line.split("\t")
        if line.startswith("#") or len(f) < 2:
            continue
        tid = f[0]
        if "-" in tid:                      # multiword token: keep the surface form, skip its words
            skip_to = int(tid.split("-")[1])
            out.append(f[1])
        elif tid.isdigit() and not (skip_to and int(tid) <= skip_to):
            skip_to = 0
            out.append(f[1])
    return compact("".join(out))


def documents(path: Path) -> dict[int, str]:
    """The <doc id="i"> blocks of a score_readit.py texts.conllu."""
    docs, cur, buf = {}, None, []
    for line in path.open(encoding="utf-8"):
        m = re.match(r'<doc id="(\d+)">', line)
        if m and cur is None:
            cur, buf = int(m.group(1)), []
        elif line.startswith("</doc>") and cur is not None:
            docs[cur] = "".join(buf)
            cur = None
        elif cur is not None:
            buf.append(line)
    return docs


def mismatches(texts: pd.Series, work: Path) -> list[int]:
    """Rows whose parse in <work>/texts.conllu does not spell the row's text."""
    docs = documents(work / "texts.conllu")
    return [i for i, t in enumerate(texts) if i in docs and spelled(docs[i]) != compact(str(t))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--column", default="prediction")
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    texts = pd.read_csv(args.input)[args.column]
    bad = mismatches(texts, args.work)
    print(f"{len(documents(args.work / 'texts.conllu'))} parses, {len(bad)} not matching their text: {bad[:20]}")


if __name__ == "__main__":
    main()
