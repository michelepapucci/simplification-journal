"""Profiling-UD features for a column of texts: stanza parse, profile every sentence, average per text.

Parsing follows GFFF/profiling/pre-processing.ipynb (stanza, Italian, tokenize,mwt,pos,lemma,depparse);
the features come from Profiling-UD (ProfilingUD/, a copy of GFFF/profiling/ProfilingUD_Code) at
sentence level with the DizionarioFondamentale dictionary, which reproduces the CLiC-it 2026
vocabulary features exactly (the NVdB_DeMauro.txt file in the same folder does not). Unlike the CLiC-it 2026 notebook, which kept only the
first sentence of each output, every sentence is profiled and the features are averaged per text,
the convention of IMPaCTS (and so of the training data). n_sentences is replaced by the real count.
Profiling-UD only writes the features that occur in its input, so absent features are filled with 0.

Stanza models are read from the local cache (~/.cache/stanza/1.11.0/resources), never downloaded.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python tools/profiling/profile_texts.py IN.csv \
        --column prediction --out OUT.csv

OUT.csv has one row per row of IN.csv: row, n_sentences, then the Profiling-UD features
(NaN where the text is empty).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STANZA_DIR = Path(os.path.expanduser("~/.cache/stanza/1.11.0/resources"))


def to_conllu(doc, row: int) -> str:
    """CoNLL-U of one parsed text, sentences tagged <row>_<k> so they can be averaged back per text."""
    blocks = []
    sentences = [s for s in "{:C}".format(doc).strip().split("\n\n") if s.strip()]
    for k, sentence in enumerate(sentences):
        lines = [line for line in sentence.split("\n") if line and not line.startswith("#")]
        blocks.append(f"# sent_id = {row}_{k}\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--column", default="prediction")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="profile only the first N rows (for tests)")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    import stanza

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    texts = [(i, t) for i, t in enumerate(df[args.column]) if isinstance(t, str) and t.strip()]
    work = args.out.with_suffix(".work").resolve()
    work.mkdir(parents=True, exist_ok=True)
    conllu = work / "texts.conllu"

    t0 = time.time()
    nlp = stanza.Pipeline(lang="it", processors="tokenize,mwt,pos,lemma,depparse", dir=str(STANZA_DIR),
                          download_method=None, use_gpu=True, logging_level="WARN")
    with open(conllu, "w", encoding="utf-8") as f:
        for start in range(0, len(texts), args.batch):
            chunk = texts[start:start + args.batch]
            docs = nlp.bulk_process([stanza.Document([], text=t) for _, t in chunk])
            for (i, _), doc in zip(chunk, docs):
                f.write(to_conllu(doc, i) + "\n\n")
    t1 = time.time()

    subprocess.run([sys.executable, str(HERE / "ProfilingUD" / "ling_monitoring.py"), "-p", str(conllu),
                    "-d", str(HERE / "ProfilingUD" / "DizionarioFondamentale"), "-t", "0"], cwd=work, check=True,
                   stdout=subprocess.DEVNULL)
    t2 = time.time()

    sent = pd.read_csv(work / "output_results" / f"{conllu.name}_sent.out", sep="\t")
    sent = sent.loc[:, ~sent.columns.str.startswith("Unnamed")]
    sent["row"] = sent["identifier"].astype(str).str.split("_").str[0].astype(int)
    feats = [c for c in sent.columns if c not in ("identifier", "row", "n_sentences")]
    per_text = sent.groupby("row")[feats].mean()
    per_text.insert(0, "n_sentences", sent.groupby("row").size())
    out = per_text.reindex(range(len(df))).rename_axis("row").reset_index()
    texted = out.n_sentences.notna()
    out.loc[texted, feats] = out.loc[texted, feats].fillna(0)
    out.to_csv(args.out, index=False)
    print(f"{len(df)} texts: parse {t1 - t0:.0f}s, profiling {t2 - t1:.0f}s; {out.n_sentences.isna().sum()} empty; wrote {args.out}")


if __name__ == "__main__":
    main()
