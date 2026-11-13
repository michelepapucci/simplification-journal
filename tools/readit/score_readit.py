"""READ-IT scores for a column of texts, computed exactly as the IMPaCTS labels were.

The IMPaCTS READ-IT columns were produced by code/local_readability_measuring.py of the old
repository: the t2k analyser with its default configuration (split, tokenize, pos, parser) on one
text at a time, then t2k's readability models, then the mean of the text's sentence scores. The
dependency parser is t2k's it_custom_parser.pt, a stanza model, and it has to run under stanza 1.10:
the t2k virtualenv has stanza 1.5.0, which builds different trees with the same model. With
stanza 1.10.1 this script reproduces the IMPaCTS labels exactly, in all four dimensions.

Setup (nothing in the old repository is modified):
- t2k_v3/ is a copy of LREC/evaluation/t2k_v3; its t2k/resources is a symlink to the original;
- stanza 1.10.1 comes from envs/pydeps-t2k, placed before the t2k virtualenv's own stanza;
- the parser reads the Italian word vectors from the stanza cache (~/.cache/stanza/1.11.0, the
  same vectors as the ones it was trained with) and never downloads anything;
- the copy differs from the original in one place: its parser's temporary files are per process
  (t2k_v3/t2k/tools/parser/parser.py). The original ones were fixed paths in /tmp, so two t2k
  processes running at once could swap parses without any error.

Each text is kept only if t2k completed all four steps (t2k logs a failed step and returns the last
file it has) and its parse spells the text (check_parses.py); otherwise it is analysed once more,
then left unscored.

Run with the t2k virtualenv, from the t2k package folder (the default configuration uses paths
relative to it):

    cd tools/readit/t2k_v3/t2k
    PYTHONPATH=<journal>/envs/pydeps-t2k:<journal>/tools/readit/t2k_v3 CUDA_VISIBLE_DEVICES=0 \\
        taskset --cpu-list 9-25 <old repo>/LREC/evaluation/.venv_t2k/bin/python \\
        <journal>/tools/readit/score_readit.py IN.csv --column prediction --out OUT.csv

OUT.csv has one row per row of IN.csv: row, readit_base, readit_lexical, readit_syntax, readit_all
(0-1, as in IMPaCTS), readit_sentences (NaN where the text is empty or could not be analysed).
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import stanza

from check_parses import compact, spelled

HERE = Path(__file__).resolve().parent
PRETRAIN = Path(os.path.expanduser("~/.cache/stanza/1.11.0/resources/it/pretrain/conll17.pt"))


def patch_parser() -> None:
    """t2k's DependencyParser, loaded under stanza 1.10 with explicit word vectors and no downloads."""
    from stanza import Pipeline
    from t2k.tools.parser import parser as t2k_parser

    def bootstrap(self, **kwargs):
        self.model = Pipeline(lang="it", processors="depparse", depparse_pretagged=True,
                              depparse_model_path=self.model_path, depparse_pretrain_path=str(PRETRAIN),
                              dir=str(PRETRAIN.parents[2]), download_method=None, logging_level="WARN")
    t2k_parser.DependencyParser.bootstrap = bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--column", default="prediction")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="score only the first N rows (for tests)")
    args = parser.parse_args()
    import t2k
    if not Path(t2k.__file__).resolve().is_relative_to(HERE):
        raise SystemExit(f"t2k is imported from {t2k.__file__}, not from tools/readit/t2k_v3")
    if not stanza.__version__.startswith("1.10"):
        raise SystemExit(f"stanza {stanza.__version__}: put envs/pydeps-t2k (stanza 1.10.1) first on PYTHONPATH")
    if Path.cwd().resolve() != (HERE / "t2k_v3" / "t2k").resolve():
        raise SystemExit("run from tools/readit/t2k_v3/t2k (the default t2k configuration uses paths relative to it)")
    patch_parser()
    from t2k.tools.analizer.analyzer import Analyzer
    from t2k.tools.readability.readability import Readability

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    work = args.out.with_suffix(".work").resolve()
    work.mkdir(parents=True, exist_ok=True)
    analyzer = Analyzer(lang="it", configuration_file=None, linguistic_chain=["split", "tokenize", "pos", "parser"])

    t0 = time.time()
    failed = 0
    with open(work / "texts.conllu", "w", encoding="utf-8") as docs:
        for i, text in enumerate(df[args.column]):
            if not (isinstance(text, str) and text.strip()):
                continue
            (work / "tmp.txt").write_text(text, encoding="utf-8")
            parsed = None
            for _ in range(2):
                try:
                    result = analyzer.analyze(str(work / "tmp.txt"), file_output=str(work / "tmp"))
                    candidate = Path(result.output_file_path).read_text(encoding="utf-8")
                except Exception:
                    continue
                if result.finished_steps[-1:] == ["parser"] and spelled(candidate) == compact(text):
                    parsed = candidate
                    break
            if parsed is None:
                failed += 1
                continue
            docs.write(f'<doc id="{i}">\n\n' + parsed + "</doc>\n\n")
    t1 = time.time()
    Readability(configs_path=None, models_folder=None, models_names=None, nvdb_path=None).analyze(
        str(work / "texts.conllu"), output_path=str(work / "texts_readability.json"))
    data = json.loads((work / "texts_readability.json").read_text(encoding="utf-8"))

    rows = []
    for i in range(len(df)):
        doc = data.get(str(i))
        if not doc or not doc.get("sentences_scores"):
            rows.append({"row": i})
            continue
        sent = list(doc["sentences_scores"].values())
        mean = lambda k: sum(s[k] for s in sent) / len(sent)
        rows.append({"row": i, "readit_base": mean("base"), "readit_lexical": mean("lexic"),
                     "readit_syntax": mean("syntax"), "readit_all": mean("all"), "readit_sentences": len(sent)})
    out = pd.DataFrame(rows, columns=["row", "readit_base", "readit_lexical", "readit_syntax", "readit_all", "readit_sentences"])
    out.to_csv(args.out, index=False)
    Path(f"/tmp/pre_processed_tags.{os.getpid()}.tmp").unlink(missing_ok=True)   # left behind by t2k's parser
    print(f"{len(df)} texts: analysis {t1 - t0:.0f}s, readability {time.time() - t1:.0f}s; {failed} failed; "
          f"{out.readit_all.isna().sum()} unscored; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
