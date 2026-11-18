"""Word-form lexicon with De Mauro's Nuovo vocabolario di base (NVdB, 2016) classes, for the logits analysis.

The model's logits are over word pieces, but NVdB lists lemmas, so each surface form needs a lemma. The forms come
from the texts the models read and write: dataset g1 (originals and simplifications), the test references and every
g1 model's test outputs. They are lemmatised with the pipeline the profiler uses (stanza 1.11.0 from the local
cache: tokenize, mwt, pos, lemma). Each lower-cased surface form gets the class of its most frequent analysis:
- FO (fondamentale), AU (alto uso), AD (alta disponibilità): the lemma, or the form itself, is in
  tools/profiling/ProfilingUD/NVdB_DeMauro.txt. This is the lookup READ-IT's lexical model uses (by lemma, then by
  form). A contracted form (della = di + la) takes the least basic class among its parts.
- out: an Italian content or function word outside NVdB;
- skip: proper nouns, numbers, punctuation, symbols and foreign words (X). READ-IT also leaves out PROPN, PUNCT and
  NUM.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python experiments/interp/lexicon.py   ->  outputs/interp/lexicon.tsv
"""

import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import G1, OUTPUTS, ROOT  # noqa: E402

STANZA_DIR = Path(os.path.expanduser("~/.cache/stanza/1.11.0/resources"))
NVDB = ROOT / "tools" / "profiling" / "ProfilingUD" / "NVdB_DeMauro.txt"
RANK = {"FO": 0, "AU": 1, "AD": 2, "out": 3}
SKIP_UPOS = {"PROPN", "NUM", "PUNCT", "SYM", "X"}
OUT = OUTPUTS / "interp" / "lexicon.tsv"


def nvdb() -> dict:
    """Lemma -> class. 30 lemmas are listed more than once with different classes (homographs, e.g. di FO and AD);
    they take their most basic class."""
    out = {}
    for line in NVDB.read_text(encoding="utf-8").split("\n"):
        if "\t" in line:
            lemma, cls = line.split("\t")
            if lemma not in out or RANK[cls] < RANK[out[lemma]]:
                out[lemma] = cls
    return out


def texts() -> list[str]:
    train = pd.read_csv(G1 / "training_set_g1.zip", usecols=["original_text", "simplification"])
    refs = pd.read_csv(G1 / "test_references_g1.csv", usecols=["simplification"])
    out = set(train.original_text) | set(train.simplification) | set(refs.simplification)
    for f in OUTPUTS.glob("g1*/predictions-test*.csv"):
        out |= set(pd.read_csv(f).prediction.dropna())
    return sorted(t for t in out if isinstance(t, str) and t.strip())


def main() -> None:
    import stanza
    vocab = nvdb()
    corpus = texts()
    print(f"{len(corpus)} texts", flush=True)
    nlp = stanza.Pipeline(lang="it", processors="tokenize,mwt,pos,lemma", dir=str(STANZA_DIR),
                          download_method=None, use_gpu=True, logging_level="WARN")
    counts: dict[str, Counter] = defaultdict(Counter)
    t0 = time.time()
    for start in range(0, len(corpus), 512):
        docs = nlp.bulk_process([stanza.Document([], text=t) for t in corpus[start:start + 512]])
        for doc in docs:
            for sent in doc.sentences:
                for tok in sent.tokens:
                    form = tok.text.lower()
                    parts = [(w.lemma or w.text).lower() for w in tok.words]
                    upos = [w.upos for w in tok.words]
                    if any(u in SKIP_UPOS for u in upos) or not any(c.isalpha() for c in form):
                        cls = "skip"
                    else:
                        ranks = [RANK[vocab.get(l) or vocab.get(w.text.lower()) or "out"] for l, w in zip(parts, tok.words)]
                        cls = [k for k, v in RANK.items() if v == max(ranks)][0]
                    counts[form][(cls, "+".join(parts), "+".join(upos))] += 1
        if start % 20480 == 0:
            print(f"{start + 512}/{len(corpus)} texts, {time.time() - t0:.0f}s", flush=True)
    rows = []
    for form, c in counts.items():
        (cls, lemma, upos), n = c.most_common(1)[0]
        rows.append((form, cls, lemma, upos, n, sum(c.values())))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["form", "class", "lemma", "upos", "count_majority", "count"]).sort_values("count", ascending=False) \
        .to_csv(OUT, sep="\t", index=False)
    print(f"wrote {OUT}: {len(rows)} forms in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
