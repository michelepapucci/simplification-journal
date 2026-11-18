"""Dependency parses for the attention and probe analyses (stanza 1.11.0, the profiler's pipeline:
tokenize, mwt, pos, lemma, depparse).

Text sets (outputs/interp/parses/<set>.json, a list with one entry per text):
- originals: the 1,000 test originals;
- references: the 7,968 test references (data/g1/test_references_g1.csv);
- outputs: g1 · guided's 7,582 test outputs;
- training: 2,000 g1 training simplifications (seed 42), for training the structural probe.

Each text has units, the surface tokens: [start, end, text, first word's UPOS, sentence index]. Its words carry
[unit, head word (-1 for the root), deprel]. Contractions such as "della" are one unit with two words.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python experiments/interp/parse.py
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import G1, OUTPUTS  # noqa: E402

STANZA_DIR = Path(os.path.expanduser("~/.cache/stanza/1.11.0/resources"))
OUT = OUTPUTS / "interp" / "parses"
SEED = 42
N_TRAINING = 2000


def sets() -> dict[str, list[str]]:
    test = pd.read_csv(G1 / "test_set_g1.csv")
    refs = pd.read_csv(G1 / "test_references_g1.csv")
    outs = pd.read_csv(OUTPUTS / "g1_guided" / "predictions-test.csv")
    train = pd.read_csv(G1 / "training_set_g1.zip", usecols=["split", "simplification"])
    train = train[train.split == "train"].sample(N_TRAINING, random_state=SEED)
    clean = lambda s: [t.strip() if isinstance(t, str) else "" for t in s]
    return {"originals": clean(test.original_text), "references": clean(refs.simplification),
            "outputs": clean(outs.prediction), "training": clean(train.simplification)}


def parse(nlp, texts: list[str]) -> list[dict]:
    import stanza
    out = []
    for start in range(0, len(texts), 256):
        chunk = texts[start:start + 256]
        docs = nlp.bulk_process([stanza.Document([], text=t if t else ".") for t in chunk])
        for t, doc in zip(chunk, docs):
            units, words = [], []
            for si, sent in enumerate(doc.sentences):
                base = len(words)
                wid = {}                                            # stanza word id -> global word index
                for tok in sent.tokens:
                    for w in tok.words:
                        wid[w.id] = len(wid) + base
                for tok in sent.tokens:
                    u = len(units)
                    units.append([tok.start_char, tok.end_char, tok.text, tok.words[0].upos, si])
                    for w in tok.words:
                        words.append([u, wid[w.head] if w.head > 0 else -1, w.deprel])
            out.append({"text": t, "units": units, "words": words})
    return out


def main() -> None:
    import stanza
    OUT.mkdir(parents=True, exist_ok=True)
    nlp = stanza.Pipeline(lang="it", processors="tokenize,mwt,pos,lemma,depparse", dir=str(STANZA_DIR),
                          download_method=None, use_gpu=True, logging_level="WARN")
    for name, texts in sets().items():
        t0 = time.time()
        res = parse(nlp, texts)
        (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False))
        print(f"{name}: {len(res)} texts in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
