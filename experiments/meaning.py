"""Meaning preservation and output quality of every generation, plus the LREC reference-based metrics.

Against the original (every file): BERTScore with bert-base-multilingual-cased, layer 9, no idf, no rescaling
(what bert_score uses for lang='italian', as in CLiC-it 2026); SBERT cosine with
distiluse-base-multilingual-cased-v1; exact copy of the original; empty output; and the three rules dataset g1
uses to clean its training pairs (scripts/build_dataset_g1.py): degenerate (repeated 4-gram or TTR < 0.4),
instruction echo, more than twice the original's words.

Against the references (test predictions only, the LREC 2026 evaluation): for every request whose original has
IMPaCTS simplifications at the requested level (data/g1/test_references_g1.csv, after g1's cleaning rules), SARI,
sacreBLEU and BERTScore F1 of the output against all of them (multi-reference; BERTScore takes the best reference).

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 PYTHONPATH=envs/pydeps python experiments/meaning.py --model g1
writes outputs/<model>/meaning-<name>.csv (row-aligned with the predictions), results/<model>/meaning.json and
results/<model>/reference_metrics.json.
"""

import argparse
import os
import json
import sys

import evaluate
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from common import G1, OUTPUTS, RESULTS, ROOT

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # quality() forks worker processes

sys.path.insert(0, str(ROOT / "scripts"))
from build_dataset_g1 import quality  # noqa: E402

BERTSCORE_MODEL, BERTSCORE_LAYER = "google-bert/bert-base-multilingual-cased", 9
SBERT_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v1"


def text_lookup() -> dict:
    both = pd.concat([pd.read_csv(G1 / f"{s}_set_g1.csv", usecols=["original_sentence_idx", "original_text"]) for s in ("test", "dev")])
    return dict(zip(both.original_sentence_idx, both.original_text))


def reference_metrics(frame: pd.DataFrame, preds: list[str], origs: list[str], bert_path: str) -> dict:
    from bert_score import score as bert_score
    refs = pd.read_csv(G1 / "test_references_g1.csv").groupby(["original_sentence_idx", "level"]).simplification.apply(list)
    key = list(zip(frame.original_sentence_idx, frame.level))
    has = np.array([k in refs.index for k in key])
    sari, bleu = evaluate.load("sari"), evaluate.load("sacrebleu")
    rows = np.flatnonzero(has)
    r_lists = [refs[key[i]] for i in rows]
    _, _, f = bert_score([preds[i] for i in rows], r_lists, model_type=bert_path, num_layers=BERTSCORE_LAYER,
                         batch_size=128, device="cuda")
    out = {"requests_with_references": int(has.sum()), "references_per_request_mean": round(float(np.mean([len(r) for r in r_lists])), 2)}

    def block(sel: np.ndarray) -> dict:
        p, s, r = [preds[rows[i]] for i in sel], [origs[rows[i]] for i in sel], [r_lists[i] for i in sel]
        width = max(len(x) for x in r)
        padded = [x + [x[0]] * (width - len(x)) for x in r]  # sacreBLEU needs the same number of references
        return {"n": int(len(sel)), "sari": round(sari.compute(sources=s, predictions=p, references=r)["sari"], 2),
                "bleu": round(bleu.compute(predictions=p, references=padded)["score"], 2),
                "bertscore_f1": round(float(f[sel].mean()), 4)}

    out["all"] = block(np.arange(len(rows)))
    levels = frame.level.to_numpy()[rows]
    out["by_level"] = {int(lv): block(np.flatnonzero(levels == lv)) for lv in sorted(set(levels))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="folder name under outputs/")
    parser.add_argument("--sets", nargs="+", help="only these files (names as in predictions-<name>.csv; samples-test for samples)")
    args = parser.parse_args()
    from bert_score import score as bert_score
    from sentence_transformers import SentenceTransformer

    bert_path = snapshot_download(BERTSCORE_MODEL, local_files_only=True)
    sbert = SentenceTransformer(snapshot_download(SBERT_MODEL, local_files_only=True), device="cuda")
    lookup = text_lookup()
    folder = OUTPUTS / args.model
    (RESULTS / args.model).mkdir(parents=True, exist_ok=True)
    summary_path, refs_path = RESULTS / args.model / "meaning.json", RESULTS / args.model / "reference_metrics.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    ref_summary = json.loads(refs_path.read_text()) if refs_path.exists() else {}
    for path in sorted(folder.glob("predictions-*.csv")) + sorted(folder.glob("samples-*.csv")):
        name = path.stem.removeprefix("predictions-")
        if args.sets and name not in args.sets:
            continue
        frame = pd.read_csv(path)
        preds = [p if isinstance(p, str) else "" for p in frame.prediction]
        origs = [lookup[i] for i in frame.original_sentence_idx]
        p, r, f = bert_score(preds, origs, model_type=bert_path, num_layers=BERTSCORE_LAYER, batch_size=128, device="cuda")
        with torch.inference_mode():
            a = sbert.encode(origs, batch_size=256, convert_to_tensor=True, normalize_embeddings=True)
            b = sbert.encode(preds, batch_size=256, convert_to_tensor=True, normalize_embeddings=True)
            cos = (a * b).sum(dim=1).cpu().numpy()
        q = quality(pd.DataFrame({"simplification": preds, "original_text": origs}))
        rows = pd.DataFrame({"bertscore_precision": p.numpy(), "bertscore_recall": r.numpy(), "bertscore_f1": f.numpy(),
                             "sbert_cosine": cos, "copy": [x.strip() == o.strip() for x, o in zip(preds, origs)],
                             "empty": [not x.strip() for x in preds], "degenerate": q.degenerate.to_numpy(),
                             "echo": q.echo.to_numpy(), "too_long": q.too_long.to_numpy()})
        rows.to_csv(folder / f"meaning-{name}.csv", index=False)
        summary[name] = {"generations": len(preds), **{k: round(float(rows[k].mean()), 4) for k in rows.columns}}
        print(name, summary[name], flush=True)
        if name.startswith("test"):
            ref_summary[name] = reference_metrics(frame, preds, origs, bert_path)
            print(name, "references:", ref_summary[name]["all"], flush=True)
    summary_path.write_text(json.dumps(summary, indent=1))
    refs_path.write_text(json.dumps(ref_summary, indent=1))


if __name__ == "__main__":
    main()
