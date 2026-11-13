"""Meaning preservation of every generation against its original, as in CLiC-it 2026, plus SBERT.

GFFF/evaluate_model_against_original.py compared the generation with the *original* sentence:
corpus sacreBLEU and aggregated ROUGE (evaluate), and BERTScore with lang='italian', which
bert_score resolves to bert-base-multilingual-cased, layer 9, no idf, no rescaling. SBERT is the
cosine similarity of sentence-transformers/distiluse-base-multilingual-cased-v1 embeddings, the model
of the lab's earlier evaluation (code/evaluation/sbert_evaluation.py).

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 PYTHONPATH=envs/pydeps python data/v1/experiments/meaning.py --model targets
writes outputs/<model>/meaning-<dim>.csv (one row per generation) and results/<model>/meaning.json.
"""

import argparse
import json

import evaluate
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from common import DIMS, OUTPUTS, RESULTS, V1

BERTSCORE_MODEL, BERTSCORE_LAYER = "google-bert/bert-base-multilingual-cased", 9
SBERT_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v1"


def originals_lookup() -> dict:
    frames = [pd.read_csv(V1 / f"{d}_test_set_v1.csv", usecols=["original_sentence_idx", "original_text"]) for d in DIMS]
    both = pd.concat(frames).drop_duplicates("original_sentence_idx")
    return dict(zip(both.original_sentence_idx, both.original_text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="folder name under outputs/")
    args = parser.parse_args()
    from bert_score import score as bert_score
    from sentence_transformers import SentenceTransformer

    bleu, rouge = evaluate.load("sacrebleu"), evaluate.load("rouge")
    bert_path = snapshot_download(BERTSCORE_MODEL, local_files_only=True)
    sbert = SentenceTransformer(snapshot_download(SBERT_MODEL, local_files_only=True), device="cuda")
    summary = {}
    lookup = originals_lookup()
    for dim in (*DIMS, "joint"):
        path = OUTPUTS / args.model / f"predictions-{dim}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        preds = [p if isinstance(p, str) else "" for p in frame.prediction]
        origs = [lookup[i] for i in frame.idx]
        p, r, f = bert_score(preds, origs, model_type=bert_path, num_layers=BERTSCORE_LAYER, batch_size=128, device="cuda")
        with torch.inference_mode():
            a = sbert.encode(origs, batch_size=256, convert_to_tensor=True, normalize_embeddings=True)
            b = sbert.encode(preds, batch_size=256, convert_to_tensor=True, normalize_embeddings=True)
            cos = (a * b).sum(dim=1).cpu().numpy()
        rows = pd.DataFrame({"bertscore_precision": p.numpy(), "bertscore_recall": r.numpy(), "bertscore_f1": f.numpy(),
                             "sbert_cosine": cos, "copy": [x.strip() == o.strip() for x, o in zip(preds, origs)],
                             "empty": [not x.strip() for x in preds]})
        rows.to_csv(OUTPUTS / args.model / f"meaning-{dim}.csv", index=False)
        rg = rouge.compute(predictions=preds, references=[[o] for o in origs], use_aggregator=True)
        summary[dim] = {
            "generations": len(preds),
            "bleu": bleu.compute(predictions=preds, references=[[o] for o in origs])["score"],
            **{k: float(rg[k]) for k in ("rouge1", "rouge2", "rougeL", "rougeLsum")},
            **{k: float(rows[k].mean()) for k in ("bertscore_precision", "bertscore_recall", "bertscore_f1", "sbert_cosine")},
            "copy_rate": float(rows["copy"].mean()), "empty_rate": float(rows["empty"].mean()),
        }
        print(dim, {k: round(v, 4) for k, v in summary[dim].items()}, flush=True)
    (RESULTS / args.model).mkdir(parents=True, exist_ok=True)
    (RESULTS / args.model / "meaning.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
