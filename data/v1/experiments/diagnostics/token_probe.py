"""Does a model read a target token? Teacher-forced NLL of held-out references under true and altered control strings.

Pairs: the validation split of dataset v2 (originals never trained on by any v1 or v2 model). For every pair with a
target in the tested dimension (lexical, then base), the reference's mean NLL per token under:
  full        the pair's own three targets
  full-any    the tested target replaced by any
  full-wrong  the tested target replaced by a wrong level: one level below the original if the true target is
              two or more below it, else level 0
  only        the tested target alone, the other two any (the condition of the main evaluation)
  only-any    all three any
  only-wrong  the wrong level alone
Each line gives, for an alteration, the mean increase of the reference's NLL in millinats per token (positive: the
true token makes the reference more likely) and the share of pairs where the true token gives the lower NLL.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python data/v1/experiments/diagnostics/token_probe.py targets v2_targets
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DIMS, ROOT, RUNS, control_string, prompt  # noqa: E402
from generate import load  # noqa: E402

CONDITIONS = ["full", "full-any", "full-wrong", "only", "only-any", "only-wrong"]
d = pd.read_csv(ROOT / "data" / "v2" / "training_set_v2.zip", dtype={f"tgt_{x}": str for x in DIMS},
                usecols=["split", "subset", "original_text", "simplification"] + [f"{p}_{x}" for p in ("src", "tgt") for x in DIMS])
d = d[d.split == "validation"].reset_index(drop=True)


def level(score: float) -> int:
    return int(min(np.floor(score / 10), 9) * 10)


@torch.inference_mode()
def nll(model, tok, ctrls, texts, refs, bs=16) -> np.ndarray:
    out = []
    for i in range(0, len(ctrls), bs):
        p = [tok(prompt(c, t), add_special_tokens=False)["input_ids"] for c, t in zip(ctrls[i:i + bs], texts[i:i + bs])]
        a = [tok(r.strip() + tok.eos_token, add_special_tokens=False)["input_ids"] for r in refs[i:i + bs]]
        seqs = [(pp + aa)[:256] for pp, aa in zip(p, a)]
        n = max(map(len, seqs))
        ids = torch.full((len(seqs), n), tok.pad_token_id)
        lab = torch.full((len(seqs), n), -100)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for j, (s, pp) in enumerate(zip(seqs, p)):
            ids[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
            lab[j, len(pp):len(s)] = torch.tensor(s[len(pp):])
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits.float()
        lp, y = torch.log_softmax(logits[:, :-1], -1), lab[:, 1:].cuda()
        tl = -lp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * (y >= 0)
        out += (tl.sum(1) / (y >= 0).sum(1)).tolist()
    return np.array(out)


def controls(rows: pd.DataFrame, dim: str) -> dict:
    out = {c: [] for c in CONDITIONS}
    for r in rows.itertuples(index=False):
        true, src = int(getattr(r, f"tgt_{dim}")), level(getattr(r, f"src_{dim}"))
        wrong = src - 10 if true <= src - 20 else 0
        full = {x: getattr(r, f"tgt_{x}") for x in DIMS}
        free = {x: "any" for x in DIMS}
        for c, t in (("full", full), ("full-any", {**full, dim: "any"}), ("full-wrong", {**full, dim: str(wrong)}),
                     ("only", {**free, dim: str(true)}), ("only-any", free), ("only-wrong", {**free, dim: str(wrong)})):
            out[c].append(control_string("targets", t))
    return out


def main() -> None:
    for name in sys.argv[1:]:
        tok, model = load(RUNS / name / "adapter")
        for dim in ("lexical", "base"):
            rows = d[d[f"tgt_{dim}"] != "any"]
            r = {c: nll(model, tok, v, rows.original_text.tolist(), rows.simplification.tolist()) for c, v in controls(rows, dim).items()}
            for sub in ("lexical_supplement", "v1"):
                m = (rows.subset == sub).values
                cell = lambda a, b: f"{1000 * (r[a][m] - r[b][m]).mean():+6.1f} ({100 * (r[a][m] > r[b][m]).mean():3.0f}%)"
                print(f"{name:11s} {dim:7s} {sub:18s} pairs {m.sum():4d} | NLL {r['full'][m].mean():.3f} | others given: any {cell('full-any', 'full')}"
                      f" wrong {cell('full-wrong', 'full')} | alone: all-any {cell('only-any', 'only')} wrong {cell('only-wrong', 'only')}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
