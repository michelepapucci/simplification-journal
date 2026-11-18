"""Structural probe (Hewitt & Manning 2019): a linear map B (rank 64) such that ||B(h_i - h_j)||^2 fits the parse-tree
distance between words i and j of a sentence. Here the words are surface units (parse.py), and h is the mean hidden
state of a unit's pieces.

Trained on 2,000 g1 training simplifications (90% train, 10% dev), each read after its original's prompt under its
own level, at layers 4, 8, 12, 16, 20, 24 and 28 (L1 loss on the tree distance, normalised by sentence length
squared, Adam, best dev epoch). Punctuation units are left out.

Evaluated on the same conditions as attention.py. For every text it gives:
- UUAS: the undirected attachment score of the minimum spanning tree of the predicted distances against the parse;
- distance Spearman: between predicted and parse distances, per sentence with 5+ units;
- the predicted dependency length: the mean and max distance in units (|i - j|) over the predicted tree's edges.
-> outputs/interp/<model>/probe.json (dev scores per layer) and probe-<condition>.npz (per text and layer).

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python experiments/interp/probe.py --model g1_guided
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.sparse.csgraph import minimum_spanning_tree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from attention import unit_of_pieces  # noqa: E402
from common import G1, OUTPUTS, RUNS, prompt, token  # noqa: E402
from generate import load  # noqa: E402

PARSES = OUTPUTS / "interp" / "parses"
LAYERS = (4, 8, 12, 16, 20, 24, 28)
RANK, EPOCHS, SEED = 64, 15, 42
LEVELS = (40, 30, 20, 10, 0)


def sentences(parse: dict) -> list[tuple[list[int], np.ndarray]]:
    """Per sentence: the non-punctuation units and their tree distances (BFS over units)."""
    units, words = parse["units"], parse["words"]
    unit = [w[0] for w in words]
    adj = {u: set() for u in range(len(units))}
    for i, (u, h, _) in enumerate(words):
        if h >= 0 and unit[h] != u:
            adj[u].add(unit[h])
            adj[unit[h]].add(u)
    out = []
    for s in sorted({u[4] for u in units}):
        keep = [k for k, u in enumerate(units) if u[4] == s and u[3] != "PUNCT"]
        if len(keep) < 2:
            continue
        D = np.zeros((len(keep), len(keep)))
        for a, src in enumerate(keep):
            dist = {src: 0}
            q = deque([src])
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if y not in dist:
                        dist[y] = dist[x] + 1
                        q.append(y)
            D[a] = [dist.get(t, np.nan) for t in keep]
        if np.isnan(D).any():
            continue
        out.append((keep, D))
    return out


@torch.inference_mode()
def unit_states(tok, model, items: list[tuple], batch: int = 16) -> list[np.ndarray | None]:
    """Mean hidden state of each unit's pieces at LAYERS: [units, layers, D] (float16) per text."""
    res = []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        seqs, spans = [], []
        for p, text, parse, in_prompt in chunk:
            if in_prompt:
                enc = tok(p, return_offsets_mapping=True, add_special_tokens=False)
                off0 = p.index(text)
                pos = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if a >= off0 and b <= off0 + len(text) and b > a]
                offs = [(enc["offset_mapping"][k][0] - off0, enc["offset_mapping"][k][1] - off0) for k in pos]
                ids = enc["input_ids"]
            else:
                p_ids = tok(p, add_special_tokens=False)["input_ids"]
                o = tok(text, return_offsets_mapping=True, add_special_tokens=False)
                ids, pos, offs = p_ids + o["input_ids"], [len(p_ids) + k for k in range(len(o["input_ids"]))], o["offset_mapping"]
            keep = [i for i, q in enumerate(pos) if q < 320]
            seqs.append(ids[:320])
            spans.append(([pos[i] for i in keep], [offs[i] for i in keep]))
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for k, s in enumerate(seqs):
            ids[k, :len(s)] = torch.tensor(s)
            mask[k, :len(s)] = 1
        hidden = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True).hidden_states
        H = torch.stack([hidden[l] for l in LAYERS], dim=2)             # [B, T, layers, D]
        for k, (p, text, parse, _) in enumerate(chunk):
            pos, offs = spans[k]
            owner = unit_of_pieces(offs, parse["units"])
            U = len(parse["units"])
            S = torch.zeros(U, len(LAYERS), H.shape[-1], device="cuda")
            c = torch.zeros(U, device="cuda")
            for q, o in zip(pos, owner):
                if o >= 0:
                    S[o] += H[k, q].float()
                    c[o] += 1
            res.append((S / c.clamp_min(1)[:, None, None]).half().cpu().numpy() if c.sum() > 0 else None)
    return res


def train(samples: list, layer: int, dev: list) -> tuple[torch.Tensor, list]:
    torch.manual_seed(SEED)
    D = samples[0][0].shape[-1]
    B = torch.nn.Parameter(torch.randn(RANK, D, device="cuda") * 0.01)
    opt = torch.optim.Adam([B], lr=1e-3)
    best, best_B, curve = -1, None, []
    rng = np.random.default_rng(SEED)
    for ep in range(EPOCHS):
        order = rng.permutation(len(samples))
        for start in range(0, len(order), 32):
            loss = 0
            for i in order[start:start + 32]:
                X, Dg = samples[i]
                Z = torch.tensor(X[:, layer], device="cuda").float() @ B.T
                pd2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
                loss = loss + (pd2 - torch.tensor(Dg, device="cuda").float()).abs().sum() / len(Dg) ** 2
            opt.zero_grad()
            (loss / 32).backward()
            opt.step()
        u = np.nanmean([evaluate(B.detach(), X[:, layer], Dg)["uuas"] for X, Dg in dev])
        curve.append(float(u))
        if u > best:
            best, best_B = u, B.detach().clone()
    return best_B, curve


def evaluate(B: torch.Tensor, X: np.ndarray, Dg: np.ndarray) -> dict:
    Z = torch.tensor(X, device="cuda").float() @ B.T
    P = ((Z[:, None] - Z[None]) ** 2).sum(-1).cpu().numpy()
    n = len(Dg)
    mst = minimum_spanning_tree(np.triu(P + 1e-6, 1)).tocoo()
    pred = {tuple(sorted((int(a), int(b)))) for a, b in zip(mst.row, mst.col)}
    goldE = {tuple(sorted((a, b))) for a in range(n) for b in range(a + 1, n) if Dg[a, b] == 1}
    iu = np.triu_indices(n, 1)
    rho = stats.spearmanr(P[iu], Dg[iu]).statistic if n >= 5 else np.nan
    lens = [abs(a - b) for a, b in pred]
    return {"uuas": len(pred & goldE) / max(1, len(goldE)), "rho": rho, "pred_len": float(np.mean(lens)) if lens else np.nan,
            "pred_max": float(max(lens)) if lens else np.nan,
            "gold_len": float(np.mean([abs(a - b) for a, b in goldE])) if goldE else np.nan}


def text_scores(Bs: dict, S: np.ndarray, parse: dict) -> np.ndarray:
    """[layers, 5]: mean over the text's sentences of uuas, rho, pred_len, pred_max, gold_len (positions in units of the
    sentence's non-punctuation sequence)."""
    out = np.full((len(LAYERS), 5), np.nan)
    sents = sentences(parse)
    if S is None or not sents:
        return out
    for li in range(len(LAYERS)):
        rows = [evaluate(Bs[li], S[keep, li].astype(np.float32), Dg) for keep, Dg in sents]
        out[li] = [np.nanmean([r[k] for r in rows]) for k in ("uuas", "rho", "pred_len", "pred_max", "gold_len")]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    out = OUTPUTS / "interp" / args.model
    tok, model = load(RUNS / args.model / "adapter")
    t0 = time.time()
    train_df = pd.read_csv(G1 / "training_set_g1.zip", usecols=["split", "original_text", "simplification", "tgt_level"])
    train_df = train_df[train_df.split == "train"].sample(2000, random_state=SEED)
    parses = json.loads((PARSES / "training.json").read_text())
    items = [(prompt(token(l), o), p["text"], p, False) for (o, l), p in zip(zip(train_df.original_text, train_df.tgt_level), parses)]
    states = unit_states(tok, model, items)
    data = []
    for S, p in zip(states, parses):
        if S is None:
            continue
        for keep, Dg in sentences(p):
            data.append((S[keep], Dg))
    cut = int(0.9 * len(data))
    tr, dev = data[:cut], data[cut:]
    print(f"probe data: {len(tr)} train / {len(dev)} dev sentences ({time.time() - t0:.0f}s)", flush=True)
    Bs, curves = {}, {}
    for li, layer in enumerate(LAYERS):
        Bs[li], curves[layer] = train(tr, li, dev)
        print(f"layer {layer}: dev UUAS by epoch {np.round(curves[layer], 3).tolist()}", flush=True)
    (out / "probe.json").write_text(json.dumps({"layers": LAYERS, "rank": RANK, "dev_uuas": curves,
                                                "train_sentences": len(tr), "dev_sentences": len(dev)}))
    torch.save({l: Bs[i].cpu() for i, l in enumerate(LAYERS)}, out / "probe_maps.pt")

    test = pd.read_csv(G1 / "test_set_g1.csv")
    otext = dict(zip(test.original_sentence_idx, test.original_text))
    P = {k: json.loads((PARSES / f"{k}.json").read_text()) for k in ("originals", "references", "outputs")}
    refs = pd.read_csv(G1 / "test_references_g1.csv", usecols=["original_sentence_idx", "level"])
    outs = pd.read_csv(OUTPUTS / "g1_guided" / "predictions-test.csv", usecols=["original_sentence_idx", "level"])
    fixed_rows = outs.index[outs.level == 40].tolist()
    conds = {f"originals@{lv}": [(prompt(token(lv), o), o, P["originals"][i], True) for i, o in enumerate(test.original_text)] for lv in LEVELS}
    conds.update({f"fixed@{lv}": [(prompt(token(lv), otext[outs.original_sentence_idx[r]]), P["outputs"][r]["text"], P["outputs"][r], False)
                                  for r in fixed_rows] for lv in LEVELS})
    conds["references"] = [(prompt(token(l), otext[i]), P["references"][k]["text"], P["references"][k], False)
                           for k, (i, l) in enumerate(zip(refs.original_sentence_idx, refs.level))]
    conds["outputs"] = [(prompt(token(l), otext[i]), P["outputs"][k]["text"], P["outputs"][k], False)
                        for k, (i, l) in enumerate(zip(outs.original_sentence_idx, outs.level))]
    for name, items in conds.items():
        t1 = time.time()
        S = unit_states(tok, model, items)
        scores = np.stack([text_scores(Bs, s, it[2]) for s, it in zip(S, items)])
        np.savez_compressed(out / f"probe-{name}.npz", scores=scores, layers=np.array(LAYERS),
                            columns=np.array(["uuas", "rho", "pred_len", "pred_max", "gold_len"]))
        print(f"{name}: {len(items)} texts, mean UUAS by layer {np.round(np.nanmean(scores[:, :, 0], 0), 3).tolist()} ({time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
