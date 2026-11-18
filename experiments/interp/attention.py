"""Attention of every head (28 layers x 16) over a text's own words: syntactic accuracy and complexity indices.

The model reads each text in its usual place:
- an original: inside the prompt, f"{token}\\n{original}\\n";
- a reference or an output: after the prompt of its original, as the model's own text.

Only the text's own span is used. Each word unit (a surface token from parse.py) attends to the earlier units,
with attention from pieces averaged over a unit's query pieces and summed over its key pieces, renormalised over the
earlier units. The first-token sink and the prompt are therefore left out. For every head and text the script stores:
- non-adjacent task: non-punctuation units whose previous unit is not a syntactic neighbour but an earlier unit is
  (the unit of a word's head, or of a dependent): how often the earlier unit they attend to most is such a neighbour,
  with fixed offsets 2 and 3 as positional baselines. (In Italian the previous unit is a neighbour 92% of the time,
  so the task on every unit is decided by adjacency.)
- per relation (Clark et al. 2019): how often a word whose head is earlier attends most to its head's unit, with
  fixed offsets 1-3 as baselines, summed over the condition's texts;
- the complexity indices, summed over non-punctuation units after the first: expected attention distance (in units),
  share of attention more than 5 units back, and entropy normalised by log(number of earlier units).

Conditions (outputs/interp/<model>/attention-<condition>.npz):
- originals@L: the 1,000 test originals in the prompt under level L (40, 30, 20, 10, 0);
- references: the 7,968 test references after their original's prompt, under their own level;
- outputs: g1 · guided's 7,582 test outputs after their request's prompt;
- fixed@L: each original's level-40 output read under level L (40, 30, 20, 10, 0).

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python experiments/interp/attention.py --model g1_guided --conditions all
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import BASE_MODEL, G1, OUTPUTS, RUNS, prompt, token  # noqa: E402

PARSES = OUTPUTS / "interp" / "parses"
LEVELS = (40, 30, 20, 10, 0)
LONG = 5


def load_eager(adapter: Path):
    tok = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.resize_token_embeddings(len(tok), pad_to_multiple_of=8)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload().to("cuda").eval()
    return tok, model


def unit_of_pieces(offsets: list, units: list) -> list[int]:
    owner = []
    for a, b in offsets:
        best, size = -1, 0
        for j, (s, e, *_ ) in enumerate(units):
            ov = min(b, e) - max(a, s)
            if ov > size:
                best, size = j, ov
        owner.append(best)
    return owner


def neighbours(parse: dict) -> list[set]:
    """For every unit, the earlier units that hold a syntactic neighbour (head or dependent) of one of its words."""
    unit = [w[0] for w in parse["words"]]
    out = [set() for _ in parse["units"]]
    for i, (u, head, _) in enumerate(parse["words"]):
        if head < 0:
            continue
        hu = unit[head]
        if hu != u:
            if hu < u:
                out[u].add(hu)
            else:
                out[hu].add(u)
    return out


def gold(parse: dict) -> dict:
    """Parser statistics: mean and max dependency length in units (non-punctuation dependents, arcs between units),
    mean over sentences of the maximum depth, units, sentences."""
    units, words = parse["units"], parse["words"]
    unit = [w[0] for w in words]
    lens = [abs(unit[h] - unit[i]) for i, (u, h, rel) in enumerate(words) if h >= 0 and rel != "punct" and unit[h] != unit[i]]
    depth = {}

    def d(i, seen=0):
        if i in depth:
            return depth[i]
        h = words[i][1]
        depth[i] = 0 if h < 0 or seen > 100 else d(h, seen + 1) + 1
        return depth[i]

    per_sent = {}
    for i in range(len(words)):
        s = units[unit[i]][4]
        per_sent[s] = max(per_sent.get(s, 0), d(i))
    return {"mean_dep_len": float(np.mean(lens)) if lens else np.nan, "max_dep_len": float(max(lens)) if lens else np.nan,
            "mean_max_depth": float(np.mean(list(per_sent.values()))) if per_sent else np.nan,
            "units": int(sum(1 for u in units if u[3] != "PUNCT")), "sentences": len(per_sent)}


def text_stats(att: torch.Tensor, owner: list[int], parse: dict) -> dict:
    """att: [heads, P, P] attention among the span's pieces (rows: queries). Returns per-head arrays."""
    U = len(parse["units"])
    P = len(owner)
    valid = [k for k in range(P) if owner[k] >= 0]
    Q = torch.zeros(U, P, device=att.device)
    K = torch.zeros(P, U, device=att.device)
    for k in valid:
        K[k, owner[k]] = 1.0
    counts = torch.zeros(U, device=att.device)
    for k in valid:
        counts[owner[k]] += 1
    for k in valid:
        Q[owner[k], k] = 1.0 / counts[owner[k]]
    Au = Q @ att @ K                                                    # [H, U, U]
    H = Au.shape[0]
    idx = torch.arange(U, device=att.device)
    earlier = idx[None, :] < idx[:, None]                               # [U, U]: key earlier than query
    W = Au * earlier
    W = W / W.sum(-1, keepdim=True).clamp_min(1e-12)
    punct = torch.tensor([u[3] == "PUNCT" for u in parse["units"]], device=att.device)
    present = counts > 0
    q_ok = (~punct) & present & (idx >= 1)
    dist = (idx[:, None] - idx[None, :]).float()
    ed = (W * dist).sum(-1)                                             # [H, U]
    lg = (W * (dist > LONG)).sum(-1)
    ent = -(W * W.clamp_min(1e-12).log()).sum(-1) / torch.log(idx.float().clamp_min(2))
    nb = neighbours(parse)
    arg = torch.where(earlier[None], Au, torch.full_like(Au, -1.0)).argmax(-1)    # [H, U]: the earlier unit attended most
    arg_np = arg.cpu().numpy()
    # non-adjacent task: the previous unit is not a syntactic neighbour, but an earlier one is
    na_c, na_t, na_b = np.zeros(H), 0, np.zeros(2)
    for u in range(1, U):
        if not bool(q_ok[u]) or not nb[u] or (u - 1) in nb[u]:
            continue
        na_t += 1
        na_c += np.isin(arg_np[:, u], list(nb[u]))
        na_b += [(u - 2) in nb[u], (u - 3) in nb[u]]
    # per relation (Clark et al. 2019): the word attends most to its own head, when the head is earlier
    unit = [w[0] for w in parse["words"]]
    rel = {}
    for i, (u, head, deprel) in enumerate(parse["words"]):
        if head < 0 or parse["units"][u][3] == "PUNCT":
            continue
        hu = unit[head]
        if hu >= u:
            continue
        r = deprel.split(":")[0]
        c, t, b = rel.get(r, (np.zeros(H), 0, np.zeros(3)))
        rel[r] = (c + (arg_np[:, u] == hu), t + 1, b + np.array([(u - k) == hu for k in (1, 2, 3)]))
    m = q_ok.float()
    return {"na_correct": na_c, "na_total": na_t, "na_base": na_b, "rel": rel,
            "dist": (ed * m).sum(-1).cpu().numpy(), "long": (lg * m).sum(-1).cpu().numpy(),
            "ent": (ent * m).sum(-1).cpu().numpy(), "nq": int(m.sum())}


@torch.inference_mode()
def run(tok, model, items: list[tuple], batch: int) -> dict:
    """items: (prompt string, text, parse, text is in the prompt). Returns stacked per-text arrays."""
    recs = []
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
                ids = p_ids + o["input_ids"]
                pos = [len(p_ids) + k for k in range(len(o["input_ids"]))]
                offs = o["offset_mapping"]
            seqs.append(ids[:320])
            keep = [i for i, q in enumerate(pos) if q < 320]
            spans.append(([pos[i] for i in keep], [offs[i] for i in keep]))
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for k, s in enumerate(seqs):
            ids[k, :len(s)] = torch.tensor(s)
            mask[k, :len(s)] = 1
        att = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), output_attentions=True).attentions
        att = torch.stack(att, dim=1)                                    # [B, L, H, T, T]
        for k, (p, text, parse, _) in enumerate(chunk):
            pos, offs = spans[k]
            if len(pos) < 2 or len(parse["units"]) < 2:
                recs.append(None)
                continue
            pt = torch.tensor(pos, device="cuda")
            a = att[k][:, :, pt][:, :, :, pt].float()                    # [L, H, P, P]
            a = a.reshape(-1, len(pos), len(pos))
            recs.append(text_stats(a, unit_of_pieces(offs, parse["units"]), parse))
        del att
    H = 28 * 16
    out = {k: np.zeros((len(recs), H)) for k in ("na_correct", "dist", "long", "ent")}
    out.update({"na_total": np.zeros(len(recs)), "nq": np.zeros(len(recs)), "na_base": np.zeros((len(recs), 2)), "ok": np.zeros(len(recs), bool)})
    rel = {}
    for i, r in enumerate(recs):
        if r is None:
            continue
        out["ok"][i] = True
        for k in ("na_correct", "dist", "long", "ent"):
            out[k][i] = r[k]
        out["na_total"][i], out["nq"][i], out["na_base"][i] = r["na_total"], r["nq"], r["na_base"]
        for name, (c, t, b) in r["rel"].items():
            C, T, B = rel.get(name, (np.zeros(H), 0, np.zeros(3)))
            rel[name] = (C + c, T + t, B + b)
    names = sorted(rel)
    out["rel_names"] = np.array(names)
    out["rel_correct"] = np.stack([rel[n][0] for n in names]) if names else np.zeros((0, H))
    out["rel_total"] = np.array([rel[n][1] for n in names])
    out["rel_base"] = np.stack([rel[n][2] for n in names]) if names else np.zeros((0, 3))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--conditions", nargs="+", default=["all"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only the first N texts of each condition (smoke tests)")
    parser.add_argument("--tag", default="", help="suffix of the output files (smoke tests)")
    parser.add_argument("--split", choices=["selection", "heldout"], help="only the texts of the first (selection) or last 500 test originals")
    args = parser.parse_args()
    out = OUTPUTS / "interp" / args.model
    tok, model = load_eager(RUNS / args.model / "adapter")
    test = pd.read_csv(G1 / "test_set_g1.csv")
    otext = dict(zip(test.original_sentence_idx, test.original_text))
    P = {k: json.loads((PARSES / f"{k}.json").read_text()) for k in ("originals", "references", "outputs")}
    refs = pd.read_csv(G1 / "test_references_g1.csv", usecols=["original_sentence_idx", "level"])
    outs = pd.read_csv(OUTPUTS / "g1_guided" / "predictions-test.csv", usecols=["original_sentence_idx", "level"])
    fixed_rows = outs.index[outs.level == 40].tolist()
    conds = {}
    for lv in LEVELS:
        conds[f"originals@{lv}"] = [(prompt(token(lv), o), o, P["originals"][i], True) for i, o in enumerate(test.original_text)]
        conds[f"fixed@{lv}"] = [(prompt(token(lv), otext[outs.original_sentence_idx[r]]), P["outputs"][r]["text"], P["outputs"][r], False)
                                for r in fixed_rows]
    conds["references"] = [(prompt(token(l), otext[i]), P["references"][k]["text"], P["references"][k], False)
                           for k, (i, l) in enumerate(zip(refs.original_sentence_idx, refs.level))]
    conds["outputs"] = [(prompt(token(l), otext[i]), P["outputs"][k]["text"], P["outputs"][k], False)
                        for k, (i, l) in enumerate(zip(outs.original_sentence_idx, outs.level))]
    origin = {f"originals@{lv}": list(test.original_sentence_idx) for lv in LEVELS}
    origin.update({f"fixed@{lv}": [outs.original_sentence_idx[r] for r in fixed_rows] for lv in LEVELS})
    origin["references"], origin["outputs"] = list(refs.original_sentence_idx), list(outs.original_sentence_idx)
    if args.split:
        first = set(test.original_sentence_idx[:500])
        want = (lambda i: i in first) if args.split == "selection" else (lambda i: i not in first)
        conds = {k: [it for it, i in zip(v, origin[k]) if want(i)] for k, v in conds.items()}
        args.tag = args.tag or f"_{args.split}"
    names = list(conds) if args.conditions == ["all"] else args.conditions
    for name in names:
        t0 = time.time()
        items = conds[name][:args.limit] if args.limit else conds[name]
        res = run(tok, model, items, args.batch)
        np.savez_compressed(out / f"attention-{name}{args.tag}.npz", **res)
        print(f"{args.model} {name}: {int(res['ok'].sum())}/{len(conds[name])} texts in {time.time() - t0:.0f}s", flush=True)
    if not args.limit and not args.split:
        golds = {k: [gold(p) for p in P[k]] for k in P}
        (out.parent / "parse_gold.json").write_text(json.dumps(golds))


if __name__ == "__main__":
    main()
