"""E: how the level token changes the probability of the text's words, by part of speech, and of ending or punctuating.

Each test original's level-40 output (the model's own) is read under the tokens of levels 40, 30, 20, 10 and 0; only the
token changes.
- Words by part of speech: the fixed text is tagged with stanza 1.11.0 (the profiler's pipeline: tokenize, mwt, pos).
  Every word of the text, whatever the number of vocabulary pieces it takes, gets its probability in context: the
  product of its pieces' probabilities. A contraction such as "della" takes its first part's tag (ADP). Punctuation
  is split into full stop, comma, semicolon or colon, and other.
- Ending and punctuating: at every step, the probability of end-of-text, of a full stop, of a comma and of a semicolon
  or colon (single vocabulary items, so these are exact). The text is followed by end-of-text, so the last step gives
  the probability of ending exactly where the text ends.

Writes outputs/interp/<model>/classes.csv (one row per original and level: summed log-probability and count of the
text's words per class, and summed probabilities per step), classes_prob.npz (summed probability of every vocabulary
item per level, for single tokens) and classes_examples.json (6 originals step by step).

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python experiments/interp/classes.py --model g1_guided
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import G1, OUTPUTS, RUNS, prompt, token  # noqa: E402
from generate import load  # noqa: E402
from logits import Scorer  # noqa: E402

STANZA_DIR = Path(os.path.expanduser("~/.cache/stanza/1.11.0/resources"))
LEVELS = (40, 30, 20, 10, 0)
FIXED_LEVEL = 40
WORD_CLASSES = ["NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "DET", "ADP", "PRON", "CCONJ", "SCONJ", "NUM",
                "full stop", "comma", "semicolon or colon", "other punctuation", "other"]
STEP_ITEMS = ["end of text", "full stop", "comma", "semicolon or colon"]
N_EXAMPLES = 6
SEED = 42


def step_items(tokenizer) -> dict[str, torch.Tensor]:
    """Vocabulary items of each step class (end of text, full stop, comma, semicolon or colon)."""
    sets = {k: [] for k in STEP_ITEMS}
    for i in range(len(tokenizer)):
        core = tokenizer.decode([i]).strip()
        if i == tokenizer.eos_token_id:
            sets["end of text"].append(i)
        elif core and core.startswith(".") and set(core) <= {".", "\n"}:
            sets["full stop"].append(i)
        elif core == ",":
            sets["comma"].append(i)
        elif core in (";", ":"):
            sets["semicolon or colon"].append(i)
    return {k: torch.tensor(v, device="cuda") for k, v in sets.items()}


def word_class(text: str, upos: str) -> str:
    if upos == "PUNCT":
        return ("full stop" if text in (".", "...") else "comma" if text == "," else
                "semicolon or colon" if text in (";", ":") else "other punctuation")
    return upos if upos in WORD_CLASSES else "other"


def tag(texts: list[str]) -> list[list[tuple[int, int, str, str]]]:
    """Surface tokens of every text: (start, end, text, class)."""
    import stanza
    nlp = stanza.Pipeline(lang="it", processors="tokenize,mwt,pos", dir=str(STANZA_DIR), download_method=None,
                          use_gpu=True, logging_level="WARN")
    out = []
    for start in range(0, len(texts), 256):
        docs = nlp.bulk_process([stanza.Document([], text=t) for t in texts[start:start + 256]])
        for doc in docs:
            out.append([(t.start_char, t.end_char, t.text, word_class(t.text, t.words[0].upos))
                        for s in doc.sentences for t in s.tokens])
    return out


def assign(offsets: list[tuple[int, int]], words: list[tuple]) -> list[int]:
    """Index of the surface token each vocabulary piece belongs to (the one it overlaps most; -1 for none)."""
    owner = []
    for a, b in offsets:
        best, size = -1, 0
        for j, (s, e, _, _) in enumerate(words):
            ov = min(b, e) - max(a, s)
            if ov > size:
                best, size = j, ov
        owner.append(best)
    return owner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=["g1", "g1_guided"])
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    out = OUTPUTS / "interp" / args.model
    tokenizer, model = load(RUNS / args.model / "adapter")
    sc = Scorer(tokenizer, model, torch.zeros(len(tokenizer), 1), torch.zeros(len(tokenizer)), args.batch)
    items = step_items(tokenizer)
    eos = tokenizer.eos_token_id

    test = pd.read_csv(G1 / "test_set_g1.csv", usecols=["original_sentence_idx", "original_text"])
    own = pd.read_csv(OUTPUTS / args.model / "predictions-test.csv")
    fixed = own[own.level == FIXED_LEVEL].set_index("original_sentence_idx").prediction.fillna("").str.strip()
    test = test[test.original_sentence_idx.isin(fixed[fixed.str.len() > 0].index)].reset_index(drop=True)
    texts = [fixed[i] for i in test.original_sentence_idx]
    words = tag(texts)
    enc = [tokenizer(t, return_offsets_mapping=True, add_special_tokens=False) for t in texts]
    o_ids = [e["input_ids"] + [eos] for e in enc]
    owners = [assign(e["offset_mapping"], w) for e, w in zip(enc, words)]
    rng = np.random.default_rng(SEED)
    pool = [k for k, (o, t) in enumerate(zip(test.original_text, texts)) if 80 <= len(o) <= 200 and 40 <= len(t) <= 220]
    example_rows = set(rng.choice(pool, size=N_EXAMPLES, replace=False).tolist())
    examples = {k: {"original": test.original_text[k], "text": texts[k],
                    "tokens": [tokenizer.decode([t]) for t in o_ids[k]],
                    "words": [{"text": w[2], "class": w[3]} for w in words[k]], "owner": owners[k] + [-1], "levels": {}}
                for k in example_rows}

    rows, item_sums, item_steps = [], {}, {}
    with torch.inference_mode():
        for lv in LEVELS:
            psum, nsteps = None, 0
            prompts = [prompt(token(lv), o) for o in test.original_text]
            for start in range(0, len(prompts), args.batch):
                p_ids = [sc.tok(x, add_special_tokens=False)["input_ids"] for x in prompts[start:start + args.batch]]
                batch_o = o_ids[start:start + args.batch]
                for k, lp in enumerate(sc.logprobs(p_ids, batch_o)):
                    r = start + k
                    tgt = torch.tensor(batch_o[k], device=lp.device)
                    tok_lp = lp.gather(1, tgt[:, None])[:, 0].double().cpu().numpy()        # log p of each actual piece
                    probs = lp.exp()
                    step = {c: probs[:, idx].sum(-1) for c, idx in items.items()}         # [steps]
                    row = {"original_sentence_idx": int(test.original_sentence_idx[r]), "level": lv, "steps": int(len(tgt)),
                           "eos_at_end": float(step["end of text"][-1]),
                           **{f"p_{c}": float(v[:-1].sum()) for c, v in step.items()}}    # every step but the final one
                    sums = {c: [0.0, 0] for c in WORD_CLASSES}
                    word_lp = np.zeros(len(words[r]))
                    for piece, owner in enumerate(owners[r]):
                        if owner >= 0:
                            word_lp[owner] += tok_lp[piece]
                    for j, w in enumerate(words[r]):
                        sums[w[3]][0] += word_lp[j]
                        sums[w[3]][1] += 1
                    for c, (s, n) in sums.items():
                        row[f"lp_{c}"], row[f"n_{c}"] = s, n
                    rows.append(row)
                    s = probs[:-1].sum(0).double()
                    psum = s if psum is None else psum + s
                    nsteps += len(tgt) - 1
                    if r in examples:
                        examples[r]["levels"][lv] = {"steps": {c: [round(float(x), 4) for x in v.cpu().numpy()] for c, v in step.items()},
                                                     "word_logp": [round(float(x), 3) for x in word_lp]}
            item_sums[lv], item_steps[lv] = psum.cpu().numpy(), nsteps
            print(f"level {lv}: {len(prompts)} texts, {nsteps} steps", flush=True)
    pd.DataFrame(rows).to_csv(out / "classes.csv", index=False)
    np.savez(out / "classes_prob.npz", levels=np.array(LEVELS), sums=np.stack([item_sums[lv] for lv in LEVELS]),
             steps=np.array([item_steps[lv] for lv in LEVELS]))
    (out / "classes_examples.json").write_text(json.dumps({"word_classes": WORD_CLASSES, "examples": [examples[k] for k in sorted(examples)]},
                                                          ensure_ascii=False))
    n_words = pd.Series([w[3] for ws in words for w in ws]).value_counts().to_dict()
    print("words per class:", n_words, flush=True)


if __name__ == "__main__":
    main()
