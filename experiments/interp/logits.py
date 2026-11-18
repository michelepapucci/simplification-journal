"""Next-token distributions at every decoding step, and how much of their probability goes to De Mauro's basic vocabulary.

Each output is read back through the model that wrote it (teacher forcing). This gives, at every step, the
distribution the token was drawn from (at temperature 1; the outputs were sampled at 0.7). With guidance (w = 3) the
distribution is the guided one, log p(.|any) + w (log p(.|level) - log p(.|any)), renormalised.

Vocabulary classes. A vocabulary item is a word when it is a space followed by letters only (" casa", " Il"). It gets
the NVdB class of that lower-cased form from outputs/interp/lexicon.tsv (lexicon.py): FO, AU, AD, or "out" (an Italian
word outside NVdB). Proper nouns, numbers, foreign words and forms never seen in the corpus are left out. Words
split into several pieces, apostrophe forms (l', dell') and a text's first word are not counted. At every step,
basic share = P(FO + AU + AD) / P(FO + AU + AD + out): of the probability given to classified words, the share
on basic vocabulary.

Two runs per model:
- generated: every test output (7,582 requests), plus the guided model's outputs at w = 3 and with the "no level"
  token -> outputs/interp/logits-<condition>.csv, one row per output;
- fixed: one fixed text per test original (the model's own output at request 40), read under all 11 tokens (levels
  0-90 and "no level"), on every original. Only the token changes, so any difference is the token's effect
  -> outputs/interp/<model>/fixed.csv, plus the summed log-probability of every vocabulary item per token
  (fixed_prob.npz: the summed probability of every item over the steps), for the items each level pushes up or down.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python experiments/interp/logits.py --model g1_guided
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import G1, NULL, OUTPUTS, RUNS, TOKENS, prompt, token  # noqa: E402
from generate import load  # noqa: E402

CLASSES = ("FO", "AU", "AD", "out")
LEXICON = OUTPUTS / "interp" / "lexicon.tsv"
FIXED_LEVEL = 40


def vocabulary(tokenizer) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """M [V, 4] one-hot class of every word item; W [V] word item (space + letters) at all."""
    lex = pd.read_csv(LEXICON, sep="\t", keep_default_na=False)
    form2cls = dict(zip(lex.form, lex["class"]))
    n = len(tokenizer)
    M, W, rows = torch.zeros(n, len(CLASSES)), torch.zeros(n), []
    for i in range(n):
        s = tokenizer.decode([i])
        if len(s) > 1 and s[0] == " " and s[1:].isalpha():
            W[i] = 1
            c = form2cls.get(s[1:].lower())
            if c in CLASSES:
                M[i, CLASSES.index(c)] = 1
                rows.append((i, s, c))
    return M, W, pd.DataFrame(rows, columns=["id", "token", "class"])


class Scorer:
    def __init__(self, tokenizer, model, M, W, batch: int):
        self.tok, self.model, self.batch = tokenizer, model, batch
        self.M, self.W = M.cuda(), W.cuda()

    def encode(self, prompts: list[str], outputs: list[str]):
        p = [self.tok(x, add_special_tokens=False)["input_ids"] for x in prompts]
        o = [self.tok(x.strip(), add_special_tokens=False)["input_ids"][:256 - len(q)] for x, q in zip(outputs, p)]
        return p, o

    @torch.inference_mode()
    def logprobs(self, p_ids, o_ids):
        """Log-probabilities at the steps that predict each output token: list of [n_i, V] (float32, on GPU)."""
        n = max(len(a) + len(b) for a, b in zip(p_ids, o_ids))
        ids = torch.full((len(p_ids), n), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for k, (a, b) in enumerate(zip(p_ids, o_ids)):
            ids[k, :len(a) + len(b)] = torch.tensor(a + b)
            mask[k, :len(a) + len(b)] = 1
        logits = self.model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
        return [logits[k, len(a) - 1:len(a) + len(b) - 1].float().log_softmax(-1) for k, (a, b) in enumerate(zip(p_ids, o_ids))]

    def summarise(self, lp: torch.Tensor, nxt: list[int]) -> dict:
        probs = lp.exp()
        mass = probs @ self.M                                   # [n, 4]
        nxt_t = torch.tensor(nxt, device=lp.device)
        boundary = self.W[nxt_t].bool()                         # the realised token starts a word
        realised = self.M[nxt_t]                                # [n, 4]
        out = {"steps": len(nxt), "boundary_steps": int(boundary.sum()),
               "entropy": float(-(probs * lp).sum(-1).mean()) if len(nxt) else np.nan}
        for j, c in enumerate(CLASSES):
            out[f"p_{c}"] = float(mass[:, j].sum())
            out[f"pb_{c}"] = float(mass[boundary, j].sum())
            out[f"n_{c}"] = int(realised[:, j].sum())
        return out

    def run(self, prompts: list[str], outputs: list[str], null_prompts: list[str] | None = None, w: float = 1.0,
            keep_logp: bool = False) -> tuple[list[dict], torch.Tensor | None]:
        rows, logp_sum = [], None
        for start in range(0, len(prompts), self.batch):
            p_ids, o_ids = self.encode(prompts[start:start + self.batch], outputs[start:start + self.batch])
            lps = self.logprobs(p_ids, o_ids)
            if null_prompts is not None:
                q_ids, _ = self.encode(null_prompts[start:start + self.batch], outputs[start:start + self.batch])
                assert [len(x) for x in q_ids] == [len(x) for x in p_ids]
                lus = self.logprobs(q_ids, o_ids)
                lps = [(lu + w * (lc - lu)).log_softmax(-1) for lc, lu in zip(lps, lus)]
            for lp, o in zip(lps, o_ids):
                rows.append(self.summarise(lp, o))
                if keep_logp:
                    s = lp.exp().sum(0).double()   # summed probability of every item over the steps
                    logp_sum = s if logp_sum is None else logp_sum + s
        return rows, logp_sum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=["g1", "g1_guided"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fixed-only", action="store_true", help="skip the generated outputs")
    args = parser.parse_args()
    out = OUTPUTS / "interp"
    (out / args.model).mkdir(parents=True, exist_ok=True)
    tokenizer, model = load(RUNS / args.model / "adapter")
    M, W, vocab = vocabulary(tokenizer)
    vocab.to_csv(out / "vocabulary_classes.csv", index=False)
    print(f"vocabulary: {len(vocab)} word items with a class: {vocab['class'].value_counts().to_dict()}", flush=True)
    sc = Scorer(tokenizer, model, M, W, args.batch)
    text = dict(pd.read_csv(G1 / "test_set_g1.csv", usecols=["original_sentence_idx", "original_text"]).values)

    conditions = [(args.model, "test", 1.0)]
    if args.model == "g1_guided":
        conditions += [("g1_guided_w3", "test-w3", 3.0), ("g1_guided_null", "test-null", 1.0)]
    for name, file, w in ([] if args.fixed_only else conditions):
        f = pd.read_csv(OUTPUTS / args.model / f"predictions-{file}.csv")
        f["prediction"] = f.prediction.fillna("")
        ctrl = f.control_token if "control_token" in f else [token(lv) for lv in f.level]
        prompts = [prompt(c, text[i]) for c, i in zip(ctrl, f.original_sentence_idx)]
        nulls = [prompt(token(NULL), text[i]) for i in f.original_sentence_idx] if w != 1.0 else None
        rows, _ = sc.run(prompts, f.prediction.tolist(), nulls, w)
        keep = [c for c in ("request", "original_sentence_idx", "level", "src_level") if c in f]
        pd.concat([f[keep].reset_index(drop=True), pd.DataFrame(rows)], axis=1).to_csv(out / f"logits-{name}.csv", index=False)
        print(f"{name}: {len(rows)} outputs", flush=True)

    own = pd.read_csv(OUTPUTS / args.model / "predictions-test.csv")
    fixed = own[own.level == FIXED_LEVEL].set_index("original_sentence_idx").prediction.fillna("")
    test = pd.read_csv(G1 / "test_set_g1.csv", usecols=["original_sentence_idx", "src_level"])
    frames, sums, steps = [], {}, {}
    for tok in TOKENS:
        lv = tok[len("<|readability_"):-2]
        sel = test[test.original_sentence_idx.isin(fixed.index)]  # every token on every original, so the sets match
        rows, s = sc.run([prompt(tok, text[i]) for i in sel.original_sentence_idx], fixed.loc[sel.original_sentence_idx].tolist(),
                         keep_logp=True)
        frames.append(pd.concat([sel.reset_index(drop=True).assign(token=lv), pd.DataFrame(rows)], axis=1))
        sums[lv], steps[lv] = s.cpu().numpy(), int(sum(r["steps"] for r in rows))
        print(f"fixed text under {tok}: {len(rows)} originals", flush=True)
    pd.concat(frames).to_csv(out / args.model / "fixed.csv", index=False)
    np.savez(out / args.model / "fixed_prob.npz", tokens=np.array(list(sums)), sums=np.stack(list(sums.values())),
             steps=np.array(list(steps.values())))


if __name__ == "__main__":
    main()
