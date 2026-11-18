"""Generate the g1 requests (data/g1/{test,dev}_requests_g1.csv): every level below each original's own.

Decoding as in the journal's earlier evaluation (and GFFF/evaluate_model_against_original.py): sampling with
temperature 0.7, top-p 1.0, at most 128 new tokens, output cut at the first end-of-sequence, seed 42.

Three modes:
- default: one output per request -> outputs/<model>/predictions-<set>.csv;
- --samples k: k more outputs per request (seed 43), for best-of-n selection together with the default output
  -> outputs/<model>/samples-<set>.csv, one row per (request, sample);
- --guidance w: classifier-free guidance against the "no level" token <|readability_any|> (only the guided model
  learned it). At every step the model reads the original twice, with the requested token and with the "no
  level" token; the next token is sampled from  log p(.|no level) + w * (log p(.|level) - log p(.|no level)),
  at temperature 0.7. w = 1 is plain generation; w > 1 strengthens what the level token changes
  -> outputs/<model>/predictions-<set>-w<w>.csv.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python experiments/generate.py --model g1 --set test
Output columns: the request's columns (without the original's text), control_token, prediction.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import BASE_MODEL, G1, NULL, OUTPUTS, RUNS, SEED, prompt, token

TEMPERATURE, MAX_NEW_TOKENS = 0.7, 128


def load(adapter: Path):
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload().to("cuda").eval()
    return tokenizer, model


def decode(tokenizer, rows) -> list[str]:
    return [tokenizer.decode(r, skip_special_tokens=True).strip() for r in rows]


@torch.inference_mode()
def generate(tokenizer, model, prompts: list[str], batch: int, samples: int = 1) -> list[str]:
    out = []
    for start in range(0, len(prompts), batch):
        enc = tokenizer(prompts[start:start + batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, do_sample=True, temperature=TEMPERATURE, top_p=1.0, max_new_tokens=MAX_NEW_TOKENS,
                             num_return_sequences=samples, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        out += decode(tokenizer, gen[:, enc["input_ids"].shape[1]:])
    return out


@torch.inference_mode()
def generate_guided(tokenizer, model, originals: list[str], levels: list[int], w: float, batch: int,
                    greedy: bool = False) -> list[str]:
    """Classifier-free guidance: the requested-level and "no level" prompts of an original have the same length,
    so they share padding, attention mask and positions, and both always receive the chosen token."""
    eos, pad, out = tokenizer.eos_token_id, tokenizer.pad_token_id, []
    for start in range(0, len(originals), batch):
        o, lv = originals[start:start + batch], levels[start:start + batch]
        b = len(o)
        enc = tokenizer([prompt(token(x), t) for x, t in zip(lv, o)] + [prompt(token(NULL), t) for t in o],
                        return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        mask = enc["attention_mask"]
        assert torch.equal(mask[:b], mask[b:]), "level and no-level prompts must have the same length"
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        res = model(input_ids=enc["input_ids"], attention_mask=mask, position_ids=pos, use_cache=True)
        done = torch.zeros(b, dtype=torch.bool, device="cuda")
        steps = []
        for _ in range(MAX_NEW_TOKENS):
            logits = res.logits[:, -1].float()
            cond, uncond = logits[:b].log_softmax(-1), logits[b:].log_softmax(-1)
            scores = uncond + w * (cond - uncond)
            nxt = scores.argmax(-1) if greedy else torch.multinomial((scores / TEMPERATURE).softmax(-1), 1).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
            steps.append(nxt)
            done |= nxt == eos
            if bool(done.all()):
                break
            mask = torch.cat([mask, mask.new_ones((2 * b, 1))], dim=1)
            pos = pos[:, -1:] + 1
            res = model(input_ids=torch.cat([nxt, nxt])[:, None], attention_mask=mask, position_ids=pos,
                        past_key_values=res.past_key_values, use_cache=True)
        gen = torch.stack(steps, dim=1)
        rows = []
        for r in gen.tolist():
            rows.append(r[: r.index(eos)] if eos in r else r)
        out += decode(tokenizer, rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="run name (runs/<model>/adapter); outputs go to outputs/<model>/")
    parser.add_argument("--set", default="test", choices=["test", "dev"])
    parser.add_argument("--samples", type=int, help="k more sampled outputs per request (best-of-n)")
    parser.add_argument("--guidance", type=float, help="classifier-free guidance weight w")
    parser.add_argument("--null", action="store_true", help="one output per original with the \"no level\" token (guided model) -> predictions-<set>-null.csv")
    parser.add_argument("--limit", type=int, help="only the first N requests (smoke tests)")
    parser.add_argument("--batch", type=int, default=96)
    args = parser.parse_args()
    out = OUTPUTS / args.model
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load(RUNS / args.model / "adapter")
    req = pd.read_csv(G1 / f"{args.set}_requests_g1.csv")
    if args.limit:
        req = req.head(args.limit)
    keep = [c for c in req.columns if c != "original_text"]
    t0 = time.time()
    if args.null:
        orig = pd.read_csv(G1 / f"{args.set}_set_g1.csv", usecols=["original_sentence_idx", "original_text", "src_level"])
        torch.manual_seed(SEED)
        preds = generate(tokenizer, model, [prompt(token(NULL), o) for o in orig.original_text], args.batch)
        frame = orig.drop(columns="original_text").assign(control_token=token(NULL), prediction=preds)
        path = out / f"predictions-{args.set}-null.csv"
    elif args.samples:
        torch.manual_seed(SEED + 1)
        preds = generate(tokenizer, model, [prompt(token(lv), o) for lv, o in zip(req.level, req.original_text)],
                         max(1, args.batch // args.samples), args.samples)
        frame = req.loc[req.index.repeat(args.samples), keep].assign(
            sample=list(range(1, args.samples + 1)) * len(req), prediction=preds)
        path = out / f"samples-{args.set}.csv"
    elif args.guidance is not None:
        torch.manual_seed(SEED)
        preds = generate_guided(tokenizer, model, req.original_text.tolist(), req.level.tolist(), args.guidance,
                                max(1, args.batch // 2))
        frame = req[keep].assign(control_token=[token(lv) for lv in req.level], guidance=args.guidance, prediction=preds)
        path = out / f"predictions-{args.set}-w{args.guidance:g}.csv"
    else:
        torch.manual_seed(SEED)
        preds = generate(tokenizer, model, [prompt(token(lv), o) for lv, o in zip(req.level, req.original_text)], args.batch)
        frame = req[keep].assign(control_token=[token(lv) for lv in req.level], prediction=preds)
        path = out / f"predictions-{args.set}.csv"
    frame.to_csv(path, index=False)
    print(f"{path.name}: {len(frame)} generations in {time.time() - t0:.0f}s, {sum(p == '' for p in preds)} empty", flush=True)


if __name__ == "__main__":
    main()
