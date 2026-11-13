"""Generate the evaluation requests (eval_requests/, built by build_requests.py) with a v1 model.

- main: eval_requests/main-<dim>.csv, one-dimension requests (every level below the original's own level
  in <dim>, the other two dimensions "any") -> outputs/<variant>/predictions-<dim>.csv;
- joint: eval_requests/joint.csv, all three dimensions set -> outputs/<variant>/predictions-joint.csv.
The source_targets variant also gets the original's three source levels. Decoding as in
GFFF/evaluate_model_against_original.py: sampling with temperature 0.7, top-p 1.0, at most 128 new
tokens, output cut at the first end-of-sequence.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python data/v1/experiments/generate.py --variant targets
Output columns: the request's columns, control_token, prediction.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import BASE_MODEL, DIMS, HERE, OUTPUTS, RUNS, SEED, V1, VARIANTS, control_string, prompt

REQUESTS = HERE / "eval_requests"
TGT = {f"tgt_{d}": str for d in DIMS}


def load(adapter: Path):
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload().to("cuda").eval()
    return tokenizer, model


@torch.inference_mode()
def generate(tokenizer, model, prompts: list[str], batch: int) -> list[str]:
    out = []
    for start in range(0, len(prompts), batch):
        enc = tokenizer(prompts[start:start + batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, do_sample=True, temperature=0.7, top_p=1.0, max_new_tokens=128,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            out.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return out


def originals() -> dict:
    frames = [pd.read_csv(V1 / f"{d}_test_set_v1.csv", usecols=["original_sentence_idx", "original_text"]) for d in DIMS]
    both = pd.concat(frames).drop_duplicates("original_sentence_idx")
    return dict(zip(both.original_sentence_idx, both.original_text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--adapter", type=Path, help="default: runs/<variant>/adapter")
    parser.add_argument("--out", type=Path, help="default: outputs/<variant>")
    parser.add_argument("--sets", nargs="+", default=[*DIMS, "joint"], choices=[*DIMS, "joint"])
    parser.add_argument("--limit", type=int, help="only the first N requests of each set (smoke tests)")
    parser.add_argument("--batch", type=int, default=96)
    args = parser.parse_args()
    adapter = args.adapter or RUNS / args.variant / "adapter"
    out = args.out or OUTPUTS / args.variant
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load(adapter)
    text = originals()

    for name in args.sets:
        req = pd.read_csv(REQUESTS / (f"main-{name}.csv" if name in DIMS else "joint.csv"), dtype=TGT)
        if args.limit:
            req = req.head(args.limit)
        ctrl = [control_string(args.variant, {d: getattr(r, f"tgt_{d}") for d in DIMS}, {d: getattr(r, f"src_{d}") for d in DIMS})
                for r in req.itertuples(index=False)]
        torch.manual_seed(SEED)
        t0 = time.time()
        preds = generate(tokenizer, model, [prompt(c, text[i]) for c, i in zip(ctrl, req.idx)], args.batch)
        req.assign(control_token=ctrl, prediction=preds).to_csv(out / f"predictions-{name}.csv", index=False)
        print(f"{name}: {len(req)} generations in {time.time() - t0:.0f}s, {sum(p == '' for p in preds)} empty", flush=True)


if __name__ == "__main__":
    main()
