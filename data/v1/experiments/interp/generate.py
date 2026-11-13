"""Greedy generations and hidden states for the interpretability requests (interp/requests.csv).

For each request:
- the greedy output (at most 128 new tokens, cut at the first end-of-sequence), so that two requests on
  the same original differ only because of the control tokens;
- the residual stream at the last prompt token (the newline after the original, where generation
  starts), at every layer (embeddings + 28 blocks), in float16.
It also saves the trained input embeddings of the 33 target tokens.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python data/v1/experiments/interp/generate.py \
        --adapter data/v1/experiments/runs/v5_targets/adapter --out data/v1/experiments/outputs/v5_targets/interp
With --reuse <folder>, requests already generated there (same original, same control string) are copied.
Writes <out>/predictions.csv (requests.csv + control_token, prediction), <out>/states.npy
[request, layer, 1024], <out>/embeddings.npz (tokens, input embeddings).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DIMS, TARGET_TOKENS, control_string, prompt  # noqa: E402
from generate import load, originals  # noqa: E402


@torch.inference_mode()
def run_batch(tokenizer, model, texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """States at the last prompt token and greedy outputs; on out-of-memory, the batch is split in two."""
    try:
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        position_ids = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        # the transformer body only: the full model would also compute logits at every prompt position
        hidden = model.model(**enc, position_ids=position_ids, output_hidden_states=True).hidden_states
        states = torch.stack([h[:, -1, :] for h in hidden], dim=1).float().cpu().numpy().astype(np.float16)
        del hidden
        gen = model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=128,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return states, [tokenizer.decode(row, skip_special_tokens=True).strip() for row in gen[:, enc["input_ids"].shape[1]:]]
    except torch.OutOfMemoryError:
        if len(texts) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(texts) // 2
        a, b = run_batch(tokenizer, model, texts[:half]), run_batch(tokenizer, model, texts[half:])
        return np.concatenate([a[0], b[0]]), a[1] + b[1]


def run(tokenizer, model, prompts: list[str], batch: int, states: np.ndarray, partial: Path) -> list[str]:
    """Batches of similar lengths; outputs are saved to <partial> every 20 batches, and a rerun resumes from it."""
    out = json.loads(partial.read_text()) if partial.exists() else {}
    out = {int(k): v for k, v in out.items()}
    if out:
        print(f"resuming: {len(out)} outputs already done", flush=True)
    todo = [i for i in range(len(prompts)) if i not in out]
    order = np.array(todo)[np.argsort([len(prompts[i]) for i in todo], kind="stable")]  # similar lengths share a batch
    t0 = time.time()
    for b, start in enumerate(range(0, len(order), batch)):
        ids = order[start:start + batch]
        st, texts = run_batch(tokenizer, model, [prompts[i] for i in ids])
        states[ids] = st
        out.update({int(i): t for i, t in zip(ids, texts)})
        if b % 20 == 0:
            states.flush()
            partial.write_text(json.dumps(out, ensure_ascii=False))
            print(f"{len(out)}/{len(prompts)} in {time.time() - t0:.0f}s", flush=True)
    return [out[i] for i in range(len(prompts))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=Path, default=HERE / "requests.csv")
    parser.add_argument("--limit", type=int, help="only the first N requests (smoke tests)")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--reuse", type=Path, help="a previous output folder of the same model: requests with the same original "
                        "and control string take its output and states instead of being generated again")
    parser.add_argument("--memory-fraction", type=float, default=0.28,
                        help="cap on this process's share of the GPU (another job trains on it)")
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    args.out.mkdir(parents=True, exist_ok=True)
    req = pd.read_csv(args.requests, dtype={f"tgt_{d}": str for d in DIMS})
    if args.limit:
        req = req.head(args.limit)
    tokenizer, model = load(args.adapter)
    text = originals()
    ctrl = [control_string("targets", {d: getattr(r, f"tgt_{d}") for d in DIMS}) for r in req.itertuples(index=False)]
    prompts = [prompt(c, text[i]) for c, i in zip(ctrl, req.idx)]

    emb = model.get_input_embeddings().weight
    ids = tokenizer.convert_tokens_to_ids(TARGET_TOKENS)
    np.savez(args.out / "embeddings.npz", tokens=np.array(TARGET_TOKENS), input=emb[ids].float().cpu().numpy(),
             mean=emb[: ids[0]].float().mean(0).cpu().numpy())

    n_layers = model.config.num_hidden_layers + 1
    partial = args.out / "predictions.partial.json"
    shape = (len(req), n_layers, model.config.hidden_size)
    if partial.exists() and (args.out / "states.npy").exists():
        states = np.load(args.out / "states.npy", mmap_mode="r+")
        assert states.shape == shape, f"states.npy has shape {states.shape}, expected {shape}"
    else:
        partial.unlink(missing_ok=True)
        states = np.lib.format.open_memmap(args.out / "states.npy", mode="w+", dtype=np.float16, shape=shape)
        if args.reuse:  # greedy outputs are deterministic: identical requests need not be generated twice
            old = pd.read_csv(args.reuse / "predictions.csv", keep_default_na=False)
            old_states = np.load(args.reuse / "states.npy", mmap_mode="r")
            where = {(i, c): k for k, (i, c) in enumerate(zip(old.idx, old.control_token))}
            hits = {k: where[(i, c)] for k, (i, c) in enumerate(zip(req.idx, ctrl)) if (i, c) in where}
            for k, j in hits.items():
                states[k] = old_states[j]
            states.flush()
            partial.write_text(json.dumps({k: old.prediction[j] for k, j in hits.items()}, ensure_ascii=False))
            print(f"reused {len(hits)} of {len(req)} requests from {args.reuse}", flush=True)
    t0 = time.time()
    preds = run(tokenizer, model, prompts, args.batch, states, partial)
    states.flush()
    req.assign(control_token=ctrl, prediction=preds).to_csv(args.out / "predictions.csv", index=False)
    partial.unlink(missing_ok=True)
    print(f"{len(req)} generations in {time.time() - t0:.0f}s, {sum(p == '' for p in preds)} empty", flush=True)


if __name__ == "__main__":
    main()
