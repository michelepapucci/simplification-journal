"""Hidden states of the control token, layer by layer (embeddings + 28 blocks), for one g1 model.

- In context: every test original (data/g1/test_set_g1.csv, 1,000) under each of the 11 tokens <|readability_0|> ...
  <|readability_90|> and <|readability_any|>. The state saved is the residual stream at the last prompt token (the
  newline after the original), where generation starts -> states.npy [original, token, layer, 1024], float16.
- Alone: each token as the only input (position 0), the token's own representation in every layer, before it sees
  any text -> alone.npy [token, layer, 1024], float32.
- The rows of the input embedding and of the output layer for the 11 tokens -> embeddings.npz.

Level 90 is never a training target and g1 never sees <|readability_any|>, so those rows keep their initialisation
(the mean of the base vocabulary) and serve as untrained references.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python experiments/interp/states.py --model g1_guided
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import G1, OUTPUTS, RUNS, TOKENS, prompt  # noqa: E402
from generate import load  # noqa: E402


@torch.inference_mode()
def last_states(tokenizer, model, prompts: list[str], batch: int) -> np.ndarray:
    out = []
    for start in range(0, len(prompts), batch):
        enc = tokenizer(prompts[start:start + batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        hidden = model.model(**enc, position_ids=pos, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -1] for h in hidden], dim=1).float().cpu().numpy().astype(np.float16))  # left padding: -1 is the last prompt token
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    out = OUTPUTS / "interp" / args.model
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load(RUNS / args.model / "adapter")
    test = pd.read_csv(G1 / "test_set_g1.csv")
    prompts = [prompt(tok, o) for o in test.original_text for tok in TOKENS]
    states = last_states(tokenizer, model, prompts, args.batch)
    states = states.reshape(len(test), len(TOKENS), *states.shape[1:])
    np.save(out / "states.npy", states)
    with torch.inference_mode():
        ids = torch.tensor(tokenizer.convert_tokens_to_ids(TOKENS), device="cuda")
        hidden = model.model(input_ids=ids[:, None], output_hidden_states=True).hidden_states
        alone = torch.stack([h[:, 0] for h in hidden], dim=1).float().cpu().numpy()
        emb = model.get_input_embeddings().weight[ids].float().cpu().numpy()
        head = model.get_output_embeddings().weight[ids].float().cpu().numpy()
    np.save(out / "alone.npy", alone)
    np.savez(out / "embeddings.npz", tokens=np.array(TOKENS), input=emb, output=head)
    (out / "states.json").write_text(json.dumps({"tokens": TOKENS, "originals": test.original_sentence_idx.tolist(),
                                                 "src_level": test.src_level.tolist(), "shape": list(states.shape)}))
    print(f"{args.model}: states {states.shape}, alone {alone.shape}", flush=True)


if __name__ == "__main__":
    main()
