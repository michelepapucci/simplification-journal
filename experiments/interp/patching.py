"""2.3: activation patching on g1 · guided, from a level-40 run into a level-0 run (greedy decoding, 500 test originals).

The two prompts differ only in the control token (one vocabulary item), so every position lines up.
- A, the request pathway: in the level-0 run, the control token's key and value projections (k_proj, v_proj outputs at
  its position) are replaced by the level-40 run's in a window of layers (0-9, 10-18, 19-27, or all 0-27). Later positions
  read the control token only through these, during the prompt and, from the cache, at every generated step. "All" must
  give exactly the level-40 output: a check on the patch.
- B, the syntax heads: during the prompt (prefill), the outputs of part F's 108 syntax heads (their slice of the input to
  o_proj) are replaced by the level-40 run's at every prompt position; generation then goes on unpatched. Control: 108
  random other heads (seed 42).
Writes outputs/g1_guided_patch/predictions-<condition>.csv (score_texts.py --model g1_guided_patch scores them).

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python experiments/interp/patching.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import G1, OUTPUTS, RESULTS, RUNS, prompt, token  # noqa: E402
from generate import load  # noqa: E402

N, SEED, BATCH, MAX_NEW = 500, 42, 25, 128
WINDOWS = {"A_0-9": range(0, 10), "A_10-18": range(10, 19), "A_19-27": range(19, 28), "A_all": range(0, 28)}
OUT = OUTPUTS / "g1_guided_patch"


class Patcher:
    def __init__(self, model):
        self.model, self.layers = model, model.model.layers
        self.mode, self.store, self.patch = None, {}, {}
        self.ctrl = None                                   # [B] index of the control token in each row
        self.kv_layers, self.heads = set(), {}             # A: layers to patch; B: layer -> head indices
        cfg = model.config
        self.d = cfg.head_dim
        for l, layer in enumerate(self.layers):
            layer.self_attn.k_proj.register_forward_hook(self._kv(l, "k"))
            layer.self_attn.v_proj.register_forward_hook(self._kv(l, "v"))
            layer.self_attn.o_proj.register_forward_pre_hook(self._heads(l))

    def _kv(self, l, which):
        def f(mod, inp, out):
            if out.shape[1] == 1:                          # a generation step: position 0 is in the cache already
                return out
            if self.mode == "capture":
                self.store[(which, l)] = out.detach().clone()
            elif self.mode == "patch" and l in self.kv_layers:
                out = out.clone()
                rows = torch.arange(out.shape[0], device=out.device)
                out[rows, self.ctrl] = self.store[(which, l)][rows, self.ctrl]
                return out
            return out
        return f

    def _heads(self, l):
        def f(mod, args):
            x = args[0]
            if x.shape[1] == 1:
                return None
            if self.mode == "capture":
                self.store[("o", l)] = x.detach().clone()
            elif self.mode == "patch" and l in self.heads:
                x = x.clone()
                src = self.store[("o", l)]
                for h in self.heads[l]:
                    x[:, :, h * self.d:(h + 1) * self.d] = src[:, :, h * self.d:(h + 1) * self.d]
                return (x,)
            return None
        return f


def generate(tok, model, prompts):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        g = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return [tok.decode(r, skip_special_tokens=True).strip() for r in g[:, enc["input_ids"].shape[1]:]], enc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok, model = load(RUNS / "g1_guided" / "adapter")
    tok.padding_side = "left"
    pt = Patcher(model)
    test = pd.read_csv(G1 / "test_set_g1.csv")
    sel = test.sample(N, random_state=SEED).sort_values("original_sentence_idx")
    syntax = json.loads((RESULTS / "attention_g1.json").read_text())["models"]["g1_guided"]["syntax_heads"]["heads"]
    rng = np.random.default_rng(SEED)
    others = [h for h in range(28 * 16) if h not in set(syntax)]
    random_heads = sorted(rng.choice(others, len(syntax), replace=False).tolist())
    group = lambda hs: {l: [h % 16 for h in hs if h // 16 == l] for l in range(28) if any(h // 16 == l for h in hs)}
    conds = ["base40", "base0", *WINDOWS, "B_syntax", "B_random"]
    texts = {c: [] for c in conds}
    t0 = time.time()
    for start in range(0, len(sel), BATCH):
        orig = sel.original_text.iloc[start:start + BATCH].tolist()
        p40, p0 = [prompt(token(40), o) for o in orig], [prompt(token(0), o) for o in orig]
        pt.mode = None
        out40, enc = generate(tok, model, p40)                       # unpatched level 40
        texts["base40"] += out40
        out0, _ = generate(tok, model, p0)
        texts["base0"] += out0
        pt.ctrl = (enc["attention_mask"] == 0).sum(1)                 # left padding: the control token follows the pads
        pt.mode = "capture"
        with torch.inference_mode():
            model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        pt.mode = "patch"
        for name, layers in WINDOWS.items():
            pt.kv_layers, pt.heads = set(layers), {}
            texts[name] += generate(tok, model, p0)[0]
        pt.kv_layers = set()
        for name, hs in (("B_syntax", syntax), ("B_random", random_heads)):
            pt.heads = group(hs)
            texts[name] += generate(tok, model, p0)[0]
        pt.mode, pt.heads = None, {}
        print(f"{start + len(orig)}/{len(sel)} originals ({time.time() - t0:.0f}s)", flush=True)
    for c in conds:
        pd.DataFrame({"original_sentence_idx": sel.original_sentence_idx.to_numpy(), "src_level": sel.src_level.to_numpy(),
                      "level": 40 if c == "base40" else 0, "condition": c, "prediction": texts[c]}).to_csv(OUT / f"predictions-{c}.csv", index=False)
    same = {c: float(np.mean([a == b for a, b in zip(texts[c], texts["base40"])])) for c in conds}
    (OUT / "patching.json").write_text(json.dumps({"n": N, "syntax_heads": syntax, "random_heads": random_heads,
                                                   "identical_to_base40": same,
                                                   "identical_to_base0": {c: float(np.mean([a == b for a, b in zip(texts[c], texts["base0"])])) for c in conds}}, indent=1))
    print("identical to the level-40 output:", {c: round(v, 3) for c, v in same.items()}, flush=True)


if __name__ == "__main__":
    main()
