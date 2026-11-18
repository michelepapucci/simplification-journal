"""G: value zeroing (Mohebbi et al. 2023) for g1 · guided, with the norm-based score (Kobayashi et al. 2020).

For a layer with input X, attention weights A (heads x queries x keys) and values V, the attention output of query i is
sum_j d_ij with d_ij = sum_h A[h,i,j] W_o^h V[h,j] (no bias). Zeroing position j's value removes d_ij, so the layer output
with j zeroed is y_ij = h_ij + MLP(LN(h_ij)) with h_ij = X_i + sum_k d_ik - d_ij. The value-zeroing score of j for i is the
cosine distance between y_ij and the unzeroed y_i, over keys j <= i, normalised to sum to 1 over j. The norm-based score is
||d_ij||, normalised the same way. Everything is in fp32 (TF32 products).

Queries are the text's pieces and the generation point (the prompt's final newline); keys are every position. Per text and
layer the script stores:
- exactness: mean cosine between the recomputed y_i and the model's own layer output (G1);
- the control token's (position 0) share by value zeroing, by attention (mean over heads) and by norm, at the text's
  pieces (mean), at the generation point and at the text's full stops (G2);
- the text's word-to-word map by value zeroing and by norm, run through attention.text_stats: per layer (instead of per
  head) syntactic accuracy per relation and the entropy and distance indices (G3), relation counts summed by split
  (selection: the first 500 test originals; heldout: the rest).

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python experiments/interp/vz.py --conditions all
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
sys.path.insert(0, str(HERE))
from attention import load_eager, text_stats, unit_of_pieces  # noqa: E402
from common import G1, OUTPUTS, RUNS, prompt, token  # noqa: E402

PARSES = OUTPUTS / "interp" / "parses"
MODEL = "g1_guided"
SEED = 42
N_REFS, N_OUT_PER_LEVEL = 2000, 250
torch.backends.cuda.matmul.allow_tf32 = True


class VZ:
    def __init__(self, model):
        self.model = model
        cfg = model.config
        self.H, self.KV, self.d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.layers = model.model.layers
        self.values = {}
        for l, layer in enumerate(self.layers):
            layer.self_attn.v_proj.register_forward_hook(self._hook(l))
        # fp32 copies of the parts of each layer that come after the attention weights
        self.Wo = [layer.self_attn.o_proj.weight.float() for layer in self.layers]          # [D, H*d]
        self.ln = [layer.post_attention_layernorm for layer in self.layers]
        self.mlp = [layer.mlp for layer in self.layers]

    def _hook(self, l):
        def f(mod, inp, out):
            self.values[l] = out.detach()
        return f

    def mlp_f32(self, l, x):
        ln, mlp = self.ln[l], self.mlp[l]
        var = x.pow(2).mean(-1, keepdim=True)
        h = x * torch.rsqrt(var + ln.variance_epsilon) * ln.weight.float()
        g = torch.nn.functional.linear(h, mlp.gate_proj.weight.float())
        u = torch.nn.functional.linear(h, mlp.up_proj.weight.float())
        return torch.nn.functional.linear(torch.nn.functional.silu(g) * u, mlp.down_proj.weight.float())

    @torch.inference_mode()
    def run(self, ids: list[int], queries: list[int]):
        """Per layer: VZ share [Q, T], norm share [Q, T], attention to position 0 [Q], exactness (mean cosine)."""
        x = torch.tensor([ids], device="cuda")
        out = self.model.model(input_ids=x, output_attentions=True, output_hidden_states=True)
        T = len(ids)
        q = torch.tensor(queries, device="cuda")
        causal = (torch.arange(T, device="cuda")[None, :] <= q[:, None]).float()        # [Q, T]
        res = []
        for l in range(len(self.layers)):
            A = out.attentions[l][0][:, q].float()                                      # [H, Q, T]
            V = self.values[l][0].float().view(T, self.KV, self.d).transpose(0, 1)       # [KV, T, d]
            V = V.repeat_interleave(self.H // self.KV, dim=0)                            # [H, T, d]
            Wo = self.Wo[l].view(-1, self.H, self.d).permute(1, 2, 0)                   # [H, d, D]
            U = torch.einsum("htd,hde->hte", V, Wo)                                    # [H, T, D]
            dmat = torch.einsum("hqt,hte->qte", A, U)                                  # [Q, T, D]
            Xin = out.hidden_states[l][0][q].float()                                    # [Q, D]
            h = Xin + dmat.sum(1)                                                       # [Q, D]
            y = h + self.mlp_f32(l, h)
            ref = out.hidden_states[l + 1][0][q].float()
            if l == len(self.layers) - 1:                                               # the last hidden state is after the final norm
                fn = self.model.model.norm
                yy = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + fn.variance_epsilon) * fn.weight.float()
            else:
                yy = y
            exact = torch.nn.functional.cosine_similarity(yy, ref, dim=-1).mean().item()
            hz = h[:, None, :] - dmat                                                   # [Q, T, D]
            vz = torch.empty(len(queries), T, device="cuda")
            for s in range(0, T, 64):                                                    # chunks of keys, to bound memory
                yz = hz[:, s:s + 64] + self.mlp_f32(l, hz[:, s:s + 64])
                vz[:, s:s + 64] = 1 - torch.nn.functional.cosine_similarity(yz, y[:, None, :], dim=-1)
            vz = vz.clamp_min(0) * causal
            vz = vz / vz.sum(-1, keepdim=True).clamp_min(1e-12)
            nm = dmat.norm(dim=-1) * causal
            nm = nm / nm.sum(-1, keepdim=True).clamp_min(1e-12)
            res.append((vz, nm, A.mean(0)[:, 0], exact))
        return res


def build(test, parses_o, parses_r, parses_out):
    otext = dict(zip(test.original_sentence_idx, test.original_text))
    refs = pd.read_csv(G1 / "test_references_g1.csv", usecols=["original_sentence_idx", "level"])
    outs = pd.read_csv(OUTPUTS / MODEL / "predictions-test.csv", usecols=["original_sentence_idx", "level"])
    fixed_rows = outs.index[outs.level == 40].tolist()
    rng = np.random.default_rng(SEED)
    ref_rows = np.sort(rng.choice(len(refs), N_REFS, replace=False))
    out_rows = np.sort(np.concatenate([rng.choice(np.where(outs.level == lv)[0], N_OUT_PER_LEVEL, replace=False)
                                       for lv in range(0, 80, 10)]))
    conds = {}
    for lv in (40, 20, 0):
        conds[f"originals@{lv}"] = [(prompt(token(lv), o), o, parses_o[i], True, i_o, None)
                                    for i, (i_o, o) in enumerate(zip(test.original_sentence_idx, test.original_text))]
        conds[f"fixed@{lv}"] = [(prompt(token(lv), otext[outs.original_sentence_idx[r]]), parses_out[r]["text"], parses_out[r], False,
                                 outs.original_sentence_idx[r], r) for r in fixed_rows]
    conds["references"] = [(prompt(token(refs.level[k]), otext[refs.original_sentence_idx[k]]), parses_r[k]["text"], parses_r[k], False,
                            refs.original_sentence_idx[k], int(k)) for k in ref_rows]
    conds["outputs"] = [(prompt(token(outs.level[k]), otext[outs.original_sentence_idx[k]]), parses_out[k]["text"], parses_out[k], False,
                         outs.original_sentence_idx[k], int(k)) for k in out_rows]
    return conds


def encode(tok, p, text, in_prompt):
    """ids, the text's piece positions and their offsets in the text, and the generation point (last prompt position)."""
    if in_prompt:
        enc = tok(p, return_offsets_mapping=True, add_special_tokens=False)
        off0 = p.index(text)
        pos = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if a >= off0 and b <= off0 + len(text) and b > a]
        offs = [(enc["offset_mapping"][k][0] - off0, enc["offset_mapping"][k][1] - off0) for k in pos]
        return enc["input_ids"], pos, offs, len(enc["input_ids"]) - 1
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    o = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    return p_ids + o["input_ids"], [len(p_ids) + k for k in range(len(o["input_ids"]))], o["offset_mapping"], len(p_ids) - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--conditions", nargs="+", default=["all"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    out = OUTPUTS / "interp" / MODEL
    tok, model = load_eager(RUNS / MODEL / "adapter")
    vzm = VZ(model)
    test = pd.read_csv(G1 / "test_set_g1.csv")
    first = set(test.original_sentence_idx[:500])
    P = {k: json.loads((PARSES / f"{k}.json").read_text()) for k in ("originals", "references", "outputs")}
    conds = build(test, P["originals"], P["references"], P["outputs"])
    names = list(conds) if args.conditions == ["all"] else args.conditions
    L = len(vzm.layers)
    for name in names:
        items = conds[name][:args.limit] if args.limit else conds[name]
        t0 = time.time()
        g2 = {k: np.full((len(items), L), np.nan) for k in ("exact", "vz_text", "vz_gen", "vz_stop", "att_text", "att_gen", "att_stop",
                                                            "norm_text", "norm_gen", "norm_stop")}
        stats = {kind: {k: np.zeros((len(items), L)) for k in ("dist", "long", "ent")} for kind in ("vz", "norm")}
        for kind in stats:
            stats[kind]["nq"] = np.zeros(len(items))
        rel = {kind: {sp: {} for sp in ("selection", "heldout")} for kind in ("vz", "norm")}
        rows = []
        for n, (p, text, parse, in_prompt, orig, row) in enumerate(items):
            rows.append(-1 if row is None else row)
            ids, pos, offs, gen = encode(tok, p, text, in_prompt)
            if len(ids) > 320 or len(pos) < 2:
                continue
            res = vzm.run(ids, pos + [gen])
            owner = unit_of_pieces(offs, parse["units"])
            stop = [k for k, o in enumerate(owner) if o >= 0 and parse["units"][o][2] in (".", "!", "?")]
            span = torch.tensor(pos, device="cuda")
            for l, (vz, nm, att0, exact) in enumerate(res):
                g2["exact"][n, l] = exact
                for key, mat in (("vz", vz[:, 0]), ("norm", nm[:, 0]), ("att", att0)):
                    g2[f"{key}_text"][n, l] = mat[:-1].mean().item()
                    g2[f"{key}_gen"][n, l] = mat[-1].item()
                    if stop:
                        g2[f"{key}_stop"][n, l] = mat[torch.tensor(stop, device="cuda")].mean().item()
            sp = "selection" if orig in first else "heldout"
            for kind, k_ in (("vz", 0), ("norm", 1)):
                M = torch.stack([r_[k_][:-1][:, span] for r_ in res])                    # [L, P, P]: the text's pieces among themselves
                st = text_stats(M, owner, parse)
                for k in ("dist", "long", "ent"):
                    stats[kind][k][n] = st[k]
                stats[kind]["nq"][n] = st["nq"]
                for rname, (c, t, b) in st["rel"].items():
                    C, Tt, B = rel[kind][sp].get(rname, (np.zeros(L), 0, np.zeros(3)))
                    rel[kind][sp][rname] = (C + c, Tt + t, B + b)
            if n % 500 == 0:
                print(f"  {name} {n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
        save = {f"g2_{k}": v for k, v in g2.items()}
        for kind in stats:
            for k, v in stats[kind].items():
                save[f"{kind}_{k}"] = v
            for sp in ("selection", "heldout"):
                names_ = sorted(rel[kind][sp])
                save[f"{kind}_{sp}_rel_names"] = np.array(names_)
                save[f"{kind}_{sp}_rel_correct"] = np.stack([rel[kind][sp][r_][0] for r_ in names_]) if names_ else np.zeros((0, L))
                save[f"{kind}_{sp}_rel_total"] = np.array([rel[kind][sp][r_][1] for r_ in names_])
                save[f"{kind}_{sp}_rel_base"] = np.stack([rel[kind][sp][r_][2] for r_ in names_]) if names_ else np.zeros((0, 3))
        save["rows"] = np.array(rows)
        save["originals"] = np.array([it[4] for it in items])
        np.savez_compressed(out / f"vz-{name}{args.tag}.npz", **save)
        print(f"{name}: {len(items)} texts in {time.time() - t0:.0f}s; exactness min over layers {np.nanmin(np.nanmean(g2['exact'], 0)):.5f}", flush=True)


if __name__ == "__main__":
    main()
