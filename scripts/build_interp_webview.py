"""Build data_interp.js for the interpretability page (interp.html of the explorer) from
data/v1/experiments/results/interp_<model>.json (written by data/v1/experiments/interp/analyze.py), one entry per model;
the page has a model selector.

    python scripts/build_interp_webview.py --model v5_targets v7_targets v7_1ep_targets --default v7_targets --out <gh-pages folder>/data_interp.js
Writes window.INTERP_MODELS = {order, default, models: {<model>: payload}}. The findings and the methods shown on the page
are kept here, next to the numbers they describe.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_webview_data import category  # noqa: E402

LABELS = {"v5_targets": "v5 · targets (Qwen3-0.6B-Base, dataset v5)",
          "v7_targets": "v7 · targets, 3 epochs (Qwen3-0.6B-Base, dataset v7)",
          "v7_1ep_targets": "v7 · targets, 1 epoch (Qwen3-0.6B-Base, dataset v7)"}
SHORT = {"v5_targets": "v5", "v7_targets": "v7 · 3 epochs", "v7_1ep_targets": "v7 · 1 epoch"}
MODEL_LINE = {
    "v5_targets": "Model: v5 · targets, Qwen3-0.6B-Base fine-tuned with LoRA on dataset v5, three target tokens (base, lexical, syntax) per example.",
    "v7_targets": "Model: v7 · targets, 3 epochs, Qwen3-0.6B-Base fine-tuned with LoRA on dataset v7 for three epochs: one example per "
                  "target with that token alone, plus one combined example per pair with all its targets.",
    "v7_1ep_targets": "Model: v7 · targets, 1 epoch, Qwen3-0.6B-Base fine-tuned with LoRA on dataset v7 for one epoch (the adapter at the lowest "
                      "validation loss): one example per target with that token alone, plus one combined example per pair with all its targets.",
}
FOOTER = {
    "v5_targets": "Greedy generations of the v5 targets model on 1,000 test originals of the IMPaCTS Wikipedia test sets. ",
    "v7_targets": "Greedy generations of the v7 targets model (three epochs) on 1,000 test originals of the IMPaCTS Wikipedia test sets. ",
    "v7_1ep_targets": "Greedy generations of the v7 targets model (one epoch) on 1,000 test originals of the IMPaCTS Wikipedia test sets. ",
}

FINDINGS = {"v5_targets": [  # from results/interp_v5_targets.md (run of 2026-11-12)
    "The three dimensions are not controlled separately (question 2). Used as in the main evaluation (other tokens “any”), the base token moves "
    "achieved syntax as much as achieved base (0.63 and 0.62 points per requested point), and the syntax token moves base 0.48 for 0.72 on syntax. "
    "Per point gained on the moved dimension, the other moves 1.01 (base token) and 0.67 (syntax token): about three times the co-movement in "
    "v5’s training data (0.34 and 0.25).",
    "No token moves READ-IT lexical (at most 0.05 points per point). The lexical token acts as a weak base token: it moves base and syntax by 0.21 "
    "and 0.20 points per point, and at equal overall READ-IT its outputs do not differ from the base token’s on any feature.",
    "Holding the other two dimensions at the sentence’s own levels does not make control specific. Lowering base with lexical and syntax "
    "held does nothing (−0.04 points per point), and half the outputs are copies: v5’s identity pairs taught it that syntax at the "
    "original’s own level means “copy”. Lowering syntax with base and lexical held still moves base by 0.28 per point for 0.46 on syntax, "
    "the same proportion as with nothing held. Lowering lexical with the others held does nothing.",
    "The tokens make mostly the same edits at different strengths (question 1): shorter sentences, fewer clauses and subordinates, shallower "
    "trees. Their effect profiles correlate 0.58–0.82. Which token was moved can be told from the change in overall READ-IT alone 49% of the time "
    "(chance 33%); the whole linguistic profile and the edit operations add 2 points (51%).",
    "What differs is mostly how they get there. As the request strengthens, the syntax token rewrites (replaced words r = 0.83, inserted 0.68, "
    "extra sentences 0.72), the base token mostly deletes (deleted 0.73, replaced 0.19). At equal overall READ-IT, base-token outputs delete more "
    "and have fewer subordinate clauses (3.5 points) than syntax-token outputs. No token simplifies vocabulary: the basic-vocabulary share "
    "barely changes (|r| ≤ 0.10).",
    "Inside the model, separate inputs merge into one signal. The three tokens’ embeddings are unrelated (cosines −0.03 to 0.28, within "
    "the ±0.34 of shuffled levels); base and syntax embeddings are ordered by level, lexical’s are not (Spearman 0.07). From layer 8 on, the "
    "state where generation starts carries the three levels on largely one direction (cosine base–syntax 0.75–0.82, base–lexical 0.90–0.94), "
    "and the base level is read along the syntax direction (0.95) as well as along its own (0.96). Base and syntax separate somewhat only in "
    "the last layers (cosine 0.55).",
    "Greedy decoding copies the original in 47% of the mildest syntax requests (one level below the original), a consequence of v5’s identity pairs.",
], "v7_targets": [  # from results/interp_v7_targets.md (run of 2026-11-13)
    "Holding the other tokens now lets the base token work. In v5, a syntax target at the original’s own level meant “copy”; in v7’s "
    "training data it does not. Lowering base with lexical and syntax held moves base by 0.51 points per point, with 18% copies (v5: −0.04, "
    "with 50% copies).",
    "But holding does not hold. With syntax held at the original’s level, the base token moves syntax by 0.52 points per point, as much as "
    "base. Lowering syntax with base held moves base by 0.34 per point for 0.48 on syntax, a larger share than with nothing held. The held "
    "tokens stop the copying, but the model does not read them as constraints.",
    "With the other tokens “any”, nothing changed from v5 (question 2). The base token moves base and syntax alike (0.61 and 0.62 points per "
    "point). Per point gained, the other dimension moves 1.01 (base token) and 0.60 (syntax token), against 0.34 and 0.25 in the training "
    "data. No token moves READ-IT lexical.",
    "The tokens make even more similar edits than v5’s (question 1). Their profiles correlate 0.84 (base–syntax; v5: 0.73), and which token "
    "was moved is told from the overall READ-IT change 44.5% of the time (v5: 48.8%; chance 33%). The syntax token still rewrites more than "
    "the base token, but by less (replaced words r = 0.58 against 0.40; v5: 0.83 against 0.19).",
    "Inside the model, base and syntax share one direction more than in v5: cosine 0.85–0.93 at layers 8–16 (v5: 0.75–0.82). The base "
    "level is read along the syntax direction (0.97–0.99) as well as along its own (0.97). The held base token, which now works, moves the "
    "state along the syntax direction too (cosine 0.73–0.82).",
    "Lexical embeddings are now partly ordered by level (Spearman 0.42; v5: 0.07); v7 shows the lexical token alone eight times as often. "
    "Its effect is still on base and syntax (0.21 and 0.24 points per point), not on READ-IT lexical (0.05).",
], "v7_1ep_targets": [  # from results/interp_v7_1ep_targets.md (run of 2026-11-13)
    "v7 trained for one epoch instead of three. On the main evaluation it does worse than the three-epoch model: it copies at mild requests "
    "too. Here it shows the same mechanism as the three-epoch model, more starkly.",
    "With the other tokens “any”, the base token moves syntax more than base (0.76 against 0.69 points per point), and the syntax token "
    "moves base by 0.41 for 0.60 on syntax.",
    "Held mode works for base (0.45 points per point, 32% copies) but does not hold syntax, which moves 0.46. The anchor request, every "
    "token at the original’s level, gives a copy 60% of the time.",
    "The tokens’ edits are almost interchangeable: profile similarity 0.93 (base–syntax) and 0.88 (base–lexical). Which token was moved "
    "is told 47% of the time from all the features together (chance 33%).",
    "Base and syntax lie on one direction at every layer from 8 to 28 (cosine 0.97–0.99), with the other tokens “any” and with them held.",
]}

METHODS = [
    "Originals: 1,000 test originals drawn at random (seed 0) among the 2,369 whose base, lexical and syntax levels are all at least 30, "
    "so that every dimension can be lowered by one, two and three levels.",
    "Requests: every level of each dimension below the original's own level, one dimension lowered at a time. Others any: the other two "
    "targets are “any”. Others held: the other two at the original's own levels (symmetric; the training data of v5 and v7 has same-level "
    "targets for base and syntax but never for lexical, so a held lexical token is outside it). 34,702 requests, decoded greedily (at most 128 new "
    "tokens), so that two outputs of the same original differ only because of the control tokens. The main evaluation samples at "
    "temperature 0.7.",
    "Scores: READ-IT base, lexical, syntax and global, computed exactly as the IMPaCTS labels; Profiling-UD features; edit operations against "
    "the original from a word-level alignment (difflib): shares of original tokens kept, deleted and replaced, share of output tokens inserted.",
    "Effects of a token: output three levels below the original against output one level below, same original (paired Wilcoxon, rank-biserial r, "
    "Benjamini–Hochberg q). Which token was moved: logistic regression on the outputs one to three levels below the original, 5-fold "
    "cross-validation grouped by original. Matched outputs: the two outputs of an original, one per token, closest in READ-IT global "
    "(within 2 points, different texts).",
    "Specificity matrix: within-original least-squares slope of each achieved score on the moved token's requested level (levels below the "
    "original only), 95% bootstrap interval over originals (2,000 resamples). Training data: slope of each score's change on the moved "
    "dimension's change across the training pairs from the v1 selection (the same pairs in v5 and v7), the natural IMPaCTS pairs (identity pairs and the "
    "lexical supplement, selected for large lexical drops, excluded).",
    "Directions: the input embeddings of the ten level tokens of each dimension, regressed on the level (null: shuffled levels, 2,000 "
    "permutations). Residual stream at the last prompt token at each of the 29 layers (embeddings and 28 blocks), regressed on the requested "
    "level within original. Readout: each half of the originals estimates the directions, the other half is projected on them; the two "
    "halves are averaged.",
]


def payload(model: str) -> dict:
    res = json.loads((ROOT / "data" / "v1" / "experiments" / "results" / f"interp_{model}.json").read_text())
    special = {"len_ratio": "Edit operations", "kept": "Edit operations", "deleted": "Edit operations", "replaced": "Edit operations",
               "inserted": "Edit operations", "copy": "Edit operations"}
    res["categories"] = [("READ-IT" if f.startswith("readit_") else special.get(f) or category(f)) for f in res["features"]]
    res["label"] = LABELS.get(model, model)
    res["short"] = SHORT.get(model, model)
    res["generated"] = datetime.date.today().isoformat()
    res["findings"] = FINDINGS.get(model, [])
    res["methods"] = [MODEL_LINE.get(model, f"Model: {model}."), *METHODS]
    res["footer"] = FOOTER.get(model, "") + "Scores: READ-IT (t2k, as the IMPaCTS labels) and Profiling-UD."
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", nargs="+", default=["v5_targets"], help="models in selector order")
    parser.add_argument("--default", help="the model shown first (default: the last of --model)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = {"order": args.model, "default": args.default or args.model[-1], "models": {m: payload(m) for m in args.model}}
    text = "window.INTERP_MODELS = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text) / 1e3:.0f} kB, models: {', '.join(args.model)})")


if __name__ == "__main__":
    main()
