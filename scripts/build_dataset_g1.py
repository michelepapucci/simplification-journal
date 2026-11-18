"""Build data/g1/: dataset g1, one control token on READ-IT global (the LREC setup) with the journal's selection rules.

Training pairs come from the Wikipedia part of IMPaCTS (scripts/impacts.py), test originals held out:
 1. hold out every original of the CLiC-it 2026 test pairs (data/source/idx_to_remove/list_idx_*.txt);
 2. original READ-IT global > 0.7;
 3. drop the release's langdetect list (data/source/idx_to_remove/list_dropped_idx_langdetect.txt);
 4. drop originals matching the double-filter proxy (a quote, guillemet or opening parenthesis, or over 400 characters);
 5. drop exact duplicates (same original, same simplification);
 6. language: langdetect 1.0.9 (seed 0) must detect Italian on both the original and the simplification;
 7. degenerate simplification: a word 4-gram occurs 3 or more times (and does not in the original), or the
    simplification's type-token ratio is below 0.4;
 8. instruction echo: the simplification repeats the generator's instructions or answers like a chat assistant
    ("La seguente frase è stata semplificata mantenendo il più possibile intatto il significato...",
    "Sulla base del paragrafo...", "Ecco..."), and the original does not contain the same phrase;
 9. length: the simplification has at most twice the original's words;
10. target: the simplification's level is below the original's (at least one control token lower). No noise margin.
11. split: 1% of the originals (seed 42) go to validation.

Levels follow the CLiC-it rule: level = 10 * floor(10 * score), capped at 90. The control token is
<|readability_{level}|> of the simplification's READ-IT global.

Test and dev originals are drawn from the held-out test originals that pass the original-level rules
(2, 4 and Italian original): 1,000 test originals and 300 dev originals, both stratified by the original's
level (seed 42); the rest stay unused. Each is requested at every level below its own (test_requests_g1.csv,
dev_requests_g1.csv). The test originals' own IMPaCTS simplifications that pass rules 3, 5-10 are the
references (test_references_g1.csv), for reference-based metrics at the requested level.

    PYTHONPATH=envs/pydeps taskset --cpu-list 9-25 ../.venv/bin/python scripts/build_dataset_g1.py
"""

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from impacts import REPO, REVISION, SCORE_COLUMNS, load_wikipedia  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LISTS = ROOT / "data" / "source" / "idx_to_remove"
OUT = ROOT / "data" / "g1"

TEST_STRATEGIES = ("base", "lexical", "syntax")
MIN_ORIGINAL_GLOBAL = 0.7
DOUBLE_FILTER_CHARS = r'["«»“”(]'
DOUBLE_FILTER_MAX_CHARS = 400
LANGUAGE = "it"
REPEATED_NGRAM, REPEATED_TIMES, MIN_TTR = 4, 3, 0.4
MAX_LENGTH_RATIO = 2.0
TEST_ORIGINALS, DEV_ORIGINALS = 1000, 300
VALIDATION_SHARE = 0.01
NEAR_DUPLICATE_SIMILARITY = 0.9
SEED = 42
BROKEN_COLUMNS = ["n_sentences_original", "n_sentences_simplification"]  # always 1 in IMPaCTS profiling
WORD = re.compile(r"\w+")
ECHO = [r"mantenendo il più possibile", r"intatto il significato", r"il più possibile il significato",
        r"\brispond[io] (solamente|solo)\b", r"\b(la|il|questa|questo) seguente (frase|testo|paragrafo|passaggio)",
        r"\b(passaggio|testo|paragrafo) fornito", r"^\W*(sommarizzare|riassumendo|semplificando|semplificare)\b",
        r"^\W*(certo|sicuro)?[,!.]?\s*ecco\b", r"\bpuò essere semplificat[ao]\b", r"\bfrase (originale|semplificata)\b",
        r"\bsemplificazione della (frase|seguente)", r"^\W*simplified\s*:", r"\bmi dispiace\b",
        r"^\W*sulla base (di questo|del|dello|della) (paragrafo|testo|passaggio)", r"\bsemplificat[ao]\s*:",
        r"\b(è|sono) stat[aoie] semplificat[aoie]\b", r"\bsemplificat[ao] (versione|frase|significato)\b", r"\bsarebbe\s*:"]
ECHO_RX = re.compile("|".join("(?:" + re.sub(r"\((?!\?)", "(?:", e) + ")" for e in ECHO), re.IGNORECASE)  # no capture groups


def read_ids(path: Path) -> set[int]:
    return {int(line) for line in path.read_text().split()}


def level(score: pd.Series) -> pd.Series:
    """READ-IT score (0-1) to control level: 0.43 -> 40; 1.0 -> 90."""
    return (np.minimum(np.floor(score * 10), 9) * 10).astype(int)


def double_filter_proxy(text: pd.Series) -> pd.Series:
    return text.str.contains(DOUBLE_FILTER_CHARS, regex=True) | (text.str.len() > DOUBLE_FILTER_MAX_CHARS)


def _init_langdetect():
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0


def _detect(text: str) -> str:
    from langdetect import detect
    try:
        return detect(text)
    except Exception:  # no letters at all
        return "unknown"


def languages(texts: list[str]) -> list[str]:
    with Pool(16, initializer=_init_langdetect) as pool:
        return pool.map(_detect, texts, chunksize=500)


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


def repetition(text: str) -> tuple[int, float, int]:
    """(most repeated word 4-gram count, type-token ratio, words)."""
    w = words(text)
    grams = Counter(zip(*(w[i:] for i in range(REPEATED_NGRAM))))
    return (max(grams.values()) if grams else 0), (len(set(w)) / len(w) if w else 0.0), len(w)


def quality(pairs: pd.DataFrame) -> pd.DataFrame:
    """Columns used by rules 7-9."""
    with Pool(16) as pool:
        s = pool.map(repetition, pairs.simplification.tolist(), chunksize=500)
        o = pool.map(repetition, pairs.original_text.tolist(), chunksize=500)
    s, o = np.array(s), np.array(o)
    return pd.DataFrame({
        "degenerate": ((s[:, 0] >= REPEATED_TIMES) & (o[:, 0] < REPEATED_TIMES)) | (s[:, 1] < MIN_TTR),
        "echo": pairs.simplification.str.contains(ECHO_RX).to_numpy() & ~pairs.original_text.str.contains(ECHO_RX).to_numpy(),
        "too_long": s[:, 2] > MAX_LENGTH_RATIO * np.maximum(o[:, 2], 1),
    }, index=pairs.index)


def _near_duplicate_differences(group: np.ndarray) -> list:
    out = []
    for a, b in combinations(group, 2):
        if a[0] == b[0]:
            continue
        sm = SequenceMatcher(None, a[0], b[0])
        if (sm.real_quick_ratio() >= NEAR_DUPLICATE_SIMILARITY and sm.quick_ratio() >= NEAR_DUPLICATE_SIMILARITY
                and sm.ratio() >= NEAR_DUPLICATE_SIMILARITY):
            out.append(100 * abs(a[1] - b[1]))
    return out


def readit_noise(pairs: pd.DataFrame) -> dict:
    """READ-IT global difference (0-100) between near-identical simplifications of the same original (documentation only)."""
    groups = [g.to_numpy() for _, g in pairs.groupby("original_sentence_idx")[["simplification", "simplification_all"]] if len(g) > 1]
    with Pool(16) as pool:
        diffs = np.array([x for r in pool.map(_near_duplicate_differences, groups, chunksize=200) for x in r])
    return {"near_duplicate_pairs": int(len(diffs)), **{f"p{q}": round(float(np.percentile(diffs, q)), 1) for q in (50, 75, 90, 95)}}


def stratified(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """n rows, allocated to the levels of `src_level` in proportion (largest remainder)."""
    share = frame.src_level.value_counts().sort_index() * n / len(frame)
    alloc = np.floor(share).astype(int)
    alloc[(share - alloc).sort_values(ascending=False).index[: n - alloc.sum()]] += 1
    parts = [g.iloc[rng.permutation(len(g))[: alloc[lv]]] for lv, g in frame.groupby("src_level")]
    return pd.concat(parts).sort_values("original_sentence_idx")


def requests(originals: pd.DataFrame) -> pd.DataFrame:
    rows = [(i, lv) for i, src in zip(originals.original_sentence_idx, originals.src_level) for lv in range(0, src, 10)]
    r = pd.DataFrame(rows, columns=["original_sentence_idx", "level"])
    r.insert(0, "request", range(len(r)))
    return r.merge(originals[["original_sentence_idx", "src_level", "original_text"]], on="original_sentence_idx")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hf-dir", type=Path, help="local snapshot of mpapucci/impacts (default: download)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    corpus = load_wikipedia(profiling=True, local_dir=args.hf_dir).drop(columns=BROKEN_COLUMNS)
    corpus = corpus.sort_values("idx").reset_index(drop=True)
    profiling = [c for c in corpus.columns if c.endswith(("_original", "_simplification")) and c not in SCORE_COLUMNS]
    core = ["idx", "original_sentence_idx", "original_text", "simplification"] + SCORE_COLUMNS
    steps = []

    def step(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        steps.append({"step": name, "pairs": int(len(frame)), "originals": int(frame.original_sentence_idx.nunique())})
        print(f"{name:66s} {len(frame):8d} pairs {frame.original_sentence_idx.nunique():7d} originals", flush=True)
        return frame

    pairs = step("release, Wikipedia", corpus[core])
    test_ids = set().union(*(read_ids(LISTS / f"list_idx_{s}.txt") for s in TEST_STRATEGIES))
    held_out = set(corpus.loc[corpus.idx.isin(test_ids), "original_sentence_idx"])
    pairs = step("1. test originals held out", pairs[~pairs.original_sentence_idx.isin(held_out)])
    pairs = step("2. original READ-IT global > 70", pairs[pairs.original_all > MIN_ORIGINAL_GLOBAL])
    langdetect_list = read_ids(LISTS / "list_dropped_idx_langdetect.txt")
    pairs = step("3. langdetect list removed", pairs[~pairs.idx.isin(langdetect_list)])
    pairs = step("4. double-filter proxy on the original", pairs[~double_filter_proxy(pairs.original_text)])
    pairs = step("5. exact duplicates removed", pairs.drop_duplicates(["original_sentence_idx", "simplification"]))

    originals = pairs.drop_duplicates("original_sentence_idx")
    lang_o = dict(zip(originals.original_sentence_idx, languages(originals.original_text.tolist())))
    lang_s = languages(pairs.simplification.tolist())
    italian = (pairs.original_sentence_idx.map(lang_o) == LANGUAGE).to_numpy() & (np.array(lang_s) == LANGUAGE)
    pairs = step("6. original and simplification detected as Italian", pairs[italian])
    q = quality(pairs)
    pairs = step("7. no degenerate simplification (repeated 4-gram x3, or TTR < 0.4)", pairs[~q.degenerate])
    pairs = step("8. no instruction echo", pairs[~q.echo.loc[pairs.index]])
    pairs = step("9. simplification at most twice the original's words", pairs[~q.too_long.loc[pairs.index]])
    pairs = pairs.assign(src_level=level(pairs.original_all), tgt_level=level(pairs.simplification_all))
    pairs = step("10. target: the simplification's level below the original's", pairs[pairs.tgt_level < pairs.src_level])
    pairs = pairs.assign(ctrl_token=[f"<|readability_{t}|>" for t in pairs.tgt_level])

    ids = np.sort(pairs.original_sentence_idx.unique())
    rng = np.random.default_rng(SEED)
    validation = set(rng.choice(ids, size=round(VALIDATION_SHARE * len(ids)), replace=False))
    pairs.insert(2, "split", np.where(pairs.original_sentence_idx.isin(validation), "validation", "train"))
    for split, frame in pairs.groupby("split"):
        step(f"11. split: {split}", frame)
    noise = readit_noise(pairs)
    print(f"READ-IT global noise between near-identical simplifications: {noise}", flush=True)
    train = pairs.merge(corpus[["idx"] + profiling], on="idx", how="left", validate="one_to_one")
    train.to_csv(OUT / "training_set_g1.zip", index=False, compression="zip")

    # test and dev originals
    pool = corpus[corpus.original_sentence_idx.isin(held_out)].drop_duplicates("original_sentence_idx")
    held = {"held-out test originals": len(pool)}
    pool = pool[pool.original_all > MIN_ORIGINAL_GLOBAL]
    held["READ-IT global > 70"] = len(pool)
    pool = pool[~double_filter_proxy(pool.original_text)]
    held["double-filter proxy"] = len(pool)
    pool = pool[np.array(languages(pool.original_text.tolist())) == LANGUAGE]
    held["Italian"] = len(pool)
    original_cols = ["original_sentence_idx", "original_text"] + [c for c in SCORE_COLUMNS if c.startswith("original_")]
    pool = pool[original_cols + [c for c in profiling if c.endswith("_original")]].assign(src_level=lambda f: level(f.original_all))
    rng = np.random.default_rng(SEED)
    test = stratified(pool, TEST_ORIGINALS, rng)
    dev = stratified(pool[~pool.original_sentence_idx.isin(test.original_sentence_idx)], DEV_ORIGINALS, rng)
    test.to_csv(OUT / "test_set_g1.csv", index=False)
    dev.to_csv(OUT / "dev_set_g1.csv", index=False)
    test_req, dev_req = requests(test), requests(dev)
    test_req.to_csv(OUT / "test_requests_g1.csv", index=False)
    dev_req.to_csv(OUT / "dev_requests_g1.csv", index=False)

    refs = corpus[corpus.original_sentence_idx.isin(test.original_sentence_idx)][core]
    ref_steps = {"IMPaCTS simplifications of the test originals": len(refs)}
    refs = refs[~refs.idx.isin(langdetect_list)].drop_duplicates(["original_sentence_idx", "simplification"])
    ref_steps["langdetect list, duplicates"] = len(refs)
    refs = refs[np.array(languages(refs.simplification.tolist())) == LANGUAGE]
    ref_steps["Italian"] = len(refs)
    rq = quality(refs)
    refs = refs[~(rq.degenerate | rq.echo | rq.too_long)]
    ref_steps["degenerate, echo, length"] = len(refs)
    refs = refs.assign(src_level=level(refs.original_all), level=level(refs.simplification_all))
    refs = refs[refs.level < refs.src_level]
    ref_steps["level below the original's"] = len(refs)
    refs = refs.merge(corpus[["idx"] + profiling], on="idx", how="left", validate="one_to_one")
    refs.to_csv(OUT / "test_references_g1.csv", index=False)
    covered = test_req.merge(refs[["original_sentence_idx", "level"]].drop_duplicates(), on=["original_sentence_idx", "level"])

    tr = pairs[pairs.split == "train"]
    construction = {
        "source": {"repo": REPO, "revision": REVISION, "part": "wikipedia"},
        "rules": {"min_original_global": MIN_ORIGINAL_GLOBAL, "level": "10 * min(floor(10 * score), 9)",
                  "double_filter_proxy": {"chars": DOUBLE_FILTER_CHARS, "max_chars": DOUBLE_FILTER_MAX_CHARS},
                  "language": f"langdetect 1.0.9, seed 0, original and simplification '{LANGUAGE}'",
                  "degenerate": f"a word {REPEATED_NGRAM}-gram >= {REPEATED_TIMES} times in the simplification and not in the original, or TTR < {MIN_TTR}",
                  "echo_patterns": ECHO, "max_length_ratio_words": MAX_LENGTH_RATIO,
                  "target": "tgt_level < src_level (no noise margin)", "token": "<|readability_{tgt_level}|>",
                  "validation_share_of_originals": VALIDATION_SHARE, "seed": SEED},
        "steps": steps,
        "rule_overlaps_before_7": {k: int(v) for k, v in q.sum().items()},
        "readit_global_noise": noise,
        "train_target_levels": {int(k): int(v) for k, v in tr.tgt_level.value_counts().sort_index().items()},
        "train_source_levels": {int(k): int(v) for k, v in tr.src_level.value_counts().sort_index().items()},
        "train_source_by_target": {int(s): {int(t): int(n) for t, n in g.tgt_level.value_counts().sort_index().items()}
                                   for s, g in tr.groupby("src_level")},
        "train_steps_down": {int(k): int(v) for k, v in ((tr.src_level - tr.tgt_level) // 10).value_counts().sort_index().items()},
        "test": {"selection": held, "test_originals": len(test), "dev_originals": len(dev),
                 "test_by_level": {int(k): int(v) for k, v in test.src_level.value_counts().sort_index().items()},
                 "dev_by_level": {int(k): int(v) for k, v in dev.src_level.value_counts().sort_index().items()},
                 "test_requests": len(test_req), "dev_requests": len(dev_req),
                 "references": ref_steps, "requests_with_a_reference": len(covered)},
    }
    (OUT / "construction_g1.json").write_text(json.dumps(construction, indent=1, ensure_ascii=False))
    print(json.dumps({k: construction[k] for k in ("train_target_levels", "train_steps_down", "test")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
