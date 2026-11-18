"""Shared definitions for the g1 experiments: one control token on READ-IT global (data/g1/).

The prompt is the LREC / CLiC-it one, f"{token}\\n{original}\\n", with token <|readability_L|> for the level L
(0-90) of the simplification's READ-IT global, or <|readability_any|> ("no level"), which only the guided model
sees in training (10% of its examples) and which classifier-free guidance uses at generation time.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # simplification-journal/
G1 = ROOT / "data" / "g1"
OLD_REPO = ROOT.parent                      # text-simplification-experiments/ (read only)
RUNS, OUTPUTS, RESULTS = HERE / "runs", HERE / "outputs", HERE / "results"

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
LEVELS = tuple(range(0, 100, 10))
NULL = "any"
TOKENS = [f"<|readability_{lv}|>" for lv in LEVELS] + [f"<|readability_{NULL}|>"]
DIMS = ("all", "base", "lexical", "syntax")   # READ-IT scores; "all" (global) is the controlled one
SEED = 42


def token(level) -> str:
    return f"<|readability_{level}|>"


def prompt(ctrl: str, original: str) -> str:
    """The LREC 2026 / CLiC-it 2026 prompt."""
    return f"{ctrl}\n{original}\n"
