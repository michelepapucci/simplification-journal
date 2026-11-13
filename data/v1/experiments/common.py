"""Shared definitions for the dataset v1 experiments (training, generation, evaluation).

Two variants of the same model, trained identically on data/v1/training_set_v1.zip:
- targets:         <|readability_base_T|><|readability_lexical_T|><|readability_syntax_T|>
- source_targets:  <|original_readability_base_S|>...<|original_readability_syntax_S|> + the same targets
T is a level 0-90 or "any"; S is the original's level 0-90.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # simplification-journal/
V1 = ROOT / "data" / "v1"
OLD_REPO = ROOT.parent                      # text-simplification-experiments/ (read only)
RUNS, OUTPUTS, RESULTS = HERE / "runs", HERE / "outputs", HERE / "results"

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
DIMS = ("base", "lexical", "syntax")
LEVELS = tuple(range(0, 100, 10))
VARIANTS = ("targets", "source_targets")
TARGET_TOKENS = [f"<|readability_{d}_{t}|>" for d in DIMS for t in [*map(str, LEVELS), "any"]]
SOURCE_TOKENS = [f"<|original_readability_{d}_{lv}|>" for d in DIMS for lv in LEVELS]
SEED = 42


def control_tokens(variant: str) -> list[str]:
    return TARGET_TOKENS + (SOURCE_TOKENS if variant == "source_targets" else [])


def control_string(variant: str, targets: dict, sources: dict | None = None) -> str:
    """targets: {dim: level or "any"}; sources: {dim: level}, used by the source_targets variant."""
    tgt = "".join(f"<|readability_{d}_{targets[d]}|>" for d in DIMS)
    if variant == "targets":
        return tgt
    return "".join(f"<|original_readability_{d}_{int(sources[d])}|>" for d in DIMS) + tgt


def prompt(ctrl: str, original: str) -> str:
    """The CLiC-it 2026 prompt (GFFF/controlled_text_simplification_ft.py)."""
    return f"{ctrl}\n{original}\n"
