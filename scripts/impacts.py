"""Load the Wikipedia part of IMPaCTS from the Hugging Face release (mpapucci/impacts).

The release is pinned to one revision so every build and analysis reads the same rows.
Pass local_dir to read a snapshot that is already on disk (the same folder layout as the
Hugging Face repository); otherwise the files are downloaded into the Hugging Face cache.
"""

from pathlib import Path

import pandas as pd

REPO = "mpapucci/impacts"
REVISION = "489c78edec079421b2c0d777cd3393ad04c96f2c"
DIMENSIONS = ("base", "lexical", "syntax", "all")
SCORE_COLUMNS = [f"{p}_{d}" for p in ("original", "simplification") for d in DIMENSIONS]


def load_wikipedia(profiling: bool = False, columns: list[str] | None = None,
                   local_dir: Path | None = None) -> pd.DataFrame:
    """Every Wikipedia pair; with profiling=True, also the Profiling-UD features."""
    config = "wikipedia_profiling" if profiling else "wikipedia"
    if local_dir is None:
        from huggingface_hub import snapshot_download
        local_dir = Path(snapshot_download(REPO, repo_type="dataset", revision=REVISION,
                                           allow_patterns=[f"{config}/*.parquet"]))
    files = sorted((Path(local_dir) / config).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {config}/*.parquet under {local_dir}")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
