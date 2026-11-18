"""A: non-linear views of the level tokens in the model's space, for the interpretability page.

For every second layer (2, 4, ..., 28) and each model:
- MDS of the 9 level means (levels 0-80, in context): the means placed by their actual distances, no axis assumed;
  the same for the token alone (outputs/interp/<model>/alone.npy);
- t-SNE and UMAP of single originals: 300 test originals (seed 42) x levels 0-80 = 2,700 states, after PCA to 50
  dimensions. Two versions: "as is", and "within original", with each original's own mean over its 9 states removed,
  which leaves only what the level changes. t-SNE and UMAP keep neighbours, not distances.

    taskset --cpu-list 9-25 python experiments/interp/projections.py   ->  outputs/interp/projections.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import OUTPUTS  # noqa: E402

MODELS = ("g1_guided", "g1")
LAYERS = list(range(2, 29, 2))
TRAINED = 9
N_ORIGINALS = 300
SEED = 42


def r(a: np.ndarray) -> list:
    return np.round(a, 3).tolist()


def scale(Y: np.ndarray) -> np.ndarray:
    Y = Y - Y.mean(0)
    return Y / (np.abs(Y).max() or 1)


def main() -> None:
    import umap
    out = {"layers": LAYERS, "n_originals": N_ORIGINALS, "models": {}}
    for model in MODELS:
        S = np.load(OUTPUTS / "interp" / model / "states.npy", mmap_mode="r")
        alone = np.load(OUTPUTS / "interp" / model / "alone.npy")
        idx = np.sort(np.random.default_rng(SEED).choice(S.shape[0], N_ORIGINALS, replace=False))
        res = {}
        for l in LAYERS:
            t0 = time.time()
            X = np.asarray(S[idx, :TRAINED, l], dtype=np.float32)            # [orig, 9, D]
            means = np.asarray(S[:, :TRAINED, l], dtype=np.float32).mean(0)
            entry = {"mds_in_context": r(scale(MDS(n_components=2, random_state=SEED, n_init=4, normalized_stress="auto").fit_transform(means))),
                     "mds_alone": r(scale(MDS(n_components=2, random_state=SEED, n_init=4, normalized_stress="auto").fit_transform(alone[:TRAINED, l])))}
            for variant, Z in (("as_is", X), ("within_original", X - X.mean(1, keepdims=True))):
                flat = Z.reshape(-1, Z.shape[-1])
                P = PCA(n_components=50, random_state=SEED).fit_transform(flat)
                entry[f"tsne_{variant}"] = r(scale(TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(P)))
                entry[f"umap_{variant}"] = r(scale(umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(P)))
            res[str(l)] = entry
            print(f"{model} layer {l}: {time.time() - t0:.0f}s", flush=True)
        out["models"][model] = {"originals": idx.tolist(), "layers": res}
    (OUTPUTS / "interp" / "projections.json").write_text(json.dumps(out))
    print("wrote", OUTPUTS / "interp" / "projections.json")


if __name__ == "__main__":
    main()
