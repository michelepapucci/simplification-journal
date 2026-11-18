# simplification-journal

Readability-controlled Italian text simplification with one control token on READ-IT global: a Qwen3-0.6B model
asked to rewrite a sentence at a requested READ-IT level.

- [`data/g1/`](data/g1/README.md): the dataset (IMPaCTS Wikipedia, complex originals, training and test originals
  kept apart), built by [`scripts/build_dataset_g1.py`](scripts/build_dataset_g1.py).
- [`experiments/`](experiments/README.md): training, generation, scoring and analysis of the g1 models.
- `tools/readit/`, `tools/profiling/`: READ-IT, which reproduces the IMPaCTS labels exactly, and Profiling-UD.
- Explorer: https://michelepapucci.github.io/simplification-journal/.

The earlier work on three control tokens (READ-IT base, lexical, syntax; datasets v0 to v7) is on the
`multidim-archive` branch, and its explorer is at
https://michelepapucci.github.io/simplification-journal/archive/multidim/.
