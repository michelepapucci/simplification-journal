# simplification-journal: multi-dimensional control tokens (archived)

This branch is a frozen snapshot of the journal work on three control tokens per pair (READ-IT base, lexical and
syntax), datasets v0 to v7 and the Qwen3-0.6B models trained on them, as of 2026-11-13. Work then moved back to a
single control token on READ-IT global (the LREC setup), on `master`.

- [`data/README.md`](data/README.md): the datasets (`clicit2026`, `v0` to `v7`) and how they were built.
- [`data/v1/experiments/README.md`](data/v1/experiments/README.md): the models, the evaluation and the results;
  [`data/v1/experiments/interp/README.md`](data/v1/experiments/interp/README.md): the interpretability experiments.
- [`reports/improving_controllability.md`](reports/improving_controllability.md): the controllability report.
- The data explorer of this period is archived at https://michelepapucci.github.io/simplification-journal/archive/multidim/.

Not in git (each file is over GitHub's 100 MB limit): the LoRA adapters (`data/v1/experiments/runs/*/adapter*/`) and
the residual-stream states of the interpretability runs (`outputs/*/interp*/states.npy`). They are kept on the
original machine.

The scripts expect this repository next to the old `text-simplification-experiments` checkout (its `.venv` and
`LREC/evaluation/.venv_t2k`), as when they were written.
