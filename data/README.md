# Data

| Version | Folder | Training set | Test | Built by | What it is |
|---|---|---|---|---|---|
| **g1** | [`g1/`](g1/) | `training_set_g1.zip`: 81,058 pairs (80,251 train + 807 validation) | 1,000 originals, 7,582 requests; 300 dev originals | [`scripts/build_dataset_g1.py`](../scripts/build_dataset_g1.py) | One control token on READ-IT global (the LREC 2026 setup), complex originals, the journal's filters and three cleaning rules. See [`g1/README.md`](g1/README.md) |

g1 comes from the Wikipedia part of IMPaCTS on Hugging Face,
[`mpapucci/impacts`](https://huggingface.co/datasets/mpapucci/impacts), pinned at revision `489c78e` in
[`scripts/impacts.py`](../scripts/impacts.py).

[`source/`](source/) holds the inputs copied from the old `GFFF/` folder:
- the CLiC-it 2026 test-pair lists `idx_to_remove/list_idx_{base,lexical,syntax}.txt`, whose originals are held
  out of training;
- the langdetect list `idx_to_remove/list_dropped_idx_langdetect.txt`.

The datasets of the three-token work (`clicit2026`, `v0` to `v7`) are on the `multidim-archive` branch.
