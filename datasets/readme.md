# Local datasets

Original image/video datasets are **not distributed** with this repository.

The supplied experiment indexes under `preprocessing/dataset_json/` use
portable paths beginning with `datasets/`. Configure `dataset_root_rgb` to
the local root containing the corresponding licensed datasets rather than
committing data into this directory.

A typical root contains the components needed by the selected experiment, for
example:

```text
<dataset_root_rgb>/
├── DF40/
├── DF40_train/
└── rgb/
    ├── FaceForensics++/
    ├── Celeb-DF-v1/
    └── Celeb-DF-v2/
```

See `preprocessing/README.md` for the exact path-resolution convention,
experiment-index definitions, and fixed 2,000-image attack subset.

Dataset access and use remain subject to the terms of the original dataset
providers. The repository `.gitignore` intentionally prevents files placed
under `datasets/` from being committed.
