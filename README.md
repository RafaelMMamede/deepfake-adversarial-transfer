# Anonymous Supplementary Code

This archive accompanies an anonymous research submission on adversarial
transfer across deepfake detectors. It contains the code and derived tabular
data used for dataset preparation, optional face-recognition pretraining,
detector training and evaluation, adversarial attack generation, and the final
statistical analysis.

The fastest reproducibility path is the analysis-only workflow below. It uses
the included CSV files and does not require the original image datasets or a
GPU. Reproducing preprocessing, training, or attacks requires the corresponding
datasets and trained model weights at user-configured local paths.

## Repository structure

```text
.
|-- README.md
|-- NOTICE_FOR_REVIEW.md
|-- environment.yml
|-- requirements.txt
|-- wheels/
|   `-- autoattack-0.1-py3-none-any.whl  # Patched Python 3.7-compatible dependency
|-- preprocessing/   # Dataset indexes, attack subset manifest, and validators
|-- pretraining/     # Face-recognition pretraining configurations and code
|-- training/        # Detector definitions, configurations, training, and testing
|-- attacks/         # AA and CW-EOT generation and transfer evaluation
`-- analysis/        # Included results, notebook, statistics, tables, and figures
```

Each workflow folder contains its own README with stage-specific inputs,
configuration, and commands.

## Methods represented in this archive

The experiments cover six detector backbones:

- Xception
- ResNet-34
- EfficientNet-B4
- DeiT-S
- ViT-B/16
- Swin-T

Models are evaluated across ImageNet and face-recognition initialization and
five training-data conditions. The transfer experiments use AutoAttack (AA)
and Carlini--Wagner with expectation over transformations (CW-EOT).

## Environment setup

Run the following commands from the repository root:

```bash
conda env create -f environment.yml
conda activate deepfake-transfer
python -m pip install -r requirements.txt
python -m pip check
```

Run these commands from the repository root so that pip can resolve the bundled
wheel at `wheels/autoattack-0.1-py3-none-any.whl`. This wheel contains the
AutoAttack version used in the experiments, with a behavior-preserving
compatibility adjustment for Python 3.7. No external AutoAttack installation is
required.

The released configuration pins PyTorch 1.12.0 with CUDA 11.3 and reflects the
Linux environment used for the experiments. GPU training and attack generation
require a compatible NVIDIA driver. The analysis workflow can run on CPU.

## Configuring local data paths

No fixed machine-specific dataset root is required. Replace path placeholders
in the relevant YAML configuration files with local paths, for example:

```yaml
dataset_root_rgb: "/path/to/deepfake-datasets"
```

On Windows, either use forward slashes or quote escaped backslashes:

```yaml
dataset_root_rgb: "C:/datasets/deepfake-datasets"
```

Dataset split indexes are stored under `preprocessing/dataset_json/`. When a
dataset is located under a different root, regenerate the indexes with the
scripts documented in `preprocessing/README.md`; do not retain absolute paths
from another machine. The attack subset is described by
`preprocessing/attack_manifest_2000.csv` and
`preprocessing/attack_dataset_2000.json`.

Checkpoint, input, and output locations are command-line arguments or shell
variables in the training and attack workflows. Set them to local locations
before running those stages.

## Reproduction workflow

The folders follow the experimental order:

1. **Preprocessing** -- build dataset JSON indexes, create the fixed attack
   subset, and validate referenced images and method labels. See
   `preprocessing/README.md`.
2. **Face-recognition pretraining** -- train the supported backbones with the
   supplied face-recognition configurations. See `pretraining/README.md`.
3. **Detector training** -- train or evaluate the six detector backbones under
   the specified initialization and training-data conditions. See
   `training/README.md`.
4. **Adversarial attacks** -- generate AA or CW-EOT examples and evaluate
   source-to-target transfer and oracle results. See `attacks/README.md`.
5. **Analysis** -- reconstruct the reported summaries, statistical tests,
   tables, and figures from the included result files. See
   `analysis/README.md`.

Run scripts from the repository root unless a folder README explicitly states
otherwise. This keeps package imports and relative output paths consistent.

## Analysis-only reproduction

The included analysis inputs are sufficient to reproduce the reported
aggregate results without rerunning model training or attacks. From the
repository root, run:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  analysis/analysis_fr_and_base.ipynb \
  --ExecutePreprocessor.timeout=-1
```

In PowerShell, use the same notebook command on one line:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace analysis/analysis_fr_and_base.ipynb --ExecutePreprocessor.timeout=-1
```

Notebook execution requires Jupyter and nbconvert to be available in the active
environment or notebook host. They are interface tools rather than experiment
runtime dependencies; the standalone statistical script does not require them.

The notebook may also be opened and run interactively from top to bottom. Its
generated artifacts are written below `analysis/output/`:

- `analysis/output/figures/`
- `analysis/output/tables/`
- `analysis/output/statistic/`

As a basic consistency check, the analysis should identify 60 detectors and
3,600 ordered source--target rows for each attack. The inferential analysis
uses leave-one-detector-out node jackknife confidence intervals and
Holm-adjusted tests.

## Anonymous-review notice

This is an anonymized review archive, not the final public software release.
Some attribution and licensing material is temporarily withheld where its
inclusion would reveal an author or institution. Third-party code remains
subject to its original terms. See `NOTICE_FOR_REVIEW.md` for the complete
review notice.
