# What Makes Adversarial Examples Transfer Across Deepfake Detectors?

Code, released models, adversarial examples, and analysis artifacts for:

**Rafael M. Mamede, Pedro C. Neto, and Ana F. Sequeira.  
"What Makes Adversarial Examples Transfer Across Deepfake Detectors?" (2026).**

- **Paper:** https://arxiv.org/abs/2609.10002
- **Models:** https://huggingface.co/poisonedchicken/deepfake-adversarial-transfer-models
- **Adversarial-example dataset:** https://huggingface.co/datasets/poisonedchicken/deepfake-adversarial-transfer

## Overview

This repository studies adversarial transfer across deepfake detectors with
different architectures, pretraining regimes, and manipulation-family training
sets.

The benchmark contains **60 detector variants** formed from:

- 6 backbones: Xception, ResNet-34, EfficientNet-B4, DeiT-S, ViT-B/16, Swin-T;
- 2 initialization regimes: ImageNet and face-recognition pretraining;
- 5 training-family settings: `ALL`, `FS`, `FR`, `FE`, and `EFS`.

The adversarial evaluation uses threshold-aware AutoAttack (AA) and a
Carlini-Wagner-style attack with expectation over transformations (CW-EOT).
The repository also contains the source-target transfer evaluation, portfolio
oracle analysis, statistical tests, tables, and figures used in the study.

## Repository and artifact split

Large experiment artifacts are intentionally kept outside Git:

| Resource | Location |
| --- | --- |
| Source code, configs, derived tables, analysis | This repository |
| Released detector checkpoints | https://huggingface.co/poisonedchicken/deepfake-adversarial-transfer-models |
| Released adversarial examples | https://huggingface.co/datasets/poisonedchicken/deepfake-adversarial-transfer |
| Paper | https://arxiv.org/abs/2609.10002 |

Model binaries such as `*.pth`, `*.pt`, and `*.ckpt` are ignored by Git.
The Hugging Face model repository is the canonical location for released
weights.

## Repository structure

```text
.
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- THIRD_PARTY_NOTICES.md
|-- environment.yml             # Exact legacy training/attack environment
|-- requirements.txt            # Exact legacy Python dependencies
|-- requirements-analysis.txt   # Modern CPU-only analysis environment
|-- wheels/
|   `-- autoattack-0.1-py3-none-any.whl
|-- preprocessing/   # Dataset indexes, fixed attack subset, validators
|-- pretraining/     # Face-recognition pretraining configurations and code
|-- training/        # Detector definitions, training, and clean evaluation
|-- attacks/         # AA/CW-EOT generation and transfer/oracle evaluation
`-- analysis/        # Included results, statistics, tables, and figures
```

Each workflow folder contains its own README with stage-specific inputs,
configuration, and commands.

## Environments

Two environments are provided for different reproducibility goals.

### Analysis-only environment (recommended for most users)

The included tables, statistical tests, and figures can be reproduced on CPU
with a modern Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

On Windows PowerShell, activate the environment with
`.venv\\Scripts\\Activate.ps1`.

This analysis environment is exercised by the repository CI, which executes the
full analysis notebook from start to finish.

### Exact training and attack environment

`environment.yml` and `requirements.txt` preserve the legacy software stack
used for detector training and adversarial-example generation: Python 3.7,
PyTorch 1.12, and CUDA 11.3.

```bash
conda env create -f environment.yml
conda activate deepfake-transfer
python -m pip install -r requirements.txt
python -m pip check
```

Python 3.7 is end-of-life, so this environment should be treated as a
reproducibility environment rather than a general-purpose or security-maintained
runtime. Use it in an isolated environment and only when exact training/attack
compatibility is required.

The bundled AutoAttack wheel is the version used in the experiments. Its single
Python 3.7 compatibility patch and SHA-256 checksum are documented in
`wheels/README.md`.

GPU training and attack generation require a compatible NVIDIA driver.

## Reproduction paths

### 1. Analysis-only reproduction

The included CSV files are sufficient to reproduce the aggregate transfer
analysis, statistical tests, tables, and figures without downloading model
weights or regenerating attacks.

Run:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  analysis/analysis_fr_and_base.ipynb \
  --ExecutePreprocessor.timeout=-1
```

The generated artifacts are written below:

```text
analysis/output/figures/
analysis/output/tables/
analysis/output/statistic/
```

A basic consistency check is that the analysis identifies 60 detectors and
3,600 ordered source-target rows for each attack.

### 2. Evaluate the released adversarial examples

Download the adversarial examples from:

https://huggingface.co/datasets/poisonedchicken/deepfake-adversarial-transfer

Then follow `attacks/README.md` to evaluate source-target transfer and the
multi-source oracle protocols.

### 3. Evaluate the released detectors

Download detector checkpoints from:

https://huggingface.co/poisonedchicken/deepfake-adversarial-transfer-models

The checkpoint paths expected by the evaluation scripts are described in
`training/README.md` and `attacks/models_to_eval.json`.

### 4. Regenerate attacks

Use the released detector checkpoints together with the fixed attack subset
described by:

- `preprocessing/attack_manifest_2000.csv`
- `preprocessing/attack_dataset_2000.json`

Attack generation commands and parameters are documented in
`attacks/README.md`.

### 5. Train detector variants

Training configurations for the six architectures, two initialization regimes,
and five training-family settings are provided under `training/config/`.

See `pretraining/README.md` and `training/README.md` for the corresponding
workflows.

## Data paths

Machine-specific absolute paths are intentionally not committed.

Set dataset locations through the relevant YAML files or command-line
arguments. For example:

```yaml
dataset_root_rgb: "/path/to/deepfake-datasets"
```

Dataset split indexes used by the released experiments are stored under
`preprocessing/dataset_json/`. The original benchmark datasets remain subject
to their respective access and licensing terms.

## Citation

If you use the code, released models, adversarial examples, or derived results,
please cite:

```bibtex
@misc{mamede2026makesadversarialexamplestransfer,
  title={What Makes Adversarial Examples Transfer Across Deepfake Detectors?},
  author={Rafael M. Mamede and Pedro C. Neto and Ana F. Sequeira},
  year={2026},
  eprint={2609.10002},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.10002}
}
```

Citation metadata is also available in `CITATION.cff`.

## License and attribution

Unless otherwise noted, the original material in this repository is released
under **CC BY-NC 4.0**. See `LICENSE`.

Parts of the detector training and evaluation infrastructure are adapted from
[DeepfakeBench](https://github.com/SCLBD/DeepfakeBench), which is also licensed
under CC BY-NC 4.0.

The bundled AutoAttack component is governed by its upstream MIT license.

See `THIRD_PARTY_NOTICES.md` for attribution and third-party licensing
details.
