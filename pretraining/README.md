# Face-Recognition Backbone Pretraining

This directory contains the face-recognition (FR) pretraining code used to initialize the six deepfake-detector backbones evaluated in this study. The code trains each backbone for identity classification on BUPT-BalancedFace using an ElasticArcFace+ head and evaluates the learned embeddings on RFW verification pairs.

This is a reduced research-release directory. Dataset files, generated indexes, experiment logs, and model checkpoints are not included. Package-level attribution and licensing information are provided in the root `README.md` and `LICENSE`.

## Pretrained backbones

| Backbone | Configuration | Construction path |
|---|---|---|
| ResNet-34 | `configs/balancedface_resnet.py` | Direct backbone class |
| Xception | `configs/balancedface_xception.py` | Direct backbone class |
| EfficientNet-B4 | `configs/balancedface_efficientnet.py` | Direct backbone class |
| DeiT-S | `configs/balancedface_deit.py` | Detector registry and `timm` |
| ViT-B/16 | `configs/balancedface_vit.py` | Detector registry and `timm` |
| Swin-T | `configs/balancedface_swin.py` | Detector registry and `timm` |

The CNN backbones are constructed directly from `training/networks/`. The transformer backbones are constructed through the detector classes registered in `training/detectors/__init__.py`:

- `deit_detector_custom`
- `vit_detector_custom`
- `swin_detector_custom`

In the transformer configurations, `pretrained=False` means that ImageNet initialization is disabled before FR pretraining. It does not disable FR pretraining.

## Directory structure

```text
pretraining/
├── README.md
├── train.py
├── configs/
│   ├── balancedface_deit.py
│   ├── balancedface_efficientnet.py
│   ├── balancedface_resnet.py
│   ├── balancedface_swin.py
│   ├── balancedface_vit.py
│   └── balancedface_xception.py
└── utils/
    ├── dataset.py
    ├── losses.py
    ├── utils.py
    └── verification_aux.py
```

The pretraining code depends on the retained implementations under `training/`; the `pretraining/` directory is therefore not intended to run as a standalone package.

## Environment

Install the environment described by the files at the package root. The principal dependencies used here are PyTorch, torchvision, timm, OpenCV, NumPy, pandas, and scikit-learn. Weights & Biases is optional.

Run every command below from the package root so that relative paths such as `./pretraining` and `./training` resolve correctly.

## Datasets

The datasets are not distributed with this source package. Obtain BUPT-BalancedFace and RFW separately and comply with their respective licenses and access conditions.

Set the dataset locations before running pretraining:

```bash
export BALANCEDFACE_ROOT=/path/to/BalancedFace/race_per_7000_aligned
export RFW_ROOT=/path/to/RFW/test_aligned
```

The submission-safe configuration pattern is:

```python
import os

rec = os.environ.get(
    "BALANCEDFACE_ROOT",
    "./data/BalancedFace/race_per_7000_aligned",
)
rfw_dir = os.environ.get(
    "RFW_ROOT",
    "./data/RFW/test_aligned",
)
```

Expected BUPT-BalancedFace layout:

```text
race_per_7000_aligned/
├── African/
│   └── <identity>/
│       └── <images>
├── Asian/
├── Caucasian/
└── Indian/
```

Expected RFW layout:

```text
test_aligned/
├── data/
│   ├── African/
│   ├── Asian/
│   ├── Caucasian/
│   └── Indian/
└── txts/
    ├── African/African_pairs.txt
    ├── Asian/Asian_pairs.txt
    ├── Caucasian/Caucasian_pairs.txt
    └── Indian/Indian_pairs.txt
```

## Running FR pretraining

Select one configuration and run:

```bash
python pretraining/train.py \
  --config pretraining/configs/balancedface_deit.py \
  --seed 42
```

Replace the configuration filename to train another backbone. The configuration files are the authoritative source for batch size, optimizer, learning-rate schedule, regularization, number of epochs, and mixed-precision settings.

The optimizer learning rate is scaled in `train.py` as:

```text
actual learning rate = configured learning rate × batch size / 512
```

Weights & Biases logging should be disabled by default for reproduction without an external account. To use it, set `use_wandb=True` in the selected configuration and authenticate separately.

## Outputs

Each run writes two checkpoints under its configured `output_dir`:

```text
best_fr_pretrained_detector_backbone.pth
last_fr_pretrained_detector_backbone.pth
```

The best checkpoint is selected using RFW validation AUC. A checkpoint contains:

- run and architecture metadata;
- the detector-backbone state under `detector_backbone`;
- the FR embedding-head state;
- the ElasticArcFace+ classification-head state;
- validation AUC, epoch, and identity-class count.

For transformer runs, the full detector state is also stored under `detector`. Downstream detector initialization should use the `detector_backbone` state rather than the FR embedding or identity-classification heads.

## Exporting detector-compatible backbone weights

Convert the raw FR checkpoints using the retained export utility:

```bash
python training/pretrained/export_fr_backbone.py \
  --root pretraining/checkpoints \
  --dry-run
```

After checking the reported files, omit `--dry-run` to create `backbone_only_compatible.pth` in each checkpoint directory:

```bash
python training/pretrained/export_fr_backbone.py \
  --root pretraining/checkpoints
```

Place each exported file in the relative location specified by its corresponding `training/config/detector/*_fr_pretrain.yaml` file.

## Import smoke test

The following test verifies registration of the transformer detectors without reading datasets or running training:

```bash
python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path("training").resolve()))
from detectors import DETECTOR

expected = {
    "deit_detector_custom",
    "vit_detector_custom",
    "swin_detector_custom",
}

missing = expected - set(DETECTOR.data)
assert not missing, f"Missing detector registrations: {sorted(missing)}"
print("FR transformer detector imports: PASS")
PY
```

This checks imports and registration only. A complete reproduction still requires the datasets, the declared software environment, and suitable compute resources.

## Reproducibility notes

- The command-line seed defaults to `42`.
- All experiments use 224 × 224 RGB inputs and a 512-dimensional FR embedding.
- The supplied configurations use ElasticArcFace+ and validate on RFW.
- GPU kernels may remain nondeterministic unless deterministic PyTorch settings are explicitly enabled.
- Dataset indexes are generated locally and must not be treated as portable submission artifacts.

## Scope

This directory preserves only the code required to reproduce the FR initialization used for the six evaluated detector backbones. It intentionally excludes unrelated FR architectures, unused analysis utilities, raw datasets, generated indexes, logs, and model checkpoints.
