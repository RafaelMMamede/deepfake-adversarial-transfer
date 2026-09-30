# Detector Training and Evaluation

This directory contains the detector training and clean-evaluation code used in the experiments. It is a reduced version of the upstream DeepfakeBench training framework and includes only the components required for the evaluated detectors.

Upstream attribution and licensing information are provided in the package-level `README.md` and `LICENSE`.

## Evaluated detectors

The study evaluates six backbones:

* ResNet-34
* Xception
* EfficientNet-B4
* DeiT-S
* ViT-B/16
* Swin-T

Each backbone is evaluated with two initialization conditions:

* ImageNet pretraining
* Face-recognition pretraining

Each configuration is trained on five manipulation-family settings:

* `FS`
* `FR`
* `FE`
* `EFS`
* `ALL`

This produces 60 detector variants in total.

## Directory structure

```text
training/
├── config/
│   ├── detector/       # Architecture and initialization configurations
│   ├── exp/            # Training-family and evaluation splits
│   └── train_config.yaml
├── dataset/            # Dataset loading and augmentation
├── detectors/          # Six evaluated detector implementations
├── networks/           # CNN backbone implementations
├── loss/               # Cross-entropy loss
├── metrics/            # Registries and evaluation metrics
├── optimizor/          # Optimizers and learning-rate scheduling
├── pretrained/         # Backbone-conversion utility
├── trainer/            # Training loop
├── train.py
└── test_with_best_val_th.py
```

## Environment

Install the software environment using the files provided at the package root before running these scripts.

All commands below must be executed from the package root. The code assumes that paths such as `./training`, `./datasets`, and `./preprocessing` are resolved from that location.

## Dataset layout

The default configuration expects:

```text
datasets/
└── <dataset files>

preprocessing/
└── dataset_json/
    └── <generated split JSON files>
```

The experiment configurations under `training/config/exp/` identify the required split JSON files.

Dataset images and generated dataset indexes are not distributed in this directory. Their preparation is described in the package-level README.

Machine-specific absolute paths should not be added to the configuration files. To use another dataset location, update `dataset_root_rgb` or pass `--dataset_root_rgb` to the evaluation script.

## Detector configurations

Detector configurations are located under:

```text
training/config/detector/
```

The files without the `_fr_pretrain` suffix use ImageNet initialization. Files ending in `_fr_pretrain` use face-recognition-pretrained backbones.

The configuration files are:

```text
resnet34.yaml
resnet34_fr_pretrain.yaml
xception.yaml
xception_fr_pretrain.yaml
efficientnetb4.yaml
efficientnetb4_fr_pretrain.yaml
deit_custom_unfreeze_all.yaml
deit_custom_unfreeze_all_fr_pretrain.yaml
vitb16.yaml
vitb16_fr_pretrain.yaml
swin_custom_unfreeze_all.yaml
swin_custom_unfreeze_all_fr_pretrain.yaml
```

## Pretrained backbones

The detector YAML files contain the expected relative paths for their initial backbone weights.

## Released detector checkpoints

The released detector weights are hosted on Hugging Face rather than committed
to this Git repository:

https://huggingface.co/poisonedchicken/deepfake-adversarial-transfer-models

Use that repository as the canonical source for the benchmark checkpoints.
Local `*.pth`, `*.pt`, and `*.ckpt` files are intentionally ignored by
Git to avoid duplicating the model release.

ImageNet- and face-recognition-pretrained backbone files are not included in this directory. Their acquisition or generation, filenames, and checksums are documented in the package-level README.

Face-recognition-pretrained checkpoints can be converted to the detector-compatible backbone format using:

```bash
python training/pretrained/export_fr_backbone.py \
  --root training/pretrained \
  --dry-run
```

After inspecting the reported files, omit `--dry-run` to perform the conversion:

```bash
python training/pretrained/export_fr_backbone.py \
  --root training/pretrained
```

The default output filename is:

```text
backbone_only_compatible.pth
```

## Training

Select one detector configuration and one experiment configuration.

For example, to train an ImageNet-pretrained Xception detector on the `FS` family:

```bash
python training/train.py \
  --detector_path training/config/detector/xception.yaml \
  --exp_config training/config/exp/train_FS_ff.json
```

To train the face-recognition-pretrained version:

```bash
python training/train.py \
  --detector_path training/config/detector/xception_fr_pretrain.yaml \
  --exp_config training/config/exp/train_FS_ff.json
```

Replace the detector and experiment files to reproduce the remaining architecture, pretraining, and training-family combinations.

Weights & Biases logging is disabled by default and is enabled only when `--wandb` is supplied.

## Clean evaluation

The evaluation script selects a classification threshold using the validation split and then applies that threshold to the configured test splits.

Example:

```bash
python training/test_with_best_val_th.py \
  --detector_path training/config/detector/xception.yaml \
  --eval_config training/config/exp/train_FS_ff.json \
  --ckpt checkpoints/imgnet_xception_FS/ckpt_best.pth \
  --save_dir results/clean_evaluation/imgnet_xception_FS \
  --dataset_root_rgb ./datasets \
  --dataset_json_folder ./preprocessing/dataset_json
```

The evaluated detector checkpoints are not included in this Git directory. Download them from the Hugging Face model repository above and place or symlink them into the local paths expected by `models_to_eval.json` and your evaluation commands.

## Import smoke test

After installing the environment, run the following command from the package root:

```bash
python - <<'PY'
from training.detectors import DETECTOR
from training.networks import BACKBONE
from training.loss import LOSSFUNC

expected_detectors = {
    "xception",
    "efficientnetb4",
    "resnet34",
    "deit_detector_custom",
    "swin_detector_custom",
    "vit_detector_custom",
}

expected_backbones = {
    "xception",
    "efficientnetb4",
    "resnet34",
}

assert expected_detectors <= set(DETECTOR.data)
assert expected_backbones <= set(BACKBONE.data)
assert "cross_entropy" in LOSSFUNC.data

print("Training imports and registries: PASS")
PY
```

This checks the retained import structure and component registration without loading datasets, checkpoints, or running training.

## Scope

This directory intentionally excludes:

* detectors not evaluated in the study;
* unused network and loss implementations;
* dataset-generation methods required only by other DeepfakeBench detectors;
* training logs and experiment-tracking files;
* optimizer states and model checkpoints;
* dataset images;
* machine-specific paths.

The objective is to preserve the exact training and evaluation code required for the reported 60-detector benchmark without distributing unrelated DeepfakeBench components.
