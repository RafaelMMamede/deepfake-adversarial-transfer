# Threshold-Aware Adversarial Attacks

This directory contains the scripts used to generate and evaluate the
threshold-aware adversarial attacks in this study. The controlled evaluation
uses 60 deepfake detectors obtained from six backbones, two pretraining
strategies, and five training families.

The two attacks are:

- Threshold-aware AutoAttack (AA).
- Threshold-aware Carlini-Wagner with Expectation over Transformation
  (CW-EOT).

Both attacks use each detector's selected operating threshold rather than a
fixed decision threshold of 0.5.

## Directory contents

| File | Purpose |
| --- | --- |
| `attack_aa_th_aware.py` | Generate threshold-aware AutoAttack examples. |
| `attack_cw_eot_th_aware.py` | Generate threshold-aware CW-EOT examples. |
| `evaluate_success_efficient.py` | Evaluate single-source white-box and black-box transfer. |
| `evaluate_transfer_oracle.py` | Evaluate single-source results and multi-source oracle transfer. |
| `models_to_eval.json` | Manifest of the 60 evaluated detectors and their relative config/checkpoint paths. |
| `oracle_rules_self_excluded_no_clip.json` | Self-excluding oracle rule. |
| `oracle_rules_no_same_train_or_arch_or_clip.json` | Strict oracle rule. |

Run all commands from the repository root. The attack scripts import detector
implementations from `training/` and therefore assume the repository layout is
preserved.

## Detector set and identifiers

The detector set contains:

- Backbones: Xception, EfficientNet-B4, ResNet-34, DeiT-S, Swin-T, and
  ViT-B/16.
- Pretraining: ImageNet (`imgnet`) and face-recognition pretraining (`fr`).
- Training families: `ALL`, `FS`, `FR`, `FE`, and `EFS`.

Target detector identifiers in `models_to_eval.json` follow:

```text
<pretraining>_<backbone>_<training_family>
```

For example:

```text
imgnet_xception_FS
fr_swin_FE
```

During attack generation, `MODEL_TAG` follows the source directory naming used
by the experiment, for example `xception_FS` or `swin_FE`. Source pretraining is
represented by the parent output group (`imgnet` or `fr_pretrain`).

## Environment

Create the project environment using the environment and lock files provided at
the repository root. The scripts require PyTorch, torchvision, pandas, Pillow,
PyYAML, tqdm, and AutoAttack, in addition to the detector dependencies used by
`training/`.

Pin the exact AutoAttack source revision and any required compatibility patch
in the repository-level environment documentation. An unpinned replacement may
change behavior or fail with the installed PyTorch version.

The examples below use the Python interpreter from the active environment:

```bash
PYTHON="${PYTHON:-python}"
```

## Required checkpoint files

Released detector checkpoints are available at:

https://huggingface.co/poisonedchicken/deepfake-adversarial-transfer-models

Each model entry in `models_to_eval.json` contains:

- `name`: canonical detector identifier.
- `config`: relative detector configuration path.
- `ckpt`: relative checkpoint path.

Each checkpoint must be accompanied by its selected detector threshold. The
attack and evaluation scripts search the checkpoint directory and its parents
for one of:

```text
best_threshold.json
bestth.json
threshold.json
```

The JSON file must contain a numeric `threshold` field strictly between 0 and
1. A typical layout is:

```text
saved_models/<model>/val/val/
├── ckpt_best.pth
└── best_threshold.json
```

The release commands do not enable threshold fallback. A missing threshold file
therefore stops execution instead of silently using 0.5.

## Released adversarial examples

The adversarial examples generated for the study are hosted on Hugging Face:

https://huggingface.co/datasets/poisonedchicken/deepfake-adversarial-transfer

Use that dataset when you want to reproduce transfer evaluation or analysis
without regenerating all AA and CW-EOT examples locally. This Git repository
keeps the attack-generation and evaluation code, while the generated image
artifacts are maintained in the Hugging Face dataset release.

## Input images

`--input_dir` must point to the clean images used to construct the attack set.
The scripts accept PNG, JPEG, BMP, and WebP files. Images are converted to RGB
and transformed using the resolution, mean, and standard deviation specified by
the selected detector configuration.

The study attacks manipulated images, so the ground-truth label is explicitly
set to `1` (fake). Both attacks are untargeted and seek to cross the
detector-specific boundary toward the authentic class.

## Generate threshold-aware AutoAttack examples

Set the paths and source identifier for one detector:

```bash
PYTHON="${PYTHON:-python}"
CLEAN_DIR="/path/to/clean_images"
ATTACK_OUT="/path/to/generated_attacks/aa_th_aware/imgnet"
CHECKPOINT="./saved_models/<model>/val/val/ckpt_best.pth"
CONFIG="./training/config/detector/<config>.yaml"
MODEL_TAG="xception_FS"
```

Run:

```bash
"$PYTHON" ./attacks/attack_aa_th_aware.py \
    --input_dir "$CLEAN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --train_config "$CONFIG" \
    --out_dir "$ATTACK_OUT" \
    --model_tag "$MODEL_TAG" \
    --device cuda \
    --label 1 \
    --batch_size 32 \
    --num_workers 0 \
    --eps 0.031372549019607843 \
    --norm Linf \
    --version custom \
    --attacks_to_run apgd-ce,fab,square
```

This corresponds to an L-infinity budget of 8/255. The evaluated
AutoAttack components are APGD-CE, FAB, and Square. The script performs an input
gradient sanity check before starting the attack.

## Generate threshold-aware CW-EOT examples

Set the paths and source identifier for one detector:

```bash
PYTHON="${PYTHON:-python}"
CLEAN_DIR="/path/to/clean_images"
ATTACK_OUT="/path/to/generated_attacks/cw_eot_th_aware/imgnet"
CHECKPOINT="./saved_models/<model>/val/val/ckpt_best.pth"
CONFIG="./training/config/detector/<config>.yaml"
MODEL_TAG="xception_FS"
```

Run:

```bash
"$PYTHON" ./attacks/attack_cw_eot_th_aware.py \
    --input_dir "$CLEAN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --train_config "$CONFIG" \
    --out_dir "$ATTACK_OUT" \
    --model_tag "$MODEL_TAG" \
    --device cuda \
    --label 1 \
    --batch_size 30 \
    --num_workers 4 \
    --eps 0.031372549019607843 \
    --norm Linf \
    --steps 100 \
    --lr 0.01 \
    --kappa 1.0 \
    --c 1.0 \
    --dist_weight 0.0 \
    --eval_every 10 \
    --eot_samples 10 \
    --eot_degrees 3 \
    --eot_translate 0.03 \
    --eot_scale_min 0.97 \
    --eot_scale_max 1.03 \
    --eot_noise_std 0.01
```

The CW-EOT configuration uses 100 optimization steps, learning rate 0.01,
`kappa=1`, `c=1`, and ten transformation samples per step. Transformations use
rotation up to 3 degrees, translation up to 0.03 of the image dimensions,
scale in `[0.97, 1.03]`, and Gaussian noise with standard deviation 0.01.

## Generated attack layout

For either attack, the output has the following structure:

```text
<ATTACK_OUT>/
├── adv/
│   └── <MODEL_TAG>/
│       └── <attack_name>/
│           └── <sample_id>.png
└── attack_results/
    └── <MODEL_TAG>/
        ├── <attack_name>.csv
        ├── <attack_name>_config.json
        └── <attack_name>_summary.json
```

Use separate output groups for ImageNet and face-recognition-pretrained source
models:

```text
<attack>/imgnet/
<attack>/fr_pretrain/
```

Repeat attack generation for the applicable detector entries in
`models_to_eval.json`.

## Evaluate single-source transfer

`evaluate_success_efficient.py` evaluates every discovered adversarial source
against every target in `models_to_eval.json`. Each adversarial root must point
to an `adv/` directory containing `<MODEL_TAG>/<attack_name>/` subdirectories.

Example for AutoAttack:

```bash
PYTHON="${PYTHON:-python}"
CLEAN_DIR="/path/to/clean_images"
AA_IMGNET_ADV="/path/to/generated_attacks/aa_th_aware/imgnet/adv"
AA_FR_ADV="/path/to/generated_attacks/aa_th_aware/fr_pretrain/adv"
RESULT_DIR="/path/to/results"

"$PYTHON" ./attacks/evaluate_success_efficient.py \
    --clean_dir "$CLEAN_DIR" \
    --adv_roots "$AA_IMGNET_ADV" "$AA_FR_ADV" \
    --models_json ./attacks/models_to_eval.json \
    --batch_size 512 \
    --num_workers 8 \
    --output_csv "$RESULT_DIR/transfer_results_aa_th_aware.csv"
```

For CW-EOT, replace the two adversarial roots and output filename with their
CW-EOT equivalents. Evaluate the two attacks separately so their result tables
remain unambiguous.

## Evaluate multi-source oracle transfer

`evaluate_transfer_oracle.py` first computes individual source-target results
and then selects, independently for each image, the eligible source example
that minimizes the target detector's fake probability.

### Self-excluding oracle

The self-excluding rule removes the source detector corresponding to the target
while retaining other eligible sources:

```bash
"$PYTHON" ./attacks/evaluate_transfer_oracle.py \
    --clean_dir "$CLEAN_DIR" \
    --adv_roots "imgnet=$AA_IMGNET_ADV" "fr=$AA_FR_ADV" \
    --models_json ./attacks/models_to_eval.json \
    --batch_size 512 \
    --num_workers 8 \
    --output_csv "$RESULT_DIR/transfer_results_aa_self_excluded_individual.csv" \
    --output_oracle_csv "$RESULT_DIR/transfer_results_aa_self_excluded_oracle.csv" \
    --output_oracle_selection_csv "$RESULT_DIR/transfer_results_aa_self_excluded_oracle_selection.csv" \
    --oracle_rules_json ./attacks/oracle_rules_self_excluded_no_clip.json
```

### Strict oracle

The strict rule additionally excludes sources sharing the target's exact
training data or exact architecture, and excludes `ALL`-trained detectors as
sources:

```bash
"$PYTHON" ./attacks/evaluate_transfer_oracle.py \
    --clean_dir "$CLEAN_DIR" \
    --adv_roots "imgnet=$AA_IMGNET_ADV" "fr=$AA_FR_ADV" \
    --models_json ./attacks/models_to_eval.json \
    --batch_size 512 \
    --num_workers 8 \
    --output_csv "$RESULT_DIR/transfer_results_aa_strict_individual.csv" \
    --output_oracle_csv "$RESULT_DIR/transfer_results_aa_strict_oracle.csv" \
    --output_oracle_selection_csv "$RESULT_DIR/transfer_results_aa_strict_oracle_selection.csv" \
    --oracle_rules_json ./attacks/oracle_rules_no_same_train_or_arch_or_clip.json
```

Use the corresponding CW-EOT roots and output names to evaluate CW-EOT. To
construct the combined AA and CW-EOT oracle, provide all four roots:

```bash
--adv_roots \
    "imgnet=$AA_IMGNET_ADV" \
    "fr=$AA_FR_ADV" \
    "imgnet=$CW_IMGNET_ADV" \
    "fr=$CW_FR_ADV"
```

The oracle evaluator writes:

- Individual source-target results to `--output_csv`.
- Aggregate oracle results to `--output_oracle_csv`.
- Per-image source selections to `--output_oracle_selection_csv`.
- Source-selection counts next to the oracle CSV using the suffix
  `_source_counts.csv`.

## Reproducibility checks

Before running a complete evaluation, verify that:

1. All paths in `models_to_eval.json` resolve from the repository root.
2. Every checkpoint has a valid threshold JSON file.
3. The clean input directory contains the expected image identifiers.
4. ImageNet and face-recognition sources are written to distinct output groups.
5. AA and CW-EOT are evaluated into distinct result files.
6. The generated `*_config.json` files record the intended parameters.

All paths shown in this README are placeholders. Keep machine-specific paths in
local job scripts or environment variables rather than committing them to the
release package.
