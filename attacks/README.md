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
| `models_to_eval.json` | Provenance manifest for the 60 evaluated detectors using the original timestamped experiment paths. |
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

Target detector identifiers in both the release and provenance manifests follow:

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

For the public release, use the `models.json` manifest distributed with that
Hugging Face model repository. Its entries point to the packaged detector
configuration, `ckpt_best.pth`, and adjacent threshold metadata for each of the
60 detector variants.

A typical downloaded model release has the following structure:

```text
deepfake-adversarial-transfer-models/
├── models.json
├── source_models.json
├── SHA256SUMS
├── imgnet/
│   └── <model>/
│       ├── config.yaml
│       ├── ckpt_best.pth
│       └── best_threshold.json
└── fr_pretrain/
    └── <model>/
        ├── config.yaml
        ├── ckpt_best.pth
        └── best_threshold.json
```

The evaluators resolve relative `config` and `ckpt` entries against the
directory containing the supplied manifest, so the model release can live
outside this code repository:

```bash
MODEL_RELEASE="/path/to/deepfake-adversarial-transfer-models"
MODELS_JSON="$MODEL_RELEASE/models.json"
```

The repository-local `attacks/models_to_eval.json` is kept for provenance. It
records the original experiment's timestamped `saved_models/` and
`saved_models_fr_pretrain/` paths and is only directly usable when that
original directory layout is reproduced.

Each model manifest entry contains:

- `name`: canonical detector identifier.
- `config`: detector configuration path.
- `ckpt`: detector checkpoint path.

Each checkpoint must be accompanied by its selected detector threshold. The
attack and evaluation scripts search the checkpoint directory and its parents
for one of:

```text
best_threshold.json
bestth.json
threshold.json
```

The JSON file must contain a numeric `threshold` field strictly between 0 and
1. A typical release-model directory is:

```text
<model>/
├── config.yaml
├── ckpt_best.pth
└── best_threshold.json
```

The release commands do not enable threshold fallback. A missing threshold file
therefore stops execution instead of silently using 0.5.

## Released adversarial examples

The adversarial examples generated for the study are hosted on Hugging Face:

https://huggingface.co/datasets/poisonedchicken/deepfake-adversarial-transfer

Use that dataset when you want to reproduce transfer evaluation without
regenerating all AA and CW-EOT examples locally. The release stores the clean
subset in `clean.tar` and the adversarial examples as one TAR per source model
under:

```text
aa_th_aware/{imgnet,fr_pretrain}/adv/
cw_eot_th_aware/{imgnet,fr_pretrain}/adv/
```

Each source-model TAR preserves the `<model>/<attack-name>/...` hierarchy
expected below `adv/`. Extracting all archives therefore reconstructs the
directory structure consumed by the evaluation scripts.

For example:

```bash
HF_DATA="/path/to/deepfake-adversarial-transfer"
EVAL_DATA="/path/to/extracted-transfer-data"

mkdir -p "$EVAL_DATA"
tar -xf "$HF_DATA/clean.tar" -C "$EVAL_DATA"

for ATTACK in aa_th_aware cw_eot_th_aware; do
  for PRETRAIN in imgnet fr_pretrain; do
    OUT="$EVAL_DATA/$ATTACK/$PRETRAIN/adv"
    mkdir -p "$OUT"
    for ARCHIVE in "$HF_DATA/$ATTACK/$PRETRAIN/adv/"*.tar; do
      tar -xf "$ARCHIVE" -C "$OUT"
    done
  done
done
```

After extraction:

```bash
CLEAN_DIR="$EVAL_DATA/clean"
AA_IMGNET_ADV="$EVAL_DATA/aa_th_aware/imgnet/adv"
AA_FR_ADV="$EVAL_DATA/aa_th_aware/fr_pretrain/adv"
CW_IMGNET_ADV="$EVAL_DATA/cw_eot_th_aware/imgnet/adv"
CW_FR_ADV="$EVAL_DATA/cw_eot_th_aware/fr_pretrain/adv"
```

The dataset release also contains `manifest.tsv` and `SHA256SUMS` for archive
inventory and integrity checking. This Git repository keeps the generation and
evaluation code; the generated image artifacts remain canonical on Hugging
Face.

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
MODEL_DIR="/path/to/deepfake-adversarial-transfer-models/imgnet/<model>"
CHECKPOINT="$MODEL_DIR/ckpt_best.pth"
CONFIG="$MODEL_DIR/config.yaml"
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
MODEL_DIR="/path/to/deepfake-adversarial-transfer-models/imgnet/<model>"
CHECKPOINT="$MODEL_DIR/ckpt_best.pth"
CONFIG="$MODEL_DIR/config.yaml"
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

Repeat attack generation for the applicable detector entries in the released
model manifest.

## Evaluate single-source transfer

`evaluate_success_efficient.py` evaluates every discovered adversarial source
against every target in the manifest supplied through `--models_json`. Each adversarial root must point
to an `adv/` directory containing `<MODEL_TAG>/<attack_name>/` subdirectories.

Example for AutoAttack:

```bash
PYTHON="${PYTHON:-python}"
MODEL_RELEASE="/path/to/deepfake-adversarial-transfer-models"
MODELS_JSON="$MODEL_RELEASE/models.json"
CLEAN_DIR="/path/to/clean_images"
AA_IMGNET_ADV="/path/to/generated_attacks/aa_th_aware/imgnet/adv"
AA_FR_ADV="/path/to/generated_attacks/aa_th_aware/fr_pretrain/adv"
RESULT_DIR="/path/to/results"

"$PYTHON" ./attacks/evaluate_success_efficient.py \
    --clean_dir "$CLEAN_DIR" \
    --adv_roots "$AA_IMGNET_ADV" "$AA_FR_ADV" \
    --models_json "$MODELS_JSON" \
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
    --models_json "$MODELS_JSON" \
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
    --models_json "$MODELS_JSON" \
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

1. The supplied model manifest resolves all 60 config/checkpoint paths. For the public model release, use its bundled `models.json`.
2. Every checkpoint has a valid threshold JSON file.
3. The clean input directory contains the expected image identifiers.
4. ImageNet and face-recognition sources are written to distinct output groups.
5. AA and CW-EOT are evaluated into distinct result files.
6. The generated `*_config.json` files record the intended parameters.

All paths shown in this README are placeholders. Keep machine-specific paths in
local job scripts or environment variables rather than committing them to the
release package.
