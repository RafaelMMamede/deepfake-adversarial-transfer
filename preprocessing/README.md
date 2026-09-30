# Preprocessing Metadata and Fixed Attack Subset

This directory contains the portable dataset indexes used for detector training and evaluation, together with the fixed 2,000-image subset used for adversarial evaluation.

The retained metadata covers four manipulation families:

| Code | Manipulation family |
|---|---|
| `FS` | Face swapping |
| `FR` | Face reenactment |
| `FE` | Face editing |
| `EFS` | Entire-face synthesis |

## Directory contents

```text
preprocessing/
├── README.md
├── attack_dataset_2000.json
├── attack_manifest_2000.csv
├── dataset_json/
│   └── <22 experiment indexes>
└── scripts/
    ├── build_attack_subset.py
    ├── create_attack_dataset_clean_from_csv.py
    ├── verify_images_in_json.py
    └── verify_methods_in_json.py
```

The JSON indexes and attack manifest are experiment-defining metadata. The distributed versions are authoritative for reproducing the reported experiments.

## Dataset paths

Every distributed frame path begins with the portable logical prefix `datasets/`. For example:

```text
datasets/DF40/VQGAN/cdf/Fake_from_Celeb-real/id10_0001/013158.png
```

The training loader replaces this leading `datasets/` component with the configured `dataset_root_rgb`. Therefore, users whose data are stored under a different root normally do **not** need to regenerate or edit the JSON files.

For example, given:

```yaml
dataset_root_rgb: /data/DeepFake
dataset_json_folder: ./preprocessing/dataset_json
```

the stored path above resolves to:

```text
/data/DeepFake/DF40/VQGAN/cdf/Fake_from_Celeb-real/id10_0001/013158.png
```

Set these values in `training/config/train_config.yaml`. Evaluation also accepts the corresponding command-line arguments:

```bash
python training/test_with_best_val_th.py \
  --dataset_root_rgb /data/DeepFake \
  --dataset_json_folder ./preprocessing/dataset_json \
  <other required arguments>
```

On Windows, forward slashes avoid YAML escaping issues:

```yaml
dataset_root_rgb: D:/datasets/DeepFake
dataset_json_folder: ./preprocessing/dataset_json
```

The expected directories beneath `dataset_root_rgb` include the applicable dataset components, for example:

```text
<dataset_root_rgb>/
├── DF40/
├── DF40_train/
└── rgb/
    ├── FaceForensics++/
    ├── Celeb-DF-v1/
    └── Celeb-DF-v2/
```

Only the components required by a selected experiment need to be present.

### When the local directory layout differs

Changing the root is supported directly. Changing the directory hierarchy beneath that root is different: the suffix after `datasets/` must still identify the corresponding local file.

For example, if an index stores:

```text
datasets/DF40/<remaining path>
```

then `<dataset_root_rgb>/DF40/<remaining path>` must exist. If the data are split across unrelated locations, use symbolic links or directory junctions to present this expected hierarchy. As a last resort, make local-only copies of the JSON files and rewrite their prefixes. Do not replace the distributed authoritative indexes with machine-specific absolute paths.

## Retained experiment indexes

The five experiment configurations under `training/config/exp/` reference the following 22 files.

Training and validation:

```text
DF40_TRAIN_real_ff.json
DF40_TRAIN_FS_ff.json
DF40_TRAIN_FR_ff.json
DF40_TRAIN_FE_ff.json
DF40_TRAIN_EFS_ff.json
```

Family-specific testing:

```text
DF40_TEST_FS_ff.json
DF40_TEST_FS_cdf.json
DF40_TEST_FS_other.json

DF40_TEST_FR_ff.json
DF40_TEST_FR_cdf.json
DF40_TEST_FR_other.json

DF40_TEST_FE_ff.json
DF40_TEST_FE_cdf.json
DF40_TEST_FE_other.json

DF40_TEST_EFS_ff.json
DF40_TEST_EFS_cdf.json
DF40_TEST_EFS_other.json
```

Real and additional Celeb-DF testing:

```text
DF40_TEST_real_ff.json
DF40_TEST_real_cdf.json
DF40_TEST_real_cdfv1.json
DF40_TEST_real_other.json
TEST_fake_cdf.json
```

Dataset names are referenced without the `.json` extension in the experiment configuration files.

## Building base dataset JSON files from another dataset root

Users starting from raw videos can generate base dataset metadata using the official [DeepfakeBench preprocessing directory](https://github.com/SCLBD/DeepfakeBench/tree/main/preprocessing) and [setup instructions](https://github.com/SCLBD/DeepfakeBench#3-preprocessing-optional):

1. Arrange the licensed source datasets using the directory structure expected by DeepfakeBench.
2. In the upstream `preprocessing/config.yaml`, set `dataset_root_path` to the user's local dataset root.
3. If cropped face frames are not already available, configure and run `preprocess.py`.
4. Configure the rearrangement output directory and run `rearrange.py` to generate the base dataset JSON metadata.
5. Store generated JSON files under `preprocessing/dataset_json/`, or change `dataset_json_folder` in the training and test configuration.
6. Normalize generated frame paths to the portable `datasets/<relative path>` convention before sharing them. Keep machine-specific, root-resolved versions local.

Upstream-generated base JSON files are not interchangeable with the 22 supplied experiment indexes: the latter encode the exact manipulation-family partitions and evaluation splits used in this work. To reproduce the reported experiments with the same datasets under a different root, retain the supplied indexes and change `dataset_root_rgb` instead.

Dataset access and use remain subject to the licenses and terms of the original dataset providers. Upstream DeepfakeBench code remains subject to its original license and attribution requirements.

## JSON schema

Each index has one top-level dataset key, real/fake branches, data splits, and per-clip frame lists. A reduced example is:

```json
{
  "DF40_TRAIN_FS_ff": {
    "real": {
      "train": {},
      "val": {}
    },
    "fake": {
      "train": {
        "faceswap+ff+001": {
          "label": "fake",
          "frames": [
            "datasets/DF40_train/faceswap/frames/001/000.png"
          ]
        }
      },
      "val": {}
    }
  }
}
```

Preserve top-level dataset names, split assignments, clip identifiers, frame order, and labels when making local path adaptations.

## Fixed adversarial-evaluation subset

`attack_manifest_2000.csv` records the exact fake-only subset used for attack generation. It contains 2,000 unique frame paths and 2,000 stable sample identifiers, with 500 images from each manipulation family.

| Family | `cdf` | `ff` | `other` | Total |
|---|---:|---:|---:|---:|
| FS | 200 | 200 | 100 | 500 |
| FR | 217 | 216 | 67 | 500 |
| FE | 55 | 193 | 252 | 500 |
| EFS | 233 | 233 | 34 | 500 |
| **Total** | **705** | **842** | **453** | **2,000** |

The requested subgroup quotas were rebalanced within each family when a subgroup did not contain enough eligible images. The manifest is therefore authoritative. It contains 2,000 unique images from 1,838 unique clips; multiple frames may originate from one clip after shortfall rebalancing.

`attack_dataset_2000.json` stores the same selected frame paths using the schema expected by the data loader.

To regenerate the fixed subset from the 12 supplied family-specific fake test indexes on a shell supporting brace expansion:

```bash
python preprocessing/scripts/build_attack_subset.py \
  --inputs preprocessing/dataset_json/DF40_TEST_{FS,FR,FE,EFS}_{ff,cdf,other}.json \
  --out-json preprocessing/attack_dataset_2000.json \
  --out-csv preprocessing/attack_manifest_2000.csv \
  --seed 42 \
  --frames-per-clip 1
```

The regenerated JSON and CSV should match the distributed files byte-for-byte. The `frames-per-clip` limit applies during initial group sampling; shortfall rebalancing can introduce additional frames from a previously selected clip.

## Validation

Check that the retained Python files parse:

```bash
python -m compileall -q preprocessing/scripts
```

Check that the metadata files are valid JSON:

```bash
python -m json.tool preprocessing/attack_dataset_2000.json > /dev/null
```

The image-verification helper can decode referenced images after the configured paths resolve from the repository root:

```bash
python preprocessing/scripts/verify_images_in_json.py \
  --json preprocessing/dataset_json/*.json \
  --workers 8 \
  --out_dir preprocessing/json_verify_reports \
  --show_progress
```


