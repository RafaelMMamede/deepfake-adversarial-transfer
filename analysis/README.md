# Transfer Analysis

This directory contains the data summaries, statistical analysis code, notebook, and generated artifacts used for the source-to-target adversarial-transfer evaluation across a bank of 60 deepfake detectors.

The released analysis covers two attacks:

- AutoAttack (`AA`)
- Carlini-Wagner with Expectation over Transformation (`CW-EOT`)

The detector bank contains six backbones, two pretraining regimes, and five manipulation-training configurations:

- Backbones: Xception, ResNet34, EfficientNet-B4, DeiT-S, ViT-B/16, and Swin-T
- Pretraining: ImageNet and face-recognition pretraining
- Training configurations: `ALL`, `FS`, `FR`, `EFS`, and `FE`

All analyses operate on the included CSV summaries. Regenerating the tables, statistical tests, and figures does not require access to the original images or attack directories.

## Directory structure

```text
analysis/
├── README.md
├── analysis_fr_and_base.ipynb
├── transfer_analysis_node_jackknife.py
├── data/
│   ├── all_model_results.csv
│   ├── transfer_results_aa_th_aware.csv
│   ├── transfer_results_cw_eot_th_aware.csv
│   ├── transfer_results_aa_strict_oracle.csv
│   ├── transfer_results_aa_self_excluded_oracle.csv
│   ├── transfer_results_cw_eot_strict_oracle.csv
│   ├── transfer_results_cw_eot_self_excluded_oracle.csv
│   ├── transfer_results_combined_strict_oracle.csv
│   └── transfer_results_combined_self_excluded_oracle.csv
└── output/
    ├── figures/
    ├── statistic/
    └── tables/
```

The output directories are created automatically if they do not already exist.

## Environment

For analysis-only reproduction, use the modern CPU environment defined at the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

This environment includes the analysis dependencies (`pandas`, `numpy`,
`scipy`, `matplotlib`, `seaborn`, `ipykernel`, and `nbconvert`) and is
validated in CI by executing this notebook from start to finish.

The legacy `environment.yml` / `requirements.txt` environment is only needed
when reproducing detector training or attack generation with the original
Python 3.7 / PyTorch 1.12 stack.

## Running the analysis

The notebook supports execution either from the repository root or from this directory. For interactive execution:

```bash
cd analysis
jupyter lab analysis_fr_and_base.ipynb
```

Restart the kernel and run all cells from top to bottom.

For non-interactive execution:

```bash
cd analysis
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    analysis_fr_and_base.ipynb
```


The notebook imports the statistical helpers from `transfer_analysis_node_jackknife.py` and writes regenerated artifacts under `output/`.

## Portable path fields

Some transfer CSVs contain an `attack_path` column with paths beginning with the literal placeholder `${DATA_ROOT}`. These strings preserve attack and source-model provenance. The loader uses components in the path suffix, such as the pretraining regime and source-model name, to reconstruct the source identifier.

The analysis never opens or resolves `${DATA_ROOT}`. It therefore does not need to be defined as an environment variable. Only replace the placeholder when intentionally reconnecting the summaries to a local copy of the corresponding attack artifacts.

## Data conventions

Detector identifiers follow the general form:

```text
<pretraining>_<backbone>_<training-configuration>
```

For example:

```text
imgnet_xception_ALL
fr_resnet34_FS
```

ASR values in the input CSVs are stored on the unit interval. Tables and figures convert them to percentages or percentage-point differences where appropriate.

The pairwise transfer files contain source-target evaluations for all 60 detectors. A source-target pair is treated as white-box when the complete source and target identifiers are equal; all other pairs are treated as black-box.


## Statistical analysis

The analysis reports:

- descriptive white-box and black-box ASR summaries;
- source-to-target transfer matrices;
- compatibility contrasts for shared backbone, architecture family, pretraining, and training data;
- directional source-target contrasts;
- the paired overall difference between CW-EOT and AA;
- strict and self-excluding multi-source oracle summaries; and
- leave-one-group-out sensitivity analyses.

Uncertainty for the prespecified contrasts is estimated with a leave-one-detector-out node jackknife. Each deletion removes the detector in both its source and target roles before recomputing the complete estimator. Approximate 95% confidence intervals and two-sided p-values use the resulting jackknife standard error and a standard-normal reference distribution. Holm correction is applied jointly across the 19 prespecified tests.


## License

See the repository-level `LICENSE` and main `README.md` for licensing, attribution, and citation information applying to this analysis package.
