# Transfer Analysis

This directory contains the data summaries, statistical analysis code, notebook, and generated artifacts used for the source-to-target adversarial-transfer evaluation across a bank of 60 deepfake detectors.

The current notebook covers three attack configurations:

- AutoAttack (`AA`)
- Carlini-Wagner without EOT (`CW without EOT`)
- Carlini-Wagner with Expectation over Transformation (`CW-EOT`)

The detector bank contains six backbones, two pretraining regimes, and five manipulation-training configurations:

- Backbones: Xception, ResNet34, EfficientNet-B4, DeiT-S, ViT-B/16, and Swin-T
- Pretraining: ImageNet and face-recognition pretraining
- Training configurations: `ALL`, `FS`, `FR`, `EFS`, and `FE`

All analyses operate on the included CSV summaries. Regenerating the tables, statistical tests, and figures does not require access to the original images or attack directories.

Some CSV columns such as `checkpoint`, `eval_config`, `root_path`,
`json_path`, and `attack_path` preserve provenance from the original
experiment runs. Treat those values as historical metadata rather than portable
input paths. The active portable detector/training configurations are the files
under `training/config/`, and released model paths are provided by the
Hugging Face model repository's `models.json`.

## Directory structure

```text
analysis/
├── README.md
├── analysis_fr_and_base.ipynb
├── transfer_analysis_node_jackknife.py  # retained legacy helper
├── transfer_analysis_three_attacks.py
├── data/
│   ├── all_model_results.csv
│   ├── transfer_results_aa_th_aware.csv
│   ├── transfer_results_cw_eot_th_aware.csv
│   ├── transfer_results_cw_no_eot_th_aware.csv
│   ├── transfer_results_cw_strict_oracle.csv
│   ├── transfer_results_cw_self_excluded_oracle.csv
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


The notebook imports `transfer_analysis_three_attacks.py` and writes regenerated artifacts under `output/three_attacks/`. The original `transfer_analysis_node_jackknife.py` remains available for historical reproduction.

`SENSITIVITY_WITH_CIS=False` skips nested jackknife intervals for group-omission diagnostics while retaining their point estimates. The primary full-bank statistical tests still include node-jackknife intervals.

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
- all three paired overall attack differences and dedicated CW-EOT versus CW-without-EOT compatibility contrasts;
- strict and self-excluding multi-source oracle summaries; and
- leave-one-group-out sensitivity analyses.

Uncertainty for the prespecified contrasts is estimated with a leave-one-detector-out node jackknife. Each deletion removes the detector in both its source and target roles before recomputing the complete estimator. Approximate 95% confidence intervals and two-sided p-values use the resulting jackknife standard error and a standard-normal reference distribution. The expanded follow-up analysis applies Holm correction jointly across 36 tests (7 per attack and 5 per attack pair). The original 19-test AA/CW-EOT correction is retained in separately labelled `p_holm_legacy` columns for reproduction. The expanded hypotheses are not described as preregistered.

CLIP sources and targets are excluded from all analysis tables and figures. The main bank contains exactly 60 detectors.

## Oracle inputs and provenance

The `cw` prefix in the new oracle filenames means CW **without** EOT. The helper also accepts `cw_no_eot`. The new CW pairwise CSV is a compact, non-CLIP projection of the uploaded `transfer_results_cw_strict_individual.csv`, with complete source IDs taken from `origin_key`. All retained metrics are unchanged. Oracle rules affect candidate selection, so the underlying individual results can be used for the pairwise summary.

The CW oracle summaries and selected-source counts are similarly restricted to non-CLIP detectors. Source counts include winners on all evaluated images, including images already misclassified when clean; they are not successful-attack counts.

The new three-attack run is included as `transfer_results_all_three_{scope}_oracle.csv`, with both scopes and non-CLIP source counts. Its original upload prefix `combined_2` is also accepted by the loader.

The legacy `combined` prefix continues to mean **AA + CW-EOT**. Use `all_three` or `aa_cw_eot_cw_no_eot` for a three-attack oracle, `aa_cw_no_eot` for AA + CW without EOT, and `cw_eot_cw_no_eot` for the two CW configurations. Missing portfolios are explicitly shown as not supplied.

The notebook writes a portfolio-comparison table and per-target differences for all three attacks versus AA + CW-EOT. These compare target-level ASRs and check counts and thresholds. Matching aggregate counts does not prove that eligible image IDs match across separate runs. The supplied combined run uses one clean prediction pass per target for all three attacks; existing pairwise summaries remain unchanged for reproduction.

Per-image `*_oracle_selection.csv` files are optional and are not committed here. Place them in `analysis/data/` to enable detailed winner/rule/count audits. Selected winners alone do not contain the clean-correct mask and cannot reconstruct every portfolio ASR. Aggregate oracle ASRs are checked against their success and clean-correct counts.

The original two-attack summaries may have different clean-correct counts from the new evaluation. The notebook reports these discrepancies; aggregate CSVs cannot establish identical eligible image IDs.


## License

See the repository-level `LICENSE` and main `README.md` for licensing, attribution, and citation information applying to this analysis package.
