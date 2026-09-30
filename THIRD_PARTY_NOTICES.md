# Third-Party Notices

This repository contains or adapts material from third-party projects. Their
original licenses and attribution requirements remain in force.

## DeepfakeBench

Parts of the detector training, evaluation, dataset-loading, and model
infrastructure are adapted from:

- **Project:** DeepfakeBench
- **Repository:** https://github.com/SCLBD/DeepfakeBench
- **Authors:** Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, and Baoyuan Wu
- **License:** Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0)

This repository contains a reduced and modified subset tailored to the detector
architectures and experiments used in *What Makes Adversarial Examples Transfer
Across Deepfake Detectors?*

## AutoAttack

The repository includes a bundled AutoAttack wheel used to reproduce the
experimental environment:

- **Project:** AutoAttack
- **Repository:** https://github.com/fra31/auto-attack
- **Authors:** Francesco Croce and Matthias Hein
- **License:** MIT

The bundled wheel contains the AutoAttack version used in the experiments with
a compatibility adjustment for Python 3.7. The upstream MIT terms continue to
apply to that component.

## Other dependencies

Python packages listed in `requirements.txt` are not relicensed by this
repository. Each dependency remains subject to its own upstream license.

The original benchmark datasets are not distributed through this Git
repository. Dataset-specific terms remain applicable to data obtained from the
respective sources.
