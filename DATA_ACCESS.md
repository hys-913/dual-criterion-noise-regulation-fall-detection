# Data Access and Provenance

This repository supports the IEEE Access resubmission:

**Dual-Criterion Noise Regulation for Capacity-Limited Visual Classification: A Frame-Level Fall-Detection Prefilter Case Study**

## Source Datasets

This study is a secondary analysis of existing visual datasets. The authors did not collect new human-subject data.

### Leeds Millennium Dataset

- Role: primary source for the fall class and part of the non-fall class.
- Access: obtained through the source distribution route documented by the dataset providers.
- Conditions: research-use conditions from the source dataset apply.
- Provenance in this package: represented in `data/split_manifest.csv`.

### COCO 2017

- Role: non-fall augmentation for training diversity.
- URL: https://cocodataset.org/#download
- License: Creative Commons Attribution 4.0.
- Provenance in this package: represented in `data/split_manifest.csv`.

### UR Fall Detection Dataset

- Role: external RGB stress evaluation.
- URL: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
- Conditions: research-use conditions from the source dataset apply.
- Not used to construct the benchmark training split.

### SBU Killbot Fall Dataset

- Role: external depth-modality stress evaluation.
- Conditions: research access from the source provider applies.
- Pixel data are not redistributed here; frame IDs used for evaluation are listed in `data/sbu_eval_manifest.csv`.
- Not used to construct the benchmark training split.

## Benchmark Split

The full repository package includes the released 5,841-image benchmark in `dataset/`:

- train: 4,248 images before training-side regulation;
- validation: 1,062 images;
- test: 531 images.

The split is subject-, video-, and scene-disjoint where grouping metadata are available. Regulation actions are applied only to the training partition. Validation and test images remain intact.

## Audit Tables

The `data/` directory contains:

- split manifest and source attribution;
- physical-quality scores for all benchmark images;
- semantic exclusion recommendations;
- retained hard-negative register;
- inter-rater agreement register;
- count-level training-regulation summary;
- external-evaluation manifests and k-of-n projection files.

Semantic audit files use anonymized review IDs. They support count verification and category-level analysis, not reconstruction of identity-linked source paths.

## Ethics and Reuse

Source images may contain visible people. Reuse is governed by the source-dataset licenses and research-access conditions. The released scripts do not use identity labels, face-recognition features, or person-identification targets.

