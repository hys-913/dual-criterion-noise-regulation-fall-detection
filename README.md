# Dual-Criterion Noise Regulation: Companion Repository

This repository accompanies the IEEE Access resubmission:

**Dual-Criterion Noise Regulation for Capacity-Limited Visual Classification: A Frame-Level Fall-Detection Prefilter Case Study**

Authors: Yusha He, Shiqi Zhang, Meian Li

## Repository Scope

This package supports reviewer inspection of the revised manuscript. It contains the full 5,841-image benchmark split used in the paper, audit tables, and scripts for the shared training/evaluation protocol.

The work is a frame-level fall/non-fall classification case study. It is positioned as an edge-prefilter component, not as a complete video-level fall-detection system.

## Contents

```text
dataset/
  train/fall/      2,620 images
  train/normal/    1,628 images
  val/fall/          655 images
  val/normal/        407 images
  test/fall/         323 images
  test/normal/       208 images

data/
  split_manifest.csv
  physical_scores.csv
  semantic_labels_removed.csv
  semantic_labels_retained.csv
  interrater_agreement.csv
  semantic_agreement_summary.csv
  semantic_audit_summary.csv
  training_regulation_summary.csv
  reproducibility_scope.csv
  source_label_counts.csv
  source_only_diagnostic.csv
  source_bias_control_report.md
  same_source_test_manifest.csv
  sbu_eval_manifest.csv
  event_level_kofn_projection.csv
  event_level_prediction_template.csv
  temporal_baseline_scope.csv

scripts/
  physical_filtering.py
  semantic_filtering.py
  source_bias_control.py
  event_level_kofn.py
  train_dual_criterion.py
  train_baseline_*.py
  evaluate.py
  evaluate_ood.py
  generate_tables.py
```

Trained checkpoints are intentionally not bundled in this repository snapshot.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Inspect the split and audit counts:

```bash
python scripts/generate_tables.py --dataset_dir ./dataset --data_dir ./data --output_markdown audit_summary.md
```

Run the proposed MobileNetV3-Small protocol for one seed:

```bash
python scripts/train_dual_criterion.py --data_dir ./dataset --seed 42 --epochs 100
```

Run representative baselines:

```bash
python scripts/train_baseline_gce.py --data_dir ./dataset --seed 42
python scripts/train_baseline_focal.py --data_dir ./dataset --seed 42
python scripts/train_baseline_curriculum.py --data_dir ./dataset --seed 42
python scripts/train_baseline_coteaching.py --data_dir ./dataset --seed 42
python scripts/train_baseline_label_smooth.py --data_dir ./dataset --seed 42
python scripts/train_baseline_reweight.py --data_dir ./dataset --seed 42
python scripts/train_baseline_random_remove.py --data_dir ./dataset --seed 42
```

Generate the analytic k-of-n projection used for temporal-extension discussion:

```bash
python scripts/event_level_kofn.py --projection --frame_sensitivity 0.923 --frame_specificity 0.887 --window 3 --output_csv data/event_level_kofn_projection.csv
```

The k-of-n calculation is an analytic projection from frame-level sensitivity/specificity. It is not reported as validated video-level false-alarm or latency performance.

## Reproducibility Notes

- The intended split rule is approximately 7:2:1, constrained by subject, video, and scene disjointness.
- The actual pre-regulation counts are 4,248 training, 1,062 validation, and 531 test images.
- Training-side regulation gives the documented regulated training count: 4,248 - 82 physical removals - 320 semantic exclusions = 3,846.
- The semantic audit contains 476 reviewed candidates: 320 exclusion recommendations and 156 retained hard negatives.
- The inter-rater agreement register reports raw agreement of 427/476 and Cohen's kappa of 0.85.
- Semantic audit tables use anonymized review IDs rather than identity-linked source file paths.

## Data and Access

See `DATA_ACCESS.md` and `LICENSE_AND_ACCESS.md`. Source-dataset licenses and access conditions govern reuse and redistribution. The images may contain visible people; the model uses fall/non-fall body-configuration evidence and does not use identity labels or face-recognition features.
