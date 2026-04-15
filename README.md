# Supplementary Materials

**Paper:** Dual-Criterion Noise Regulation for Capacity-Limited Visual Classification: A Fall Detection Case Study
**Journal:** Applied Intelligence (Springer Nature)
**Authors:** Yusha He, Meian Li, Shiqi Zhang

---

## Contents

```
supplementary/
  README.md                          <- This file
  DATA_ACCESS.md                     <- Dataset access protocol, URLs, licenses
  requirements.txt                   <- Python dependencies
  dataset/                           <- Complete original dataset (5,841 images)
    train/
      fall/                          <- Training fall samples (2,620 images)
      normal/                        <- Training normal samples (1,628 images)
    val/
      fall/                          <- Validation fall samples (655 images)
      normal/                        <- Validation normal samples (407 images)
    test/
      fall/                          <- Test fall samples (323 images)
      normal/                        <- Test normal samples (208 images)
  data/
    split_manifest.csv               <- 5,841 samples: sample_id, source, split, label
    physical_scores.csv              <- Per-sample blur/occlusion/entropy scores
    semantic_labels_removed.csv      <- 320 removed semantic-noise samples with categories
    semantic_labels_retained.csv     <- 156 retained hard negatives with F1 impact
    interrater_agreement.csv         <- Two-expert annotation agreement (476 samples)
    sbu_eval_manifest.csv            <- SBU Killbot frame IDs for OOD evaluation
  scripts/
    physical_filtering.py            <- Algorithm 1 Phase 1: physical noise filtering
    semantic_filtering.py            <- Algorithm 1 Phase 2: semantic noise filtering
    train_dual_criterion.py          <- Main training pipeline (proposed method)
    train_baseline_gce.py            <- Baseline: GCE (Zhang & Sabuncu, NeurIPS 2018)
    train_baseline_focal.py          <- Baseline: Focal Loss (Lin et al., ICCV 2017)
    train_baseline_curriculum.py     <- Baseline: Curriculum Learning (Bengio et al., ICML 2009)
    train_baseline_coteaching.py     <- Baseline: Co-teaching (Han et al., NeurIPS 2018)
    train_baseline_label_smooth.py   <- Baseline: Label Smoothing
    train_baseline_reweight.py       <- Baseline: Sample Reweighting
    train_baseline_random_remove.py  <- Control: Random Removal (12.1% matched fraction)
    evaluate.py                      <- Evaluation: accuracy, F1, sensitivity, specificity, AUC
    evaluate_ood.py                  <- OOD evaluation on external datasets (UR Fall, SBU, Leeds)
    generate_tables.py               <- Reproduce all 10 manuscript tables
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset

The complete original dataset (5,841 images) is included in the `dataset/` directory with pre-organized train/val/test splits. No additional download or reconstruction is needed.

For external OOD evaluation datasets, see `DATA_ACCESS.md` for download URLs and licenses.

### 3. Train proposed method

```bash
python scripts/train_dual_criterion.py --data_dir ./dataset --seed 42 --epochs 100
```

### 4. Train all baselines (seeds 42-46)

```bash
for seed in 42 43 44 45 46; do
    python scripts/train_dual_criterion.py --data_dir ./dataset --seed $seed
    python scripts/train_baseline_gce.py --data_dir ./dataset --seed $seed
    python scripts/train_baseline_focal.py --data_dir ./dataset --seed $seed
    python scripts/train_baseline_curriculum.py --data_dir ./dataset --seed $seed
    python scripts/train_baseline_coteaching.py --data_dir ./dataset --seed $seed
    python scripts/train_baseline_label_smooth.py --data_dir ./dataset --seed $seed
    python scripts/train_baseline_reweight.py --data_dir ./dataset --seed $seed
    python scripts/train_baseline_random_remove.py --data_dir ./dataset --seed $seed
done
```

### 5. Evaluate

```bash
python scripts/evaluate.py --data_dir ./dataset --checkpoint_dir ./checkpoints
```

### 6. External OOD evaluation

```bash
python scripts/evaluate_ood.py --dataset urfall --data_dir ./ur_fall --checkpoint_dir ./checkpoints
python scripts/evaluate_ood.py --dataset sbu --data_dir ./sbu --checkpoint_dir ./checkpoints
```

### 7. Generate manuscript tables

```bash
python scripts/generate_tables.py --output paper_tables.tex
```

---

## Artifact-to-Manuscript Mapping

| # | Artifact | Manuscript Reference | File |
|---|----------|---------------------|------|
| 1 | Original dataset (5,841 images) | Section 3, 4 | `dataset/` (train/val/test splits) |
| 2 | Split manifest | Section 4.1, Table 3 | `data/split_manifest.csv` |
| 3 | Physical noise scores | Section 3.3, Tables 1--2 | `data/physical_scores.csv` |
| 4 | Physical filtering script | Algorithm 1, Phase 1 | `scripts/physical_filtering.py` |
| 5 | Semantic labels (removed) | Section 3.4 | `data/semantic_labels_removed.csv` |
| 6 | Semantic labels (retained) | Section 5.6, Table 10 | `data/semantic_labels_retained.csv` |
| 7 | Inter-rater agreement | Section 3.4 | `data/interrater_agreement.csv` |
| 8 | Training scripts (all) | Section 4.2 | `scripts/train_*.py` |
| 9 | Evaluation script | Section 5 | `scripts/evaluate.py` |
| 10 | OOD evaluation script | Section 5.5, Table 9 | `scripts/evaluate_ood.py` |
| 11 | SBU frame manifest | Section 5.5 | `data/sbu_eval_manifest.csv` |
| 12 | Table generation | All tables | `scripts/generate_tables.py` |

---

## Expected Runtime

- Training (single seed, single method): ~30 min on NVIDIA RTX 3060
- Full reproduction (9 methods x 5 seeds): ~22 hours on single GPU
- Evaluation only: <5 min

---

## Notes

- The complete original dataset (5,841 images) is included in the `dataset/` directory
- All baseline scripts import shared utilities from `train_dual_criterion.py`
- Random seed controls both data shuffling and weight initialization
