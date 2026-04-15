# Data Access Protocol

**Paper:** Dual-Criterion Noise Regulation for Capacity-Limited Visual Classification: A Fall Detection Case Study

---

## 1. Source Datasets

This study uses four publicly available datasets. No new human-subject data was collected.

### 1.1 Leeds Millennium Dataset (Primary)

| Field | Value |
|-------|-------|
| Role | Training, validation, and test benchmark (RGB frames) |
| Provider | School of Computing, University of Leeds |
| Access | Available upon request from the dataset maintainers. We obtained the dataset through direct correspondence with the original authors; access is typically granted within 2 weeks of request. |
| License | Research use only; no redistribution without permission |
| Redistribution | The source images are included in the `dataset/` directory of this supplementary package under the research-use license for the purpose of peer review and reproducibility |
| Content | RGB and depth video sequences of daily activities and fall events, collected under informed consent with ethical approval from the University of Leeds |
| Our usage | RGB frames extracted at 1 fps; split into train/val/test per `data/split_manifest.csv` |
| **Reviewer reproduction** | Reviewers can evaluate all results directly using the images in `dataset/`. For independent access, contact the Leeds dataset maintainers. |

### 1.2 COCO 2017 (Negative-class augmentation)

| Field | Value |
|-------|-------|
| Role | Negative-class (non-fall) samples for training diversity |
| URL | https://cocodataset.org/#download |
| License | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Content | 330K images, 80 object categories |
| Our usage | Person-containing images sampled as non-fall examples; sample IDs in `data/split_manifest.csv` with `source_dataset=coco2017` |

### 1.3 UR Fall Detection Dataset (External RGB evaluation)

| Field | Value |
|-------|-------|
| Role | Truly external out-of-distribution evaluation (RGB) |
| URL | http://fenix.ur.edu.pl/~mkepski/ds/uf.html |
| Reference | Kwolek & Kepski, Computer Methods and Programs in Biomedicine, 2014 |
| License | Research use only |
| Content | 70 sequences (30 falls, 40 ADL) from two Microsoft Kinect cameras |
| Our usage | RGB frames only; evaluated with no threshold retuning (Table 6: 87.3% accuracy) |
| **Not used in benchmark construction** | This dataset was held out entirely for external validation |

### 1.4 SBU Killbot Fall Dataset (External depth evaluation)

| Field | Value |
|-------|-------|
| Role | Truly external out-of-distribution evaluation (cross-modality depth) |
| Access | Available from SBU Computer Vision Lab upon request. We obtained the dataset through the lab's academic access program. |
| License | Research use only |
| Redistribution | Not redistributed in this supplementary package due to license restrictions. Evaluation frames used in our study are listed in `data/sbu_eval_manifest.csv` (frame IDs only, no pixel data) |
| Content | Depth map sequences of fall and non-fall activities captured with Microsoft Kinect |
| Our usage | Depth frames converted to grayscale; evaluated with no threshold retuning (Table 6: 76.2% accuracy) |
| **Not used in benchmark construction** | This dataset was held out entirely for cross-modality validation |
| **Reviewer reproduction** | To reproduce SBU results: (1) obtain dataset from SBU CV Lab, (2) extract frames listed in `data/sbu_eval_manifest.csv`, (3) run `scripts/evaluate_ood.py --dataset sbu` |

---

## 2. Benchmark Reconstruction

The benchmark used in this paper can be exactly reconstructed from `data/split_manifest.csv`:

```
Step 1: Obtain source datasets (see URLs above)
Step 2: Extract frames per source dataset instructions
Step 3: Use split_manifest.csv to assign each frame to train/val/test
        - Column 'sample_id': relative path within the source dataset
        - Column 'source_dataset': which source dataset the frame belongs to
        - Column 'split': train / val / test
        - Column 'label': fall / normal
Step 4: Apply physical filtering (scripts/physical_filtering.py)
        - Thresholds: blur > 15 px, occlusion > 40%, entropy < 2.0 bits
        - Scores for all retained samples: data/physical_scores.csv
Step 5: Apply semantic filtering (scripts/semantic_filtering.py)
        - Removed samples: data/semantic_labels_removed.csv (320 samples)
        - Retained hard negatives: data/semantic_labels_retained.csv (156 samples)
```

**Note**: The complete dataset (5,841 images with train/val/test splits) is included in the `dataset/` directory of this supplementary package. The split manifest (`data/split_manifest.csv`) provides full traceability of each sample to its original source dataset.

---

## 3. Annotation Protocol

- **Annotators**: Two domain experts with computer vision background
- **Task**: Assign semantic noise category (theatrical / occupational / transitional) to ambiguous samples
- **Agreement**: Cohen's kappa = 0.79 (substantial agreement)
- **Raw data**: `data/interrater_agreement.csv` (476 annotated samples)
- **Rubric**: Detailed annotation guidelines are described in manuscript Section 3.4

---

## 4. Reproducibility Commitment

- All experiments use 5 random seeds (42, 43, 44, 45, 46)
- Mean +/- SD reported for all metrics
- Training scripts with exact hyperparameters are provided in `scripts/`
- Model checkpoints (when provided) correspond to seed 42
- Code and all artifacts will be released on GitHub upon paper acceptance

---

## 5. Ethical Considerations

- All source datasets were obtained through their official distribution channels
- No personally identifiable information was used or generated
- Fall detection is a safety-critical application; we clearly state frame-level classification limitations and do not claim system-level performance
- Deployment metrics are reported for transparency, not as deployment recommendations

---

## 6. Contact

For questions about data access or reproduction:

**Corresponding Author:** Meian Li
**Email:** limeian1973@126.com
**Affiliation:** School of Computer and Information Engineering, Inner Mongolia Agricultural University
