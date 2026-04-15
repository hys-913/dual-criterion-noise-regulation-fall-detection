#!/usr/bin/env python3
"""
Physical Noise Filtering (Algorithm 1, Phase 1)
================================================
Filters images with severe sensory degradation:
  - Motion blur > BLUR_THRESH px
  - Occlusion ratio > OCC_THRESH
  - Illumination entropy < ENT_THRESH bits

Usage:
    python physical_filtering.py --data_dir <path> --manifest ../data/split_manifest.csv
"""

import argparse
import csv
import os

import cv2
import numpy as np


# ── Default thresholds (Table 8 in manuscript) ──
BLUR_THRESH = 15.0      # pixels
OCC_THRESH = 0.40        # ratio
ENT_THRESH = 2.0         # bits


def compute_blur_score(img_gray: np.ndarray) -> float:
    """Laplacian variance as blur proxy (lower = blurrier)."""
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(lap.var())


def blur_to_px(laplacian_var: float, img_width: int = 224) -> float:
    """Heuristic mapping: Laplacian variance → effective blur in pixels."""
    if laplacian_var < 1e-6:
        return float(img_width)
    return img_width / np.sqrt(laplacian_var)


def compute_occlusion_ratio(img_gray: np.ndarray) -> float:
    """Fraction of near-zero pixels (proxy for occlusion / black patches)."""
    return float(np.mean(img_gray < 10))


def compute_illumination_entropy(img_gray: np.ndarray) -> float:
    """Shannon entropy of the grayscale histogram (bits)."""
    hist, _ = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def score_image(path: str):
    """Return (blur_px, occlusion, entropy) for one image."""
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224))

    lap_var = compute_blur_score(gray)
    blur_px = blur_to_px(lap_var)
    occ = compute_occlusion_ratio(gray)
    ent = compute_illumination_entropy(gray)
    return blur_px, occ, ent


def should_remove(blur_px, occ, ent):
    return blur_px > BLUR_THRESH or occ > OCC_THRESH or ent < ENT_THRESH


def main():
    parser = argparse.ArgumentParser(description="Physical noise filtering")
    parser.add_argument("--data_dir", required=True, help="Root of image dataset")
    parser.add_argument("--manifest", default="../data/split_manifest.csv")
    parser.add_argument("--output", default="../data/physical_scores.csv")
    parser.add_argument("--blur_thresh", type=float, default=BLUR_THRESH)
    parser.add_argument("--occ_thresh", type=float, default=OCC_THRESH)
    parser.add_argument("--ent_thresh", type=float, default=ENT_THRESH)
    args = parser.parse_args()

    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    results = []
    removed_count = 0
    for row in samples:
        img_path = os.path.join(args.data_dir, row["sample_id"])
        scores = score_image(img_path)
        if scores is None:
            continue
        blur_px, occ, ent = scores
        rm = should_remove(blur_px, occ, ent)
        if rm:
            removed_count += 1
        results.append({
            "sample_id": row["sample_id"],
            "blur_score_px": f"{blur_px:.2f}",
            "occlusion_ratio": f"{occ:.3f}",
            "illumination_entropy_bits": f"{ent:.3f}",
            "removed": str(rm),
        })

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print(f"Scored {len(results)} images, removed {removed_count} "
          f"({removed_count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
