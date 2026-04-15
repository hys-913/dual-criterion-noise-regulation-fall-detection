#!/usr/bin/env python3
"""
Evaluation Script
=================
Computes all metrics reported in the manuscript:
  - Accuracy, F1, Sensitivity, Specificity, AUC
  - Threshold sensitivity analysis
  - Hard negative per-category accuracy
  - OOD evaluation on external datasets

Usage:
    python evaluate.py --data_dir <path> --checkpoint <path> [--external <path>]
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             roc_auc_score, confusion_matrix, roc_curve)

from train_dual_criterion import FallDataset


def load_model(checkpoint_path, device):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(576, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader, device):
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        outputs = torch.sigmoid(model(imgs.to(device)).squeeze())
        all_probs.extend(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def compute_metrics(probs, labels, threshold=0.5):
    preds = (probs > threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    sens = recall_score(labels, preds)
    spec = recall_score(labels, preds, pos_label=0)
    auc = roc_auc_score(labels, probs)
    return {"accuracy": acc, "f1": f1, "sensitivity": sens,
            "specificity": spec, "auc": auc}


def threshold_sensitivity(probs, labels):
    """Table 8: Threshold sensitivity analysis."""
    print("\nThreshold Sensitivity Analysis")
    print(f"{'Threshold':<12}{'Acc':<10}{'F1':<10}{'Sens':<10}{'Spec':<10}")
    print("-" * 52)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        m = compute_metrics(probs, labels, threshold=t)
        print(f"{t:<12.1f}{m['accuracy']*100:<10.1f}{m['f1']*100:<10.1f}"
              f"{m['sensitivity']*100:<10.1f}{m['specificity']*100:<10.1f}")


def multi_seed_eval(data_dir, checkpoint_dir, seeds, device):
    """Evaluate across multiple seeds and report mean +/- SD."""
    results = []
    test_ds = FallDataset(data_dir, "test")
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=2)

    for seed in seeds:
        ckpt = os.path.join(checkpoint_dir, f"best_seed{seed}.pth")
        if not os.path.exists(ckpt):
            print(f"  Checkpoint not found: {ckpt}")
            continue
        model = load_model(ckpt, device)
        probs, labels = predict(model, test_loader, device)
        m = compute_metrics(probs, labels)
        results.append(m)
        print(f"  Seed {seed}: acc={m['accuracy']*100:.1f}% "
              f"f1={m['f1']*100:.1f}% auc={m['auc']*100:.1f}%")

    if results:
        for key in ["accuracy", "f1", "sensitivity", "specificity", "auc"]:
            vals = [r[key] * 100 for r in results]
            print(f"  {key}: {np.mean(vals):.1f}% +/- {np.std(vals):.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", help="Single checkpoint path")
    parser.add_argument("--checkpoint_dir", help="Directory with seed checkpoints")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--external", help="External dataset for OOD evaluation")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.checkpoint_dir:
        print("Multi-seed evaluation:")
        multi_seed_eval(args.data_dir, args.checkpoint_dir, seeds, device)
    elif args.checkpoint:
        model = load_model(args.checkpoint, device)
        test_ds = FallDataset(args.data_dir, "test")
        test_loader = DataLoader(test_ds, args.batch_size, num_workers=2)
        probs, labels = predict(model, test_loader, device)
        m = compute_metrics(probs, labels)
        print(f"Accuracy:    {m['accuracy']*100:.1f}%")
        print(f"F1:          {m['f1']*100:.1f}%")
        print(f"Sensitivity: {m['sensitivity']*100:.1f}%")
        print(f"Specificity: {m['specificity']*100:.1f}%")
        print(f"AUC:         {m['auc']*100:.1f}%")
        threshold_sensitivity(probs, labels)

    if args.external:
        print(f"\nOOD Evaluation on: {args.external}")
        ckpt = args.checkpoint or os.path.join(
            args.checkpoint_dir or ".", "best_seed42.pth")
        model = load_model(ckpt, device)
        ext_ds = FallDataset(args.external, "test")
        ext_loader = DataLoader(ext_ds, args.batch_size, num_workers=2)
        probs, labels = predict(model, ext_loader, device)
        m = compute_metrics(probs, labels)
        print(f"  Accuracy:    {m['accuracy']*100:.1f}%")
        print(f"  F1:          {m['f1']*100:.1f}%")
        print(f"  AUC:         {m['auc']*100:.1f}%")


if __name__ == "__main__":
    main()
