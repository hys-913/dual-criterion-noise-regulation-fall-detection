#!/usr/bin/env python3
"""
Semantic Audit Reference Implementation (Algorithm 1, Phase 2)
==============================================================
Demonstrates the thresholding logic used for semantically ambiguous samples:
  - Theatrical performances (non-physiological postures)
  - Occupational activities (mimicking fall kinematics)
  - Transitional states (sitting-to-lying, standing-to-sitting)

The reviewer package ships anonymized semantic-audit tables rather than a
path-resolved post-exclusion directory tree. This script therefore serves as a
reference implementation for the leave-one-out thresholding logic used during
the study rather than a one-command replay of the released semantic audit.

Usage:
    python semantic_filtering.py \
        --data_dir <path> \
        --labels ../data/semantic_labels_removed.csv \
        --retained ../data/semantic_labels_retained.csv \
        --f1_thresh 0.5
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from sklearn.metrics import f1_score
from PIL import Image


F1_THRESH = 0.5  # percentage-point improvement threshold


class FallDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = os.path.join(root, split)
        self.samples = []
        for label_idx, cls in enumerate(["normal", "fall"]):
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label_idx))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def quick_train_eval(train_loader, val_loader, device, epochs=5):
    """Train MobileNetV3-Small briefly and return validation F1."""
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier = nn.Sequential(
        nn.Linear(576, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
    )
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9,
                          weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            outputs = torch.sigmoid(model(imgs.to(device)).squeeze())
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_labels.extend(labels.numpy())
    return f1_score(all_labels, all_preds, average="binary") * 100


def semantic_leave_one_out(data_dir, candidate_indices, f1_thresh, device):
    """For each candidate, check if removing it improves val F1 > threshold."""
    train_ds = FallDataset(data_dir, "train")
    val_ds = FallDataset(data_dir, "val")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Baseline F1 with all samples
    full_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    baseline_f1 = quick_train_eval(full_loader, val_loader, device)
    print(f"Baseline validation F1: {baseline_f1:.2f}%")

    removed, retained = [], []
    for idx in candidate_indices:
        keep = list(range(len(train_ds)))
        keep.remove(idx)
        subset = Subset(train_ds, keep)
        loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=0)
        new_f1 = quick_train_eval(loader, val_loader, device)
        delta = new_f1 - baseline_f1
        if delta > f1_thresh:
            removed.append((idx, delta))
        else:
            retained.append((idx, delta))

    return removed, retained, baseline_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--labels", default="../data/semantic_labels_removed.csv")
    parser.add_argument("--retained", default="../data/semantic_labels_retained.csv")
    parser.add_argument("--f1_thresh", type=float, default=F1_THRESH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Semantic filtering with F1 threshold = {args.f1_thresh}%")
    print(f"Device: {device}")
    print("NOTE: The reviewer package distributes anonymized semantic-audit "
          "registers. This script documents the reference thresholding logic; "
          "the released CSVs support count verification and category analysis.")


if __name__ == "__main__":
    main()
