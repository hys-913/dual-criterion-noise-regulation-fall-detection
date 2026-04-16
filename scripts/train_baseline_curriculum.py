#!/usr/bin/env python3
"""
Baseline: Curriculum Learning
=============================
Bengio et al., ICML 2009.
Samples ordered by loss magnitude; easy samples first.

Usage:
    python train_baseline_curriculum.py --data_dir <path> --seed 42
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models

from train_dual_criterion import FallDataset, evaluate


def compute_sample_difficulty(model, dataset, device):
    """Return per-sample loss as difficulty proxy."""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    losses = []
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            out = model(imgs).squeeze()
            losses.extend(criterion(out, labels).cpu().numpy())
    return np.array(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output_dir", default="./checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = FallDataset(args.data_dir, "train", augment=True)
    val_ds = FallDataset(args.data_dir, "val")
    test_ds = FallDataset(args.data_dir, "test")
    val_loader = DataLoader(val_ds, args.batch_size, num_workers=2)
    test_loader = DataLoader(test_ds, args.batch_size, num_workers=2)

    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier = nn.Sequential(
        nn.Linear(576, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    model = model.to(device)

    # Initial difficulty ranking
    difficulty = compute_sample_difficulty(model, train_ds, device)
    sorted_idx = np.argsort(difficulty)  # easy first

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=1e-4)

    best_f1, patience_ctr = 0, 0
    os.makedirs(args.output_dir, exist_ok=True)
    n = len(train_ds)

    for epoch in range(1, args.epochs + 1):
        # Gradually increase data fraction (linear pacing)
        frac = min(1.0, 0.3 + 0.7 * epoch / args.epochs)
        k = int(frac * n)
        subset = Subset(train_ds, sorted_idx[:k].tolist())
        loader = DataLoader(subset, args.batch_size, shuffle=True, num_workers=2)

        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            criterion(model(imgs).squeeze(), labels).backward()
            optimizer.step()

        val = evaluate(model, val_loader, device)
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            patience_ctr = 0
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"curriculum_seed{args.seed}.pth"))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                break

    model.load_state_dict(torch.load(
        os.path.join(args.output_dir, f"curriculum_seed{args.seed}.pth"),
        map_location=device))
    t = evaluate(model, test_loader, device)
    print(f"Curriculum seed={args.seed}: "
          f"acc={t['accuracy']*100:.1f}% f1={t['f1']*100:.1f}% auc={t['auc']*100:.1f}%")


if __name__ == "__main__":
    main()
