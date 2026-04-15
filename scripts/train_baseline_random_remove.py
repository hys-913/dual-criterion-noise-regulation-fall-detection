#!/usr/bin/env python3
"""
Control: Random Removal (12.1% matched fraction)
=================================================
Removes 12.1% of training data uniformly at random to match
the fraction removed by dual-criterion regulation.
Confirms that gains stem from targeted regulation, not dataset reduction.

Usage:
    python train_baseline_random_remove.py --data_dir <path> --seed 42
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


REMOVE_FRACTION = 0.121  # 12.1%, matching dual-criterion removal rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--remove_frac", type=float, default=REMOVE_FRACTION)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output_dir", default="./checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = FallDataset(args.data_dir, "train", augment=True)
    val_ds = FallDataset(args.data_dir, "val")
    test_ds = FallDataset(args.data_dir, "test")

    # Random removal
    n = len(train_ds)
    n_keep = int(n * (1 - args.remove_frac))
    indices = np.random.permutation(n)[:n_keep].tolist()
    train_subset = Subset(train_ds, indices)
    print(f"Random removal: {n} -> {n_keep} ({args.remove_frac*100:.1f}% removed)")

    train_loader = DataLoader(train_subset, args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, args.batch_size, num_workers=2)
    test_loader = DataLoader(test_ds, args.batch_size, num_workers=2)

    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier = nn.Sequential(
        nn.Linear(576, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1, patience_ctr = 0, 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            criterion(model(imgs).squeeze(), labels).backward()
            optimizer.step()

        val = evaluate(model, val_loader, device)
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            patience_ctr = 0
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"random_rm_seed{args.seed}.pth"))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                break

    model.load_state_dict(torch.load(
        os.path.join(args.output_dir, f"random_rm_seed{args.seed}.pth"),
        map_location=device))
    t = evaluate(model, test_loader, device)
    print(f"RandomRemoval ({args.remove_frac*100:.1f}%) seed={args.seed}: "
          f"acc={t['accuracy']*100:.1f}% f1={t['f1']*100:.1f}% auc={t['auc']*100:.1f}%")


if __name__ == "__main__":
    main()
