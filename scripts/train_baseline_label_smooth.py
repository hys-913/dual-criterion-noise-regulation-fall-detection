#!/usr/bin/env python3
"""
Baseline: Label Smoothing
=========================
Soft targets with smoothing factor epsilon.

Usage:
    python train_baseline_label_smooth.py --data_dir <path> --seed 42
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from train_dual_criterion import FallDataset, evaluate


class LabelSmoothingBCE(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        smoothed = targets * (1 - self.epsilon) + 0.5 * self.epsilon
        return self.bce(logits.squeeze(), smoothed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output_dir", default="./checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = FallDataset(args.data_dir, "train", augment=True)
    val_ds = FallDataset(args.data_dir, "val")
    test_ds = FallDataset(args.data_dir, "test")
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, args.batch_size, num_workers=2)
    test_loader = DataLoader(test_ds, args.batch_size, num_workers=2)

    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier = nn.Sequential(
        nn.Linear(576, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    model = model.to(device)

    criterion = LabelSmoothingBCE(epsilon=args.epsilon)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=1e-4)

    best_f1, patience_ctr = 0, 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            criterion(model(imgs), labels).backward()
            optimizer.step()

        val = evaluate(model, val_loader, device)
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            patience_ctr = 0
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"labelsmooth_seed{args.seed}.pth"))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                break

    model.load_state_dict(torch.load(
        os.path.join(args.output_dir, f"labelsmooth_seed{args.seed}.pth"),
        map_location=device))
    t = evaluate(model, test_loader, device)
    print(f"LabelSmoothing (eps={args.epsilon}) seed={args.seed}: "
          f"acc={t['accuracy']*100:.1f}% f1={t['f1']*100:.1f}% auc={t['auc']*100:.1f}%")


if __name__ == "__main__":
    main()
