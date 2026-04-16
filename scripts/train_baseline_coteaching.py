#!/usr/bin/env python3
"""
Baseline: Co-teaching
=====================
Han et al., NeurIPS 2018.
Two networks teach each other by selecting small-loss samples.

Usage:
    python train_baseline_coteaching.py --data_dir <path> --seed 42
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


def build_model(device):
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier = nn.Sequential(
        nn.Linear(576, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    return model.to(device)


def co_teaching_step(model1, model2, imgs, labels, criterion, forget_rate):
    """Each network selects small-loss samples for the other."""
    out1 = model1(imgs).squeeze()
    out2 = model2(imgs).squeeze()
    loss1 = criterion(out1, labels)
    loss2 = criterion(out2, labels)

    n = len(labels)
    n_keep = max(1, int((1 - forget_rate) * n))

    # Network 1 selects for network 2
    _, idx1 = torch.topk(loss1, n_keep, largest=False)
    # Network 2 selects for network 1
    _, idx2 = torch.topk(loss2, n_keep, largest=False)

    loss_update1 = criterion(out1[idx2], labels[idx2]).mean()
    loss_update2 = criterion(out2[idx1], labels[idx1]).mean()
    return loss_update1, loss_update2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--forget_rate", type=float, default=0.2)
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

    model1 = build_model(device)
    model2 = build_model(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    opt1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=0.9,
                     weight_decay=1e-4)
    opt2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9,
                     weight_decay=1e-4)

    best_f1, patience_ctr = 0, 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model1.train()
        model2.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            loss1, loss2 = co_teaching_step(
                model1, model2, imgs, labels, criterion, args.forget_rate)
            opt1.zero_grad()
            loss1.backward()
            opt1.step()
            opt2.zero_grad()
            loss2.backward()
            opt2.step()

        val = evaluate(model1, val_loader, device)
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            patience_ctr = 0
            torch.save(model1.state_dict(),
                       os.path.join(args.output_dir, f"coteach_seed{args.seed}.pth"))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                break

    model1.load_state_dict(torch.load(
        os.path.join(args.output_dir, f"coteach_seed{args.seed}.pth"),
        map_location=device))
    t = evaluate(model1, test_loader, device)
    print(f"Co-teaching seed={args.seed}: "
          f"acc={t['accuracy']*100:.1f}% f1={t['f1']*100:.1f}% auc={t['auc']*100:.1f}%")


if __name__ == "__main__":
    main()
