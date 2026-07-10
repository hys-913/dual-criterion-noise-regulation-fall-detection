#!/usr/bin/env python3
"""
Dual-Criterion Benchmark Training Pipeline
==========================================
Main training script for the protocol released in the reviewer package.

The package ships the released benchmark split directly. This script trains the
proposed MobileNetV3-Small configuration under the shared optimizer, early
stopping, and evaluation protocol described in the manuscript.

Usage:
    python train_dual_criterion.py --data_dir <path> --seed 42 --epochs 100
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, roc_auc_score, confusion_matrix)
from PIL import Image, ImageOps


class ResizePad:
    """Aspect-ratio preserving resize followed by symmetric padding."""
    def __init__(self, size, fill=0):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.fill = fill

    def __call__(self, img):
        target_w, target_h = self.size
        img = ImageOps.contain(img, self.size, method=Image.BILINEAR)
        pad_w = target_w - img.size[0]
        pad_h = target_h - img.size[1]
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        return ImageOps.expand(img, (left, top, right, bottom), fill=self.fill)


def build_transform(augment=False, resize_mode="stretch"):
    if resize_mode == "letterbox":
        ops = [ResizePad((224, 224))]
        if augment:
            ops.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=5, translate=(0.03, 0.03),
                                        scale=(0.95, 1.05), fill=0),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            ])
    else:
        if augment:
            ops = [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            ]
        else:
            ops = [transforms.Resize((224, 224))]

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(ops)


class FallDataset(Dataset):
    def __init__(self, root, split="train", augment=False, resize_mode="stretch"):
        self.root = os.path.join(root, split)
        self.samples = []
        for label_idx, cls in enumerate(["normal", "fall"]):
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label_idx))

        self.transform = build_transform(augment=augment, resize_mode=resize_mode)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_model(pretrained=True, head_width=128):
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier = nn.Sequential(
        nn.Linear(576, head_width),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(head_width, 1),
    )
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs).squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        outputs = torch.sigmoid(model(imgs.to(device)).squeeze())
        all_probs.extend(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    sens = recall_score(labels, preds)            # sensitivity = recall
    spec = recall_score(labels, preds, pos_label=0)  # specificity
    auc = roc_auc_score(labels, probs)

    return {"accuracy": acc, "f1": f1, "sensitivity": sens,
            "specificity": spec, "auc": auc, "probs": probs, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Root directory with train/val/test splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--resize_mode", choices=["stretch", "letterbox"],
                        default="stretch",
                        help="stretch keeps the original benchmark preprocessing; "
                             "letterbox preserves posture aspect ratio by resize+pad")
    parser.add_argument("--pretrained", choices=["imagenet", "none"],
                        default="imagenet",
                        help="Backbone initialization")
    parser.add_argument("--head_width", type=int, default=128,
                        help="Classifier hidden width used in the released protocol")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {args.seed} | Resize: {args.resize_mode} | "
          f"Head: {args.head_width}")

    train_ds = FallDataset(args.data_dir, "train", augment=True,
                           resize_mode=args.resize_mode)
    val_ds = FallDataset(args.data_dir, "val", resize_mode=args.resize_mode)
    test_ds = FallDataset(args.data_dir, "test", resize_mode=args.resize_mode)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = get_model(pretrained=args.pretrained == "imagenet",
                      head_width=args.head_width).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
              f"val_acc {val_metrics['accuracy']:.4f} | "
              f"val_f1 {val_metrics['f1']:.4f} | {elapsed:.1f}s")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            ckpt_path = os.path.join(args.output_dir, f"best_seed{args.seed}.pth")
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(
        os.path.join(args.output_dir, f"best_seed{args.seed}.pth"),
        map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "=" * 50)
    print(f"TEST RESULTS (seed={args.seed})")
    print(f"  Accuracy:    {test_metrics['accuracy']*100:.1f}%")
    print(f"  F1:          {test_metrics['f1']*100:.1f}%")
    print(f"  Sensitivity: {test_metrics['sensitivity']*100:.1f}%")
    print(f"  Specificity: {test_metrics['specificity']*100:.1f}%")
    print(f"  AUC:         {test_metrics['auc']*100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
