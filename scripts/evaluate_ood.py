#!/usr/bin/env python3
"""
Out-of-Distribution Evaluation Script
======================================
Evaluates trained models on external datasets (UR Fall, SBU Killbot)
without threshold retuning.

Reproduces Table 6 in the manuscript:
  - UR Fall Detection (external RGB): 87.3% accuracy
  - Leeds Millennium (same-source held-out): 84.7% accuracy
  - SBU Killbot (external depth): 76.2% accuracy

Usage:
    python evaluate_ood.py --dataset urfall --data_dir <path> --checkpoint_dir <path>
    python evaluate_ood.py --dataset sbu --data_dir <path> --checkpoint_dir <path>
    python evaluate_ood.py --dataset leeds --data_dir <path> --checkpoint_dir <path>
    python evaluate_ood.py --dataset leeds --data_dir <path> --checkpoint_dir <path>

For SBU reproduction:
    1. Obtain dataset from SBU CV Lab (see DATA_ACCESS.md)
    2. Extract frames listed in data/sbu_eval_manifest.csv
    3. Run: python evaluate_ood.py --dataset sbu --data_dir <sbu_path> --checkpoint_dir <ckpt_path>
"""

import argparse
import os
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from PIL import Image


# Same model architecture as train_dual_criterion.py
def load_model(checkpoint_path, device):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(576, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


class OODDataset(Dataset):
    """Dataset loader for out-of-distribution evaluation.

    Expects directory structure:
        data_dir/
            fall/    (or positive/)
            normal/  (or negative/ or adl/)
    """

    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # For SBU depth images: convert single-channel depth to 3-channel
    DEPTH_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, data_dir, depth_mode=False, manifest_path=None):
        self.data_dir = data_dir
        self.depth_mode = depth_mode
        self.samples = []

        if manifest_path and os.path.exists(manifest_path):
            # Load specific frames from manifest
            with open(manifest_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fpath = os.path.join(data_dir, row["frame_id"])
                    if os.path.exists(fpath):
                        label = 1 if row["label"] == "fall" else 0
                        self.samples.append((fpath, label))
        else:
            # Auto-discover from directory structure
            for label_name, label_val in [
                ("fall", 1), ("positive", 1),
                ("normal", 0), ("negative", 0), ("adl", 0)
            ]:
                label_dir = os.path.join(data_dir, label_name)
                if os.path.isdir(label_dir):
                    for fname in sorted(os.listdir(label_dir)):
                        if fname.lower().endswith((".jpg", ".png", ".bmp")):
                            self.samples.append(
                                (os.path.join(label_dir, fname), label_val))

        print(f"  Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.depth_mode:
            img = self.DEPTH_TRANSFORM(img)
        else:
            img = self.TRANSFORM(img)
        return img, label


@torch.no_grad()
def predict(model, loader, device):
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        outputs = torch.sigmoid(model(imgs.to(device)).squeeze())
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
        all_probs.extend(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def evaluate_dataset(name, data_dir, checkpoint_dir, seeds, device,
                     depth_mode=False, manifest_path=None, batch_size=32):
    """Evaluate on a single OOD dataset across all seeds."""
    print(f"\n{'='*60}")
    print(f"OOD Evaluation: {name}")
    print(f"{'='*60}")

    dataset = OODDataset(data_dir, depth_mode=depth_mode,
                         manifest_path=manifest_path)
    if len(dataset) == 0:
        print(f"  ERROR: No samples found in {data_dir}")
        return None

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    results = []

    for seed in seeds:
        ckpt = os.path.join(checkpoint_dir, f"best_seed{seed}.pth")
        if not os.path.exists(ckpt):
            print(f"  Checkpoint not found: {ckpt}")
            continue
        model = load_model(ckpt, device)
        probs, labels = predict(model, loader, device)
        acc = accuracy_score(labels, (probs > 0.5).astype(int))
        f1 = f1_score(labels, (probs > 0.5).astype(int))
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
        results.append({"accuracy": acc, "f1": f1, "auc": auc})
        print(f"  Seed {seed}: acc={acc*100:.1f}% f1={f1*100:.1f}% "
              f"auc={auc*100:.1f}%")

    if results:
        for key in ["accuracy", "f1", "auc"]:
            vals = [r[key] * 100 for r in results]
            print(f"  {key}: {np.mean(vals):.1f}% +/- {np.std(vals):.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Out-of-distribution evaluation for fall detection")
    parser.add_argument("--dataset", required=True,
                        choices=["urfall", "sbu", "leeds", "all"],
                        help="Which OOD dataset to evaluate")
    parser.add_argument("--data_dir", required=True,
                        help="Root directory of the OOD dataset")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing seed checkpoints")
    parser.add_argument("--manifest",
                        help="Path to frame manifest CSV (for SBU)")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]

    datasets_config = {
        "urfall": {
            "name": "UR Fall Detection (External RGB)",
            "depth_mode": False,
            "manifest_path": None,
        },
        "sbu": {
            "name": "SBU Killbot (External Depth → Grayscale)",
            "depth_mode": True,
            "manifest_path": args.manifest or os.path.join(
                os.path.dirname(__file__), "..", "data",
                "sbu_eval_manifest.csv"),
        },
        "leeds": {
            "name": "Leeds Millennium (Same-Source Held-Out RGB)",
            "depth_mode": False,
            "manifest_path": None,
        },
    }

    if args.dataset == "all":
        for ds_key, ds_cfg in datasets_config.items():
            evaluate_dataset(
                ds_cfg["name"], args.data_dir, args.checkpoint_dir,
                seeds, device, depth_mode=ds_cfg["depth_mode"],
                manifest_path=ds_cfg["manifest_path"],
                batch_size=args.batch_size)
    else:
        cfg = datasets_config[args.dataset]
        evaluate_dataset(
            cfg["name"], args.data_dir, args.checkpoint_dir,
            seeds, device, depth_mode=cfg["depth_mode"],
            manifest_path=cfg["manifest_path"],
            batch_size=args.batch_size)


if __name__ == "__main__":
    main()
