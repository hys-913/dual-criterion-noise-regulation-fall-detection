#!/usr/bin/env python3
"""
Minimal k-of-n event-level aggregation for sequence predictions.

The manuscript studies single-frame fall-presence classification. For deployment,
a practical edge camera can reduce false alarms by triggering an event only
when at least k of n recent frames are positive. This script implements that
post-processing rule without changing the frame classifier.

Two modes are provided:

1. Real event-level evaluation from model predictions:
       python event_level_kofn.py --manifest ../data/sbu_eval_manifest.csv \
           --predictions_csv predictions.csv --output_csv ../data/event_level_kofn.csv

   predictions.csv must contain:
       frame_id,prob_fall
   or:
       frame_id,pred_label

2. A transparent projection from measured per-frame sensitivity/specificity:
       python event_level_kofn.py --projection \
           --frame_sensitivity 0.923 --frame_specificity 0.887

The projection is not a substitute for sequence-level validation; it is a
deployment calculation that shows the expected effect of k-of-n confirmation
under an independent-frame assumption.
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


POS_LABEL = "fall"


def _as_binary(labels: Iterable[str]) -> np.ndarray:
    return np.array([1 if str(v).lower() == POS_LABEL else 0 for v in labels])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return {
        "events": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def parse_k_values(text: str) -> List[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def write_template(manifest: pd.DataFrame, path: Path) -> None:
    out = manifest.copy()
    out["prob_fall"] = ""
    out.to_csv(path, index=False)


def load_predictions(manifest: pd.DataFrame, pred_path: Path, threshold: float) -> pd.DataFrame:
    pred = pd.read_csv(pred_path)
    if "frame_id" not in pred.columns:
        raise ValueError("Prediction CSV must contain frame_id.")
    merged = manifest.merge(pred, on="frame_id", how="inner", validate="one_to_one")
    if len(merged) == 0:
        raise ValueError("No prediction rows matched the manifest frame_id values.")
    if "prob_fall" in merged.columns:
        merged["frame_pred"] = (merged["prob_fall"].astype(float) >= threshold).astype(int)
    elif "pred_label" in merged.columns:
        merged["frame_pred"] = _as_binary(merged["pred_label"])
    else:
        raise ValueError("Prediction CSV must contain prob_fall or pred_label.")
    return merged


def detection_frame_for_group(group: pd.DataFrame, k: int) -> float:
    positives = group[group["frame_pred"].eq(1)].sort_values("frame_number")
    if len(positives) < k:
        return np.nan
    return float(positives.iloc[k - 1]["frame_number"])


def evaluate_kofn(pred_df: pd.DataFrame, k_values: List[int]) -> pd.DataFrame:
    rows = []
    for k in k_values:
        event_labels = []
        event_preds = []
        detection_frames = []
        for sequence, group in pred_df.groupby("sequence"):
            group = group.sort_values("frame_number")
            n = len(group)
            kk = min(k, n)
            label = group["label"].iloc[0]
            event_pred = int(group["frame_pred"].sum() >= kk)
            event_labels.append(1 if label == POS_LABEL else 0)
            event_preds.append(event_pred)
            if label == POS_LABEL and event_pred == 1:
                detection_frames.append(detection_frame_for_group(group, kk))

        y_true = np.array(event_labels, dtype=int)
        y_pred = np.array(event_preds, dtype=int)
        row = {
            "rule": f"{k}-of-n",
            "k": k,
            "n_min": int(pred_df.groupby("sequence").size().min()),
            "n_max": int(pred_df.groupby("sequence").size().max()),
        }
        row.update(_metrics(y_true, y_pred))
        row["false_alarm_rate_per_normal_sequence"] = (
            row["fp"] / (row["fp"] + row["tn"]) if (row["fp"] + row["tn"]) else np.nan
        )
        row["median_detection_frame"] = float(np.nanmedian(detection_frames)) if detection_frames else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def projected_event_rates(sensitivity: float, specificity: float, n: int, k_values: List[int]) -> pd.DataFrame:
    fpr = 1.0 - specificity
    rows = []
    for k in k_values:
        event_sens = sum(comb(n, i) * sensitivity**i * (1 - sensitivity) ** (n - i) for i in range(k, n + 1))
        event_fpr = sum(comb(n, i) * fpr**i * (1 - fpr) ** (n - i) for i in range(k, n + 1))
        rows.append({
            "rule": f"{k}-of-{n}",
            "k": k,
            "n": n,
            "frame_sensitivity": sensitivity,
            "frame_specificity": specificity,
            "projected_event_sensitivity": event_sens,
            "projected_event_specificity": 1.0 - event_fpr,
            "projected_false_alarm_probability_per_window": event_fpr,
            "assumption": "independent frame errors; projection, not sequence validation",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="../data/sbu_eval_manifest.csv")
    parser.add_argument("--predictions_csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k_values", default="1,2,3")
    parser.add_argument("--output_csv", default="../data/event_level_kofn.csv")
    parser.add_argument("--write_template", action="store_true")
    parser.add_argument("--template_csv", default="../data/event_level_prediction_template.csv")
    parser.add_argument("--projection", action="store_true")
    parser.add_argument("--frame_sensitivity", type=float, default=0.923)
    parser.add_argument("--frame_specificity", type=float, default=0.887)
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    required = {"frame_id", "sequence", "label", "frame_number"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    k_values = parse_k_values(args.k_values)

    if args.write_template:
        template_path = Path(args.template_csv)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        write_template(manifest, template_path)
        print(f"Wrote {template_path}")

    if args.projection:
        out = projected_event_rates(args.frame_sensitivity, args.frame_specificity, args.window, k_values)
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")
        return

    if not args.predictions_csv:
        counts = manifest.groupby("label")["sequence"].nunique().reset_index(name="sequences")
        frames = manifest.groupby("label").size().reset_index(name="frames")
        summary = counts.merge(frames, on="label")
        print(summary.to_string(index=False))
        print("\nNo predictions supplied. Use --write_template to create a prediction template,")
        print("or use --projection for a transparent k-of-n deployment calculation.")
        return

    pred_df = load_predictions(manifest, Path(args.predictions_csv), args.threshold)
    out = evaluate_kofn(pred_df, k_values)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
