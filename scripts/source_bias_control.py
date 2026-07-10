#!/usr/bin/env python3
"""
Source-shortcut diagnostic for the released fall-presence benchmark.

The benchmark combines Leeds Millennium fall/normal frames with COCO normal
frames. Because COCO contributes only non-fall images, a reviewer may ask
whether a classifier can obtain high raw accuracy by recognizing the source
dataset rather than the fall state. This script makes that risk explicit.

It reports:
  1. label/source counts for each split;
  2. a source-only diagnostic rule:
       leeds_millennium -> fall, coco2017 -> normal;
  3. optional per-source metrics for a model prediction CSV.

Prediction CSV format, when supplied:
    sample_id,prob_fall

or:
    sample_id,pred_label

The script does not require torch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


POS_LABEL = "fall"
NEG_LABEL = "normal"


def _as_binary(labels: Iterable[str]) -> np.ndarray:
    return np.array([1 if str(v).lower() == POS_LABEL else 0 for v in labels])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "n": 0,
            "accuracy": np.nan,
            "f1": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "balanced_accuracy": np.nan,
            "tp": 0,
            "fn": 0,
            "fp": 0,
            "tn": 0,
        }

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return {
        "n": int(len(y_true)),
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


def source_only_prediction(df: pd.DataFrame) -> np.ndarray:
    """Predict fall for Leeds samples and normal for COCO samples."""
    return np.where(df["source_dataset"].eq("leeds_millennium"), 1, 0)


def summarize_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["split", "source_dataset", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "source_dataset", "label"])
    )
    return counts


def summarize_source_rule(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test", "all"]:
        part = df if split == "all" else df[df["split"].eq(split)]
        y_true = _as_binary(part["label"])
        y_pred = source_only_prediction(part)
        row = {"split": split, "rule": "leeds_millennium=>fall; coco2017=>normal"}
        row.update(_metrics(y_true, y_pred))
        rows.append(row)
    return pd.DataFrame(rows)


def model_predictions(df: pd.DataFrame, pred_csv: Path, threshold: float) -> pd.DataFrame:
    pred = pd.read_csv(pred_csv)
    if "sample_id" not in pred.columns:
        raise ValueError("Prediction CSV must contain a sample_id column.")

    merged = df.merge(pred, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) == 0:
        raise ValueError("No prediction rows matched the manifest sample_id values.")

    if "prob_fall" in merged.columns:
        y_pred = (merged["prob_fall"].astype(float).to_numpy() >= threshold).astype(int)
    elif "pred_label" in merged.columns:
        y_pred = _as_binary(merged["pred_label"])
    else:
        raise ValueError("Prediction CSV must contain prob_fall or pred_label.")

    merged = merged.copy()
    merged["pred_binary"] = y_pred

    rows = []
    subsets = {
        "all_test": merged[merged["split"].eq("test")],
        "leeds_test": merged[merged["split"].eq("test") & merged["source_dataset"].eq("leeds_millennium")],
        "coco_test": merged[merged["split"].eq("test") & merged["source_dataset"].eq("coco2017")],
        "all_splits": merged,
    }
    for name, part in subsets.items():
        y_true = _as_binary(part["label"])
        y_hat = part["pred_binary"].to_numpy(dtype=int)
        row = {"subset": name}
        row.update(_metrics(y_true, y_hat))
        rows.append(row)
    return pd.DataFrame(rows)


def write_same_source_manifest(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    test = df[df["split"].eq("test")]
    same_source_parts = []
    for source, part in test.groupby("source_dataset"):
        if part["label"].nunique() == 2:
            same_source_parts.append(part)
    if not same_source_parts:
        return None
    out = pd.concat(same_source_parts).sort_values(["source_dataset", "label", "sample_id"])
    out_path = out_dir / "same_source_test_manifest.csv"
    out.to_csv(out_path, index=False)
    return out_path


def write_markdown_report(
    counts: pd.DataFrame,
    source_rule: pd.DataFrame,
    out_path: Path,
    model_metrics: Optional[pd.DataFrame] = None,
) -> None:
    test_row = source_rule[source_rule["split"].eq("test")].iloc[0]
    lines = [
        "# Source-Shortcut Diagnostic",
        "",
        "## Why This Check Is Needed",
        "",
        "The released split contains Leeds Millennium fall and normal images, while COCO2017 contributes normal images only. "
        "A model could therefore appear strong if it learns source-specific cues. This diagnostic quantifies that shortcut.",
        "",
        "## Source-Only Rule",
        "",
        "Rule: `leeds_millennium => fall`, `coco2017 => normal`.",
        "",
        (
            f"On the held-out test split this rule obtains {test_row['accuracy'] * 100:.1f}% raw accuracy, "
            f"{test_row['sensitivity'] * 100:.1f}% sensitivity, {test_row['specificity'] * 100:.1f}% specificity, "
            f"and {test_row['balanced_accuracy'] * 100:.1f}% balanced accuracy."
        ),
        "",
        "The high raw accuracy shows that source shortcuts are a real confound. The low specificity shows the shortcut fails on Leeds normal images.",
        "",
        "## Source/Label Counts",
        "",
        counts.to_markdown(index=False),
        "",
        "## Source-Only Metrics",
        "",
        source_rule.to_markdown(index=False, floatfmt=".4f"),
    ]
    if model_metrics is not None:
        lines.extend([
            "",
            "## Model Metrics by Source Subset",
            "",
            model_metrics.to_markdown(index=False, floatfmt=".4f"),
        ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="../data/split_manifest.csv")
    parser.add_argument("--output_dir", default="../data")
    parser.add_argument("--predictions_csv", help="Optional sample_id/prob_fall or sample_id/pred_label CSV")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)
    required = {"sample_id", "source_dataset", "split", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    counts = summarize_counts(df)
    source_rule = summarize_source_rule(df)

    counts_path = out_dir / "source_label_counts.csv"
    source_rule_path = out_dir / "source_only_diagnostic.csv"
    counts.to_csv(counts_path, index=False)
    source_rule.to_csv(source_rule_path, index=False)

    same_source_path = write_same_source_manifest(df, out_dir)

    model_metrics = None
    if args.predictions_csv:
        model_metrics = model_predictions(df, Path(args.predictions_csv), args.threshold)
        model_metrics.to_csv(out_dir / "source_bias_model_metrics.csv", index=False)

    write_markdown_report(
        counts,
        source_rule,
        out_dir / "source_bias_control_report.md",
        model_metrics=model_metrics,
    )

    print(f"Wrote {counts_path}")
    print(f"Wrote {source_rule_path}")
    if same_source_path:
        print(f"Wrote {same_source_path}")
    if model_metrics is not None:
        print(f"Wrote {out_dir / 'source_bias_model_metrics.csv'}")
    print(f"Wrote {out_dir / 'source_bias_control_report.md'}")


if __name__ == "__main__":
    main()
