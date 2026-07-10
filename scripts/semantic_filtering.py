#!/usr/bin/env python3
"""
Semantic-audit summary and leakage check.

The semantic stage is an offline audit of visually clear but label-ambiguous
training candidates. It is not an inference-time module and it does not remove
validation or test images. The released reviewer package intentionally stores
anonymized audit identifiers rather than path-resolved image names, so this
script verifies the released audit registers and reports category counts and
annotator agreement without requiring torch.

Released CSVs:
  - semantic_labels_removed.csv: candidates recommended for exclusion
  - semantic_labels_retained.csv: retained hard negatives
  - interrater_agreement.csv: second-pass category agreement
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


EXPECTED_CATEGORIES = {"theatrical", "occupational", "transitional"}


def category_counts(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = (
        df.groupby("category")
        .size()
        .reindex(sorted(EXPECTED_CATEGORIES), fill_value=0)
        .reset_index(name="count")
    )
    out.insert(0, "register", name)
    return out


def check_anonymized_ids(df: pd.DataFrame, id_col: str) -> bool:
    """Return True when no released semantic ID exposes train/val/test paths."""
    return not df[id_col].astype(str).str.contains(r"/|\\|train|val|test", case=False, regex=True).any()


def agreement_summary(agreement_df: pd.DataFrame) -> pd.DataFrame:
    raw_agreement = agreement_df["agreed"].astype(str).str.lower().eq("true").mean()
    kappa = cohen_kappa_score(agreement_df["expert1_category"], agreement_df["expert2_category"])
    return pd.DataFrame([{
        "reviewed_candidates": len(agreement_df),
        "raw_agreement": raw_agreement,
        "cohens_kappa": kappa,
    }])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--removed", default="../data/semantic_labels_removed.csv")
    parser.add_argument("--retained", default="../data/semantic_labels_retained.csv")
    parser.add_argument("--agreement", default="../data/interrater_agreement.csv")
    parser.add_argument("--output_dir", default="../data")
    args = parser.parse_args()

    removed = pd.read_csv(args.removed)
    retained = pd.read_csv(args.retained)
    agreement = pd.read_csv(args.agreement)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in [("removed", removed), ("retained", retained)]:
        missing = EXPECTED_CATEGORIES - set(df["category"])
        if missing:
            raise ValueError(f"{name} register is missing categories: {sorted(missing)}")
        if not check_anonymized_ids(df, "sample_id"):
            raise ValueError(f"{name} register exposes split/path-like identifiers.")

    counts = pd.concat([
        category_counts(removed, "excluded"),
        category_counts(retained, "retained_hard_negative"),
    ], ignore_index=True)
    agreement_out = agreement_summary(agreement)

    counts_path = out_dir / "semantic_audit_summary.csv"
    agreement_path = out_dir / "semantic_agreement_summary.csv"
    counts.to_csv(counts_path, index=False)
    agreement_out.to_csv(agreement_path, index=False)

    print("Semantic audit summary")
    print(counts.to_string(index=False))
    print("\nAgreement")
    print(agreement_out.to_string(index=False))
    print("\nLeakage check: semantic registers use anonymized IDs; no train/val/test paths are exposed.")
    print(f"Wrote {counts_path}")
    print(f"Wrote {agreement_path}")


if __name__ == "__main__":
    main()
