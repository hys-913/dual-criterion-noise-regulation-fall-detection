#!/usr/bin/env python3
"""
Generate audit-summary tables from the bundled reviewer artifacts.

This utility inspects the released benchmark split plus the physical/semantic
audit CSV files and emits compact LaTeX tables for supplementary use.
"""

import argparse
import csv
import os
from collections import Counter, defaultdict


def count_benchmark(dataset_dir):
    counts = defaultdict(int)
    for split in ("train", "val", "test"):
        for label in ("fall", "normal"):
            folder = os.path.join(dataset_dir, split, label)
            if not os.path.isdir(folder):
                continue
            counts[(split, label)] = sum(
                1 for name in os.listdir(folder)
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            )
    return counts


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_split_table(counts):
    split_totals = {}
    for split in ("train", "val", "test"):
        fall = counts.get((split, "fall"), 0)
        normal = counts.get((split, "normal"), 0)
        split_totals[split] = (fall, normal, fall + normal)

    return rf"""
\begin{{table}}[tb]
\caption{{Released benchmark split bundled in the reviewer package.}}
\label{{tab:artifact_split}}
\centering
\begin{{tabular}}{{lccc}}
\toprule
Split & Fall & Non-fall & Total \\
\midrule
Train & {split_totals["train"][0]} & {split_totals["train"][1]} & {split_totals["train"][2]} \\
Validation & {split_totals["val"][0]} & {split_totals["val"][1]} & {split_totals["val"][2]} \\
Test & {split_totals["test"][0]} & {split_totals["test"][1]} & {split_totals["test"][2]} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def build_physical_table(rows):
    flag_key = "quality_flag" if rows and "quality_flag" in rows[0] else "removed"
    by_split = Counter()
    total_flagged = 0
    for row in rows:
        flagged = str(row.get(flag_key, "")).lower() == "true"
        if flagged:
            total_flagged += 1
            by_split[row.get("split", "unknown")] += 1

    return rf"""
\begin{{table}}[tb]
\caption{{Physical-quality audit summary from \texttt{{physical\_scores.csv}}.}}
\label{{tab:artifact_physical}}
\centering
\begin{{tabular}}{{lcc}}
\toprule
Scope & Samples & Flagged \\
\midrule
Train & {sum(1 for r in rows if r.get("split") == "train")} & {by_split["train"]} \\
Validation & {sum(1 for r in rows if r.get("split") == "val")} & {by_split["val"]} \\
Test & {sum(1 for r in rows if r.get("split") == "test")} & {by_split["test"]} \\
\midrule
All benchmark images & {len(rows)} & {total_flagged} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def build_semantic_table(removed_rows, retained_rows):
    removed_by_category = Counter(row.get("category", "unknown") for row in removed_rows)
    retained_by_category = Counter(row.get("category", "unknown") for row in retained_rows)
    categories = sorted(set(removed_by_category) | set(retained_by_category))

    body = []
    for category in categories:
        body.append(
            f"{category.title()} & {removed_by_category[category]} & "
            f"{retained_by_category[category]} \\\\"
        )

    body_str = "\n".join(body)
    return rf"""
\begin{{table}}[tb]
\caption{{Semantic-audit register summary from the bundled reviewer CSV files.}}
\label{{tab:artifact_semantic}}
\centering
\begin{{tabular}}{{lcc}}
\toprule
Category & Exclusion recommendations & Retained hard negatives \\
\midrule
{body_str}
\midrule
Total & {len(removed_rows)} & {len(retained_rows)} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def write_markdown(path, counts, physical_rows, removed_rows, retained_rows):
    flag_key = "quality_flag" if physical_rows and "quality_flag" in physical_rows[0] else "removed"
    total_flagged = sum(1 for row in physical_rows if str(row.get(flag_key, "")).lower() == "true")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Audit Summary\n\n")
        handle.write("## Benchmark split\n")
        for split in ("train", "val", "test"):
            fall = counts.get((split, "fall"), 0)
            normal = counts.get((split, "normal"), 0)
            handle.write(f"- {split}: {fall} fall, {normal} non-fall, {fall + normal} total\n")

        handle.write("\n## Physical audit\n")
        handle.write(f"- Scored samples: {len(physical_rows)}\n")
        handle.write(f"- Flagged by default thresholds: {total_flagged}\n")

        handle.write("\n## Semantic audit\n")
        handle.write(f"- Exclusion recommendations: {len(removed_rows)}\n")
        handle.write(f"- Retained hard negatives: {len(retained_rows)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact_root",
        default=os.path.join(os.path.dirname(__file__), ".."),
        help="Root of the supplementary artifact package",
    )
    parser.add_argument("--output", default="./paper_tables.tex")
    parser.add_argument("--summary", default="./artifact_summary.md")
    args = parser.parse_args()

    artifact_root = os.path.abspath(args.artifact_root)
    dataset_dir = os.path.join(artifact_root, "dataset")
    data_dir = os.path.join(artifact_root, "data")

    counts = count_benchmark(dataset_dir)
    physical_rows = load_csv(os.path.join(data_dir, "physical_scores.csv"))
    removed_rows = load_csv(os.path.join(data_dir, "semantic_labels_removed.csv"))
    retained_rows = load_csv(os.path.join(data_dir, "semantic_labels_retained.csv"))

    tables = [
        build_split_table(counts),
        build_physical_table(physical_rows),
        build_semantic_table(removed_rows, retained_rows),
    ]

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("%% Auto-generated audit-summary tables\n\n")
        for table in tables:
            handle.write(table.strip())
            handle.write("\n\n")

    write_markdown(args.summary, counts, physical_rows, removed_rows, retained_rows)
    print(f"Wrote {len(tables)} tables to {args.output}")
    print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
